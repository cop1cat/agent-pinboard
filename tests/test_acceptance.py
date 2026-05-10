"""Phase 1 acceptance tests, one per criterion in README §16.

Each ``test_acN_…`` corresponds directly to AC #N. If any of these
regress, Phase 1 is not done.
"""

from __future__ import annotations

import threading
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent_pinboard import (
    AgentPinBoardValidationError,
    Entity,
    OnDuplicate,
    make_graph_tools,
    node,
    pin,
)
from agent_pinboard.session import get_or_load_session

# --------------------------------------------------------------------------- #
# Common fixtures.                                                            #
# --------------------------------------------------------------------------- #

IP = Entity(name="IP", description="ipv4/ipv6 address")
User = Entity(name="User", description="user/account")


class Actor(BaseModel):
    user_arn: str | None = node(type=User, description="who", default=None)


class CloudTrailEvent(BaseModel):
    src_ip: str | None = node(type=IP, description="src ip", default=None)
    actor: Actor | None = None
    action: str | None = Field(default=None, description="action name")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _build(tools, store):
    g = StateGraph(State)
    g.add_node("seed", lambda s: {})
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "seed")
    g.add_edge("seed", "tools")
    g.add_edge("tools", END)
    return g.compile(store=store)


def _call(graph, name: str, args: dict, thread_id: str = "tid") -> str:
    out = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": args, "id": "c", "type": "tool_call"}],
                )
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    return out["messages"][-1].content


# --------------------------------------------------------------------------- #
# AC1 — extraction on nested model.                                           #
# --------------------------------------------------------------------------- #

def test_ac1_nested_extraction(store: InMemoryStore) -> None:
    @pin(model=CloudTrailEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {
            "src_ip": "1.2.3.4",
            "actor": {"user_arn": "arn:aws:iam::123:user/admin"},
            "action": "AssumeRole",
        }

    graph = _build([fetch], store)
    _call(graph, "fetch", {"value": "x"})

    g = get_or_load_session(store, "tid")
    # 1 EventNode, 1 IP, 1 User.
    assert len(list(g.search_by_type("Event"))) == 1
    assert len(list(g.search_by_type("IP"))) == 1
    assert len(list(g.search_by_type("User"))) == 1

    # Nested edge uses declaring class (Actor, not CloudTrailEvent).
    user_id = g.find_by_value("User", "arn:aws:iam::123:user/admin")
    assert user_id is not None
    event_id = list(g.search_by_type("Event"))[0]
    edges = g.edges_for_event(event_id)
    edge_types = {e.edge_type for e in edges}
    assert "Actor.user_arn" in edge_types
    assert "CloudTrailEvent.src_ip" in edge_types


# --------------------------------------------------------------------------- #
# AC2 — autolink + dedup across two tools.                                    #
# --------------------------------------------------------------------------- #

def test_ac2_autolink_and_dedup(store: InMemoryStore) -> None:
    class Report(BaseModel):
        ip: str = node(type=IP, description="queried")

    @pin(model=CloudTrailEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": "1.2.3.4", "action": "x"}

    @pin(model=Report)
    @tool
    def vt(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"ip": "1.2.3.4"}

    graph = _build([fetch, vt], store)
    _call(graph, "fetch", {"value": "a"})
    _call(graph, "vt", {"value": "b"})

    g = get_or_load_session(store, "tid")
    ip_ids = list(g.search_by_type("IP"))
    assert len(ip_ids) == 1, "same IP from two tools must be one node"
    n = g.get(ip_ids[0])
    assert n.source_tools == {"fetch", "vt"}  # type: ignore[union-attr]
    assert len(n.source_events) == 2  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# AC3 — recursion guard (eager-scan terminates on recursive models).          #
# --------------------------------------------------------------------------- #

def test_ac3_recursion_guard(store: InMemoryStore) -> None:
    class Process(BaseModel):
        pid: str | None = node(type=IP, description="pid", default=None)
        parent: Process | None = None

    Process.model_rebuild()

    @pin(model=Process)
    @tool
    def scan(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"pid": "1", "parent": {"pid": "2", "parent": None}}

    # If recursion guard is broken, decoration itself hangs.
    graph = _build([scan], store)
    _call(graph, "scan", {"value": "x"})
    g = get_or_load_session(store, "tid")
    assert len(list(g.search_by_type("IP"))) == 2


# --------------------------------------------------------------------------- #
# AC4 — concurrent ingestion: 10 parallel calls → 10 nodes, no losses.        #
# --------------------------------------------------------------------------- #

def test_ac4_concurrent_ingestion(store: InMemoryStore) -> None:
    """Drive @pin wrappers in parallel threads; rely on per-session RLock."""

    @pin(model=CloudTrailEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": value, "action": "x"}

    graph = _build([fetch], store)
    N = 10
    barrier = threading.Barrier(N)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=2)
            _call(graph, "fetch", {"value": f"10.0.0.{i}"}, thread_id="tid")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    g = get_or_load_session(store, "tid")
    assert len(list(g.search_by_type("IP"))) == N


# --------------------------------------------------------------------------- #
# AC5 — duplicate detection (skip): inner function not invoked on repeat.     #
# --------------------------------------------------------------------------- #

def test_ac5_duplicate_skip(store: InMemoryStore) -> None:
    invocations: list[int] = []

    class H:
        def on_ingest_complete(self, *_a, **_kw):
            invocations.append(0)

        def __getattr__(self, name):
            return lambda *a, **kw: None

    @pin(model=CloudTrailEvent, on_duplicate=OnDuplicate.SKIP, hooks=H())  # type: ignore[arg-type]
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        invocations.append(1)
        return {"src_ip": "1.2.3.4", "action": "x"}

    graph = _build([fetch], store)
    _call(graph, "fetch", {"value": "x"})
    _call(graph, "fetch", {"value": "x"})  # duplicate
    _call(graph, "fetch", {"value": "y"})  # different
    # Tool ran twice: first + third (second skipped). Hook fired only on real ingestion.
    assert invocations.count(1) == 2
    assert invocations.count(0) == 2


# --------------------------------------------------------------------------- #
# AC6 — fail-loud on validation: graph stays empty.                            #
# --------------------------------------------------------------------------- #

def test_ac6_fail_loud_validation(store: InMemoryStore) -> None:
    class Strict(BaseModel):
        src_ip: str = node(type=IP, description="src")  # required

    @pin(model=Strict)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"unrelated": 42}

    graph = _build([fetch], store)
    with pytest.raises(AgentPinBoardValidationError):
        _call(graph, "fetch", {"value": "x"})
    g = get_or_load_session(store, "tid")
    assert list(g.search_by_type("IP")) == []
    assert list(g.search_by_type("Event")) == []


# --------------------------------------------------------------------------- #
# AC7 — session isolation by thread_id.                                       #
# --------------------------------------------------------------------------- #

def test_ac7_session_isolation(store: InMemoryStore) -> None:
    @pin(model=CloudTrailEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": value, "action": "x"}

    graph = _build([fetch], store)
    _call(graph, "fetch", {"value": "1.1.1.1"}, thread_id="alpha")
    _call(graph, "fetch", {"value": "2.2.2.2"}, thread_id="beta")

    ga = get_or_load_session(store, "alpha")
    gb = get_or_load_session(store, "beta")
    a_ids = {ga.get(i).value for i in ga.search_by_type("IP")}  # type: ignore[union-attr]
    b_ids = {gb.get(i).value for i in gb.search_by_type("IP")}  # type: ignore[union-attr]
    assert a_ids == {"1.1.1.1"}
    assert b_ids == {"2.2.2.2"}


# --------------------------------------------------------------------------- #
# AC8 — discovery without ingestion: graph_summary shows known types.         #
# --------------------------------------------------------------------------- #

def test_ac8_discovery_without_ingestion(store: InMemoryStore) -> None:
    @pin(model=CloudTrailEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": "1.2.3.4"}

    tools = [fetch, *make_graph_tools()]
    graph = _build(tools, store)

    out = _call(graph, "graph_summary", {})
    # Both Entity types declared by the model are visible.
    assert "IP" in out
    assert "User" in out
    assert "0 in graph" in out
    # Their descriptions are surfaced for the LLM.
    assert "ipv4/ipv6" in out.lower()
