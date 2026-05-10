"""Regression tests for the issues found by Phase-1 code review.

Each test is named after the original review finding (B1–B9 / S1) so any
future regression points back at exactly which contract was broken.
"""

from __future__ import annotations

import logging
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

from langchain_core.callbacks import BaseCallbackHandler

import agent_pinboard
from agent_pinboard import (
    AgentPinBoardConfigError,
    Direction,
    Entity,
    make_graph_tools,
    node,
    pin,
)
from agent_pinboard.decorator import INGEST_EVENT

IP = Entity(name="IP", description="ipv4/ipv6", normalizer=lambda v: str(v).lower())
User = Entity(name="User", description="acting principal")


class Actor(BaseModel):
    user_arn: str | None = node(type=User, description="who", default=None)


class CTEvent(BaseModel):
    src_ip: str | None = node(type=IP, description="src ip", default=None)
    dst_ip: str | None = node(type=IP, description="dst ip", default=None)
    actor: Actor | None = None
    action: str | None = Field(default=None, description="action")


class _S(TypedDict):
    messages: Annotated[list, add_messages]


def _build(tools, store):
    g = StateGraph(_S)
    g.add_node("seed", lambda s: {})
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "seed")
    g.add_edge("seed", "tools")
    g.add_edge("tools", END)
    return g.compile(store=store)


def _call(
    graph,
    name: str,
    args: dict,
    thread_id: str = "tid",
    call_id: str = "c-1",
    callbacks: list | None = None,
) -> str:
    config: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    out = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
                )
            ]
        },
        config=config,
    )
    return out["messages"][-1].content


class _IngestRecorder(BaseCallbackHandler):
    """Test callback that records every agent_pinboard:ingest event payload."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def on_custom_event(self, name, data, *, run_id, tags=None, metadata=None, **kwargs):
        if name == INGEST_EVENT:
            self.events.append(data)


# --------------------------------------------------------------------------- #
# B1 — explore depth + direction.                                             #
# --------------------------------------------------------------------------- #

def test_b1_explore_depth_traverses_multi_hop(store: InMemoryStore) -> None:
    """Depth=2 must reach facts that share an event with a directly-related fact."""

    @pin(model=CTEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2"}

    @pin(model=CTEvent)
    @tool
    def other(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": "2.2.2.2", "dst_ip": "3.3.3.3"}

    graph_tools = make_graph_tools()
    graph = _build([fetch, other, *graph_tools], store)
    _call(graph, "fetch", {"value": "x"}, call_id="a")
    _call(graph, "other", {"value": "y"}, call_id="b")

    out_d1 = _call(graph, "explore", {"node_type": "IP", "value": "1.1.1.1", "depth": 1}, call_id="d1")
    out_d2 = _call(graph, "explore", {"node_type": "IP", "value": "1.1.1.1", "depth": 2}, call_id="d2")
    # depth=1 sees 2.2.2.2 (shared event with 1.1.1.1) but NOT 3.3.3.3
    assert "2.2.2.2" in out_d1
    assert "3.3.3.3" not in out_d1
    # depth=2 reaches 3.3.3.3 via 2.2.2.2
    assert "3.3.3.3" in out_d2


def test_b1_explore_depth_zero_is_just_start(store: InMemoryStore) -> None:
    @pin(model=CTEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2"}

    graph_tools = make_graph_tools()
    graph = _build([fetch, *graph_tools], store)
    _call(graph, "fetch", {"value": "x"})
    out = _call(graph, "explore", {"node_type": "IP", "value": "1.1.1.1", "depth": 0})
    assert "2.2.2.2" not in out
    assert "no related facts" in out.lower()


def test_b1_explore_direction_with_events_visible(store: InMemoryStore) -> None:
    """direction=IN with skip_events=False follows the inverse of edges."""

    @pin(model=CTEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": "1.1.1.1"}

    graph_tools = make_graph_tools()
    graph = _build([fetch, *graph_tools], store)
    _call(graph, "fetch", {"value": "x"})

    # IP has only inbound edges (from EventNode). With skip_events=False:
    #   - direction=IN finds the EventNode
    #   - direction=OUT finds nothing
    in_out = _call(graph, "explore", {
        "node_type": "IP", "value": "1.1.1.1",
        "skip_events": False, "direction": Direction.IN.value, "depth": 1,
    }, call_id="in")
    out_out = _call(graph, "explore", {
        "node_type": "IP", "value": "1.1.1.1",
        "skip_events": False, "direction": Direction.OUT.value, "depth": 1,
    }, call_id="out")
    assert "Event" in in_out
    assert "no related facts" in out_out.lower()


# --------------------------------------------------------------------------- #
# B2 — on_link_found fires when an existing fact is re-linked.                #
# --------------------------------------------------------------------------- #

def test_b2_linked_facts_surface_in_dispatched_event(store: InMemoryStore) -> None:
    """The second ingest of the same canonical fact reports it as linked
    in the dispatched ``agent_pinboard:ingest`` payload."""
    rec = _IngestRecorder()

    @pin(model=CTEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": value}

    graph = _build([fetch], store)
    _call(graph, "fetch", {"value": "1.1.1.1"}, call_id="a", callbacks=[rec])
    _call(graph, "fetch", {"value": "1.1.1.1"}, call_id="b", callbacks=[rec])
    # First ingest: src_ip is brand new, no linked facts.
    assert rec.events[0]["linked_facts"] == []
    # Second ingest: same canonical → reported as linked.
    linked_values = [f.value for f in rec.events[1]["linked_facts"]]
    assert linked_values == ["1.1.1.1"]


# --------------------------------------------------------------------------- #
# B3 — what_have_i_done filter normalises value via Entity.normalizer.        #
# --------------------------------------------------------------------------- #

def test_b3_what_have_i_done_normalises_value(store: InMemoryStore) -> None:
    """Filter should match by canonical form, not by raw display string."""

    @pin(model=CTEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": value}

    graph_tools = make_graph_tools()
    graph = _build([fetch, *graph_tools], store)
    # Stored canonical for IP normalizer is .lower().
    _call(graph, "fetch", {"value": "ABC"})
    out = _call(graph, "what_have_i_done", {"node_type": "IP", "value": "ABC"})
    assert "fetch(" in out
    out_lower = _call(graph, "what_have_i_done", {"node_type": "IP", "value": "abc"})
    assert "fetch(" in out_lower  # lower-case matches via canonical normalisation


# --------------------------------------------------------------------------- #
# B4 — node() rejects BaseModel-typed fields at registration time.            #
# --------------------------------------------------------------------------- #

def test_b4_node_on_basemodel_field_raises_at_registration() -> None:
    Person = Entity(name="Person", description="p")

    class Inner(BaseModel):
        x: str = "x"

    class Bad(BaseModel):
        inner: Inner = node(type=Person, description="nope", default_factory=Inner)

    with pytest.raises(AgentPinBoardConfigError, match="BaseModel"):
        @pin(model=Bad)
        @tool
        def t(value: str, runtime: ToolRuntime) -> dict:
            """."""
            return {}


# --------------------------------------------------------------------------- #
# B6 — linked_nodes counts distinct existing nodes, not edges.                #
# --------------------------------------------------------------------------- #

def test_b6_linked_nodes_dedup(store: InMemoryStore) -> None:
    """Two fields pointing to the same canonical → linked_nodes increments by 1."""

    rec = _IngestRecorder()

    @pin(model=CTEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """."""
        return {"src_ip": "1.1.1.1"}

    @pin(model=CTEvent)
    @tool
    def repeat(value: str, runtime: ToolRuntime) -> dict:
        """."""
        # Both src_ip and dst_ip canonicalise to "1.1.1.1" — one linked node.
        return {"src_ip": "1.1.1.1", "dst_ip": "1.1.1.1"}

    graph = _build([fetch, repeat], store)
    _call(graph, "fetch", {"value": "x"}, call_id="a", callbacks=[rec])
    _call(graph, "repeat", {"value": "y"}, call_id="b", callbacks=[rec])
    # First call: new node, linked=0. Second call: 0 new, 1 linked (deduped).
    assert [e["result"].linked_nodes for e in rec.events] == [0, 1]


# --------------------------------------------------------------------------- #
# B9 — extract.py no longer carries dead code.                                #
# --------------------------------------------------------------------------- #

def test_b9_no_dead_helper() -> None:
    from agent_pinboard import extract as ex_mod
    assert not hasattr(ex_mod, "_check_unsupported_dict")


# --------------------------------------------------------------------------- #
# S1 — public API does not advertise unimplemented hooks.                     #
# --------------------------------------------------------------------------- #

def test_s1_hooks_module_no_longer_exists() -> None:
    """Observability is wired through LangChain callbacks now —
    agent_pinboard.hooks was removed in PR #3."""
    with pytest.raises(ImportError):
        from agent_pinboard import hooks  # noqa: F401


# --------------------------------------------------------------------------- #
# Sanity — public API surface intact and importable.                          #
# --------------------------------------------------------------------------- #

def test_public_api_intact() -> None:
    for name in agent_pinboard.__all__:
        assert hasattr(agent_pinboard, name), name


# --------------------------------------------------------------------------- #
# Caplog smoke for fixed warnings (no behaviour change, just verifying).      #
# --------------------------------------------------------------------------- #

def test_register_model_basemodel_field_logs_or_raises(caplog: pytest.LogCaptureFixture) -> None:
    Person = Entity(name="Person2", description="p")

    class Inner(BaseModel):
        x: str = "x"

    class Bad(BaseModel):
        inner: Inner = node(type=Person, description="nope", default_factory=Inner)

    from agent_pinboard.registry import register_model

    with caplog.at_level(logging.WARNING), pytest.raises(AgentPinBoardConfigError):
        register_model(Bad)
