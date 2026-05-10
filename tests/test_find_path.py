from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from typing_extensions import TypedDict

from pinboard import Entity, fact, make_graph_tools, node


IP = Entity(name="IP", description="ipv4/ipv6 address")
User = Entity(name="User", description="acting principal")


class Event(BaseModel):
    src_ip: str | None = node(type=IP, description="src ip", default=None)
    dst_ip: str | None = node(type=IP, description="dst ip", default=None)
    actor: str | None = node(type=User, description="who", default=None)


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


def _call(graph, name, args, *, call_id="c"):
    out = graph.invoke(
        {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        )]},
        config={"configurable": {"thread_id": "tid"}},
    )
    return out["messages"][-1].content


@fact(model=Event)
@tool
def fetch(value: str, runtime: ToolRuntime) -> dict:
    """."""
    # The value-string maps to a fixed scenario via prefix.
    if value == "ab":
        return {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2"}
    if value == "bc":
        return {"src_ip": "2.2.2.2", "dst_ip": "3.3.3.3"}
    if value == "ad":
        return {"src_ip": "1.1.1.1", "actor": "alice"}
    if value == "de":
        return {"actor": "alice", "dst_ip": "3.3.3.3"}
    return {"src_ip": value}


class TestFindPathBasic:
    def test_one_hop(self, store: InMemoryStore) -> None:
        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"value": "ab"}, call_id="1")
        out = _call(graph, "find_path", {
            "from_type": "IP", "from_value": "1.1.1.1",
            "to_type": "IP", "to_value": "2.2.2.2",
        }, call_id="2")
        assert "Path 1 (1 hop)" in out
        assert "1.1.1.1" in out and "2.2.2.2" in out

    def test_two_hops_chain(self, store: InMemoryStore) -> None:
        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"value": "ab"}, call_id="1")
        _call(graph, "fetch", {"value": "bc"}, call_id="2")
        out = _call(graph, "find_path", {
            "from_type": "IP", "from_value": "1.1.1.1",
            "to_type": "IP", "to_value": "3.3.3.3",
        }, call_id="3")
        assert "Path 1 (2 hops)" in out
        assert "2.2.2.2" in out  # midpoint

    def test_no_path_returns_message_not_exception(self, store: InMemoryStore) -> None:
        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"value": "ab"}, call_id="1")
        # 9.9.9.9 not in graph
        out = _call(graph, "find_path", {
            "from_type": "IP", "from_value": "1.1.1.1",
            "to_type": "IP", "to_value": "9.9.9.9",
        }, call_id="2")
        assert "No node found" in out

    def test_disconnected_facts(self, store: InMemoryStore) -> None:
        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"value": "ab"}, call_id="1")
        _call(graph, "fetch", {"value": "bc"}, call_id="2")
        # 1.1.1.1 → ?? → 5.5.5.5 doesn't exist
        _call(graph, "fetch", {"value": "5.5.5.5"}, call_id="3")
        out = _call(graph, "find_path", {
            "from_type": "IP", "from_value": "1.1.1.1",
            "to_type": "IP", "to_value": "5.5.5.5",
        }, call_id="4")
        assert "no path" in out.lower()


class TestFindPathTopN:
    def test_top_returns_multiple_paths(self, store: InMemoryStore) -> None:
        """Two events both link 1.1.1.1 to alice → two distinct paths of equal length."""
        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"value": "ab"}, call_id="1")
        _call(graph, "fetch", {"value": "bc"}, call_id="2")
        _call(graph, "fetch", {"value": "ad"}, call_id="3")
        _call(graph, "fetch", {"value": "de"}, call_id="4")
        out = _call(graph, "find_path", {
            "from_type": "IP", "from_value": "1.1.1.1",
            "to_type": "IP", "to_value": "3.3.3.3",
            "top": 5,
        }, call_id="5")
        # Two paths: through 2.2.2.2 and through alice.
        assert "Path 1" in out and "Path 2" in out


class TestFindPathSkipEvents:
    def test_skip_events_false_doubles_path_length(self, store: InMemoryStore) -> None:
        """Without skip_events, path goes fact → event → fact."""
        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"value": "ab"}, call_id="1")
        out = _call(graph, "find_path", {
            "from_type": "IP", "from_value": "1.1.1.1",
            "to_type": "IP", "to_value": "2.2.2.2",
            "skip_events": False, "max_depth": 3,
        }, call_id="2")
        # 1.1.1.1 → Event → 2.2.2.2 is two hops (the EventNode counts).
        assert "2 hops" in out
        assert "Event" in out


class TestFindPathSelfLoop:
    def test_same_node(self, store: InMemoryStore) -> None:
        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"value": "ab"}, call_id="1")
        out = _call(graph, "find_path", {
            "from_type": "IP", "from_value": "1.1.1.1",
            "to_type": "IP", "to_value": "1.1.1.1",
        }, call_id="2")
        assert "same node" in out


class TestFindPathDepthLimit:
    def test_max_depth_truncates(self, store: InMemoryStore) -> None:
        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"value": "ab"}, call_id="1")
        _call(graph, "fetch", {"value": "bc"}, call_id="2")
        out = _call(graph, "find_path", {
            "from_type": "IP", "from_value": "1.1.1.1",
            "to_type": "IP", "to_value": "3.3.3.3",
            "max_depth": 1,
        }, call_id="3")
        # The shortest path is 2 hops; max_depth=1 truncates, no result.
        assert "no path" in out.lower()
