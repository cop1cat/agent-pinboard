from __future__ import annotations

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

from pinboard import Entity, fact, make_graph_tools, node


IP = Entity(name="IP", description="ipv4/ipv6 address")
User = Entity(name="User", description="acting principal")


class CloudTrailEvent(BaseModel):
    src_ip: str | None = node(type=IP, description="src ip", default=None)
    actor: str | None = node(type=User, description="who", default=None)
    action: str | None = Field(default=None, description="api action")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_graph(extra_tools, store):
    @fact(model=CloudTrailEvent)
    @tool
    def fetch(value: str, runtime: ToolRuntime) -> dict:
        """Returns one event."""
        return {"src_ip": value, "actor": "admin", "action": "AssumeRole"}

    all_tools = [fetch, *extra_tools]
    g = StateGraph(State)
    g.add_node("seed", lambda s: {})
    g.add_node("tools", ToolNode(all_tools))
    g.add_edge(START, "seed")
    g.add_edge("seed", "tools")
    g.add_edge("tools", END)
    return g.compile(store=store), {t.name: t for t in all_tools}


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


class TestGraphSummaryDiscovery:
    def test_known_types_visible_before_ingestion(self, store: InMemoryStore) -> None:
        """README §16 AC8 — discovery without ingestion."""
        graph_tools = make_graph_tools()
        graph, by_name = _build_graph(graph_tools, store)
        out = _call(graph, "graph_summary", {})
        assert "IP" in out
        assert "User" in out
        assert "0 in graph" in out

    def test_after_ingestion_shows_counts(self, store: InMemoryStore) -> None:
        graph_tools = make_graph_tools()
        graph, _ = _build_graph(graph_tools, store)
        _call(graph, "fetch", {"value": "1.2.3.4"})
        out = _call(graph, "graph_summary", {})
        assert "IP (1 in graph)" in out
        assert "User (1 in graph)" in out


class TestSearchNodes:
    def test_lists_facts_and_hides_events_by_default(self, store: InMemoryStore) -> None:
        graph_tools = make_graph_tools()
        graph, _ = _build_graph(graph_tools, store)
        _call(graph, "fetch", {"value": "1.2.3.4"})
        out = _call(graph, "search_nodes", {"node_type": "IP"})
        assert "1.2.3.4" in out
        # No events shown by default.
        out_events = _call(graph, "search_nodes", {"node_type": "Event"})
        assert "no matches" in out_events.lower()
        out_with = _call(graph, "search_nodes", {"node_type": "Event", "include_events": True})
        assert "Event" in out_with

    def test_value_pattern_glob(self, store: InMemoryStore) -> None:
        graph_tools = make_graph_tools()
        graph, _ = _build_graph(graph_tools, store)
        _call(graph, "fetch", {"value": "1.2.3.4"})
        out = _call(graph, "search_nodes", {"node_type": "IP", "value_pattern": "1.*"})
        assert "1.2.3.4" in out


class TestExploreSkipEvents:
    def test_skip_events_shows_facts_directly(self, store: InMemoryStore) -> None:
        graph_tools = make_graph_tools()
        graph, _ = _build_graph(graph_tools, store)
        _call(graph, "fetch", {"value": "1.2.3.4"})
        out = _call(graph, "explore", {"node_type": "IP", "value": "1.2.3.4"})
        # Other fact (User) is reachable via the shared event without an event hop.
        assert "User: admin" in out
        # Description from node() is rendered on the edge.
        assert "who" in out


class TestTimeline:
    def test_lists_events(self, store: InMemoryStore) -> None:
        graph_tools = make_graph_tools()
        graph, _ = _build_graph(graph_tools, store)
        _call(graph, "fetch", {"value": "1.2.3.4"})
        out = _call(graph, "timeline", {"node_type": "IP", "value": "1.2.3.4"})
        assert "fetch" in out


class TestWhatHaveIDoneSemantics:
    def test_value_without_node_type_rejected(self, store: InMemoryStore) -> None:
        graph_tools = make_graph_tools()
        graph, _ = _build_graph(graph_tools, store)
        _call(graph, "fetch", {"value": "1.2.3.4"})
        out = _call(graph, "what_have_i_done", {"value": "1.2.3.4"})
        assert "value" in out and "node_type" in out

    def test_filter_by_tool_name(self, store: InMemoryStore) -> None:
        graph_tools = make_graph_tools()
        graph, _ = _build_graph(graph_tools, store)
        _call(graph, "fetch", {"value": "1.2.3.4"})
        out = _call(graph, "what_have_i_done", {"tool_name": "fetch"})
        assert "fetch(" in out

    def test_filter_by_entity(self, store: InMemoryStore) -> None:
        graph_tools = make_graph_tools()
        graph, _ = _build_graph(graph_tools, store)
        _call(graph, "fetch", {"value": "1.2.3.4"})
        _call(graph, "fetch", {"value": "5.5.5.5"})
        out = _call(graph, "what_have_i_done", {"node_type": "IP", "value": "1.2.3.4"})
        # Only the first call referenced 1.2.3.4
        assert "1.2.3.4" in out
        assert "5.5.5.5" not in out


class TestMissingNode:
    def test_explore_unknown_returns_hint(self, store: InMemoryStore) -> None:
        graph_tools = make_graph_tools()
        graph, _ = _build_graph(graph_tools, store)
        out = _call(graph, "explore", {"node_type": "IP", "value": "9.9.9.9"})
        assert "No node found" in out
        assert "search_nodes" in out
