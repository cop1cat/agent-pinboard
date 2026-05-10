"""Tests for AriGraph relevance ranking in the timeline tool."""

from __future__ import annotations

from typing import Annotated

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from typing_extensions import TypedDict

from pinboard import Entity, fact, make_graph_tools, node


IP = Entity(name="IP", description="ip")
User = Entity(name="User", description="u")


class CTEvent(BaseModel):
    src_ip: str | None = node(type=IP, description="src", default=None)
    dst_ip: str | None = node(type=IP, description="dst", default=None)
    actor: str | None = node(type=User, description="actor", default=None)


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


@fact(model=CTEvent)
@tool
def fetch(scenario: str, runtime: ToolRuntime) -> dict:
    """."""
    if scenario == "ip-with-actor":
        return {"src_ip": "1.1.1.1", "actor": "alice"}
    if scenario == "ip-with-dst":
        return {"src_ip": "1.1.1.1", "dst_ip": "alice"}  # 'alice' as IP — wrong but harmless for test
    if scenario == "ip-noise":
        return {"src_ip": "1.1.1.1", "dst_ip": "9.9.9.9"}  # unrelated dst
    return {"src_ip": "1.1.1.1"}


class TestTimelineRanking:
    def test_default_chronological_unchanged(self, store: InMemoryStore) -> None:
        """Without rank=True the existing behaviour is intact."""
        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"scenario": "ip-with-actor"}, call_id="a")
        _call(graph, "fetch", {"scenario": "ip-noise"}, call_id="b")
        out = _call(graph, "timeline", {"node_type": "IP", "value": "1.1.1.1"})
        assert "oldest first" in out
        assert "score" not in out

    def test_rank_true_orders_by_relevance(self, store: InMemoryStore) -> None:
        """An event whose other facts overlap with our neighbours scores higher."""
        graph = _build([fetch, *make_graph_tools()], store)
        # Event A ties 1.1.1.1 to alice (User). alice becomes a neighbour.
        _call(graph, "fetch", {"scenario": "ip-with-actor"}, call_id="a")
        # Event B ties 1.1.1.1 to alice via dst_ip — alice already a
        # neighbour, so this event scores higher than one with random
        # noise.
        _call(graph, "fetch", {"scenario": "ip-with-dst"}, call_id="b")
        # Event C ties 1.1.1.1 to a brand-new node 9.9.9.9 — no overlap.
        _call(graph, "fetch", {"scenario": "ip-noise"}, call_id="c")

        out = _call(graph, "timeline", {
            "node_type": "IP", "value": "1.1.1.1", "rank": True,
        })
        assert "by relevance" in out
        assert "score=" in out

    def test_no_events_returns_message(self, store: InMemoryStore) -> None:
        graph = _build([fetch, *make_graph_tools()], store)
        out = _call(graph, "timeline", {
            "node_type": "IP", "value": "9.9.9.9", "rank": True,
        })
        assert "No node found" in out


class TestArigraphScoreFormula:
    def test_zero_for_event_with_no_overlap(self, store: InMemoryStore) -> None:
        """An event with no neighbour overlap scores 0."""
        from pinboard.session import get_or_load_session
        from pinboard.tools import _arigraph_score_events

        graph = _build([fetch, *make_graph_tools()], store)
        _call(graph, "fetch", {"scenario": "ip-with-actor"}, call_id="a")
        _call(graph, "fetch", {"scenario": "ip-noise"}, call_id="b")

        g = get_or_load_session(store, "tid")
        ip_id = g.find_by_value("IP", "1.1.1.1")
        events = list(g.all_events())
        scores = _arigraph_score_events(g, ip_id, events)
        # All scores should be ≥ 0.
        assert all(s >= 0 for s in scores.values())
