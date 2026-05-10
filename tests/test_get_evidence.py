"""Tests for get_evidence + @pin(store_raw=True)."""

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

from agent_pinboard import Entity, make_graph_tools, node, pin

IP = Entity(name="IP", description="ip")


class CTEvent(BaseModel):
    src_ip: str | None = node(type=IP, description="src", default=None)


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


@pin(model=CTEvent, store_raw=True)
@tool
def fetch_with_raw(value: str, runtime: ToolRuntime) -> dict:
    """Persists raw return."""
    return {"src_ip": value, "extra": {"score": 87, "tags": ["foo", "bar"]}}


@pin(model=CTEvent)  # store_raw=False (default)
@tool
def fetch_without_raw(value: str, runtime: ToolRuntime) -> dict:
    """."""
    return {"src_ip": value}


class TestStoreRaw:
    def test_get_evidence_returns_raw_when_stored(self, store: InMemoryStore) -> None:
        from agent_pinboard.session import get_or_load_session

        graph = _build([fetch_with_raw, *make_graph_tools()], store)
        _call(graph, "fetch_with_raw", {"value": "1.2.3.4"}, call_id="a")

        g = get_or_load_session(store, "tid")
        events = list(g.all_events())
        assert len(events) == 1
        ev_id = events[0].id

        out = _call(graph, "get_evidence", {"event_id": ev_id}, call_id="b")
        assert "1.2.3.4" in out
        assert "score" in out and "87" in out
        assert "tags" in out and "foo" in out

    def test_get_evidence_hint_when_not_stored(self, store: InMemoryStore) -> None:
        from agent_pinboard.session import get_or_load_session

        graph = _build([fetch_without_raw, *make_graph_tools()], store)
        _call(graph, "fetch_without_raw", {"value": "1.1.1.1"}, call_id="a")

        ev_id = list(get_or_load_session(store, "tid").all_events())[0].id
        out = _call(graph, "get_evidence", {"event_id": ev_id}, call_id="b")
        assert "store_raw=True" in out
        assert "fetch_without_raw" in out

    def test_unknown_event_returns_hint(self, store: InMemoryStore) -> None:
        graph = _build([fetch_with_raw, *make_graph_tools()], store)
        out = _call(graph, "get_evidence", {"event_id": "nonexistent"}, call_id="a")
        assert "no event with id" in out
        assert "timeline" in out or "what_have_i_done" in out


class TestRawNamespaceIsolation:
    def test_different_threads_isolated(self, store: InMemoryStore) -> None:
        from agent_pinboard import store as store_io

        graph = _build([fetch_with_raw, *make_graph_tools()], store)
        # Two thread_ids, each gets its own raw payload.
        graph.invoke(
            {"messages": [AIMessage(
                content="",
                tool_calls=[{"name": "fetch_with_raw", "args": {"value": "1.1.1.1"}, "id": "x", "type": "tool_call"}],
            )]},
            config={"configurable": {"thread_id": "alpha"}},
        )
        graph.invoke(
            {"messages": [AIMessage(
                content="",
                tool_calls=[{"name": "fetch_with_raw", "args": {"value": "2.2.2.2"}, "id": "y", "type": "tool_call"}],
            )]},
            config={"configurable": {"thread_id": "beta"}},
        )

        from agent_pinboard.session import get_or_load_session
        ev_alpha = list(get_or_load_session(store, "alpha").all_events())[0].id
        ev_beta = list(get_or_load_session(store, "beta").all_events())[0].id

        # alpha event raw not visible in beta namespace
        assert store_io.load_raw_event(store, "alpha", ev_alpha) is not None
        assert store_io.load_raw_event(store, "beta", ev_alpha) is None
        assert store_io.load_raw_event(store, "beta", ev_beta) is not None
