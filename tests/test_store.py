from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langgraph.store.memory import InMemoryStore

from pinboard import (
    Entity,
    EventNode,
    FactEdge,
    FactGraph,
    ToolCallRecord,
)
from pinboard import store as store_io


def _ip(normalizer=None) -> Entity:
    return Entity(name="IP", description="ip", normalizer=normalizer)


class TestRoundtrip:
    def test_persist_and_load_graph(self, store: InMemoryStore) -> None:
        g = FactGraph()
        ev = EventNode(id="e-1", source_tool="t", timestamp=datetime.now(timezone.utc))
        g.add_event(ev)
        nid, _ = g.upsert_fact(_ip(), "1.2.3.4", ev.id, "t")
        edge = FactEdge(event_id=ev.id, target_id=nid, edge_type="M.f", description="d")
        g.add_edge(edge)

        store_io.persist_delta(store, "tid", [ev, g.get(nid)], [edge])

        loaded = store_io.load_graph(store, "tid")
        assert loaded.find_by_value("IP", "1.2.3.4") == nid
        edges = loaded.edges_for_event(ev.id)
        assert len(edges) == 1
        assert edges[0].edge_type == "M.f"

    def test_namespace_isolation(self, store: InMemoryStore) -> None:
        g_a = FactGraph()
        ev_a = EventNode(id="ea", source_tool="t", timestamp=datetime.now(timezone.utc))
        g_a.add_event(ev_a)
        nid_a, _ = g_a.upsert_fact(_ip(), "1.1.1.1", ev_a.id, "t")
        store_io.persist_delta(store, "alpha", [ev_a, g_a.get(nid_a)], [])

        g_b = FactGraph()
        ev_b = EventNode(id="eb", source_tool="t", timestamp=datetime.now(timezone.utc))
        g_b.add_event(ev_b)
        nid_b, _ = g_b.upsert_fact(_ip(), "2.2.2.2", ev_b.id, "t")
        store_io.persist_delta(store, "beta", [ev_b, g_b.get(nid_b)], [])

        loaded_a = store_io.load_graph(store, "alpha")
        loaded_b = store_io.load_graph(store, "beta")
        assert loaded_a.find_by_value("IP", "1.1.1.1") is not None
        assert loaded_a.find_by_value("IP", "2.2.2.2") is None
        assert loaded_b.find_by_value("IP", "2.2.2.2") is not None
        assert loaded_b.find_by_value("IP", "1.1.1.1") is None


class TestToolCalls:
    def test_persist_and_load(self, store: InMemoryStore) -> None:
        rec1 = ToolCallRecord(
            tool_name="vt",
            args_repr="{}",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            event_id="e-1",
            summary="ok",
            duration_ms=10,
        )
        rec2 = ToolCallRecord(
            tool_name="vt",
            args_repr="{}",
            timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
            event_id="e-2",
            summary="ok",
            duration_ms=12,
        )
        store_io.persist_tool_call(store, "tid", rec2)
        store_io.persist_tool_call(store, "tid", rec1)
        loaded = store_io.load_tool_calls(store, "tid")
        assert [r.event_id for r in loaded] == ["e-1", "e-2"]


class TestAsyncSymmetry:
    @pytest.mark.asyncio
    async def test_aload_persist(self, store: InMemoryStore) -> None:
        g = FactGraph()
        ev = EventNode(id="e", source_tool="t", timestamp=datetime.now(timezone.utc))
        g.add_event(ev)
        nid, _ = g.upsert_fact(_ip(), "9.9.9.9", ev.id, "t")
        await store_io.apersist_delta(store, "tid", [ev, g.get(nid)], [])
        loaded = await store_io.aload_graph(store, "tid")
        assert loaded.find_by_value("IP", "9.9.9.9") == nid
