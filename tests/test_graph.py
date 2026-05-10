from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from agent_pinboard import (
    AgentPinBoardNormalizerError,
    Entity,
    EventNode,
    FactEdge,
    FactGraph,
    FactNode,
)


def _ip_entity(normalizer=None) -> Entity:
    return Entity(name="IP", description="ip", normalizer=normalizer)


def _make_event(tool: str = "fetch", eid: str = "e-1") -> EventNode:
    return EventNode(id=eid, source_tool=tool, timestamp=datetime.now(UTC))


class TestUpsertFact:
    def test_inserts_new_node(self) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        nid, was_new = g.upsert_fact(_ip_entity(), "1.2.3.4", ev.id, "fetch")
        assert was_new
        assert nid is not None
        n = g.get(nid)
        assert isinstance(n, FactNode)
        assert n.value == "1.2.3.4"
        assert n.source_events == [ev.id]
        assert n.source_tools == {"fetch"}

    def test_autolinks_same_canonical(self) -> None:
        """README §16 AC2 — one IP from two events → one node."""
        g = FactGraph()
        ev1 = _make_event(tool="fetch", eid="e-1")
        ev2 = _make_event(tool="vt", eid="e-2")
        g.add_event(ev1)
        g.add_event(ev2)

        nid1, new1 = g.upsert_fact(_ip_entity(), "1.2.3.4", ev1.id, "fetch")
        nid2, new2 = g.upsert_fact(_ip_entity(), "1.2.3.4", ev2.id, "vt")

        assert nid1 == nid2
        assert new1 is True
        assert new2 is False
        n = g.get(nid1)
        assert isinstance(n, FactNode)
        assert n.source_events == ["e-1", "e-2"]
        assert n.source_tools == {"fetch", "vt"}

    def test_normalizer_canonicalises(self) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        e = _ip_entity(normalizer=str.lower)
        nid_a, _ = g.upsert_fact(e, "ABC", ev.id, "t")
        nid_b, _ = g.upsert_fact(e, "abc", ev.id, "t")
        assert nid_a == nid_b

    def test_normalizer_exception_fails_loud(self) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)

        def boom(_: object) -> str:
            raise ValueError("kaboom")

        e = Entity(name="IP", description="ip", normalizer=boom)
        with pytest.raises(AgentPinBoardNormalizerError):
            g.upsert_fact(e, "x", ev.id, "t")

    def test_empty_canonical_dropped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        e = Entity(name="X", description="x", normalizer=lambda _: "")
        warnings: list[str] = []
        with caplog.at_level(logging.WARNING):
            nid, was_new = g.upsert_fact(e, "abc", ev.id, "t", warnings=warnings)
        assert nid is None and was_new is False
        assert any("empty canonical" in w for w in warnings)

    def test_none_value_returns_none(self) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        nid, was_new = g.upsert_fact(_ip_entity(), None, ev.id, "t")
        assert nid is None and was_new is False


class TestAddEdgeAndQueries:
    def test_add_edge_and_lookup(self) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        nid, _ = g.upsert_fact(_ip_entity(), "1.1.1.1", ev.id, "t")
        assert nid
        edge = FactEdge(event_id=ev.id, target_id=nid, edge_type="M.f", description="d")
        g.add_edge(edge)
        edges = g.edges_for_event(ev.id)
        assert edges == [edge]

    def test_search_by_type(self) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        n1, _ = g.upsert_fact(_ip_entity(), "1.1.1.1", ev.id, "t")
        n2, _ = g.upsert_fact(_ip_entity(), "2.2.2.2", ev.id, "t")
        assert set(g.search_by_type("IP")) == {n1, n2}
        assert g.search_by_type("Nope") == []

    def test_find_by_value(self) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        nid, _ = g.upsert_fact(_ip_entity(), "5.5.5.5", ev.id, "t")
        assert g.find_by_value("IP", "5.5.5.5") == nid
        assert g.find_by_value("IP", "9.9.9.9") is None

    def test_all_facts_excludes_events(self) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        g.upsert_fact(_ip_entity(), "1.1.1.1", ev.id, "t")
        facts = list(g.all_facts())
        assert len(facts) == 1
        assert all(isinstance(f, FactNode) for f in facts)


class TestBackfillProvenance:
    def test_orphan_fact_has_empty_provenance(self) -> None:
        """A FactNode loaded with no incoming edges has empty provenance.
        first_seen/last_seen are left as the snapshot supplied them
        (the Store deserializer seeds them with the EPOCH sentinel for
        orphans loaded from disk; in-memory orphans keep whatever the
        constructor set)."""
        orphan = FactNode(
            id="x", node_type="IP", value="1.1.1.1",
            canonical_value="1.1.1.1",
            first_seen=datetime.now(UTC),
            last_seen=datetime.now(UTC),
        )
        g = FactGraph.from_snapshot([orphan], [])
        loaded = g.get("x")
        assert isinstance(loaded, FactNode)
        assert loaded.source_events == []
        assert loaded.source_tools == set()

    def test_edge_to_missing_event_is_ignored(self) -> None:
        """A FactEdge whose event_id is not in the snapshot does not contribute
        to provenance — defensive against partially-loaded snapshots."""
        ev = _make_event(eid="e-real")
        fact = FactNode(
            id="x", node_type="IP", value="2.2.2.2",
            canonical_value="2.2.2.2",
            first_seen=datetime.now(UTC), last_seen=datetime.now(UTC),
        )
        real_edge = FactEdge(event_id="e-real", target_id="x",
                             edge_type="M.f", description="")
        ghost_edge = FactEdge(event_id="e-missing", target_id="x",
                              edge_type="M.f", description="")
        g = FactGraph.from_snapshot([ev, fact], [real_edge, ghost_edge])
        loaded = g.get("x")
        assert isinstance(loaded, FactNode)
        assert loaded.source_events == ["e-real"]

    def test_ordering_deterministic_by_timestamp_then_id(self) -> None:
        """Two events with the same timestamp tie-break on event_id."""
        same_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        ev_b = EventNode(id="e-bbb", source_tool="t", timestamp=same_ts)
        ev_a = EventNode(id="e-aaa", source_tool="t", timestamp=same_ts)
        fact = FactNode(
            id="x", node_type="IP", value="3.3.3.3",
            canonical_value="3.3.3.3",
            first_seen=datetime.now(UTC), last_seen=datetime.now(UTC),
        )
        edges = [
            FactEdge(event_id=ev_b.id, target_id="x", edge_type="M.f", description=""),
            FactEdge(event_id=ev_a.id, target_id="x", edge_type="M.f", description=""),
        ]
        g = FactGraph.from_snapshot([ev_b, ev_a, fact], edges)
        loaded = g.get("x")
        assert isinstance(loaded, FactNode)
        assert loaded.source_events == ["e-aaa", "e-bbb"]


class TestSnapshotRoundtrip:
    def test_roundtrip(self) -> None:
        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        nid, _ = g.upsert_fact(_ip_entity(), "1.2.3.4", ev.id, "t")
        edge = FactEdge(event_id=ev.id, target_id=nid, edge_type="M.f", description="d")
        g.add_edge(edge)

        nodes = [g.get(nid) for nid in g.g.nodes]
        edges = []
        for u, v, data in g.g.edges(data=True):
            edges.append(data["obj"])

        g2 = FactGraph.from_snapshot(nodes, edges)
        assert g2.find_by_value("IP", "1.2.3.4") == nid
        assert g2.edges_for_event(ev.id) == [edge]
