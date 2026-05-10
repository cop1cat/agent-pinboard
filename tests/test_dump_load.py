from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pinboard import Entity, EventNode, FactEdge, FactGraph


def _make_filled_graph() -> tuple[FactGraph, str, str]:
    g = FactGraph()
    IP = Entity(name="IP", description="ip")
    User = Entity(name="User", description="u")
    ev = EventNode(id="e-1", source_tool="fetch", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    g.add_event(ev)
    nid_ip, _ = g.upsert_fact(IP, "1.2.3.4", ev.id, "fetch")
    nid_user, _ = g.upsert_fact(User, "alice", ev.id, "fetch")
    g.add_edge(FactEdge(event_id=ev.id, target_id=nid_ip, edge_type="E.ip", description="src"))
    g.add_edge(FactEdge(event_id=ev.id, target_id=nid_user, edge_type="E.actor", description="who"))
    return g, nid_ip, nid_user


class TestDumpLoad:
    def test_round_trip(self) -> None:
        g, nid_ip, nid_user = _make_filled_graph()
        payload = g.dump_to_dict()

        # Schema and version present.
        assert payload["schema"] == "pinboard.factgraph"
        assert "pinboard_version" in payload
        assert len(payload["nodes"]) == 3   # 1 event + 2 facts
        assert len(payload["edges"]) == 2

        restored = FactGraph.load_from_dict(payload)
        assert restored.find_by_value("IP", "1.2.3.4") == nid_ip
        assert restored.find_by_value("User", "alice") == nid_user
        edges = restored.edges_for_event("e-1")
        assert {e.edge_type for e in edges} == {"E.ip", "E.actor"}

    def test_json_serialisable(self) -> None:
        import json

        g, _, _ = _make_filled_graph()
        payload = g.dump_to_dict()
        s = json.dumps(payload)
        again = json.loads(s)
        restored = FactGraph.load_from_dict(again)
        assert restored.find_by_value("IP", "1.2.3.4") is not None

    def test_empty_graph(self) -> None:
        g = FactGraph()
        payload = g.dump_to_dict()
        restored = FactGraph.load_from_dict(payload)
        assert list(restored.all_facts()) == []


class TestLoadValidation:
    def test_rejects_non_dict(self) -> None:
        with pytest.raises(ValueError, match="dict"):
            FactGraph.load_from_dict([])  # type: ignore[arg-type]

    def test_rejects_wrong_schema(self) -> None:
        with pytest.raises(ValueError, match="schema"):
            FactGraph.load_from_dict({"schema": "something.else", "nodes": [], "edges": []})

    def test_missing_nodes_edges_treated_as_empty(self) -> None:
        # Forward-compat: tolerate a payload that lacks these keys.
        restored = FactGraph.load_from_dict({"schema": "pinboard.factgraph"})
        assert list(restored.all_facts()) == []
