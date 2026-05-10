from __future__ import annotations

from datetime import UTC, datetime

from agent_pinboard import EventNode, FactEdge, FactNode, IngestResult, ToolCallRecord
from agent_pinboard.models import EVENT_NODE_TYPE, fact_node_id


class TestFactNodeId:
    def test_deterministic(self) -> None:
        assert fact_node_id("IP", "1.2.3.4") == fact_node_id("IP", "1.2.3.4")

    def test_distinct_per_type_and_value(self) -> None:
        assert fact_node_id("IP", "1.2.3.4") != fact_node_id("IP", "1.2.3.5")
        assert fact_node_id("IP", "1.2.3.4") != fact_node_id("Host", "1.2.3.4")

    def test_short(self) -> None:
        assert len(fact_node_id("IP", "1.2.3.4")) == 16


class TestFactEdge:
    def test_id_property_derived(self) -> None:
        e = FactEdge(
            event_id="e-1",
            target_id="n-abc",
            edge_type="MyModel.field",
            description="d",
        )
        assert e.id == "e-1|MyModel.field|n-abc"

    def test_frozen(self) -> None:
        e = FactEdge(event_id="e", target_id="t", edge_type="x.y", description="d")
        try:
            e.target_id = "other"  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("FactEdge should be frozen")


class TestEventNode:
    def test_default_node_type(self) -> None:
        e = EventNode(id="x", source_tool="t", timestamp=datetime.now(UTC))
        assert e.node_type == EVENT_NODE_TYPE


class TestFactNodeShape:
    def test_defaults(self) -> None:
        now = datetime.now(UTC)
        n = FactNode(
            id="abc",
            node_type="IP",
            value="1.1.1.1",
            canonical_value="1.1.1.1",
            properties={},
            first_seen=now,
            last_seen=now,
        )
        assert n.source_events == []
        assert n.source_tools == set()


class TestIngestResultDefaults:
    def test_warnings_default_empty(self) -> None:
        r = IngestResult(event_ids=[], new_nodes=0, linked_nodes=0, new_edges=0)
        assert r.warnings == []


class TestToolCallRecordFrozen:
    def test_immutable(self) -> None:
        r = ToolCallRecord(
            tool_name="t",
            args_repr="{}",
            timestamp=datetime.now(UTC),
            event_id="e",
            summary="ok",
            duration_ms=1,
        )
        try:
            r.summary = "x"  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("ToolCallRecord should be frozen")
