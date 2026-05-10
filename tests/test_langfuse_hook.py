from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from pinboard import Entity, EventNode, FactEdge, FactGraph, IngestResult


pytest.importorskip("langfuse")


from pinboard.integrations.langfuse_hook import LangfuseHook, render_mermaid


def _add(g: FactGraph, ent: Entity, value: str, ev_id: str, tool: str, edge_type: str) -> None:
    nid, _ = g.upsert_fact(ent, value, ev_id, tool)
    if nid is not None:
        g.add_edge(FactEdge(event_id=ev_id, target_id=nid, edge_type=edge_type, description=""))


@pytest.fixture
def graph() -> FactGraph:
    g = FactGraph()
    IP = Entity(name="IP", description="ip")
    User = Entity(name="User", description="u")
    ev = EventNode(id="e-1", source_tool="fetch", timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    g.add_event(ev)
    _add(g, IP, "1.1.1.1", ev.id, "fetch", "Event.src_ip")
    _add(g, User, "alice", ev.id, "fetch", "Event.actor")
    return g


class TestMermaidRendering:
    def test_basic_structure(self, graph: FactGraph) -> None:
        out = render_mermaid(graph)
        assert out.startswith("flowchart LR")
        # Both facts present.
        assert "IP: 1.1.1.1" in out
        assert "User: alice" in out
        # Event present and connected to both facts.
        assert "fetch@12:00:00" in out
        assert "-->" in out

    def test_truncation_with_extra_marker(self, graph: FactGraph) -> None:
        IP = Entity(name="IP", description="ip")
        ev = EventNode(id="e-2", source_tool="vt", timestamp=datetime(2026, 1, 1, 12, 1, 0, tzinfo=timezone.utc))
        graph.add_event(ev)
        for i in range(40):
            _add(graph, IP, f"10.0.0.{i}", ev.id, "vt", "Event.related_ip")
        out = render_mermaid(graph, max_facts=10)
        assert "... and" in out and "more facts" in out

    def test_orphan_events_omitted(self, graph: FactGraph) -> None:
        # Add an event with NO facts attached.
        graph.add_event(EventNode(
            id="e-orphan", source_tool="lonely",
            timestamp=datetime(2026, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
        ))
        out = render_mermaid(graph)
        assert "lonely@" not in out

    def test_quotes_escaped(self) -> None:
        g = FactGraph()
        IP = Entity(name="IP", description="ip")
        ev = EventNode(id="e", source_tool="t", timestamp=datetime.now(timezone.utc))
        g.add_event(ev)
        _add(g, IP, 'with "quote"', ev.id, "t", "Event.x")
        out = render_mermaid(g)
        assert '\\"quote\\"' in out


class TestLangfuseHookEmits:
    def test_on_ingest_complete_calls_start_observation(self) -> None:
        client = MagicMock()
        client.start_observation.return_value = MagicMock()

        hook = LangfuseHook(client, emit_snapshots=False)
        result = IngestResult(event_ids=["e-1"], new_nodes=2, linked_nodes=1, new_edges=3)
        hook.on_ingest_complete(result)

        assert client.start_observation.called
        kwargs = client.start_observation.call_args.kwargs
        assert kwargs["name"] == "pinboard.ingest"
        assert kwargs["output"]["new_nodes"] == 2
        assert kwargs["metadata"]["event_ids"] == ["e-1"]

    def test_on_graph_changed_emits_mermaid(self, graph: FactGraph) -> None:
        client = MagicMock()
        client.start_observation.return_value = MagicMock()

        hook = LangfuseHook(client)
        hook.on_graph_changed(graph)

        # Two calls would be made if both ingest+snapshot fired together,
        # but on_graph_changed alone only emits the snapshot.
        assert client.start_observation.called
        kwargs = client.start_observation.call_args.kwargs
        assert kwargs["name"] == "pinboard.graph_snapshot"
        assert "mermaid" in kwargs["metadata"]
        assert kwargs["metadata"]["mermaid"].startswith("flowchart LR")

    def test_emit_snapshots_false_skips_graph_calls(self, graph: FactGraph) -> None:
        client = MagicMock()
        hook = LangfuseHook(client, emit_snapshots=False)
        hook.on_graph_changed(graph)
        assert not client.start_observation.called

    def test_warnings_set_warning_level(self) -> None:
        client = MagicMock()
        client.start_observation.return_value = MagicMock()

        hook = LangfuseHook(client, emit_snapshots=False)
        result = IngestResult(
            event_ids=["e"], new_nodes=0, linked_nodes=0, new_edges=0,
            warnings=["empty canonical: x"],
        )
        hook.on_ingest_complete(result)
        assert client.start_observation.call_args.kwargs["level"] == "WARNING"


class TestLangfuseHookIsolation:
    def test_failing_client_logs_and_continues(
        self, graph: FactGraph, caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = MagicMock()
        client.start_observation.side_effect = RuntimeError("LF down")

        hook = LangfuseHook(client)
        with caplog.at_level(logging.ERROR):
            hook.on_ingest_complete(
                IngestResult(event_ids=[], new_nodes=0, linked_nodes=0, new_edges=0)
            )
            hook.on_graph_changed(graph)

        # Both callbacks attempted; both logged ERRORs; nothing raised.
        msgs = [r.message for r in caplog.records]
        assert any("on_ingest_complete failed" in m for m in msgs)
        assert any("on_graph_changed failed" in m for m in msgs)


class TestLangfuseHookConstructor:
    def test_rejects_none_client(self) -> None:
        with pytest.raises(ValueError):
            LangfuseHook(None)  # type: ignore[arg-type]
