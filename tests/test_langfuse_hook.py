from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agent_pinboard import Entity, EventNode, FactEdge, FactGraph, IngestResult
from agent_pinboard.decorator import INGEST_EVENT

pytest.importorskip("langfuse")


from agent_pinboard.integrations.langfuse_hook import LangfuseHook, render_mermaid


def _add(g: FactGraph, ent: Entity, value: str, ev_id: str, tool: str, edge_type: str) -> None:
    nid, _ = g.upsert_fact(ent, value, ev_id, tool)
    if nid is not None:
        g.add_edge(FactEdge(event_id=ev_id, target_id=nid, edge_type=edge_type, description=""))


@pytest.fixture
def graph() -> FactGraph:
    g = FactGraph()
    IP = Entity(name="IP", description="ip")
    User = Entity(name="User", description="u")
    ev = EventNode(id="e-1", source_tool="fetch", timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))
    g.add_event(ev)
    _add(g, IP, "1.1.1.1", ev.id, "fetch", "Event.src_ip")
    _add(g, User, "alice", ev.id, "fetch", "Event.actor")
    return g


def _payload(graph: FactGraph, *, result: IngestResult | None = None) -> dict:
    return {
        "thread_id": "tid",
        "tool_name": "fetch",
        "result": result or IngestResult(event_ids=["e-1"], new_nodes=2, linked_nodes=1, new_edges=3),
        "events": [],
        "new_facts": [],
        "linked_facts": [],
        "new_edges": [],
        "graph": graph,
    }


def _fire(handler: LangfuseHook, payload: dict) -> None:
    handler.on_custom_event(INGEST_EVENT, payload, run_id=uuid4())


class TestMermaidRendering:
    def test_basic_structure(self, graph: FactGraph) -> None:
        out = render_mermaid(graph)
        assert out.startswith("flowchart LR")
        assert "IP: 1.1.1.1" in out
        assert "User: alice" in out
        assert "fetch@12:00:00" in out
        assert "-->" in out

    def test_truncation_with_extra_marker(self, graph: FactGraph) -> None:
        IP = Entity(name="IP", description="ip")
        ev = EventNode(id="e-2", source_tool="vt", timestamp=datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC))
        graph.add_event(ev)
        for i in range(40):
            _add(graph, IP, f"10.0.0.{i}", ev.id, "vt", "Event.related_ip")
        out = render_mermaid(graph, max_facts=10)
        assert "... and" in out and "more facts" in out

    def test_orphan_events_omitted(self, graph: FactGraph) -> None:
        graph.add_event(EventNode(
            id="e-orphan", source_tool="lonely",
            timestamp=datetime(2026, 1, 1, 12, 5, 0, tzinfo=UTC),
        ))
        out = render_mermaid(graph)
        assert "lonely@" not in out

    def test_quotes_escaped(self) -> None:
        g = FactGraph()
        IP = Entity(name="IP", description="ip")
        ev = EventNode(id="e", source_tool="t", timestamp=datetime.now(UTC))
        g.add_event(ev)
        _add(g, IP, 'with "quote"', ev.id, "t", "Event.x")
        out = render_mermaid(g)
        assert '\\"quote\\"' in out


class TestLangfuseHookEmits:
    def test_ingest_event_calls_start_observation(self, graph: FactGraph) -> None:
        client = MagicMock()
        client.start_observation.return_value = MagicMock()

        handler = LangfuseHook(client, emit_snapshots=False)
        _fire(handler, _payload(graph))

        assert client.start_observation.called
        kwargs = client.start_observation.call_args_list[0].kwargs
        assert kwargs["name"] == "agent_pinboard.ingest"
        assert kwargs["output"]["new_nodes"] == 2
        assert kwargs["metadata"]["event_ids"] == ["e-1"]

    def test_ingest_event_with_snapshots_emits_two_spans(self, graph: FactGraph) -> None:
        client = MagicMock()
        client.start_observation.return_value = MagicMock()

        handler = LangfuseHook(client, emit_snapshots=True)
        _fire(handler, _payload(graph))

        names = [c.kwargs["name"] for c in client.start_observation.call_args_list]
        assert "agent_pinboard.ingest" in names
        assert "agent_pinboard.graph_snapshot" in names
        snap_call = next(c for c in client.start_observation.call_args_list
                         if c.kwargs["name"] == "agent_pinboard.graph_snapshot")
        assert snap_call.kwargs["metadata"]["mermaid"].startswith("flowchart LR")

    def test_emit_snapshots_false_only_emits_ingest(self, graph: FactGraph) -> None:
        client = MagicMock()
        client.start_observation.return_value = MagicMock()

        handler = LangfuseHook(client, emit_snapshots=False)
        _fire(handler, _payload(graph))

        names = [c.kwargs["name"] for c in client.start_observation.call_args_list]
        assert names == ["agent_pinboard.ingest"]

    def test_warnings_set_warning_level(self, graph: FactGraph) -> None:
        client = MagicMock()
        client.start_observation.return_value = MagicMock()

        handler = LangfuseHook(client, emit_snapshots=False)
        result = IngestResult(
            event_ids=["e"], new_nodes=0, linked_nodes=0, new_edges=0,
            warnings=["empty canonical: x"],
        )
        _fire(handler, _payload(graph, result=result))

        kwargs = client.start_observation.call_args_list[0].kwargs
        assert kwargs["level"] == "WARNING"

    def test_other_event_names_ignored(self, graph: FactGraph) -> None:
        client = MagicMock()
        handler = LangfuseHook(client)
        handler.on_custom_event(
            "some_other_event", _payload(graph), run_id=uuid4(),
        )
        assert not client.start_observation.called


class TestLangfuseHookIsolation:
    def test_failing_client_logs_and_continues(
        self, graph: FactGraph, caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = MagicMock()
        client.start_observation.side_effect = RuntimeError("LF down")

        handler = LangfuseHook(client)
        with caplog.at_level(logging.ERROR):
            _fire(handler, _payload(graph))

        msgs = [r.message for r in caplog.records]
        assert any("LangfuseHook ingest dispatch failed" in m for m in msgs)


class TestLangfuseHookConstructor:
    def test_rejects_none_client(self) -> None:
        with pytest.raises(ValueError):
            LangfuseHook(None)  # type: ignore[arg-type]
