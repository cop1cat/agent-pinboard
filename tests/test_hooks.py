from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from pinboard import EventNode, FactEdge, FactGraph, FactNode, IngestResult
from pinboard.hooks import CompositeHook, LoggingHook, PinBoardHooks, fire

_FactGraph_for_test = FactGraph()


def _ev() -> EventNode:
    return EventNode(id="e", source_tool="t", timestamp=datetime.now(timezone.utc))


def _fact() -> FactNode:
    now = datetime.now(timezone.utc)
    return FactNode(
        id="n", node_type="IP", value="1.1.1.1", canonical_value="1.1.1.1",
        properties={}, first_seen=now, last_seen=now,
    )


def _edge() -> FactEdge:
    return FactEdge(event_id="e", target_id="n", edge_type="M.f", description="d")


class TestBaseClassNoOp:
    def test_methods_are_noops(self) -> None:
        h = PinBoardHooks()
        h.on_node_added(_fact())
        h.on_edge_added(_edge())
        h.on_link_found(_fact(), "e")
        h.on_ingest_complete(IngestResult(event_ids=[], new_nodes=0, linked_nodes=0, new_edges=0))
        h.on_graph_changed(_FactGraph_for_test)


class TestLoggingHook:
    def test_logs_each_callback(self, caplog: pytest.LogCaptureFixture) -> None:
        h = LoggingHook(level=logging.INFO)
        with caplog.at_level(logging.INFO):
            h.on_node_added(_fact())
            h.on_edge_added(_edge())
            h.on_link_found(_fact(), "e")
            h.on_ingest_complete(
                IngestResult(event_ids=["e"], new_nodes=1, linked_nodes=0, new_edges=1)
            )
        assert any("node_added" in r.message for r in caplog.records)
        assert any("edge_added" in r.message for r in caplog.records)
        assert any("link_found" in r.message for r in caplog.records)
        assert any("ingest_complete" in r.message for r in caplog.records)


class TestCompositeHookFanOut:
    def test_calls_each_in_order(self) -> None:
        events: list[str] = []

        class A(PinBoardHooks):
            def on_node_added(self, node):
                events.append("a")

        class B(PinBoardHooks):
            def on_node_added(self, node):
                events.append("b")

        c = CompositeHook([A(), B()])
        c.on_node_added(_fact())
        assert events == ["a", "b"]


class TestHookFailureIsolation:
    def test_one_hook_raises_others_still_called(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        events: list[str] = []

        class Boom(PinBoardHooks):
            def on_node_added(self, node):
                raise RuntimeError("boom")

        class Good(PinBoardHooks):
            def on_node_added(self, node):
                events.append("good")

        c = CompositeHook([Boom(), Good()])
        with caplog.at_level(logging.ERROR):
            c.on_node_added(_fact())
        assert events == ["good"]
        assert any("hook" in r.message for r in caplog.records)


class TestFireHelper:
    def test_none_hooks_is_noop(self) -> None:
        fire(None, "on_node_added", _fact())  # must not raise

    def test_routes_to_method_by_name(self) -> None:
        seen: list[FactNode] = []

        class H(PinBoardHooks):
            def on_node_added(self, node):
                seen.append(node)  # type: ignore[arg-type]

        h = H()
        f = _fact()
        fire(h, "on_node_added", f)
        assert seen == [f]

    def test_swallows_exceptions(self, caplog: pytest.LogCaptureFixture) -> None:
        class Boom(PinBoardHooks):
            def on_node_added(self, node):
                raise RuntimeError("x")

        with caplog.at_level(logging.ERROR):
            fire(Boom(), "on_node_added", _fact())
        assert any("hook" in r.message for r in caplog.records)
