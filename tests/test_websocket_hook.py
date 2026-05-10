"""Tests for WebSocketHook + serve_websocket."""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_pinboard import Entity, EventNode, FactEdge, FactGraph, IngestResult
from agent_pinboard.decorator import INGEST_EVENT

pytest.importorskip("websockets")


from agent_pinboard.integrations.websocket_hook import (
    WebSocketHook,
    _build_snapshot,
    _edge_to_payload,
    _node_to_payload,
    serve_websocket,
)


def _ev(eid: str = "e-1") -> EventNode:
    return EventNode(id=eid, source_tool="fetch",
                     timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))


def _make_graph() -> tuple[FactGraph, str]:
    g = FactGraph()
    IP = Entity(name="IP", description="ip")
    ev = _ev()
    g.add_event(ev)
    nid, _ = g.upsert_fact(IP, "1.2.3.4", ev.id, "fetch")
    g.add_edge(FactEdge(event_id=ev.id, target_id=nid,
                        edge_type="E.src_ip", description="src"))
    return g, nid  # type: ignore[return-value]


def _ingest_payload(
    *,
    graph: FactGraph,
    events: list[EventNode] | None = None,
    new_facts: list | None = None,
    linked_facts: list | None = None,
    new_edges: list[FactEdge] | None = None,
    result: IngestResult | None = None,
) -> dict:
    return {
        "thread_id": "tid",
        "tool_name": "fetch",
        "result": result or IngestResult(
            event_ids=[e.id for e in (events or [])],
            new_nodes=len(new_facts or []),
            linked_nodes=len(linked_facts or []),
            new_edges=len(new_edges or []),
        ),
        "events": list(events or []),
        "new_facts": list(new_facts or []),
        "linked_facts": list(linked_facts or []),
        "new_edges": list(new_edges or []),
        "graph": graph,
    }


def _fire(handler: WebSocketHook, payload: dict) -> None:
    handler.on_custom_event(INGEST_EVENT, payload, run_id=uuid4())


# --------------------------------------------------------------------------- #
# Handler unit tests                                                          #
# --------------------------------------------------------------------------- #

class TestIngestEventDispatch:
    def test_emits_per_node_per_edge_and_summary(self) -> None:
        from agent_pinboard import FactNode

        graph = FactGraph()
        ev = _ev()
        graph.add_event(ev)
        new_fact = FactNode(
            id="x", node_type="IP", value="1.1.1.1",
            canonical_value="1.1.1.1", properties={},
            first_seen=datetime.now(UTC), last_seen=datetime.now(UTC),
        )
        edge = FactEdge(event_id=ev.id, target_id="x",
                        edge_type="E.src_ip", description="")

        h = WebSocketHook(thread_id_label="T")
        _fire(h, _ingest_payload(
            graph=graph,
            events=[ev],
            new_facts=[new_fact],
            new_edges=[edge],
            result=IngestResult(
                event_ids=[ev.id], new_nodes=1, linked_nodes=0, new_edges=1,
            ),
        ))

        out = h.drain_pending()
        types = [d["type"] for d in out]
        # event node, fact node, edge, ingest_complete
        assert types == ["node_added", "node_added", "edge_added", "ingest_complete"]
        for d in out:
            assert d["thread_id"] == "T"

    def test_linked_fact_emits_link_found(self) -> None:
        from agent_pinboard import FactNode

        graph = FactGraph()
        ev = _ev()
        graph.add_event(ev)
        linked = FactNode(
            id="y", node_type="IP", value="2.2.2.2",
            canonical_value="2.2.2.2", properties={},
            first_seen=datetime.now(UTC), last_seen=datetime.now(UTC),
        )
        h = WebSocketHook(thread_id_label="T")
        _fire(h, _ingest_payload(
            graph=graph, events=[ev], linked_facts=[linked],
            result=IngestResult(
                event_ids=[ev.id], new_nodes=0, linked_nodes=1, new_edges=0,
            ),
        ))

        out = h.drain_pending()
        link_msgs = [d for d in out if d["type"] == "link_found"]
        assert link_msgs == [
            {"type": "link_found", "node_id": "y",
             "event_id": ev.id, "thread_id": "T"}
        ]

    def test_other_event_names_ignored(self) -> None:
        h = WebSocketHook()
        h.on_custom_event("unrelated_event", {"foo": "bar"}, run_id=uuid4())
        assert h.drain_pending() == []

    def test_drain_empties_queue(self) -> None:
        graph, _ = _make_graph()
        h = WebSocketHook()
        _fire(h, _ingest_payload(graph=graph, events=[_ev()]))
        assert len(h.drain_pending()) > 0
        assert h.drain_pending() == []


class TestSnapshot:
    def test_ingest_event_stores_snapshot(self) -> None:
        h = WebSocketHook(thread_id_label="demo")
        graph, _ = _make_graph()
        assert h.latest_snapshot() is None
        _fire(h, _ingest_payload(graph=graph))
        snap = h.latest_snapshot()
        assert snap is not None
        assert snap["type"] == "snapshot"
        assert snap["thread_id"] == "demo"
        # 1 event + 1 fact node
        assert len(snap["nodes"]) == 2
        assert len(snap["edges"]) == 1


class TestQueueOverflow:
    def test_drops_oldest_when_full(self) -> None:
        graph = FactGraph()
        edges = [
            FactEdge(event_id="e", target_id=f"n{i}",
                     edge_type="x", description="")
            for i in range(10)
        ]
        h = WebSocketHook(queue_maxsize=3)
        _fire(h, _ingest_payload(graph=graph, new_edges=edges))
        out = h.drain_pending()
        # Bound is 3 → only 3 entries survive.
        assert len(out) == 3


# --------------------------------------------------------------------------- #
# Payload serialisation                                                        #
# --------------------------------------------------------------------------- #

class TestPayloads:
    def test_event_node_payload(self) -> None:
        ev = _ev()
        payload = _node_to_payload(ev)
        assert payload["kind"] == "event"
        assert payload["source_tool"] == "fetch"
        assert payload["label"] == f"fetch@{ev.timestamp.strftime('%H:%M:%S')}"

    def test_fact_node_payload(self) -> None:
        from agent_pinboard import FactNode
        n = FactNode(id="n", node_type="IP", value="1.1.1.1",
                     canonical_value="1.1.1.1", properties={},
                     first_seen=datetime.now(UTC),
                     last_seen=datetime.now(UTC))
        payload = _node_to_payload(n)
        assert payload["kind"] == "fact"
        assert payload["node_type"] == "IP"
        assert payload["value"] == "1.1.1.1"

    def test_edge_payload_label_short_form(self) -> None:
        e = FactEdge(event_id="e", target_id="n",
                     edge_type="MyModel.field_name", description="")
        p = _edge_to_payload(e)
        assert p["label"] == "field_name"
        assert p["source"] == "e" and p["target"] == "n"

    def test_snapshot_round_trips_to_json(self) -> None:
        g, _ = _make_graph()
        snap = _build_snapshot(g, thread_id_label="x")
        s = json.dumps(snap)
        again = json.loads(s)
        assert again["type"] == "snapshot"


# --------------------------------------------------------------------------- #
# Server smoke test                                                            #
# --------------------------------------------------------------------------- #

class TestServerSmoke:
    @pytest.mark.asyncio
    async def test_client_receives_snapshot_then_delta(self) -> None:
        from websockets.asyncio.client import connect

        graph, _ = _make_graph()
        h = WebSocketHook(thread_id_label="T")
        _fire(h, _ingest_payload(graph=graph))  # primes latest_snapshot

        port = 8987
        server_task = asyncio.create_task(
            serve_websocket(h, host="127.0.0.1", port=port, poll_interval=0.01)
        )
        try:
            await asyncio.sleep(0.1)
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                # Drain the snapshot + the prime-ingest deltas.
                msgs: list[dict] = []
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        break
                    msgs.append(json.loads(raw))
                assert any(m["type"] == "snapshot" and m["thread_id"] == "T" for m in msgs)

                # Push a fresh ingest with a single edge and expect to see the delta.
                edge = FactEdge(event_id="e-2", target_id="n-2",
                                edge_type="M.f", description="d")
                _fire(h, _ingest_payload(graph=graph, new_edges=[edge]))
                delta_msgs: list[dict] = []
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        break
                    delta_msgs.append(json.loads(raw))
                assert any(m["type"] == "edge_added" and m["thread_id"] == "T"
                           for m in delta_msgs)
        finally:
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task

    @pytest.mark.asyncio
    async def test_http_get_serves_html_when_html_path_given(self, tmp_path) -> None:
        """A plain browser request to / returns the HTML file, not a WS error."""
        import urllib.request

        h = WebSocketHook()
        html_file = tmp_path / "page.html"
        html_file.write_text("<!doctype html><title>agent_pinboard demo</title>")

        port = 8988
        server_task = asyncio.create_task(
            serve_websocket(
                h, host="127.0.0.1", port=port, poll_interval=0.01,
                html_path=str(html_file),
            )
        )
        try:
            await asyncio.sleep(0.1)
            data = await asyncio.to_thread(
                lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
            )
            assert b"agent_pinboard demo" in data
            from websockets.asyncio.client import connect
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                edge = FactEdge(event_id="e", target_id="n",
                                edge_type="x", description="")
                graph = FactGraph()
                _fire(h, _ingest_payload(graph=graph, new_edges=[edge]))
                # Grab the first delta we receive.
                msgs: list[dict] = []
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        break
                    msgs.append(json.loads(raw))
                assert any(m["type"] == "edge_added" for m in msgs)
        finally:
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task
