"""Tests for WebSocketHook + serve_websocket."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest

from agent_pinboard import Entity, EventNode, FactEdge, FactGraph, IngestResult

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


# --------------------------------------------------------------------------- #
# Hook unit tests                                                              #
# --------------------------------------------------------------------------- #

class TestHookEnqueues:
    def test_node_added_drained(self) -> None:
        h = WebSocketHook(thread_id_label="T")
        ev = _ev()
        h.on_node_added(ev)
        out = h.drain_pending()
        assert len(out) == 1
        assert out[0]["type"] == "node_added"
        assert out[0]["thread_id"] == "T"
        assert out[0]["node"]["kind"] == "event"

    def test_edge_added_drained(self) -> None:
        h = WebSocketHook()
        h.on_edge_added(FactEdge(
            event_id="e", target_id="n", edge_type="M.f", description="d",
        ))
        out = h.drain_pending()
        assert out[0]["type"] == "edge_added"
        assert out[0]["edge"]["edge_type"] == "M.f"

    def test_link_found_drained(self) -> None:
        from agent_pinboard import FactNode

        h = WebSocketHook()
        n = FactNode(id="x", node_type="IP", value="1.1.1.1",
                     canonical_value="1.1.1.1", properties={},
                     first_seen=datetime.now(UTC),
                     last_seen=datetime.now(UTC))
        h.on_link_found(n, "e-1")
        out = h.drain_pending()
        assert out[0]["type"] == "link_found"
        assert out[0]["node_id"] == "x"
        assert out[0]["event_id"] == "e-1"

    def test_ingest_complete_drained(self) -> None:
        h = WebSocketHook()
        h.on_ingest_complete(
            IngestResult(event_ids=["e-1"], new_nodes=2, linked_nodes=1, new_edges=3)
        )
        out = h.drain_pending()
        assert out[0]["type"] == "ingest_complete"
        assert out[0]["result"]["new_nodes"] == 2

    def test_drain_empties_queue(self) -> None:
        h = WebSocketHook()
        h.on_edge_added(FactEdge(event_id="e", target_id="n",
                                 edge_type="x", description=""))
        assert len(h.drain_pending()) == 1
        assert h.drain_pending() == []


class TestHookSnapshot:
    def test_on_graph_changed_stores_snapshot(self) -> None:
        h = WebSocketHook(thread_id_label="demo")
        g, _ = _make_graph()
        assert h.latest_snapshot() is None
        h.on_graph_changed(g)
        snap = h.latest_snapshot()
        assert snap is not None
        assert snap["type"] == "snapshot"
        assert snap["thread_id"] == "demo"
        # 1 event + 1 fact node
        assert len(snap["nodes"]) == 2
        assert len(snap["edges"]) == 1


class TestQueueOverflow:
    def test_drops_oldest_when_full(self) -> None:
        h = WebSocketHook(queue_maxsize=3)
        for i in range(10):
            h.on_edge_added(FactEdge(
                event_id="e", target_id=f"n{i}",
                edge_type="x", description="",
            ))
        # Bound is 3 → only 3 entries survive.
        out = h.drain_pending()
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
        # Base label; frontend enriches using connected-fact edges.
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
        # Must serialise cleanly.
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

        h = WebSocketHook(thread_id_label="T")
        g, _ = _make_graph()
        h.on_graph_changed(g)  # primes latest_snapshot

        # Pick an ephemeral port to avoid collisions.
        port = 8987
        server_task = asyncio.create_task(
            serve_websocket(h, host="127.0.0.1", port=port, poll_interval=0.01)
        )
        try:
            await asyncio.sleep(0.1)  # let the server bind
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                # 1. Snapshot first.
                snap_raw = await asyncio.wait_for(ws.recv(), timeout=2)
                snap = json.loads(snap_raw)
                assert snap["type"] == "snapshot"
                assert snap["thread_id"] == "T"

                # 2. Push a delta and expect to receive it.
                h.on_edge_added(FactEdge(
                    event_id="e-2", target_id="n-2",
                    edge_type="M.f", description="d",
                ))
                delta_raw = await asyncio.wait_for(ws.recv(), timeout=2)
                delta = json.loads(delta_raw)
                assert delta["type"] == "edge_added"
                assert delta["thread_id"] == "T"
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

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
            # Plain HTTP GET — must succeed with the HTML body.
            data = await asyncio.to_thread(
                lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
            )
            assert b"agent_pinboard demo" in data
            # WS upgrade still works on the same port.
            from websockets.asyncio.client import connect
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                # No snapshot was set, so no message is sent on connect.
                # Send a delta and confirm round-trip.
                h.on_edge_added(FactEdge(
                    event_id="e", target_id="n", edge_type="x", description="",
                ))
                msg = await asyncio.wait_for(ws.recv(), timeout=2)
                assert json.loads(msg)["type"] == "edge_added"
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
