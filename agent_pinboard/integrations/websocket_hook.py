"""WebSocket hook — streams graph deltas to connected clients in real time.

Optional dependency. Install with::

    uv add 'agent_pinboard[ws]'        # or:  pip install agent_pinboard[ws]

Designed to drive a live visualisation (e.g. the Cytoscape.js demo in
``examples/web/``). The hook itself is just a delta producer with a
fan-out queue; the ``serve_websocket`` coroutine spins up an actual
``websockets.serve`` server that broadcasts the deltas to every
connected client.

Wire-format (JSON, one message per line):

* ``{"type": "snapshot", "thread_id": "...", "nodes": [...], "edges": [...]}`` —
  full graph dump, sent on initial client connect.
* ``{"type": "node_added", "thread_id": "...", "node": {...}}``
* ``{"type": "edge_added", "thread_id": "...", "edge": {...}}``
* ``{"type": "link_found", "thread_id": "...", "node_id": "...", "event_id": "..."}``
* ``{"type": "ingest_complete", "thread_id": "...", "result": {...}}``

The ``thread_id`` field lets a client UI filter or label sessions if
multiple are streaming through one server.

Threading
---------
The hook itself is sync (it's called from inside ``@pin`` ingestion
under a threading lock). Deltas are pushed into a thread-safe queue;
the asyncio server drains the queue from the loop thread. This keeps
ingestion latency unaffected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, override

from agent_pinboard.graph import FactGraph
from agent_pinboard.hooks import AgentPinBoardHooks
from agent_pinboard.models import EVENT_NODE_TYPE, EventNode, FactEdge, FactNode, IngestResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEPENDENCY_HINT = (
    "WebSocketHook + serve_websocket() require the websockets package: "
    "install with `pip install agent_pinboard[ws]` or `pip install websockets`."
)


# --------------------------------------------------------------------------- #
# Hook                                                                         #
# --------------------------------------------------------------------------- #

class WebSocketHook(AgentPinBoardHooks):
    """Pushes graph deltas into a queue for a separate WS server to drain.

    Construct one hook per agent and pass it both to ``@pin(hooks=...)``
    *and* to ``serve_websocket(hook, ...)`` (in the asyncio main).

    Parameters
    ----------
    thread_id_label:
        Optional label to send with every delta — useful when a single
        WS server fans out multiple sessions and the UI needs to colour
        them. If ``None``, every delta carries an empty string in
        ``thread_id``.
    queue_maxsize:
        Bounded queue size; older deltas are dropped silently when the
        queue is full (avoids unbounded memory when no client is
        connected). Default 1024.
    """

    def __init__(
        self,
        *,
        thread_id_label: str | None = None,
        queue_maxsize: int = 1024,
    ) -> None:
        self._label = thread_id_label or ""
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_maxsize)
        self._latest_snapshot: dict[str, Any] | None = None

    # ---- public surface for the server to consume -----------------------

    def drain_pending(self) -> list[dict[str, Any]]:
        """Pop every queued delta. Server calls this from the asyncio loop."""
        out: list[dict[str, Any]] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def latest_snapshot(self) -> dict[str, Any] | None:
        """Most recent full snapshot, or ``None`` if no ingest has happened."""
        return self._latest_snapshot

    # ---- AgentPinBoardHooks overrides ----------------------------------------

    @override
    def on_node_added(self, node: FactNode | EventNode) -> None:
        self._enqueue({"type": "node_added", "node": _node_to_payload(node)})

    @override
    def on_edge_added(self, edge: FactEdge) -> None:
        self._enqueue({"type": "edge_added", "edge": _edge_to_payload(edge)})

    @override
    def on_link_found(self, existing: FactNode, event_id: str) -> None:
        self._enqueue({
            "type": "link_found",
            "node_id": existing.id,
            "event_id": event_id,
        })

    @override
    def on_ingest_complete(self, result: IngestResult) -> None:
        self._enqueue({"type": "ingest_complete", "result": asdict(result)})

    @override
    def on_graph_changed(self, graph: FactGraph) -> None:
        snapshot = _build_snapshot(graph, thread_id_label=self._label)
        self._latest_snapshot = snapshot
        # Snapshot is always sent on connect; we don't push it to the
        # delta queue because it would dwarf the deltas. Clients get the
        # current snapshot at handshake and apply deltas thereafter.

    # ---- internals -------------------------------------------------------

    def _enqueue(self, delta: dict[str, Any]) -> None:
        delta["thread_id"] = self._label
        try:
            self._queue.put_nowait(delta)
        except queue.Full:
            # Drop oldest by popping one and trying again. Best-effort —
            # if the second put also fails, we just lose this delta.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(delta)
            except (queue.Empty, queue.Full):
                logger.warning(
                    "WebSocketHook queue full; dropping delta type=%s",
                    delta.get("type"),
                )


# --------------------------------------------------------------------------- #
# Server                                                                       #
# --------------------------------------------------------------------------- #

async def serve_websocket(
    hook: WebSocketHook,
    *,
    host: str = "localhost",
    port: int = 8765,
    poll_interval: float = 0.05,
    html_path: str | None = None,
) -> None:
    """Run a WebSocket server that broadcasts ``hook``'s deltas to all clients.

    Each client receives the latest snapshot on connect (if any), then
    every subsequent delta. The server runs forever — wrap in a task or
    use ``asyncio.run`` at the top level.

    Optional ``html_path`` makes plain HTTP ``GET /`` (and ``GET /index.html``)
    return the contents of the given file, so navigating to
    ``http://localhost:<port>/`` in a browser shows the visualisation
    page instead of the raw "you need a WebSocket client" error. The
    same port still serves the WS upgrade for ``ws://localhost:<port>/``
    requests.

    Example::

        import asyncio
        from agent_pinboard.integrations.websocket_hook import (
            WebSocketHook, serve_websocket,
        )

        hook = WebSocketHook(thread_id_label="demo")

        @pin(model=MyModel, hooks=hook)
        @tool
        def my_tool(...): ...

        async def main():
            server = asyncio.create_task(serve_websocket(
                hook, html_path="examples/web/index.html",
            ))
            # ... drive your agent ...

        asyncio.run(main())
    """
    try:
        from websockets.asyncio.server import serve  # type: ignore[import-not-found]
        from websockets.datastructures import Headers
        from websockets.http11 import Response
    except ImportError as exc:
        raise ImportError(_DEPENDENCY_HINT) from exc

    clients: set[Any] = set()

    html_bytes: bytes | None = None
    if html_path is not None:
        from pathlib import Path

        html_bytes = Path(html_path).read_bytes()

    def http_route(_connection: Any, request: Any) -> Response | None:
        """Serve the HTML page on plain HTTP GET; let WS upgrades through."""
        # Upgrade requests carry the websockets headers — let them through
        # by returning None.
        if "upgrade" in request.headers.get("Connection", "").lower():
            return None
        if html_bytes is None:
            return None  # no HTML configured; default 426 Upgrade Required
        path = request.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return Response(
                200, "OK",
                Headers([
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(html_bytes))),
                    ("Cache-Control", "no-store"),
                ]),
                body=html_bytes,
            )
        return Response(
            404, "Not Found",
            Headers([("Content-Type", "text/plain")]),
            body=b"not found\n",
        )

    async def handler(ws: Any) -> None:
        clients.add(ws)
        try:
            snap = hook.latest_snapshot()
            if snap is not None:
                await ws.send(json.dumps(snap))
            try:
                async for _ in ws:  # we don't process inbound messages
                    pass
            except Exception:  # noqa: BLE001
                pass
        finally:
            clients.discard(ws)

    async def broadcaster() -> None:
        while True:
            await asyncio.sleep(poll_interval)
            deltas = hook.drain_pending()
            if not deltas or not clients:
                continue
            messages = [json.dumps(d) for d in deltas]
            dead: list[Any] = []
            for ws in clients:
                for msg in messages:
                    try:
                        await ws.send(msg)
                    except Exception:  # noqa: BLE001 — peer gone
                        dead.append(ws)
                        break
            for ws in dead:
                clients.discard(ws)

    async with serve(handler, host, port, process_request=http_route):
        logger.info("AgentPinBoard server on http://%s:%d  (ws on same port)", host, port)
        await broadcaster()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _node_to_payload(node: FactNode | EventNode) -> dict[str, Any]:
    if isinstance(node, EventNode):
        return {
            "kind": "event",
            "id": node.id,
            "node_type": EVENT_NODE_TYPE,
            "source_tool": node.source_tool,
            "timestamp": node.timestamp.isoformat(),
            # Base label; the frontend may enrich it using connected
            # fact edges (see index.html :: refreshEventLabels).
            "label": f"{node.source_tool}@{node.timestamp.strftime('%H:%M:%S')}",
        }
    return {
        "kind": "fact",
        "id": node.id,
        "node_type": node.node_type,
        "value": node.value,
        "label": f"{node.node_type}: {node.value}",
    }


def _edge_to_payload(edge: FactEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source": edge.event_id,
        "target": edge.target_id,
        "edge_type": edge.edge_type,
        "label": edge.edge_type.split(".", 1)[-1],
    }


def _build_snapshot(graph: FactGraph, *, thread_id_label: str) -> dict[str, Any]:
    nodes_payload: list[dict[str, Any]] = []
    for nid in graph.g.nodes:
        n = graph.get(nid)
        if n is not None:
            nodes_payload.append(_node_to_payload(n))
    edges_payload: list[dict[str, Any]] = []
    for _src, _tgt, data in graph.g.edges(data=True):
        edge = data.get("obj")
        if isinstance(edge, FactEdge):
            edges_payload.append(_edge_to_payload(edge))
    return {
        "type": "snapshot",
        "thread_id": thread_id_label,
        "nodes": nodes_payload,
        "edges": edges_payload,
    }


__all__ = ["WebSocketHook", "serve_websocket"]


# Re-export AsyncIterator only when explicitly imported; satisfies linters.
_ = AsyncIterator  # noqa: F841
