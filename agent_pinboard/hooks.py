"""Observability hooks.

AgentPinBoard fires hooks on every graph change. Each call is wrapped in
``try/except`` and any exception is logged at ERROR — the hook never
breaks ingestion.

The base class is plain Python (no ``ABC``) so users can override only
what they need; the rest stay no-ops. Subclasses are encouraged to use
``@typing.override`` (Python 3.12+) to catch typos in method names.
"""

from __future__ import annotations

import logging

from agent_pinboard.graph import FactGraph
from agent_pinboard.models import EventId, EventNode, FactEdge, FactNode, IngestResult

logger = logging.getLogger(__name__)


class AgentPinBoardHooks:
    """Override any subset of these to observe graph mutations."""

    def on_node_added(self, node: FactNode | EventNode) -> None: ...

    def on_edge_added(self, edge: FactEdge) -> None: ...

    def on_link_found(self, existing: FactNode, event_id: EventId) -> None: ...

    def on_ingest_complete(self, result: IngestResult) -> None: ...

    def on_graph_changed(self, graph: FactGraph) -> None: ...


class LoggingHook(AgentPinBoardHooks):
    """Emit a log line for every event. Useful while wiring an agent up."""

    def __init__(self, *, level: int = logging.INFO) -> None:
        self._level = level

    def on_node_added(self, node: FactNode | EventNode) -> None:
        logger.log(self._level, "node_added: %s id=%s", node.node_type, node.id)

    def on_edge_added(self, edge: FactEdge) -> None:
        logger.log(self._level, "edge_added: %s -> %s", edge.event_id, edge.target_id)

    def on_link_found(self, existing: FactNode, event_id: EventId) -> None:
        logger.log(
            self._level,
            "link_found: %s value=%r in event=%s",
            existing.node_type,
            existing.value,
            event_id,
        )

    def on_ingest_complete(self, result: IngestResult) -> None:
        logger.log(
            self._level,
            "ingest_complete: +%d nodes, +%d linked, +%d edges",
            result.new_nodes,
            result.linked_nodes,
            result.new_edges,
        )


class CompositeHook(AgentPinBoardHooks):
    """Forward each callback to a list of underlying hooks, in order."""

    def __init__(self, hooks: list[AgentPinBoardHooks]) -> None:
        self._hooks = list(hooks)

    def on_node_added(self, node: FactNode | EventNode) -> None:
        for h in self._hooks:
            _safe_call(h.on_node_added, node)

    def on_edge_added(self, edge: FactEdge) -> None:
        for h in self._hooks:
            _safe_call(h.on_edge_added, edge)

    def on_link_found(self, existing: FactNode, event_id: EventId) -> None:
        for h in self._hooks:
            _safe_call(h.on_link_found, existing, event_id)

    def on_ingest_complete(self, result: IngestResult) -> None:
        for h in self._hooks:
            _safe_call(h.on_ingest_complete, result)

    def on_graph_changed(self, graph: FactGraph) -> None:
        for h in self._hooks:
            _safe_call(h.on_graph_changed, graph)


def _safe_call(fn, *args) -> None:
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 — hook isolation contract
        logger.error("hook %s failed", getattr(fn, "__qualname__", fn), exc_info=True)


def fire(hooks: AgentPinBoardHooks | None, method: str, *args) -> None:
    """Invoke a hook method by name with the log-and-continue contract."""
    if hooks is None:
        return
    fn = getattr(hooks, method, None)
    if fn is None:
        return
    _safe_call(fn, *args)


__all__ = ["CompositeHook", "LoggingHook", "AgentPinBoardHooks", "fire"]
