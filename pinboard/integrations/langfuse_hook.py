"""Langfuse hook — sends ingest spans (with a Mermaid graph snapshot) to Langfuse.

Optional dependency. Install with::

    pip install pinboard[langfuse]
    # or:  uv add pinboard[langfuse]

Usage::

    from langfuse import Langfuse
    from pinboard.integrations.langfuse_hook import LangfuseHook

    client = Langfuse(public_key=..., secret_key=..., host=...)
    hooks = LangfuseHook(client)

    @fact(model=MyModel, hooks=hooks)
    @tool
    def my_tool(...): ...

What the hook emits
-------------------

* On every ``on_ingest_complete`` — a Langfuse span ``"pinboard.ingest"``
  with input = ingest summary, metadata = the per-ingest
  ``IngestResult`` dataclass.
* On every ``on_graph_changed`` — a Langfuse span
  ``"pinboard.graph_snapshot"`` whose metadata carries a
  Mermaid-flowchart rendering of the current top-N facts and the events
  that connect them. Langfuse renders Markdown/Mermaid in metadata,
  giving you a visual graph alongside the trace.

The hook never raises; failures are logged at ERROR (the underlying
``PinBoardHooks`` contract is preserved).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, override

from pinboard.graph import FactGraph
from pinboard.hooks import PinBoardHooks
from pinboard.models import EVENT_NODE_TYPE, EventNode, FactNode, IngestResult

if TYPE_CHECKING:
    from langfuse import Langfuse  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

_DEPENDENCY_HINT = (
    "LangfuseHook requires the langfuse package: install with "
    "`pip install pinboard[langfuse]` or `pip install langfuse`."
)


class LangfuseHook(PinBoardHooks):
    """PinBoard hook that fans graph events into Langfuse spans."""

    def __init__(
        self,
        client: Langfuse,
        *,
        max_facts_in_snapshot: int = 30,
        emit_snapshots: bool = True,
    ) -> None:
        try:
            from langfuse import Langfuse  # noqa: F401
        except ImportError as exc:
            raise ImportError(_DEPENDENCY_HINT) from exc

        if client is None:
            raise ValueError("LangfuseHook requires a non-None Langfuse client")

        self._client = client
        self._max_facts = max_facts_in_snapshot
        self._emit_snapshots = emit_snapshots

    @override
    def on_ingest_complete(self, result: IngestResult) -> None:
        try:
            self._client.start_observation(
                name="pinboard.ingest",
                as_type="span",
                input=_summary(result),
                output={
                    "new_nodes": result.new_nodes,
                    "linked_nodes": result.linked_nodes,
                    "new_edges": result.new_edges,
                },
                metadata={
                    "event_ids": result.event_ids,
                    "warnings": result.warnings,
                    "result": asdict(result),
                },
                level="WARNING" if result.warnings else "DEFAULT",
            ).end()
        except Exception:  # noqa: BLE001
            logger.error("LangfuseHook.on_ingest_complete failed", exc_info=True)

    @override
    def on_graph_changed(self, graph: FactGraph) -> None:
        if not self._emit_snapshots:
            return
        try:
            mermaid = render_mermaid(graph, max_facts=self._max_facts)
            counts = {
                t: len(ids)
                for t, ids in graph.nodes_by_type.items()
                if t != EVENT_NODE_TYPE
            }
            event_count = len(graph.nodes_by_type.get(EVENT_NODE_TYPE, set()))
            self._client.start_observation(
                name="pinboard.graph_snapshot",
                as_type="span",
                input={"counts": counts, "events": event_count},
                metadata={
                    "mermaid": mermaid,
                    "counts_by_type": counts,
                    "event_count": event_count,
                },
            ).end()
        except Exception:  # noqa: BLE001
            logger.error("LangfuseHook.on_graph_changed failed", exc_info=True)


# --------------------------------------------------------------------------- #
# Mermaid renderer — also useful standalone in tests / debug scripts.         #
# --------------------------------------------------------------------------- #

def render_mermaid(graph: FactGraph, *, max_facts: int = 30) -> str:
    """Render the current graph as a Mermaid flowchart string.

    Top-`max_facts` facts (by event count) are kept; the rest are summarised
    as a single ``... (N more)`` node. Events that connect kept facts are
    rendered; orphan events are omitted.
    """
    facts: list[FactNode] = []
    for ntype, ids in graph.nodes_by_type.items():
        if ntype == EVENT_NODE_TYPE:
            continue
        for nid in ids:
            n = graph.get(nid)
            if isinstance(n, FactNode):
                facts.append(n)
    facts.sort(key=lambda f: -len(f.source_events))
    keep = facts[:max_facts]
    extra = max(0, len(facts) - len(keep))
    keep_ids = {f.id for f in keep}

    lines = ["flowchart LR"]
    seen_event_ids: set[str] = set()
    for f in keep:
        lines.append(f'  {_safe(f.id)}["{_label(f)}"]')
    for f in keep:
        for ev_id in f.source_events:
            if ev_id in seen_event_ids:
                continue
            ev = graph.get(ev_id)
            if not isinstance(ev, EventNode):
                continue
            edges = graph.edges_for_event(ev_id)
            connected = [e for e in edges if e.target_id in keep_ids]
            if len(connected) < 1:
                continue
            seen_event_ids.add(ev_id)
            ev_label = f"{ev.source_tool}@{ev.timestamp.strftime('%H:%M:%S')}"
            lines.append(f'  {_safe(ev_id)}(("{ev_label}"))')
            for edge in connected:
                lines.append(f"  {_safe(ev_id)} --> {_safe(edge.target_id)}")
    if extra > 0:
        lines.append(f'  more[/"... and {extra} more facts"/]')
    return "\n".join(lines)


def _label(f: FactNode) -> str:
    # Mermaid requires escaping double quotes inside labels.
    val = f.value.replace('"', '\\"')
    return f"{f.node_type}: {val}"


def _safe(node_id: str) -> str:
    """Mermaid IDs cannot contain dashes etc. — strip to a safe alnum prefix."""
    return "n" + "".join(c for c in node_id if c.isalnum())[:24]


def _summary(result: IngestResult) -> str:
    return (
        f"+{result.new_nodes} new, +{result.linked_nodes} linked, "
        f"+{result.new_edges} edges"
    )


__all__ = ["LangfuseHook", "render_mermaid"]
