"""Langfuse callback handler — sends ingest spans (with a Mermaid graph snapshot) to Langfuse.

Optional dependency. Install with::

    pip install agent-pinboard[langfuse]
    # or:  uv add agent-pinboard[langfuse]

Usage::

    from langfuse import Langfuse
    from agent_pinboard.integrations.langfuse_hook import LangfuseHook

    client = Langfuse(public_key=..., secret_key=..., host=...)
    handler = LangfuseHook(client)

    result = await agent.ainvoke(
        {"messages": [...]},
        config={
            "callbacks": [handler],
            "configurable": {"thread_id": "session-42"},
        },
    )

What the handler emits
----------------------

After every successful ``@pin`` ingest the decorator dispatches an
``agent_pinboard:ingest`` custom event. ``LangfuseHook`` handles it by
emitting:

* a span ``"agent_pinboard.ingest"`` with input = ingest summary,
  metadata = the per-ingest ``IngestResult`` dataclass.
* (optional) a span ``"agent_pinboard.graph_snapshot"`` whose metadata
  carries a Mermaid-flowchart rendering of the current top-N facts and
  the events that connect them. Langfuse renders Markdown/Mermaid in
  metadata, giving you a visual graph alongside the trace.

Failures inside the handler are logged at ERROR — the handler never
breaks the surrounding agent run.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from agent_pinboard.decorator import INGEST_EVENT
from agent_pinboard.graph import FactGraph
from agent_pinboard.models import EVENT_NODE_TYPE, IngestResult

if TYPE_CHECKING:
    from langfuse import Langfuse  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

_DEPENDENCY_HINT = (
    "LangfuseHook requires the langfuse package: install with "
    "`pip install agent-pinboard[langfuse]` or `pip install langfuse`."
)


class LangfuseHook(BaseCallbackHandler):
    """LangChain callback handler that fans AgentPinBoard ingest events into Langfuse spans.

    Pass an instance via ``config={"callbacks": [LangfuseHook(client)]}``
    on ``agent.invoke`` / ``ainvoke``. The handler subscribes to the
    ``agent_pinboard:ingest`` custom event emitted by ``@pin``-decorated
    tools after each successful ingest.
    """

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

    # LangChain calls handlers regardless of which run they belong to;
    # `raise_error=False` (the default) ensures exceptions inside any
    # handler method do not break the run.

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if name != INGEST_EVENT:
            return
        try:
            self._handle_ingest(data)
        except Exception:  # noqa: BLE001
            logger.error("LangfuseHook ingest dispatch failed", exc_info=True)

    # ------------------------------------------------------------------ #
    # Internals.                                                         #
    # ------------------------------------------------------------------ #

    def _handle_ingest(self, data: dict[str, Any]) -> None:
        result: IngestResult = data["result"]
        graph: FactGraph = data["graph"]
        self._emit_ingest_span(result)
        if self._emit_snapshots:
            self._emit_snapshot_span(graph)

    def _emit_ingest_span(self, result: IngestResult) -> None:
        self._client.start_observation(
            name="agent_pinboard.ingest",
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

    def _emit_snapshot_span(self, graph: FactGraph) -> None:
        mermaid = graph.to_mermaid(max_facts=self._max_facts)
        counts = {
            t: len(ids)
            for t, ids in graph.nodes_by_type.items()
            if t != EVENT_NODE_TYPE
        }
        event_count = len(graph.nodes_by_type.get(EVENT_NODE_TYPE, set()))
        self._client.start_observation(
            name="agent_pinboard.graph_snapshot",
            as_type="span",
            input={"counts": counts, "events": event_count},
            metadata={
                "mermaid": mermaid,
                "counts_by_type": counts,
                "event_count": event_count,
            },
        ).end()


def render_mermaid(graph: FactGraph, *, max_facts: int = 30) -> str:
    """Backward-compatible alias for ``FactGraph.to_mermaid``.

    Prefer the method form: ``graph.to_mermaid(max_facts=...)``.
    """
    return graph.to_mermaid(max_facts=max_facts)


def _summary(result: IngestResult) -> str:
    return (
        f"+{result.new_nodes} new, +{result.linked_nodes} linked, "
        f"+{result.new_edges} edges"
    )


__all__ = ["LangfuseHook", "render_mermaid"]
