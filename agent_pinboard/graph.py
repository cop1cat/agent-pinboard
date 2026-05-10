"""``FactGraph`` — runtime container for the session graph.

NetworkX MultiDiGraph is the source of truth for topology; sidecar dicts
provide O(1) lookups that NetworkX would otherwise require a linear scan
for. Sidecars are derived state, rebuilt from scratch by
:meth:`from_snapshot`.

Every operation here is in-memory only — persistence is handled by
``agent_pinboard.store``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import networkx as nx

from agent_pinboard.entity import Entity
from agent_pinboard.exceptions import AgentPinBoardNormalizerError
from agent_pinboard.models import (
    EVENT_NODE_TYPE,
    EventId,
    EventNode,
    FactEdge,
    FactNode,
    NodeId,
    fact_node_id,
)

logger = logging.getLogger(__name__)


def _mermaid_label(f: FactNode) -> str:
    val = f.value.replace('"', '\\"')
    return f"{f.node_type}: {val}"


def _mermaid_safe(node_id: str) -> str:
    """Mermaid IDs cannot contain dashes etc. — strip to a safe alnum prefix."""
    return "n" + "".join(c for c in node_id if c.isalnum())[:24]


def _backfill_fact_provenance(
    facts: list[FactNode],
    edges: list[FactEdge],
    events_by_id: dict[EventId, EventNode],
) -> None:
    """Recompute mutable FactNode fields (source_events / source_tools / first_seen / last_seen)
    from the canonical edge + event records loaded from the Store.

    Order of source_events follows event timestamp (then event_id for stability),
    so two processes loading the same Store see the same list.
    """
    edges_by_target: dict[NodeId, list[FactEdge]] = {}
    for e in edges:
        edges_by_target.setdefault(e.target_id, []).append(e)

    for fact in facts:
        incoming = edges_by_target.get(fact.id, [])
        # Distinct event ids; deterministic order (timestamp, then id).
        seen_ev: set[EventId] = set()
        events: list[EventNode] = []
        for edge in incoming:
            ev = events_by_id.get(edge.event_id)
            if ev is None or ev.id in seen_ev:
                continue
            seen_ev.add(ev.id)
            events.append(ev)
        events.sort(key=lambda e: (e.timestamp, e.id))

        fact.source_events = [e.id for e in events]
        fact.source_tools = {e.source_tool for e in events}
        if events:
            fact.first_seen = events[0].timestamp
            fact.last_seen = events[-1].timestamp


class FactGraph:
    """In-memory graph + sidecar indices.

    Topology lives in ``self.g`` (a ``MultiDiGraph``). Sidecars accelerate
    the hot paths:

    * ``nodes_by_key`` — autolink lookup by ``(type, canonical_value)``.
    * ``nodes_by_type`` — fast slicing for ``search_nodes`` / ``graph_summary``.
    """

    __slots__ = ("g", "nodes_by_key", "nodes_by_type")

    def __init__(self) -> None:
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()
        self.nodes_by_key: dict[tuple[str, str], NodeId] = {}
        self.nodes_by_type: dict[str, set[NodeId]] = {}

    # ------------------------------------------------------------------ #
    # Mutations.                                                         #
    # ------------------------------------------------------------------ #

    def add_event(self, event: EventNode) -> None:
        """Add an EventNode to the graph (always new — UUID4 IDs)."""
        self.g.add_node(event.id, kind="event", obj=event)
        self.nodes_by_type.setdefault(EVENT_NODE_TYPE, set()).add(event.id)

    def upsert_fact(
        self,
        entity: Entity,
        value: Any,
        event_id: EventId,
        source_tool: str,
        *,
        warnings: list[str] | None = None,
    ) -> tuple[NodeId | None, bool]:
        """Insert-or-link a fact node.

        Returns ``(node_id, was_new)``. If the value canonicalises to an
        empty string, returns ``(None, False)`` and appends a warning to
        ``warnings`` (the caller — typically ``IngestResult`` — can surface it).

        Raises :class:`AgentPinBoardNormalizerError` if the user-supplied
        normalizer crashes — this is fail-loud per README §6.5.
        """
        if value is None:
            return None, False

        try:
            canonical = entity.normalizer(value) if entity.normalizer else str(value)
        except Exception as exc:  # noqa: BLE001 — wrap any user error
            raise AgentPinBoardNormalizerError(
                f"normalizer for Entity({entity.name!r}) failed on value {value!r}: {exc}"
            ) from exc

        if not canonical:
            msg = (
                f"empty canonical_value for Entity({entity.name!r}) value={value!r}; "
                "fact dropped to avoid graph poisoning"
            )
            logger.warning(msg)
            if warnings is not None:
                warnings.append(msg)
            return None, False

        key = (entity.name, canonical)
        now = datetime.now(UTC)
        existing_id = self.nodes_by_key.get(key)

        if existing_id is not None:
            n = self.g.nodes[existing_id]["obj"]
            assert isinstance(n, FactNode)
            n.last_seen = now
            n.source_events.append(event_id)
            n.source_tools.add(source_tool)
            return existing_id, False

        node_id = fact_node_id(entity.name, canonical)
        fact = FactNode(
            id=node_id,
            node_type=entity.name,
            value=str(value),
            canonical_value=canonical,
            first_seen=now,
            last_seen=now,
            source_events=[event_id],
            source_tools={source_tool},
        )
        self.g.add_node(node_id, kind="fact", obj=fact)
        self.nodes_by_key[key] = node_id
        self.nodes_by_type.setdefault(entity.name, set()).add(node_id)
        return node_id, True

    def add_edge(self, edge: FactEdge) -> None:
        """Add a FactEdge to the graph. Multi-edges between the same pair are allowed."""
        self.g.add_edge(
            edge.event_id,
            edge.target_id,
            key=edge.id,
            obj=edge,
        )

    # ------------------------------------------------------------------ #
    # Queries.                                                           #
    # ------------------------------------------------------------------ #

    def get(self, node_id: NodeId) -> FactNode | EventNode | None:
        data = self.g.nodes.get(node_id)
        if data is None:
            return None
        return data["obj"]

    def search_by_type(self, node_type: str) -> list[NodeId]:
        return list(self.nodes_by_type.get(node_type, set()))

    def find_by_value(self, node_type: str, value: str) -> NodeId | None:
        """Find a fact node by displayed ``value`` (not canonical) within a type."""
        for nid in self.nodes_by_type.get(node_type, ()):
            n = self.g.nodes[nid]["obj"]
            if isinstance(n, FactNode) and n.value == value:
                return nid
        return None

    def edges_for_event(self, event_id: EventId) -> list[FactEdge]:
        out: list[FactEdge] = []
        if event_id not in self.g:
            return out
        for _, _, data in self.g.out_edges(event_id, data=True):
            edge = data.get("obj")
            if isinstance(edge, FactEdge):
                out.append(edge)
        return out

    def all_events(self) -> list[EventNode]:
        result: list[EventNode] = []
        for nid in self.nodes_by_type.get(EVENT_NODE_TYPE, ()):
            n = self.g.nodes[nid]["obj"]
            if isinstance(n, EventNode):
                result.append(n)
        return result

    def all_facts(self) -> Iterable[FactNode]:
        for ntype, ids in self.nodes_by_type.items():
            if ntype == EVENT_NODE_TYPE:
                continue
            for nid in ids:
                n = self.g.nodes[nid]["obj"]
                if isinstance(n, FactNode):
                    yield n

    # ------------------------------------------------------------------ #
    # Snapshot / restore — used by Store layer.                          #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Rendering.                                                         #
    # ------------------------------------------------------------------ #

    def to_mermaid(self, *, max_facts: int = 30) -> str:
        """Render the current graph as a Mermaid flowchart string.

        Top-``max_facts`` facts (by event count) are kept; the rest
        are summarised as a single ``... (N more)`` node. Events that
        connect at least one kept fact are rendered; orphan events
        are omitted.

        Useful standalone (debugging, ad-hoc dumps) and as the data
        source for ``LangfuseHook``'s ``agent_pinboard.graph_snapshot``
        span.
        """
        facts: list[FactNode] = []
        for ntype, ids in self.nodes_by_type.items():
            if ntype == EVENT_NODE_TYPE:
                continue
            for nid in ids:
                n = self.get(nid)
                if isinstance(n, FactNode):
                    facts.append(n)
        facts.sort(key=lambda f: -len(f.source_events))
        keep = facts[:max_facts]
        extra = max(0, len(facts) - len(keep))
        keep_ids = {f.id for f in keep}

        lines = ["flowchart LR"]
        seen_event_ids: set[str] = set()
        for f in keep:
            lines.append(f'  {_mermaid_safe(f.id)}["{_mermaid_label(f)}"]')
        for f in keep:
            for ev_id in f.source_events:
                if ev_id in seen_event_ids:
                    continue
                ev = self.get(ev_id)
                if not isinstance(ev, EventNode):
                    continue
                edges = self.edges_for_event(ev_id)
                connected = [e for e in edges if e.target_id in keep_ids]
                if len(connected) < 1:
                    continue
                seen_event_ids.add(ev_id)
                ev_label = f"{ev.source_tool}@{ev.timestamp.strftime('%H:%M:%S')}"
                lines.append(f'  {_mermaid_safe(ev_id)}(("{ev_label}"))')
                for edge in connected:
                    lines.append(f"  {_mermaid_safe(ev_id)} --> {_mermaid_safe(edge.target_id)}")
        if extra > 0:
            lines.append(f'  more[/"... and {extra} more facts"/]')
        return "\n".join(lines)

    @classmethod
    def from_snapshot(
        cls,
        nodes: Iterable[FactNode | EventNode],
        edges: Iterable[FactEdge],
    ) -> FactGraph:
        """Rebuild a FactGraph (and its sidecars) from a flat snapshot.

        FactNodes are persisted as their immutable subset only
        (id / type / value / canonical_value). The mutable fields
        (``source_events``, ``source_tools``, ``first_seen``, ``last_seen``)
        are derived here by walking the loaded edges and EventNodes, so
        two processes upserting the same canonical fact never lose each
        other's links.
        """
        g = cls()
        events_by_id: dict[EventId, EventNode] = {}
        facts: list[FactNode] = []
        for n in nodes:
            if isinstance(n, EventNode):
                g.g.add_node(n.id, kind="event", obj=n)
                g.nodes_by_type.setdefault(EVENT_NODE_TYPE, set()).add(n.id)
                events_by_id[n.id] = n
            else:
                g.g.add_node(n.id, kind="fact", obj=n)
                g.nodes_by_key[(n.node_type, n.canonical_value)] = n.id
                g.nodes_by_type.setdefault(n.node_type, set()).add(n.id)
                facts.append(n)
        edge_list = list(edges)
        for e in edge_list:
            g.g.add_edge(e.event_id, e.target_id, key=e.id, obj=e)
        _backfill_fact_provenance(facts, edge_list, events_by_id)
        return g

    # ------------------------------------------------------------------ #
    # Portable dump / load (JSON-friendly).                              #
    # ------------------------------------------------------------------ #

    def dump_to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict for snapshotting / archival.

        The format is stable for the duration of one library minor
        version; ``agent_pinboard_version`` is included so future migrators
        can detect older dumps.
        """
        # Local import avoids a top-level circular dependency.
        from agent_pinboard import __version__ as _agent_pinboard_version
        from agent_pinboard import store as store_io

        nodes_payload: list[dict[str, Any]] = []
        for nid in self.g.nodes:
            n = self.g.nodes[nid]["obj"]
            nodes_payload.append(store_io._node_to_dict(n))
        edges_payload: list[dict[str, Any]] = []
        for _src, _tgt, data in self.g.edges(data=True):
            edge = data.get("obj")
            if isinstance(edge, FactEdge):
                edges_payload.append(store_io._edge_to_dict(edge))
        return {
            "agent_pinboard_version": _agent_pinboard_version,
            "schema": "agent_pinboard.factgraph",
            "nodes": nodes_payload,
            "edges": edges_payload,
        }

    @classmethod
    def load_from_dict(cls, payload: dict[str, Any]) -> FactGraph:
        """Inverse of :meth:`dump_to_dict`.

        Raises :class:`ValueError` if the payload is missing required
        keys. Cross-version migration is the caller's responsibility:
        check ``payload["agent_pinboard_version"]`` if the format may have
        changed since the dump was written.
        """
        from agent_pinboard import store as store_io

        if not isinstance(payload, dict):
            raise ValueError("FactGraph.load_from_dict expects a dict payload")
        if payload.get("schema") != "agent_pinboard.factgraph":
            raise ValueError(
                "payload schema mismatch: expected 'agent_pinboard.factgraph', "
                f"got {payload.get('schema')!r}"
            )
        nodes = [store_io._node_from_dict(d) for d in payload.get("nodes", [])]
        edges = [store_io._edge_from_dict(d) for d in payload.get("edges", [])]
        return cls.from_snapshot(nodes, edges)
