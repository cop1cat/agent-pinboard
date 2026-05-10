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
            properties={},
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

    @classmethod
    def from_snapshot(
        cls,
        nodes: Iterable[FactNode | EventNode],
        edges: Iterable[FactEdge],
    ) -> FactGraph:
        """Rebuild a FactGraph (and its sidecars) from a flat snapshot."""
        g = cls()
        for n in nodes:
            if isinstance(n, EventNode):
                g.g.add_node(n.id, kind="event", obj=n)
                g.nodes_by_type.setdefault(EVENT_NODE_TYPE, set()).add(n.id)
            else:
                g.g.add_node(n.id, kind="fact", obj=n)
                g.nodes_by_key[(n.node_type, n.canonical_value)] = n.id
                g.nodes_by_type.setdefault(n.node_type, set()).add(n.id)
        for e in edges:
            g.g.add_edge(e.event_id, e.target_id, key=e.id, obj=e)
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
        from agent_pinboard import __version__ as _agent_agent_pinboard_version
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
            "agent_pinboard_version": _agent_agent_pinboard_version,
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
