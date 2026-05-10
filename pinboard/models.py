"""Graph data models — internal containers, NOT Pydantic.

Pydantic is reserved for user-defined tool-response models. PinBoard's own
graph state uses ``@dataclass(slots=True)`` for speed and clarity.

ID strategy:
* ``FactNode.id`` — stable hash of ``(node_type, canonical_value)``.
* ``EventNode.id`` — fresh UUID4 per tool invocation.
* ``FactEdge.id`` — derived property from ``(event_id, edge_type, target_id)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Public type aliases (PEP 695). They are plain strings, but the named
# aliases make signatures readable.
type NodeId = str
type EventId = str

# Reserved node_type used for technical event nodes. Users cannot declare
# an ``Entity(name=EVENT_NODE_TYPE)`` (validated at registration time).
EVENT_NODE_TYPE = "Event"


def fact_node_id(node_type: str, canonical_value: str) -> NodeId:
    """Stable, deterministic ID for a fact node."""
    return hashlib.sha256(f"{node_type}|{canonical_value}".encode()).hexdigest()[:16]


@dataclass(slots=True)
class FactNode:
    """A semantic entity in the graph (e.g. a specific IP, user, ARN)."""

    id: NodeId
    node_type: str
    value: str
    canonical_value: str
    properties: dict[str, Any]
    first_seen: datetime
    last_seen: datetime
    source_events: list[EventId] = field(default_factory=list)
    source_tools: set[str] = field(default_factory=set)


@dataclass(slots=True)
class EventNode:
    """A technical node representing one tool invocation.

    Always created (even if no facts were extracted), so ``what_have_i_done``
    and provenance queries see every call.
    """

    id: EventId
    source_tool: str
    timestamp: datetime
    properties: dict[str, Any] = field(default_factory=dict)
    node_type: str = EVENT_NODE_TYPE


@dataclass(slots=True, frozen=True)
class FactEdge:
    """An edge from an EventNode to a FactNode.

    Topology is always ``EventNode → FactNode`` (star around event). Every
    other piece of metadata that *could* be denormalised onto the edge
    (``source_tool``, ``created_at``) is reachable via the parent
    ``EventNode`` instead.
    """

    event_id: EventId
    target_id: NodeId
    edge_type: str  # "{ModelClass}.{field_name}"
    description: str

    @property
    def id(self) -> str:
        return f"{self.event_id}|{self.edge_type}|{self.target_id}"


@dataclass(slots=True)
class IngestResult:
    """Summary of one ``@fact`` invocation, passed to hooks and transforms."""

    event_ids: list[EventId]
    new_nodes: int
    linked_nodes: int
    new_edges: int
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ToolCallRecord:
    """One entry in the per-session tool-call log."""

    tool_name: str
    args_repr: str
    timestamp: datetime
    event_id: EventId | None
    summary: str
    duration_ms: int
