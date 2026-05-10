"""Sharded persistence layer over LangGraph ``BaseStore``.

The graph is split across four namespaces under ``("agent_pinboard", thread_id, ...)``:

* ``"nodes"`` — one key per ``FactNode`` / ``EventNode``.
* ``"edges"`` — one key per ``FactEdge``.
* ``"entities"`` — single ``"registry"`` key with the session entity registry.
* ``"tool_calls"`` — one key per ``ToolCallRecord``.

Loading a session = one ``store.search`` per namespace; ingestion writes only
the touched keys (delta), never the whole graph as a blob.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_pinboard.entity import Entity
from agent_pinboard.graph import FactGraph
from agent_pinboard.models import (
    EVENT_NODE_TYPE,
    EventNode,
    FactEdge,
    FactNode,
    ToolCallRecord,
)

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore


# Generous per-namespace read limit. Session-scope graphs are not expected
# to exceed this; if they do, README §16 reaches "out of scope (sharded
# pagination — Phase 3)".
_SCAN_LIMIT = 10_000


# --------------------------------------------------------------------------- #
# Namespace builders.                                                         #
# --------------------------------------------------------------------------- #

def _ns_root(thread_id: str) -> tuple[str, ...]:
    return ("agent_pinboard", thread_id)


def _ns_nodes(thread_id: str) -> tuple[str, ...]:
    return ("agent_pinboard", thread_id, "nodes")


def _ns_edges(thread_id: str) -> tuple[str, ...]:
    return ("agent_pinboard", thread_id, "edges")


def _ns_entities(thread_id: str) -> tuple[str, ...]:
    return ("agent_pinboard", thread_id, "entities")


def _ns_tool_calls(thread_id: str) -> tuple[str, ...]:
    return ("agent_pinboard", thread_id, "tool_calls")


def _ns_raw_events(thread_id: str) -> tuple[str, ...]:
    return ("agent_pinboard", thread_id, "raw_events")


# --------------------------------------------------------------------------- #
# Serialization (dataclass <-> dict).                                         #
# --------------------------------------------------------------------------- #

def _node_to_dict(n: FactNode | EventNode) -> dict[str, Any]:
    if isinstance(n, EventNode):
        return {
            "kind": "event",
            "id": n.id,
            "node_type": n.node_type,
            "source_tool": n.source_tool,
            "timestamp": n.timestamp.isoformat(),
            "properties": n.properties,
        }
    return {
        "kind": "fact",
        "id": n.id,
        "node_type": n.node_type,
        "value": n.value,
        "canonical_value": n.canonical_value,
        "properties": n.properties,
        "first_seen": n.first_seen.isoformat(),
        "last_seen": n.last_seen.isoformat(),
        "source_events": list(n.source_events),
        "source_tools": sorted(n.source_tools),
    }


def _node_from_dict(d: dict[str, Any]) -> FactNode | EventNode:
    if d["kind"] == "event":
        return EventNode(
            id=d["id"],
            node_type=d.get("node_type", EVENT_NODE_TYPE),
            source_tool=d["source_tool"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            properties=dict(d.get("properties") or {}),
        )
    return FactNode(
        id=d["id"],
        node_type=d["node_type"],
        value=d["value"],
        canonical_value=d["canonical_value"],
        properties=dict(d.get("properties") or {}),
        first_seen=datetime.fromisoformat(d["first_seen"]),
        last_seen=datetime.fromisoformat(d["last_seen"]),
        source_events=list(d.get("source_events") or []),
        source_tools=set(d.get("source_tools") or ()),
    )


def _edge_to_dict(e: FactEdge) -> dict[str, Any]:
    return {
        "id": e.id,
        "event_id": e.event_id,
        "target_id": e.target_id,
        "edge_type": e.edge_type,
        "description": e.description,
    }


def _edge_from_dict(d: dict[str, Any]) -> FactEdge:
    return FactEdge(
        event_id=d["event_id"],
        target_id=d["target_id"],
        edge_type=d["edge_type"],
        description=d["description"],
    )


def _tool_call_to_dict(r: ToolCallRecord) -> dict[str, Any]:
    return {
        "tool_name": r.tool_name,
        "args_repr": r.args_repr,
        "timestamp": r.timestamp.isoformat(),
        "event_id": r.event_id,
        "summary": r.summary,
        "duration_ms": r.duration_ms,
    }


def _tool_call_from_dict(d: dict[str, Any]) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=d["tool_name"],
        args_repr=d["args_repr"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        event_id=d.get("event_id"),
        summary=d["summary"],
        duration_ms=d["duration_ms"],
    )


def _entity_to_dict(e: Entity) -> dict[str, Any]:
    # Normalizer is a callable — not portable across processes. We stash
    # the qualified name for diagnostics only; the live runtime entity stays
    # in the in-process registry.
    norm_name = (
        f"{e.normalizer.__module__}.{e.normalizer.__qualname__}"
        if e.normalizer is not None
        else None
    )
    return {
        "name": e.name,
        "description": e.description,
        "normalizer": norm_name,
    }


# --------------------------------------------------------------------------- #
# Sync API.                                                                   #
# --------------------------------------------------------------------------- #

def load_graph(store: BaseStore, thread_id: str) -> FactGraph:
    """Load the session graph from sharded keys."""
    node_items = store.search(_ns_nodes(thread_id), limit=_SCAN_LIMIT)
    edge_items = store.search(_ns_edges(thread_id), limit=_SCAN_LIMIT)
    nodes = [_node_from_dict(item.value) for item in node_items]
    edges = [_edge_from_dict(item.value) for item in edge_items]
    return FactGraph.from_snapshot(nodes, edges)


def persist_delta(
    store: BaseStore,
    thread_id: str,
    new_or_updated_nodes: list[FactNode | EventNode],
    new_edges: list[FactEdge],
) -> None:
    """Write only the touched keys."""
    nodes_ns = _ns_nodes(thread_id)
    edges_ns = _ns_edges(thread_id)
    for n in new_or_updated_nodes:
        store.put(nodes_ns, n.id, _node_to_dict(n))
    for e in new_edges:
        store.put(edges_ns, e.id, _edge_to_dict(e))


def persist_tool_call(store: BaseStore, thread_id: str, record: ToolCallRecord) -> None:
    key = f"{record.timestamp.isoformat()}|{record.tool_name}"
    store.put(_ns_tool_calls(thread_id), key, _tool_call_to_dict(record))


def load_tool_calls(store: BaseStore, thread_id: str) -> list[ToolCallRecord]:
    items = store.search(_ns_tool_calls(thread_id), limit=_SCAN_LIMIT)
    records = [_tool_call_from_dict(i.value) for i in items]
    records.sort(key=lambda r: r.timestamp)
    return records


def save_entities_registry(
    store: BaseStore, thread_id: str, registry: dict[str, Entity]
) -> None:
    payload = {name: _entity_to_dict(e) for name, e in registry.items()}
    store.put(_ns_entities(thread_id), "registry", {"entities": payload})


def persist_raw_event(
    store: BaseStore, thread_id: str, event_id: str, raw: Any
) -> None:
    """Stash the tool's raw return for a specific event under ``raw_events``."""
    store.put(_ns_raw_events(thread_id), event_id, {"raw": _jsonable(raw)})


def load_raw_event(
    store: BaseStore, thread_id: str, event_id: str
) -> Any | None:
    """Return the stashed raw return, or ``None`` if it was not stored."""
    item = store.get(_ns_raw_events(thread_id), event_id)
    return None if item is None else item.value.get("raw")


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-serialisable structure (mirrors decorator helper)."""
    from datetime import date, datetime

    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


# --------------------------------------------------------------------------- #
# Async API — symmetrical, used by async tools.                               #
# --------------------------------------------------------------------------- #

async def aload_graph(store: BaseStore, thread_id: str) -> FactGraph:
    node_items = await store.asearch(_ns_nodes(thread_id), limit=_SCAN_LIMIT)
    edge_items = await store.asearch(_ns_edges(thread_id), limit=_SCAN_LIMIT)
    nodes = [_node_from_dict(item.value) for item in node_items]
    edges = [_edge_from_dict(item.value) for item in edge_items]
    return FactGraph.from_snapshot(nodes, edges)


async def apersist_delta(
    store: BaseStore,
    thread_id: str,
    new_or_updated_nodes: list[FactNode | EventNode],
    new_edges: list[FactEdge],
) -> None:
    nodes_ns = _ns_nodes(thread_id)
    edges_ns = _ns_edges(thread_id)
    for n in new_or_updated_nodes:
        await store.aput(nodes_ns, n.id, _node_to_dict(n))
    for e in new_edges:
        await store.aput(edges_ns, e.id, _edge_to_dict(e))


async def apersist_tool_call(
    store: BaseStore, thread_id: str, record: ToolCallRecord
) -> None:
    key = f"{record.timestamp.isoformat()}|{record.tool_name}"
    await store.aput(_ns_tool_calls(thread_id), key, _tool_call_to_dict(record))


async def aload_tool_calls(store: BaseStore, thread_id: str) -> list[ToolCallRecord]:
    items = await store.asearch(_ns_tool_calls(thread_id), limit=_SCAN_LIMIT)
    records = [_tool_call_from_dict(i.value) for i in items]
    records.sort(key=lambda r: r.timestamp)
    return records


async def asave_entities_registry(
    store: BaseStore, thread_id: str, registry: dict[str, Entity]
) -> None:
    payload = {name: _entity_to_dict(e) for name, e in registry.items()}
    await store.aput(_ns_entities(thread_id), "registry", {"entities": payload})


async def apersist_raw_event(
    store: BaseStore, thread_id: str, event_id: str, raw: Any
) -> None:
    await store.aput(_ns_raw_events(thread_id), event_id, {"raw": _jsonable(raw)})


async def aload_raw_event(
    store: BaseStore, thread_id: str, event_id: str
) -> Any | None:
    item = await store.aget(_ns_raw_events(thread_id), event_id)
    return None if item is None else item.value.get("raw")


# Re-exported for the test suite.
__all__ = [
    "aload_graph",
    "aload_raw_event",
    "aload_tool_calls",
    "apersist_delta",
    "apersist_raw_event",
    "apersist_tool_call",
    "asave_entities_registry",
    "load_graph",
    "load_raw_event",
    "load_tool_calls",
    "persist_delta",
    "persist_raw_event",
    "persist_tool_call",
    "save_entities_registry",
]
