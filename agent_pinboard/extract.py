"""Extract :class:`FactNode` / :class:`FactEdge` deltas from a Pydantic model.

Implements the five extraction rules from README §4.1 via structural
pattern matching. Pure function — does not mutate the graph; returns a
``(nodes, edges, properties, warnings)`` quadruple that the ``@fact``
decorator merges under the session lock.

``edge_type`` is always ``"{ModelClass}.{field_name}"``, where
``ModelClass`` is the class in which the field is **declared** (resolved
via MRO walk on ``__annotations__``), so fields inherited from a base
class — or reused via composition — produce stable, sharable edge labels.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from pinboard.exceptions import PinBoardExtractionError
from pinboard.fields import field_entity
from pinboard.graph import FactGraph
from pinboard.models import EventId, FactEdge, FactNode


def extract(
    model: BaseModel,
    graph: FactGraph,
    event_id: EventId,
    source_tool: str,
) -> tuple[list[FactNode], list[FactNode], list[FactEdge], list[str]]:
    """Walk ``model``, mutating ``graph`` with new facts and edges.

    Returns ``(new_facts, linked_facts, new_edges, warnings)``:

    * ``new_facts`` — fact nodes created during this extraction (was_new=True).
    * ``linked_facts`` — distinct existing fact nodes that were re-linked
      (was_new=False). Used to fire ``on_link_found`` and to count
      ``IngestResult.linked_nodes`` correctly (one count per node, not per edge).
    * ``new_edges`` — every edge created (one per fact occurrence in the model).
    * ``warnings`` — non-fatal messages (empty canonical, etc.).
    """
    new_facts: list[FactNode] = []
    linked_facts: list[FactNode] = []
    seen_linked: set[str] = set()
    new_edges: list[FactEdge] = []
    warnings: list[str] = []
    _walk(
        model, graph, event_id, source_tool,
        new_facts, linked_facts, seen_linked, new_edges, warnings,
    )
    return new_facts, linked_facts, new_edges, warnings


def event_properties(model: BaseModel) -> dict[str, Any]:
    """Collect rule-5 scalars: non-node fields that go on the EventNode."""
    out: dict[str, Any] = {}
    for name, info in type(model).model_fields.items():
        if field_entity(info) is not None:
            continue
        value = getattr(model, name)
        if value is None:
            continue
        if isinstance(value, (BaseModel, list, dict, tuple)):
            continue
        out[name] = value
    return out


# --------------------------------------------------------------------------- #
# Internals.                                                                  #
# --------------------------------------------------------------------------- #

def _walk(
    model: BaseModel,
    graph: FactGraph,
    event_id: EventId,
    source_tool: str,
    new_facts: list[FactNode],
    linked_facts: list[FactNode],
    seen_linked: set[str],
    new_edges: list[FactEdge],
    warnings: list[str],
) -> None:
    model_cls = type(model)
    for name, info in model_cls.model_fields.items():
        value = getattr(model, name)
        entity = field_entity(info)
        edge_type = f"{_declaring_class(model_cls, name)}.{name}"
        edge_desc = info.description or ""

        match value, entity:
            # Rule 2 — None skipped.
            case None, _:
                continue

            # Rule 3 — list[primitive] with node() metadata.
            case list(), _ if entity is not None:
                for item in value:
                    if isinstance(item, (BaseModel, list, dict, tuple)):
                        raise PinBoardExtractionError(
                            f"field {model_cls.__name__}.{name}: list with node() expects "
                            f"primitives, got {type(item).__name__}"
                        )
                    _emit_one(
                        item, entity, graph, event_id, source_tool,
                        edge_type, edge_desc,
                        new_facts, linked_facts, seen_linked, new_edges, warnings,
                    )

            # Rule 4 (BaseModel variant) — recurse into nested model.
            case BaseModel(), None:
                _walk(
                    value, graph, event_id, source_tool,
                    new_facts, linked_facts, seen_linked, new_edges, warnings,
                )

            # Rule 4 (list variant) — recurse into each BaseModel-element.
            case list(), None:
                for item in value:
                    if isinstance(item, BaseModel):
                        _walk(
                            item, graph, event_id, source_tool,
                            new_facts, linked_facts, seen_linked, new_edges, warnings,
                        )
                    elif isinstance(item, (dict, list, tuple)):
                        raise PinBoardExtractionError(
                            f"field {model_cls.__name__}.{name}: list of {type(item).__name__} "
                            "is not supported (use list[BaseModel] or list[primitive] with node())"
                        )
                    # Plain primitives without node() are silently ignored
                    # (rule 5 doesn't apply to list elements).

            # Rule 1 — primitive with node() metadata.
            case _, _ if entity is not None:
                if isinstance(value, (dict, tuple)):
                    raise PinBoardExtractionError(
                        f"field {model_cls.__name__}.{name}: node() expects a primitive, "
                        f"got {type(value).__name__}"
                    )
                _emit_one(
                    value, entity, graph, event_id, source_tool,
                    edge_type, edge_desc,
                    new_facts, linked_facts, seen_linked, new_edges, warnings,
                )

            # Rule 5 falls through — handled by event_properties().
            case _:
                pass


def _emit_one(
    value: Any,
    entity,
    graph: FactGraph,
    event_id: EventId,
    source_tool: str,
    edge_type: str,
    edge_desc: str,
    new_facts: list[FactNode],
    linked_facts: list[FactNode],
    seen_linked: set[str],
    new_edges: list[FactEdge],
    warnings: list[str],
) -> None:
    target_id, was_new = graph.upsert_fact(
        entity, value, event_id, source_tool, warnings=warnings
    )
    if target_id is None:
        return
    edge = FactEdge(
        event_id=event_id,
        target_id=target_id,
        edge_type=edge_type,
        description=edge_desc,
    )
    graph.add_edge(edge)
    new_edges.append(edge)

    node = graph.get(target_id)
    if not isinstance(node, FactNode):
        return
    if was_new:
        new_facts.append(node)
    elif target_id not in seen_linked:
        seen_linked.add(target_id)
        linked_facts.append(node)


def _declaring_class(model_cls: type[BaseModel], field_name: str) -> str:
    """Walk the MRO to find the class that physically declares ``field_name``.

    Handles inheritance: a field defined in a base reuses the base class
    name in ``edge_type``, so ``Actor.user_arn`` stays the same regardless
    of which event-model embeds ``Actor``.
    """
    for klass in model_cls.__mro__:
        if klass is object:
            break
        anns = klass.__dict__.get("__annotations__")
        if anns and field_name in anns:
            return klass.__name__
    return model_cls.__name__


__all__ = ["extract", "event_properties"]
