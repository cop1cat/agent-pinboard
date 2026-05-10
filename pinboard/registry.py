"""Process-global registry of declared :class:`Entity` instances.

Populated eagerly when ``@fact(model=X)`` is applied — the model is scanned
and every ``Entity`` referenced via ``node()`` is recorded. This lets
``graph_summary`` show the full known schema before any ingestion has
happened.

If the same ``Entity.name`` is declared with different attributes (different
description / normalizer), a warning is emitted and the first registration
wins. The library does not police user code beyond that.
"""

from __future__ import annotations

import logging
from typing import get_args, get_origin

from pydantic import BaseModel

from pinboard.entity import Entity
from pinboard.exceptions import PinBoardConfigError
from pinboard.fields import field_entity

logger = logging.getLogger(__name__)


# Process-global. Reset between tests via :func:`_reset`.
_declared_entities: dict[str, Entity] = {}
_origin_locations: dict[str, str] = {}


def known_entities() -> dict[str, Entity]:
    """Snapshot of currently-declared entities, keyed by name."""
    return dict(_declared_entities)


def register_model(model: type[BaseModel]) -> None:
    """Eagerly scan ``model`` and register every ``Entity`` it references.

    Recursively walks ``model_fields``, descending into nested ``BaseModel``
    types and ``list[BaseModel]``. Recursion is bounded by ``seen``, so models
    like ``Process(parent: Process | None)`` do not loop.
    """
    seen: set[type[BaseModel]] = set()
    _scan(model, seen)


def _scan(model: type[BaseModel], seen: set[type[BaseModel]]) -> None:
    if model in seen:
        return
    seen.add(model)

    for name, info in model.model_fields.items():
        entity = field_entity(info)
        if entity is not None:
            _check_node_field_shape(model, name, info.annotation)
            _register_entity(entity, location=f"{model.__name__}.{name}")
            # node()-marked fields are leaves (rule 1/3): no nested-model recursion needed.
            continue

        for nested in _nested_models(info.annotation):
            _scan(nested, seen)


def _check_node_field_shape(model: type[BaseModel], name: str, annotation: object) -> None:
    """Reject ``node()``-marked fields whose annotation is a ``BaseModel`` (or list of).

    README §3.2: node() expects a primitive or list[primitive]. A BaseModel-typed
    field with node() is semantically wrong (it would have to be both a node
    *and* a structured object) and would silently fall through extractor rules.
    """
    if _is_basemodel_typed(annotation):
        raise PinBoardConfigError(
            f"node() on field {model.__name__}.{name}: BaseModel-typed fields "
            "cannot be marked as nodes (they are containers, not values). "
            "Either mark a primitive field with node(), or remove node() and "
            "let the extractor recurse into the nested model."
        )


def _is_basemodel_typed(annotation: object) -> bool:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_is_basemodel_typed(arg) for arg in get_args(annotation) if arg is not type(None))


def _register_entity(entity: Entity, *, location: str) -> None:
    existing = _declared_entities.get(entity.name)
    if existing is None:
        _declared_entities[entity.name] = entity
        _origin_locations[entity.name] = location
        return
    if existing != entity:
        logger.warning(
            "Entity name collision: %r already registered at %s with different "
            "attributes; second declaration at %s ignored. "
            "Define each Entity once and import it where needed.",
            entity.name,
            _origin_locations.get(entity.name, "<unknown>"),
            location,
        )


def _nested_models(annotation: object) -> list[type[BaseModel]]:
    """Return the BaseModel subclasses reachable from a field annotation.

    Handles ``T``, ``T | None``, ``list[T]``, ``Optional[T]``. Other shapes
    return an empty list (extractor will report unsupported types at runtime).
    """
    if annotation is None:
        return []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]

    origin = get_origin(annotation)
    if origin is None:
        return []

    nested: list[type[BaseModel]] = []
    for arg in get_args(annotation):
        nested.extend(_nested_models(arg))
    return nested


def _reset() -> None:
    """Clear all declared entities. Test-only."""
    _declared_entities.clear()
    _origin_locations.clear()
