"""``node(...)`` factory — marks a Pydantic field as a graph node."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic.fields import FieldInfo

from pinboard.entity import Entity
from pinboard.exceptions import PinBoardConfigError

# Key inside a field's ``json_schema_extra`` that holds PinBoard metadata.
META_KEY = "__pinboard__"


def node(
    *,
    type: Entity,
    description: str,
    **field_kwargs: Any,
) -> Any:
    """Build a Pydantic ``FieldInfo`` carrying PinBoard node-marker metadata.

    Use in place of ``pydantic.Field`` for fields whose values should become
    nodes in the fact graph::

        class CloudTrailEvent(BaseModel):
            src_ip: str | None = node(
                type=IP,
                description="IP from which the API call was made",
                default=None,
            )

    Parameters
    ----------
    type:
        The :class:`Entity` describing what this field's values represent.
        Must be an ``Entity`` instance — strings or other types raise
        :class:`PinBoardConfigError`.
    description:
        Mandatory, non-empty. Describes how this field relates to its event;
        rendered on edges in ``explore`` / ``timeline`` output.
    **field_kwargs:
        Forwarded to :func:`pydantic.Field` (e.g. ``default``,
        ``default_factory``, ``ge``, ``alias``).

    Returns
    -------
    FieldInfo
        A standard Pydantic field with a ``__pinboard__`` entry in
        ``json_schema_extra``.
    """
    if not isinstance(type, Entity):
        raise PinBoardConfigError(
            f"node(type=...) expects an Entity instance, got {type!r}. "
            "Define `MyEntity = Entity(name=..., description=...)` and pass it."
        )
    if not isinstance(description, str) or not description.strip():
        raise PinBoardConfigError(
            "node(description=...) must be a non-empty string. "
            "Describe how this specific field relates to its event "
            "(Entity.description covers what the type means)."
        )

    schema_extra = field_kwargs.pop("json_schema_extra", None) or {}
    if not isinstance(schema_extra, dict):
        raise PinBoardConfigError(
            "node(...) does not support callable json_schema_extra"
        )
    schema_extra = {**schema_extra, META_KEY: {"entity": type}}

    return Field(
        description=description,
        json_schema_extra=schema_extra,
        **field_kwargs,
    )


def field_entity(field_info: FieldInfo) -> Entity | None:
    """Read the :class:`Entity` attached to a field, or ``None`` if not a node-field."""
    extra = field_info.json_schema_extra
    if not isinstance(extra, dict):
        return None
    meta = extra.get(META_KEY)
    if not isinstance(meta, dict):
        return None
    entity = meta.get("entity")
    return entity if isinstance(entity, Entity) else None
