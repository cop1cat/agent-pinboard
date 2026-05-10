"""``Entity`` — the user-facing description of a node type."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class Entity:
    """A node-type descriptor.

    Created once per type by the user (typically in an ``entities.py`` module),
    then referenced from ``node(type=...)`` markers in their tool-response
    Pydantic models.

    The ``normalizer`` (if provided) canonicalises field values for autolinking:
    ``("IP", "192.168.001.001")`` and ``("IP", "192.168.1.1")`` collapse to one
    node when ``canonical_ip`` returns the same string for both.

    ``Entity`` is a plain value object — frozen, slotted, no side effects on
    construction.
    """

    name: str
    description: str
    normalizer: Callable[[Any], str] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Entity.name must be a non-empty string")
        if not self.description or not self.description.strip():
            raise ValueError("Entity.description must be a non-empty string")
