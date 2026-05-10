"""Public enums.

Both use :class:`enum.StrEnum` (Python 3.11+) for natural string interop:
JSON serialization, dict keys, and ``==`` against strings all work.
"""

from __future__ import annotations

from enum import StrEnum


class Direction(StrEnum):
    """Edge traversal direction for ``explore``."""

    OUT = "out"
    IN = "in"
    BOTH = "both"


class OnDuplicate(StrEnum):
    """Behaviour of ``@fact`` when a tool is called twice with identical args."""

    ALWAYS = "always"
    SKIP = "skip"
    CACHE = "cache"
