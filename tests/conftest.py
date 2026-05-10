"""Shared pytest fixtures.

PinBoard relies on a small amount of process-global state (declared-entities
registry, session graph cache, configure() settings). These fixtures reset
that state between tests so the suite is order-independent.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langgraph.store.memory import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    """Fresh in-memory store per test."""
    return InMemoryStore()


@pytest.fixture(autouse=True)
def reset_pinboard_state() -> Iterator[None]:
    """Wipe process-level state before and after each test.

    Modules are imported lazily — if a module isn't imported yet, its reset
    hook is simply skipped. This keeps the fixture decoupled from the
    module dependency order.
    """
    _reset_all()
    yield
    _reset_all()


def _reset_all() -> None:
    for modname in (
        "pinboard.registry",
        "pinboard.session",
        "pinboard.config",
    ):
        try:
            mod = __import__(modname, fromlist=["_reset"])
        except ImportError:
            continue
        reset = getattr(mod, "_reset", None)
        if reset is not None:
            reset()
