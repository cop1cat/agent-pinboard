"""Per-session FactGraph cache and concurrency lock.

The graph is loaded from the Store once per session and kept in process
memory thereafter. The lock protects the read-modify-write block of
``@pin`` (step 4 in README §6.1) so parallel ingestion does not lose
updates.

Concurrency primitive
---------------------
README originally promised ``anyio.Lock``. In practice ``anyio.Lock`` is
async-only and refuses sync ``with`` use, so we use ``threading.Lock``,
which is safe in both contexts: in async code its acquire/release is a
microsecond-level CPU operation around the in-memory delta-merge, so
the brief event-loop block is acceptable.

Thread-id resolution
--------------------
``runtime.config["configurable"]["thread_id"]`` is the source of truth.
Absent → fresh UUID4 with a warning so two parallel runs never collide
on a shared default key.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from typing import TYPE_CHECKING

from agent_pinboard import store as store_io
from agent_pinboard.graph import FactGraph

if TYPE_CHECKING:
    from langgraph.prebuilt import ToolRuntime
    from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


# Process-global state. Reset between tests via :func:`_reset`.
_session_graphs: dict[str, FactGraph] = {}
_session_locks: dict[str, threading.RLock] = {}
_async_load_locks: dict[str, asyncio.Lock] = {}
_registry_lock = threading.Lock()


def lock_for(thread_id: str) -> threading.RLock:
    """Return the per-session reentrant lock, creating it on first use.

    Re-entrancy matters because ``@pin`` holds the lock for the
    read-modify-write block and may call ``get_or_load_session`` from
    inside that block (the latter also acquires the lock for its own
    double-checked load).
    """
    with _registry_lock:
        lock = _session_locks.get(thread_id)
        if lock is None:
            lock = threading.RLock()
            _session_locks[thread_id] = lock
        return lock


def get_or_load_session(store: BaseStore, thread_id: str) -> FactGraph:
    """Return the cached FactGraph, loading it from Store on first access."""
    cached = _session_graphs.get(thread_id)
    if cached is not None:
        return cached
    with lock_for(thread_id):
        cached = _session_graphs.get(thread_id)
        if cached is None:
            cached = store_io.load_graph(store, thread_id)
            _session_graphs[thread_id] = cached
    return cached


async def aget_or_load_session(store: BaseStore, thread_id: str) -> FactGraph:
    """Async variant — uses the async store API for the initial load.

    Uses an ``asyncio.Lock`` to serialize concurrent loads for the same
    ``thread_id``, so two parallel awaiters do not both call the store and
    then race to install their copy.
    """
    cached = _session_graphs.get(thread_id)
    if cached is not None:
        return cached
    async with _async_lock_for(thread_id):
        cached = _session_graphs.get(thread_id)
        if cached is not None:
            return cached
        g = await store_io.aload_graph(store, thread_id)
        _session_graphs[thread_id] = g
        return g


def _async_lock_for(thread_id: str) -> asyncio.Lock:
    """Per-session asyncio.Lock; created lazily under the registry mutex.

    Lives only as long as the asyncio event loop that created it; created
    on demand inside ``aget_or_load_session``.
    """
    with _registry_lock:
        lock = _async_load_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            _async_load_locks[thread_id] = lock
        return lock


def thread_id_from(runtime: ToolRuntime) -> str:
    """Resolve the session ``thread_id`` from a ToolRuntime.

    Falls back to a fresh UUID4 with a warning — parallel runs without
    explicit thread_id are then guaranteed to be isolated.
    """
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if isinstance(configurable, dict):
        tid = configurable.get("thread_id")
        if isinstance(tid, str) and tid:
            return tid

    fallback = f"unset-{uuid.uuid4()}"
    logger.warning(
        "thread_id not found in runtime.config.configurable; "
        "generated isolated fallback %r. Pass thread_id explicitly via "
        "graph.invoke(config={'configurable': {'thread_id': ...}}) to "
        "avoid surprises across calls.",
        fallback,
    )
    return fallback


def _reset() -> None:
    """Wipe all per-session state. Test-only."""
    _session_graphs.clear()
    _session_locks.clear()
    _async_load_locks.clear()
