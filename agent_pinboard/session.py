"""Per-session lock + thread_id resolution.

The session graph is **not** cached in process memory. Every ``@pin``
ingest and every read tool loads the graph fresh from the Store, so a
multi-process deployment (gunicorn / Celery / Ray workers sharing a
PostgresStore) sees consistent state without invalidation logic.

The only in-process state is a per-``thread_id`` ``threading.RLock``
that serializes the read-modify-write window of a single ingest within
one process — preventing two threads in the same worker from racing on
their reload+persist cycle. Cross-process correctness comes from the
mergeable data model: stored ``FactNode`` keys carry only the immutable
subset (``id``, ``node_type``, ``value``, ``canonical_value``); the
mutable provenance fields (``source_events``, ``source_tools``,
``first_seen``, ``last_seen``) are derived from the canonical edges +
EventNodes at load time, so two processes upserting the same canonical
fact never lose each other's links.

Thread-id resolution
--------------------
``runtime.config["configurable"]["thread_id"]`` is the source of truth.
Absent → fresh UUID4 with a warning so two parallel runs never collide
on a shared default key.
"""

from __future__ import annotations

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
_session_locks: dict[str, threading.RLock] = {}
_registry_lock = threading.Lock()


def lock_for(thread_id: str) -> threading.RLock:
    """Return the per-session reentrant lock, creating it on first use.

    Reentrant because ``@pin`` holds the lock for the read-modify-write
    block and the read tools may re-enter via ``get_or_load_session``
    while the same thread already holds it.
    """
    with _registry_lock:
        lock = _session_locks.get(thread_id)
        if lock is None:
            lock = threading.RLock()
            _session_locks[thread_id] = lock
        return lock


def get_or_load_session(store: BaseStore, thread_id: str) -> FactGraph:
    """Always load the session graph fresh from the Store.

    The name is preserved for backward-compat with internal callers; in
    practice this is now a thin wrapper around ``store_io.load_graph``.
    """
    return store_io.load_graph(store, thread_id)


async def aget_or_load_session(store: BaseStore, thread_id: str) -> FactGraph:
    """Async variant — uses the async store API for the load."""
    return await store_io.aload_graph(store, thread_id)


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
    _session_locks.clear()
