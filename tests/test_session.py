from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langgraph.store.memory import InMemoryStore

from pinboard import Entity, EventNode
from pinboard import store as store_io
from pinboard.session import (
    aget_or_load_session,
    get_or_load_session,
    lock_for,
    thread_id_from,
)


def _make_runtime(thread_id: str | None) -> object:
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    return SimpleNamespace(config=config, store=None)


class TestThreadIdResolution:
    def test_explicit(self) -> None:
        rt = _make_runtime("session-x")
        assert thread_id_from(rt) == "session-x"  # type: ignore[arg-type]

    def test_missing_falls_back_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rt = _make_runtime(None)
        with caplog.at_level(logging.WARNING):
            tid = thread_id_from(rt)  # type: ignore[arg-type]
        assert tid.startswith("unset-")
        assert any("thread_id" in r.message for r in caplog.records)

    def test_two_missing_runtimes_get_distinct_ids(self) -> None:
        rt1 = _make_runtime(None)
        rt2 = _make_runtime(None)
        tid1 = thread_id_from(rt1)  # type: ignore[arg-type]
        tid2 = thread_id_from(rt2)  # type: ignore[arg-type]
        assert tid1 != tid2


class TestSessionCacheAndLoad:
    def test_load_then_cached(self, store: InMemoryStore) -> None:
        g1 = get_or_load_session(store, "tid")
        g2 = get_or_load_session(store, "tid")
        assert g1 is g2  # same in-memory object cached

    def test_load_picks_up_persisted_state(self, store: InMemoryStore) -> None:
        ev = EventNode(id="e", source_tool="t", timestamp=datetime.now(timezone.utc))
        store_io.persist_delta(store, "tid", [ev], [])
        g = get_or_load_session(store, "tid")
        assert g.get("e") is not None

    def test_session_isolation(self, store: InMemoryStore) -> None:
        """README §16 AC7 — different thread_ids never share data."""
        ev_a = EventNode(id="ea", source_tool="t", timestamp=datetime.now(timezone.utc))
        ev_b = EventNode(id="eb", source_tool="t", timestamp=datetime.now(timezone.utc))
        store_io.persist_delta(store, "alpha", [ev_a], [])
        store_io.persist_delta(store, "beta", [ev_b], [])

        ga = get_or_load_session(store, "alpha")
        gb = get_or_load_session(store, "beta")
        assert ga is not gb
        assert ga.get("ea") is not None and ga.get("eb") is None
        assert gb.get("eb") is not None and gb.get("ea") is None


class TestLockBehaviour:
    def test_same_thread_id_returns_same_lock(self) -> None:
        a = lock_for("tid")
        b = lock_for("tid")
        assert a is b

    def test_different_thread_ids_distinct_locks(self) -> None:
        assert lock_for("a") is not lock_for("b")

    def test_concurrent_writes_do_not_lose_updates(self, store: InMemoryStore) -> None:
        """README §16 AC4 — N parallel ingest steps → all N writes survive."""
        g = get_or_load_session(store, "tid")
        IP = Entity(name="IP", description="ip")
        N = 10
        barrier = threading.Barrier(N)
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                barrier.wait(timeout=2)
                with lock_for("tid"):
                    g.upsert_fact(IP, f"10.0.0.{i}", f"e-{i}", f"tool-{i}")
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(list(g.all_facts())) == N


class TestAsyncSession:
    @pytest.mark.asyncio
    async def test_aget_or_load_caches(self, store: InMemoryStore) -> None:
        g1 = await aget_or_load_session(store, "tid")
        g2 = await aget_or_load_session(store, "tid")
        assert g1 is g2
