"""Cross-process concurrency: mergeable FactNode storage preserves all links.

Simulates two worker processes that share a Store but do **not** share
the in-process lock (which is what `lock_for` would otherwise enforce
inside one process). Both workers ingest an event that extracts the
same canonical fact at the same time. With the mergeable storage model
the post-reload FactNode references both events in ``source_events``;
without it, last-write-wins would erase one worker's link.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime

import pytest
from langgraph.store.memory import InMemoryStore

from agent_pinboard import Entity, EventNode, FactEdge
from agent_pinboard import store as store_io


@pytest.fixture
def shared_store() -> InMemoryStore:
    return InMemoryStore()


def _ingest_one(
    store: InMemoryStore,
    thread_id: str,
    entity: Entity,
    canonical_value: str,
    tool_name: str,
    barrier: threading.Barrier,
) -> str:
    """Run one ingest cycle without the in-process lock — i.e. as a
    separate 'process' would. Returns the EventNode id it created.
    """
    barrier.wait(timeout=5)
    g = store_io.load_graph(store, thread_id)
    ev = EventNode(
        id=str(uuid.uuid4()),
        source_tool=tool_name,
        timestamp=datetime.now(UTC),
    )
    g.add_event(ev)
    barrier.wait(timeout=5)  # force interleaving before the write
    nid, _ = g.upsert_fact(entity, canonical_value, ev.id, tool_name)
    edge = FactEdge(
        event_id=ev.id, target_id=nid, edge_type="X.ip", description=""
    )
    g.add_edge(edge)
    fact = g.get(nid)
    store_io.persist_delta(store, thread_id, [ev, fact], [edge])
    return ev.id


class TestCrossProcessConcurrency:
    def test_overlapping_facts_merge_on_reload(
        self, shared_store: InMemoryStore
    ) -> None:
        """Two workers extract the SAME canonical IP at the same time;
        after both persist, a fresh load sees both events and both edges
        attached to the single FactNode.
        """
        IP = Entity(name="IP", description="ip")
        barrier = threading.Barrier(2)
        results: dict[int, str] = {}

        def worker(idx: int) -> None:
            results[idx] = _ingest_one(
                shared_store, "shared", IP, "8.8.8.8", f"tool-{idx}", barrier
            )

        t1 = threading.Thread(target=worker, args=(0,))
        t2 = threading.Thread(target=worker, args=(1,))
        t1.start(); t2.start(); t1.join(); t2.join()

        ev_ids = sorted(results.values())
        assert len(ev_ids) == 2

        g = store_io.load_graph(shared_store, "shared")
        # Both events survived (unique ids, append-only).
        assert {e.id for e in g.all_events()} == set(ev_ids)
        # Single canonical FactNode (deterministic id from
        # (node_type, canonical_value)).
        facts = list(g.all_facts())
        assert len(facts) == 1
        fact = facts[0]
        # source_events backfilled from edges → both workers' links present.
        assert sorted(fact.source_events) == ev_ids
        assert fact.source_tools == {"tool-0", "tool-1"}

    def test_disjoint_facts_both_persist(self, shared_store: InMemoryStore) -> None:
        """Two workers extract DIFFERENT canonical IPs concurrently — both
        FactNodes and both events are present in the Store afterwards.
        """
        IP = Entity(name="IP", description="ip")
        barrier = threading.Barrier(2)
        ev_ids: list[str] = []
        lock = threading.Lock()

        def worker(canonical: str, tool: str) -> None:
            ev = _ingest_one(shared_store, "tid2", IP, canonical, tool, barrier)
            with lock:
                ev_ids.append(ev)

        t1 = threading.Thread(target=worker, args=("1.1.1.1", "t-A"))
        t2 = threading.Thread(target=worker, args=("2.2.2.2", "t-B"))
        t1.start(); t2.start(); t1.join(); t2.join()

        g = store_io.load_graph(shared_store, "tid2")
        assert len(list(g.all_events())) == 2
        canonical_values = {f.canonical_value for f in g.all_facts()}
        assert canonical_values == {"1.1.1.1", "2.2.2.2"}
