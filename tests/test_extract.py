from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, Field

from agent_pinboard import (
    AgentPinBoardExtractionError,
    AgentPinBoardNormalizerError,
    Entity,
    EventNode,
    FactGraph,
    node,
)
from agent_pinboard.extract import event_properties, extract


def _ip(normalizer=None) -> Entity:
    return Entity(name="IP", description="ip", normalizer=normalizer)


def _user() -> Entity:
    return Entity(name="User", description="u")


def _make_event() -> EventNode:
    return EventNode(id="e-1", source_tool="t", timestamp=datetime.now(UTC))


class TestRule1Primitive:
    def test_emits_fact_and_edge(self) -> None:
        IP = _ip()

        class M(BaseModel):
            src_ip: str | None = node(type=IP, description="source", default=None)

        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        m = M(src_ip="1.1.1.1")
        new_facts, _linked, new_edges, warnings = extract(m, g, ev.id, "t")
        assert len(new_facts) == 1 and new_facts[0].value == "1.1.1.1"
        assert len(new_edges) == 1
        assert new_edges[0].edge_type == "M.src_ip"
        assert new_edges[0].description == "source"
        assert warnings == []


class TestRule2None:
    def test_skipped(self) -> None:
        IP = _ip()

        class M(BaseModel):
            src_ip: str | None = node(type=IP, description="src", default=None)

        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        new_facts, _linked, new_edges, _ = extract(M(), g, ev.id, "t")
        assert new_facts == [] and new_edges == []


class TestRule3ListPrimitives:
    def test_emits_node_per_element(self) -> None:
        IP = _ip()

        class M(BaseModel):
            ips: list[str] = node(type=IP, description="all", default_factory=list)

        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        new_facts, _linked, new_edges, _ = extract(
            M(ips=["1.1.1.1", "2.2.2.2"]), g, ev.id, "t"
        )
        assert {f.value for f in new_facts} == {"1.1.1.1", "2.2.2.2"}
        assert all(e.edge_type == "M.ips" for e in new_edges)

    def test_list_of_dicts_with_node_rejected(self) -> None:
        IP = _ip()

        class M(BaseModel):
            ips: list[dict] = node(type=IP, description="x", default_factory=list)

        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        with pytest.raises(AgentPinBoardExtractionError):
            extract(M(ips=[{"a": 1}]), g, ev.id, "t")


class TestRule4NestedModel:
    def test_recurse_into_nested_basemodel(self) -> None:
        """README §16 AC1 — nested Actor.user_arn → edge labelled Actor.user_arn."""
        User = _user()

        class Actor(BaseModel):
            user_arn: str | None = node(type=User, description="who", default=None)

        class CloudTrailEvent(BaseModel):
            actor: Actor | None = None

        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        m = CloudTrailEvent(actor=Actor(user_arn="arn:aws:iam::123:user/admin"))
        new_facts, _linked, new_edges, _ = extract(m, g, ev.id, "t")

        assert len(new_facts) == 1
        assert new_facts[0].node_type == "User"
        assert new_facts[0].value == "arn:aws:iam::123:user/admin"
        assert len(new_edges) == 1
        # Edge_type uses the *declaring* class (Actor), NOT the outer (CloudTrailEvent).
        assert new_edges[0].edge_type == "Actor.user_arn"

    def test_list_of_basemodel_recurses(self) -> None:
        User = _user()

        class Actor(BaseModel):
            user_arn: str | None = node(type=User, description="who", default=None)

        class M(BaseModel):
            actors: list[Actor] = []

        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        m = M(actors=[Actor(user_arn="a"), Actor(user_arn="b")])
        new_facts, _l, _, _ = extract(m, g, ev.id, "t")
        assert {f.value for f in new_facts} == {"a", "b"}


class TestRule5EventProperties:
    def test_scalar_non_node_in_event_properties(self) -> None:
        IP = _ip()

        class M(BaseModel):
            src_ip: str | None = node(type=IP, description="src", default=None)
            action: str = Field(description="API action")
            ts: int = Field(description="ts", default=0)

        m = M(src_ip="1.1.1.1", action="AssumeRole", ts=42)
        props = event_properties(m)
        assert props == {"action": "AssumeRole", "ts": 42}


class TestEdgeTypeDeclaringClass:
    def test_inherited_field_uses_base_class_name(self) -> None:
        IP = _ip()

        class Base(BaseModel):
            src_ip: str | None = node(type=IP, description="src", default=None)

        class Derived(Base):
            extra: str | None = None

        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        m = Derived(src_ip="1.1.1.1")
        _, _l, edges, _ = extract(m, g, ev.id, "t")
        assert edges[0].edge_type == "Base.src_ip"


class TestNormalizerErrorPropagates:
    def test_extract_propagates_normalizer_error(self) -> None:
        def boom(_: object) -> str:
            raise ValueError("nope")

        E = Entity(name="X", description="x", normalizer=boom)

        class M(BaseModel):
            x: str | None = node(type=E, description="x", default=None)

        g = FactGraph()
        ev = _make_event()
        g.add_event(ev)
        with pytest.raises(AgentPinBoardNormalizerError):
            extract(M(x="hi"), g, ev.id, "t")


class TestAutolinkAcrossExtractions:
    def test_two_extractions_same_value_one_node(self) -> None:
        """README §16 AC2 — autolink + dedup integration check."""
        IP = _ip()

        class M(BaseModel):
            src_ip: str | None = node(type=IP, description="src", default=None)

        g = FactGraph()
        ev1 = _make_event()
        ev1.id = "e-1"
        ev2 = EventNode(id="e-2", source_tool="vt", timestamp=datetime.now(UTC))
        g.add_event(ev1)
        g.add_event(ev2)

        new1, _l1, _, _ = extract(M(src_ip="1.2.3.4"), g, ev1.id, "fetch")
        new2, _l2, _, _ = extract(M(src_ip="1.2.3.4"), g, ev2.id, "vt")
        assert len(new1) == 1
        assert len(new2) == 0  # already linked
        nid = g.find_by_value("IP", "1.2.3.4")
        n = g.get(nid)  # type: ignore[arg-type]
        assert n.source_tools == {"fetch", "vt"}  # type: ignore[union-attr]
        assert n.source_events == [ev1.id, ev2.id]  # type: ignore[union-attr]
