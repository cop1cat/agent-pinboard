from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agent_pinboard import Entity, node
from agent_pinboard.registry import known_entities, register_model


class TestRegistryEagerScan:
    def test_collects_top_level_node_fields(self) -> None:
        IP = Entity(name="IP", description="ip")
        User = Entity(name="User", description="u")

        class M(BaseModel):
            ip: str | None = node(type=IP, description="src", default=None)
            user: str | None = node(type=User, description="who", default=None)
            other: int = Field(default=0)

        register_model(M)
        known = known_entities()
        assert known == {"IP": IP, "User": User}

    def test_collects_nested_via_basemodel(self) -> None:
        User = Entity(name="User", description="u")
        IP = Entity(name="IP", description="ip")

        class Actor(BaseModel):
            user_arn: str | None = node(type=User, description="arn", default=None)

        class M(BaseModel):
            actor: Actor | None = None
            src_ip: str | None = node(type=IP, description="src", default=None)

        register_model(M)
        assert set(known_entities()) == {"User", "IP"}

    def test_collects_via_list_of_basemodel(self) -> None:
        User = Entity(name="User", description="u")

        class Actor(BaseModel):
            user_arn: str | None = node(type=User, description="arn", default=None)

        class M(BaseModel):
            actors: list[Actor] = []

        register_model(M)
        assert "User" in known_entities()

    def test_recursion_guard(self) -> None:
        """README §16 AC3: recursive models do not loop."""
        IP = Entity(name="IP", description="ip")

        class Process(BaseModel):
            pid: str | None = node(type=IP, description="pid-as-IP-just-for-test", default=None)
            parent: Process | None = None

        Process.model_rebuild()
        register_model(Process)  # must terminate
        assert "IP" in known_entities()


class TestRegistryConflictHandling:
    def test_duplicate_name_same_attrs_no_warning(
        self, caplog: logging.LogCaptureFixture
    ) -> None:
        IP = Entity(name="IP", description="ip")

        class A(BaseModel):
            ip: str | None = node(type=IP, description="a", default=None)

        class B(BaseModel):
            ip: str | None = node(type=IP, description="b", default=None)

        with caplog.at_level(logging.WARNING):
            register_model(A)
            register_model(B)
        assert not any("collision" in r.message for r in caplog.records)

    def test_duplicate_name_different_attrs_warns(
        self, caplog: logging.LogCaptureFixture
    ) -> None:
        IP1 = Entity(name="IP", description="first")
        IP2 = Entity(name="IP", description="second — different")

        class A(BaseModel):
            ip: str | None = node(type=IP1, description="a", default=None)

        class B(BaseModel):
            ip: str | None = node(type=IP2, description="b", default=None)

        with caplog.at_level(logging.WARNING):
            register_model(A)
            register_model(B)

        assert any("collision" in r.message for r in caplog.records)
        # First wins.
        assert known_entities()["IP"] is IP1
