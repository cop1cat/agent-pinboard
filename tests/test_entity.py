from __future__ import annotations

import pytest

from agent_pinboard import Entity


class TestEntityValidation:
    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Entity(name="", description="x")

    def test_whitespace_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Entity(name="   ", description="x")

    def test_empty_description_rejected(self) -> None:
        with pytest.raises(ValueError, match="description"):
            Entity(name="IP", description="")

    def test_whitespace_description_rejected(self) -> None:
        with pytest.raises(ValueError, match="description"):
            Entity(name="IP", description="   ")


class TestEntityValueSemantics:
    def test_equal_when_same_attrs_and_normalizer(self) -> None:
        f = str.lower
        a = Entity(name="X", description="d", normalizer=f)
        b = Entity(name="X", description="d", normalizer=f)
        assert a == b
        assert hash(a) == hash(b)

    def test_unequal_when_normalizers_differ_by_identity(self) -> None:
        a = Entity(name="X", description="d", normalizer=lambda v: str(v))
        b = Entity(name="X", description="d", normalizer=lambda v: str(v))
        assert a != b  # Different lambda objects.

    def test_frozen(self) -> None:
        e = Entity(name="X", description="d")
        with pytest.raises(Exception):  # FrozenInstanceError
            e.name = "Y"  # type: ignore[misc]
