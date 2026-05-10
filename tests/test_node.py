from __future__ import annotations

import pytest
from pydantic import BaseModel

from pinboard import Entity, PinBoardConfigError, node
from pinboard.fields import META_KEY, field_entity


@pytest.fixture
def IP() -> Entity:
    return Entity(name="IP", description="IPv4/IPv6", normalizer=str.strip)


class TestNodeFactoryValidation:
    def test_string_type_rejected(self, IP: Entity) -> None:
        with pytest.raises(PinBoardConfigError, match="Entity instance"):
            node(type="IP", description="src ip")  # type: ignore[arg-type]

    def test_empty_description_rejected(self, IP: Entity) -> None:
        with pytest.raises(PinBoardConfigError, match="description"):
            node(type=IP, description="")

    def test_whitespace_description_rejected(self, IP: Entity) -> None:
        with pytest.raises(PinBoardConfigError, match="description"):
            node(type=IP, description="   ")


class TestNodeFactoryShape:
    def test_returns_pydantic_field_with_meta(self, IP: Entity) -> None:
        info = node(type=IP, description="src", default=None)
        assert info.description == "src"
        assert isinstance(info.json_schema_extra, dict)
        assert info.json_schema_extra[META_KEY]["entity"] is IP

    def test_field_entity_helper_reads_meta(self, IP: Entity) -> None:
        class M(BaseModel):
            ip: str | None = node(type=IP, description="src", default=None)
            other: int = 0

        assert field_entity(M.model_fields["ip"]) is IP
        assert field_entity(M.model_fields["other"]) is None


class TestNodeFactoryPydanticIntegration:
    def test_optional_with_default_none(self, IP: Entity) -> None:
        class M(BaseModel):
            ip: str | None = node(type=IP, description="src", default=None)

        m = M.model_validate({})
        assert m.ip is None
        m = M.model_validate({"ip": "1.2.3.4"})
        assert m.ip == "1.2.3.4"

    def test_required_field(self, IP: Entity) -> None:
        class M(BaseModel):
            ip: str = node(type=IP, description="src")

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            M.model_validate({})

    def test_list_with_default_factory(self, IP: Entity) -> None:
        class M(BaseModel):
            ips: list[str] = node(type=IP, description="all", default_factory=list)

        m = M.model_validate({})
        assert m.ips == []
        m = M.model_validate({"ips": ["1.1.1.1", "2.2.2.2"]})
        assert m.ips == ["1.1.1.1", "2.2.2.2"]

    def test_extra_field_kwargs_pass_through(self, IP: Entity) -> None:
        class M(BaseModel):
            ip: str = node(type=IP, description="src", alias="source_ip")

        m = M.model_validate({"source_ip": "1.1.1.1"})
        assert m.ip == "1.1.1.1"
