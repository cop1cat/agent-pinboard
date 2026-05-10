"""Smoke tests for the foundation layer."""

from __future__ import annotations

import pytest

from agent_pinboard import (
    AgentPinBoardConfigError,
    AgentPinBoardError,
    AgentPinBoardExtractionError,
    AgentPinBoardNormalizerError,
    AgentPinBoardValidationError,
    Direction,
    OnDuplicate,
)


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            AgentPinBoardConfigError,
            AgentPinBoardValidationError,
            AgentPinBoardNormalizerError,
            AgentPinBoardExtractionError,
        ],
    )
    def test_subclass_of_agent_pinboard_error(self, exc_cls: type[Exception]) -> None:
        assert issubclass(exc_cls, AgentPinBoardError)

    def test_can_be_raised_and_caught_via_base(self) -> None:
        with pytest.raises(AgentPinBoardError):
            raise AgentPinBoardConfigError("boom")


class TestEnums:
    def test_direction_str_compatible(self) -> None:
        assert Direction.OUT == "out"
        assert Direction.IN == "in"
        assert Direction.BOTH == "both"

    def test_on_duplicate_str_compatible(self) -> None:
        assert OnDuplicate.ALWAYS == "always"
        assert OnDuplicate.SKIP == "skip"
        assert OnDuplicate.CACHE == "cache"
