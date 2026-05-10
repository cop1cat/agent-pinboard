"""Smoke tests for the foundation layer."""

from __future__ import annotations

import pytest

from pinboard import (
    Direction,
    OnDuplicate,
    PinBoardConfigError,
    PinBoardError,
    PinBoardExtractionError,
    PinBoardNormalizerError,
    PinBoardValidationError,
)


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            PinBoardConfigError,
            PinBoardValidationError,
            PinBoardNormalizerError,
            PinBoardExtractionError,
        ],
    )
    def test_subclass_of_pinboard_error(self, exc_cls: type[Exception]) -> None:
        assert issubclass(exc_cls, PinBoardError)

    def test_can_be_raised_and_caught_via_base(self) -> None:
        with pytest.raises(PinBoardError):
            raise PinBoardConfigError("boom")


class TestEnums:
    def test_direction_str_compatible(self) -> None:
        assert Direction.OUT == "out"
        assert Direction.IN == "in"
        assert Direction.BOTH == "both"

    def test_on_duplicate_str_compatible(self) -> None:
        assert OnDuplicate.ALWAYS == "always"
        assert OnDuplicate.SKIP == "skip"
        assert OnDuplicate.CACHE == "cache"
