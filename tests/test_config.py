from __future__ import annotations

import pytest

import agent_pinboard
from agent_pinboard.config import _reset, configure, get_config


class TestConfigure:
    def test_default_soft_limit(self) -> None:
        assert get_config().tool_log_soft_limit == 500

    def test_can_change(self) -> None:
        configure(tool_log_soft_limit=10)
        assert get_config().tool_log_soft_limit == 10

    def test_resets_between_tests(self) -> None:
        # The conftest autouse fixture wipes state; check default is back.
        assert get_config().tool_log_soft_limit == 500

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            configure(tool_log_soft_limit=0)

    def test_explicit_reset(self) -> None:
        configure(tool_log_soft_limit=42)
        _reset()
        assert get_config().tool_log_soft_limit == 500


class TestPublicAPISurface:
    def test_exports(self) -> None:
        # Verify every name in __all__ is importable from the package root.
        for name in agent_pinboard.__all__:
            assert hasattr(agent_pinboard, name), name
