"""Process-global config knobs.

Sparse on purpose — anything that varies per session belongs in
``ToolRuntime.config``, not here. Only true global toggles live here
(currently just ``tool_log_soft_limit``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class _Config:
    tool_log_soft_limit: int = 500


_config = _Config()


def configure(*, tool_log_soft_limit: int | None = None) -> None:
    """Mutate process-wide AgentPinBoard settings.

    Settings apply to every session in this process; per-session overrides
    are out of scope for the MVP.
    """
    if tool_log_soft_limit is not None:
        if tool_log_soft_limit < 1:
            raise ValueError("tool_log_soft_limit must be >= 1")
        _config.tool_log_soft_limit = tool_log_soft_limit


def get_config() -> _Config:
    return _config


def _reset() -> None:
    """Restore defaults. Test-only."""
    _config.tool_log_soft_limit = 500
