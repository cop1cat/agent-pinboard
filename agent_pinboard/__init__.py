"""AgentPinBoard — LLM-agent working memory as a fact graph.

Public API surface — see README for usage. Symbols are added by their owning
modules and re-exported here as the single import point.
"""

from __future__ import annotations

__version__ = "0.1.0"

from agent_pinboard.config import configure
from agent_pinboard.decorator import pin
from agent_pinboard.entity import Entity
from agent_pinboard.enums import Direction, OnDuplicate
from agent_pinboard.exceptions import (
    AgentPinBoardConfigError,
    AgentPinBoardError,
    AgentPinBoardExtractionError,
    AgentPinBoardNormalizerError,
    AgentPinBoardValidationError,
)
from agent_pinboard.fields import node
from agent_pinboard.graph import FactGraph
from agent_pinboard.hooks import AgentPinBoardHooks, CompositeHook, LoggingHook
from agent_pinboard.models import (
    EventId,
    EventNode,
    FactEdge,
    FactNode,
    IngestResult,
    NodeId,
    ToolCallRecord,
)
from agent_pinboard.tools import make_graph_tools

__all__ = [
    # Decorator + global config
    "configure",
    "pin",
    # Read tools
    "make_graph_tools",
    # Hooks
    "CompositeHook",
    "LoggingHook",
    "AgentPinBoardHooks",
    # Markers / factories
    "Entity",
    "node",
    # Enums
    "Direction",
    "OnDuplicate",
    # Models
    "EventId",
    "EventNode",
    "FactEdge",
    "FactGraph",
    "FactNode",
    "IngestResult",
    "NodeId",
    "ToolCallRecord",
    # Exceptions
    "AgentPinBoardConfigError",
    "AgentPinBoardError",
    "AgentPinBoardExtractionError",
    "AgentPinBoardNormalizerError",
    "AgentPinBoardValidationError",
]
