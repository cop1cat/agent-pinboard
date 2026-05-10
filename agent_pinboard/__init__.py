"""PinBoard — LLM-agent working memory as a fact graph.

Public API surface — see README for usage. Symbols are added by their owning
modules and re-exported here as the single import point.
"""

from __future__ import annotations

__version__ = "0.1.0"

from pinboard.config import configure
from pinboard.decorator import fact
from pinboard.entity import Entity
from pinboard.enums import Direction, OnDuplicate
from pinboard.exceptions import (
    PinBoardConfigError,
    PinBoardError,
    PinBoardExtractionError,
    PinBoardNormalizerError,
    PinBoardValidationError,
)
from pinboard.fields import node
from pinboard.graph import FactGraph
from pinboard.hooks import CompositeHook, LoggingHook, PinBoardHooks
from pinboard.models import (
    EventId,
    EventNode,
    FactEdge,
    FactNode,
    IngestResult,
    NodeId,
    ToolCallRecord,
)
from pinboard.tools import make_graph_tools

__all__ = [
    # Decorator + global config
    "configure",
    "fact",
    # Read tools
    "make_graph_tools",
    # Hooks
    "CompositeHook",
    "LoggingHook",
    "PinBoardHooks",
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
    "PinBoardConfigError",
    "PinBoardError",
    "PinBoardExtractionError",
    "PinBoardNormalizerError",
    "PinBoardValidationError",
]
