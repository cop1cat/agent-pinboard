"""Exception hierarchy for AgentPinBoard.

All library-raised exceptions inherit from :class:`AgentPinBoardError`. Users can
catch this single base to handle any failure originating from AgentPinBoard.
"""

from __future__ import annotations


class AgentPinBoardError(Exception):
    """Base for all AgentPinBoard-raised exceptions."""


class AgentPinBoardConfigError(AgentPinBoardError):
    """Configuration / setup error.

    Raised at registration / decoration time for misconfigured ``Entity``,
    ``node()``, decorator stack order, missing store, or sync/async mismatch.
    """


class AgentPinBoardValidationError(AgentPinBoardError):
    """Pydantic validation of a tool return failed.

    Wraps the underlying ``pydantic.ValidationError`` (available as ``__cause__``).
    Raised by ``@pin`` when the tool's return cannot be parsed into the
    declared model.
    """


class AgentPinBoardNormalizerError(AgentPinBoardError):
    """An ``Entity.normalizer`` raised on its input.

    Wraps the original exception. Indicates either a buggy normalizer or
    malformed input data — either way, fail-loud surfaces the issue.
    """


class AgentPinBoardExtractionError(AgentPinBoardError):
    """Extraction encountered an unsupported field shape.

    Raised for ``dict[str, BaseModel]``, ``Union[NodeA, NodeB]``, ``tuple``,
    or lists with mixed-type elements — see README §4.1.
    """
