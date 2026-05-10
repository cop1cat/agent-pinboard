"""Exception hierarchy for PinBoard.

All library-raised exceptions inherit from :class:`PinBoardError`. Users can
catch this single base to handle any failure originating from PinBoard.
"""

from __future__ import annotations


class PinBoardError(Exception):
    """Base for all PinBoard-raised exceptions."""


class PinBoardConfigError(PinBoardError):
    """Configuration / setup error.

    Raised at registration / decoration time for misconfigured ``Entity``,
    ``node()``, decorator stack order, missing store, or sync/async mismatch.
    """


class PinBoardValidationError(PinBoardError):
    """Pydantic validation of a tool return failed.

    Wraps the underlying ``pydantic.ValidationError`` (available as ``__cause__``).
    Raised by ``@fact`` when the tool's return cannot be parsed into the
    declared model.
    """


class PinBoardNormalizerError(PinBoardError):
    """An ``Entity.normalizer`` raised on its input.

    Wraps the original exception. Indicates either a buggy normalizer or
    malformed input data — either way, fail-loud surfaces the issue.
    """


class PinBoardExtractionError(PinBoardError):
    """Extraction encountered an unsupported field shape.

    Raised for ``dict[str, BaseModel]``, ``Union[NodeA, NodeB]``, ``tuple``,
    or lists with mixed-type elements — see README §4.1.
    """
