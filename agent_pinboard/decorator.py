"""``@pin`` — the user-facing entry point.

Wraps a LangChain ``BaseTool`` (created by ``@tool``) so that every
invocation, in addition to its normal return:

1. validates the return value against ``model``,
2. extracts a delta of ``FactNode`` / ``FactEdge``,
3. merges the delta into the session graph under a per-thread lock,
4. records a ``ToolCallRecord``,
5. fires hooks,
6. optionally rewrites the return via ``response_transform``.

Stack order
-----------
``@pin`` must be **above** ``@tool`` (see README §6.2). We detect the
inverse and raise :class:`AgentPinBoardConfigError` with a hint.

Sync vs async
-------------
Decoded from whether the underlying ``BaseTool`` has ``func`` (sync) or
``coroutine`` (async). The decorator builds a matching wrapper. The bulk
of the pipeline is shared as pure helpers; only the actual store I/O
(load tool calls, persist delta, persist record) differs between paths.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import adispatch_custom_event, dispatch_custom_event
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from agent_pinboard import store as store_io
from agent_pinboard.config import get_config
from agent_pinboard.enums import OnDuplicate
from agent_pinboard.exceptions import (
    AgentPinBoardConfigError,
    AgentPinBoardValidationError,
)
from agent_pinboard.extract import event_properties, extract
from agent_pinboard.graph import FactGraph
from agent_pinboard.models import EventNode, FactEdge, FactNode, IngestResult, ToolCallRecord
from agent_pinboard.registry import register_model
from agent_pinboard.session import (
    aget_or_load_session,
    get_or_load_session,
    lock_for,
    thread_id_from,
)

# Custom-event name dispatched after every successful ingest. Subscribers
# implement ``BaseCallbackHandler.on_custom_event`` and filter on this
# name. Payload schema: see ``_dispatch_ingest_event``.
INGEST_EVENT = "agent_pinboard:ingest"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public decorator.                                                           #
# --------------------------------------------------------------------------- #

def pin(
    *,
    model: type[BaseModel],
    many: bool = False,
    on_duplicate: OnDuplicate = OnDuplicate.ALWAYS,
    mask_args: list[str] | None = None,
    response_transform: Callable[[Any, IngestResult], Any] | None = None,
    store_raw: bool = False,
) -> Callable[[BaseTool], BaseTool]:
    """Decorate a LangChain tool so its results are extracted into the fact graph.

    Observability is wired through the standard LangChain callback chain
    — pass any ``BaseCallbackHandler`` subclass via
    ``config={"callbacks": [...]}`` on ``invoke`` / ``ainvoke``. After
    every successful ingest the decorator dispatches an
    ``agent_pinboard:ingest`` custom event whose payload carries the
    delta and the post-ingest graph; handlers (e.g.
    :class:`agent_pinboard.integrations.langfuse_hook.LangfuseHook`,
    :class:`agent_pinboard.integrations.websocket_hook.WebSocketHook`)
    pick it up via ``on_custom_event``.

    ``store_raw=True`` additionally stashes the tool's raw return JSON
    under ``("agent_pinboard", thread_id, "raw_events", event_id)`` for each
    event the call produced, so the ``get_evidence`` tool can replay it.
    Default off to keep storage lean.
    """
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise AgentPinBoardConfigError(
            f"@pin(model=...) expects a Pydantic BaseModel subclass, got {model!r}"
        )
    on_duplicate = OnDuplicate(on_duplicate)
    masked = list(mask_args or ())

    # Eager-scan model into the declared-entities registry. Done at
    # decoration time so graph_summary() works before the first call.
    register_model(model)

    def decorator(target: Any) -> BaseTool:
        if not isinstance(target, BaseTool):
            raise AgentPinBoardConfigError(
                "@pin must be placed ABOVE @tool. "
                "Correct order: `@pin(...) / @tool / def my_tool(...)`."
            )

        is_async = target.coroutine is not None
        original_func = target.coroutine if is_async else target.func
        if original_func is None:
            raise AgentPinBoardConfigError(
                f"BaseTool {target.name!r} has neither func nor coroutine; cannot wrap"
            )

        if (
            response_transform is not None
            and not is_async
            and asyncio.iscoroutinefunction(response_transform)
        ):
            raise AgentPinBoardConfigError(
                "sync tool cannot use an async response_transform — "
                "the result is needed synchronously"
            )

        ctx = _Ctx(
            model=model,
            many=many,
            on_duplicate=on_duplicate,
            mask_args=masked,
            response_transform=response_transform,
            tool_name=target.name,
            original_signature=inspect.signature(original_func),
            store_raw=store_raw,
        )

        if is_async:
            target.coroutine = _make_async_wrapper(original_func, ctx)
            target.func = None
        else:
            target.func = _make_sync_wrapper(original_func, ctx)

        return target

    return decorator


# --------------------------------------------------------------------------- #
# Decorator state.                                                            #
# --------------------------------------------------------------------------- #

class _Ctx:
    """Frozen-ish bag of decorator parameters, shared by the wrapper closure."""

    __slots__ = (
        "many",
        "mask_args",
        "model",
        "on_duplicate",
        "original_signature",
        "response_transform",
        "store_raw",
        "tool_name",
    )

    def __init__(
        self,
        *,
        model: type[BaseModel],
        many: bool,
        on_duplicate: OnDuplicate,
        mask_args: list[str],
        response_transform: Callable[[Any, IngestResult], Any] | None,
        tool_name: str,
        original_signature: inspect.Signature,
        store_raw: bool = False,
    ) -> None:
        self.model = model
        self.many = many
        self.on_duplicate = on_duplicate
        self.mask_args = mask_args
        self.response_transform = response_transform
        self.tool_name = tool_name
        self.original_signature = original_signature
        self.store_raw = store_raw


# Sentinels for the duplicate-detection branches. Anything else returned
# from the duplicate check is the cached payload.
_PROCEED = object()
_SKIP = object()


# --------------------------------------------------------------------------- #
# Sync / async wrapper bodies — thin shells over shared pure helpers.         #
# --------------------------------------------------------------------------- #

def _make_sync_wrapper(original: Callable, ctx: _Ctx) -> Callable:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        store, thread_id = _resolve_runtime(args, kwargs, ctx)
        args_repr = _canonical_args_repr(args, kwargs, ctx)

        # Duplicate handling.
        last_record = _find_duplicate_sync(store, thread_id, ctx, args_repr)
        if last_record is not None:
            return _handle_duplicate_sync(store, thread_id, ctx, args_repr, last_record)

        # Tool execution (outside lock).
        start = time.perf_counter()
        try:
            raw = original(*args, **kwargs)
        except Exception:
            store_io.persist_tool_call(
                store, thread_id, _build_record(ctx, args_repr, None, "error: tool raised", start)
            )
            raise

        validated_items = _validate(raw, ctx)

        with lock_for(thread_id):
            graph = get_or_load_session(store, thread_id)
            payload, result = _build_payload(graph, validated_items, ctx)
            store_io.persist_delta(store, thread_id, payload.nodes, payload.edges)

        if ctx.store_raw:
            for ev_id in result.event_ids:
                store_io.persist_raw_event(store, thread_id, ev_id, raw)

        store_io.persist_tool_call(
            store, thread_id,
            _build_record(ctx, args_repr, _first_event(result), _summary(result), start),
        )
        _maybe_warn_soft_limit(store, thread_id)
        _dispatch_ingest_event(ctx, thread_id, payload, result, graph)

        return _apply_transform(ctx, raw, result)

    return wrapper


def _make_async_wrapper(original: Callable, ctx: _Ctx) -> Callable:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        store, thread_id = _resolve_runtime(args, kwargs, ctx)
        args_repr = _canonical_args_repr(args, kwargs, ctx)

        last_record = await _afind_duplicate(store, thread_id, ctx, args_repr)
        if last_record is not None:
            return await _ahandle_duplicate(store, thread_id, ctx, args_repr, last_record)

        start = time.perf_counter()
        try:
            raw = await original(*args, **kwargs)
        except Exception:
            await store_io.apersist_tool_call(
                store, thread_id, _build_record(ctx, args_repr, None, "error: tool raised", start)
            )
            raise

        validated_items = _validate(raw, ctx)

        with lock_for(thread_id):
            graph = await aget_or_load_session(store, thread_id)
            payload, result = _build_payload(graph, validated_items, ctx)
            await store_io.apersist_delta(store, thread_id, payload.nodes, payload.edges)

        if ctx.store_raw:
            for ev_id in result.event_ids:
                await store_io.apersist_raw_event(store, thread_id, ev_id, raw)

        await store_io.apersist_tool_call(
            store, thread_id,
            _build_record(ctx, args_repr, _first_event(result), _summary(result), start),
        )
        await _amaybe_warn_soft_limit(store, thread_id)
        await _adispatch_ingest_event(ctx, thread_id, payload, result, graph)

        out = _apply_transform(ctx, raw, result)
        if asyncio.iscoroutine(out):
            out = await out
        return out

    return wrapper


# --------------------------------------------------------------------------- #
# Pure helpers used by both paths.                                            #
# --------------------------------------------------------------------------- #

class _Payload:
    """Accumulator of nodes/edges produced by one ingest pass."""

    __slots__ = ("edges", "linked_facts", "new_facts", "nodes")

    def __init__(self) -> None:
        self.nodes: list[FactNode | EventNode] = []
        self.edges: list[FactEdge] = []
        self.new_facts: list[FactNode] = []
        self.linked_facts: list[FactNode] = []


def _resolve_runtime(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ctx: _Ctx,
) -> tuple[Any, str]:
    """Locate the ToolRuntime in the call's arguments, return (store, thread_id)."""
    bound = ctx.original_signature.bind_partial(*args, **kwargs)
    runtime = bound.arguments.get("runtime")
    if runtime is None:
        for v in (*args, *kwargs.values()):
            if hasattr(v, "store") and hasattr(v, "config"):
                runtime = v
                break
    if runtime is None:
        raise AgentPinBoardConfigError(
            f"tool {ctx.tool_name!r} must declare a `runtime: ToolRuntime` parameter"
        )
    store = getattr(runtime, "store", None)
    if store is None:
        raise AgentPinBoardConfigError(
            "graph must be compiled with .compile(store=...) to use @pin-decorated tools"
        )
    return store, thread_id_from(runtime)


def _canonical_args_repr(
    args: tuple[Any, ...], kwargs: dict[str, Any], ctx: _Ctx
) -> str:
    bound = ctx.original_signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    payload: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name == "runtime":
            continue
        if name in ctx.mask_args:
            payload[name] = "***"
            continue
        payload[name] = _jsonable(value)
    return json.dumps(payload, sort_keys=True, default=str)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _validate(raw: Any, ctx: _Ctx) -> list[BaseModel]:
    """Coerce ``raw`` into a list of validated model instances. Fail-loud."""
    raw_items = raw if ctx.many else [raw]
    if ctx.many and not isinstance(raw_items, list):
        raise AgentPinBoardValidationError(
            f"@pin(many=True): tool {ctx.tool_name!r} must return a list, "
            f"got {type(raw_items).__name__}"
        )
    out: list[BaseModel] = []
    for i, item in enumerate(raw_items):
        if isinstance(item, ctx.model):
            out.append(item)
            continue
        try:
            out.append(ctx.model.model_validate(item))
        except ValidationError as exc:
            raise AgentPinBoardValidationError(
                f"validation failed for tool {ctx.tool_name!r}"
                f"{f' item #{i}' if ctx.many else ''}: {exc}"
            ) from exc
    return out


def _build_payload(
    graph: FactGraph, items: list[BaseModel], ctx: _Ctx
) -> tuple[_Payload, IngestResult]:
    """Mutate the graph with ``items``; return the delta and a summary.

    Caller is responsible for holding the per-session lock and persisting
    ``payload.nodes`` and ``payload.edges`` to the store.
    """
    payload = _Payload()
    event_ids: list[str] = []
    new_node_count = 0
    linked_node_count = 0
    new_edge_count = 0
    warnings: list[str] = []

    for item in items:
        event = EventNode(
            id=str(uuid.uuid4()),
            source_tool=ctx.tool_name,
            timestamp=datetime.now(UTC),
            properties=event_properties(item),
        )
        graph.add_event(event)
        payload.nodes.append(event)
        event_ids.append(event.id)

        new_facts, linked_facts, edges, w = extract(item, graph, event.id, ctx.tool_name)
        warnings.extend(w)
        new_node_count += len(new_facts)
        linked_node_count += len(linked_facts)  # one count per distinct linked node
        new_edge_count += len(edges)

        payload.new_facts.extend(new_facts)
        payload.linked_facts.extend(linked_facts)
        payload.nodes.extend(new_facts)
        payload.edges.extend(edges)

    # Updated-node refresh: every fact node touched (new or linked) needs its
    # latest source_events / source_tools / last_seen persisted.
    refreshed_ids: set[str] = set()
    for f in (*payload.new_facts, *payload.linked_facts):
        if f.id in refreshed_ids:
            continue
        refreshed_ids.add(f.id)
        if f not in payload.nodes:
            payload.nodes.append(f)

    result = IngestResult(
        event_ids=event_ids,
        new_nodes=new_node_count,
        linked_nodes=linked_node_count,
        new_edges=new_edge_count,
        warnings=warnings,
    )
    return payload, result


def _build_ingest_payload(
    ctx: _Ctx,
    thread_id: str,
    payload: _Payload,
    result: IngestResult,
    graph: FactGraph,
) -> dict[str, Any]:
    """Build the data dict for the ``agent_pinboard:ingest`` custom event."""
    events: list[EventNode] = [n for n in payload.nodes if isinstance(n, EventNode)]
    return {
        "thread_id": thread_id,
        "tool_name": ctx.tool_name,
        "result": result,
        "events": events,
        "new_facts": list(payload.new_facts),
        "linked_facts": list(payload.linked_facts),
        "new_edges": list(payload.edges),
        "graph": graph,
    }


def _dispatch_ingest_event(
    ctx: _Ctx,
    thread_id: str,
    payload: _Payload,
    result: IngestResult,
    graph: FactGraph,
) -> None:
    """Dispatch the per-ingest custom event into the LangChain callback chain.

    Outside a runnable context (e.g. unit tests calling the wrapped tool
    directly) ``dispatch_custom_event`` raises ``RuntimeError``; we
    swallow it so the decorator stays usable in those settings.
    """
    data = _build_ingest_payload(ctx, thread_id, payload, result, graph)
    try:
        dispatch_custom_event(INGEST_EVENT, data)
    except RuntimeError:
        # No callback manager in scope — nothing to dispatch to.
        pass
    except Exception:  # noqa: BLE001 — observability never breaks ingestion
        logger.error("agent_pinboard:ingest dispatch failed", exc_info=True)


async def _adispatch_ingest_event(
    ctx: _Ctx,
    thread_id: str,
    payload: _Payload,
    result: IngestResult,
    graph: FactGraph,
) -> None:
    data = _build_ingest_payload(ctx, thread_id, payload, result, graph)
    try:
        await adispatch_custom_event(INGEST_EVENT, data)
    except RuntimeError:
        pass
    except Exception:  # noqa: BLE001
        logger.error("agent_pinboard:ingest dispatch failed", exc_info=True)


def _apply_transform(ctx: _Ctx, raw: Any, result: IngestResult) -> Any:
    if ctx.response_transform is None:
        return raw
    return ctx.response_transform(raw, result)


def _build_record(
    ctx: _Ctx,
    args_repr: str,
    event_id: str | None,
    summary: str,
    start: float,
) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=ctx.tool_name,
        args_repr=args_repr,
        timestamp=datetime.now(UTC),
        event_id=event_id,
        summary=summary,
        duration_ms=int((time.perf_counter() - start) * 1000),
    )


def _summary(result: IngestResult) -> str:
    return (
        f"+{result.new_nodes} nodes, +{result.linked_nodes} linked, "
        f"+{result.new_edges} edges"
    )


def _first_event(result: IngestResult) -> str | None:
    return result.event_ids[0] if result.event_ids else None


# --------------------------------------------------------------------------- #
# Duplicate detection — sync.                                                 #
# --------------------------------------------------------------------------- #

def _find_duplicate_sync(
    store: Any, thread_id: str, ctx: _Ctx, args_repr: str
) -> ToolCallRecord | None:
    if ctx.on_duplicate is OnDuplicate.ALWAYS:
        return None
    records = store_io.load_tool_calls(store, thread_id)
    return _last_matching(records, ctx.tool_name, args_repr)


def _handle_duplicate_sync(
    store: Any, thread_id: str, ctx: _Ctx, args_repr: str, last: ToolCallRecord
) -> Any:
    if ctx.on_duplicate is OnDuplicate.SKIP:
        store_io.persist_tool_call(
            store, thread_id, _build_record(ctx, args_repr, None, "duplicate (skipped)", time.perf_counter())
        )
        _maybe_warn_soft_limit(store, thread_id)
        return "duplicate call skipped"
    # CACHE
    store_io.persist_tool_call(
        store, thread_id,
        _build_record(ctx, args_repr, None, "duplicate (cached)", time.perf_counter()),
    )
    _maybe_warn_soft_limit(store, thread_id)
    return f"duplicate (cached at {last.timestamp.isoformat()})"


def _maybe_warn_soft_limit(store: Any, thread_id: str) -> None:
    limit = get_config().tool_log_soft_limit
    records = store_io.load_tool_calls(store, thread_id)
    if len(records) > limit:
        logger.warning(
            "tool_log soft limit exceeded for thread_id=%s: %d > %d records",
            thread_id, len(records), limit,
        )


# --------------------------------------------------------------------------- #
# Duplicate detection — async (mirrors sync, swaps store calls).              #
# --------------------------------------------------------------------------- #

async def _afind_duplicate(
    store: Any, thread_id: str, ctx: _Ctx, args_repr: str
) -> ToolCallRecord | None:
    if ctx.on_duplicate is OnDuplicate.ALWAYS:
        return None
    records = await store_io.aload_tool_calls(store, thread_id)
    return _last_matching(records, ctx.tool_name, args_repr)


async def _ahandle_duplicate(
    store: Any, thread_id: str, ctx: _Ctx, args_repr: str, last: ToolCallRecord
) -> Any:
    if ctx.on_duplicate is OnDuplicate.SKIP:
        await store_io.apersist_tool_call(
            store, thread_id, _build_record(ctx, args_repr, None, "duplicate (skipped)", time.perf_counter())
        )
        await _amaybe_warn_soft_limit(store, thread_id)
        return "duplicate call skipped"
    await store_io.apersist_tool_call(
        store, thread_id,
        _build_record(ctx, args_repr, None, "duplicate (cached)", time.perf_counter()),
    )
    await _amaybe_warn_soft_limit(store, thread_id)
    return f"duplicate (cached at {last.timestamp.isoformat()})"


async def _amaybe_warn_soft_limit(store: Any, thread_id: str) -> None:
    limit = get_config().tool_log_soft_limit
    records = await store_io.aload_tool_calls(store, thread_id)
    if len(records) > limit:
        logger.warning(
            "tool_log soft limit exceeded for thread_id=%s: %d > %d records",
            thread_id, len(records), limit,
        )


def _last_matching(
    records: list[ToolCallRecord], tool_name: str, args_repr: str
) -> ToolCallRecord | None:
    for r in reversed(records):
        if (
            r.tool_name == tool_name
            and r.args_repr == args_repr
            and not r.summary.startswith("duplicate")
            and not r.summary.startswith("error")
        ):
            return r
    return None


__all__ = ["pin"]
