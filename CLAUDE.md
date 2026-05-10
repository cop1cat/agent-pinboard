# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**All three planned phases are complete.** 171 tests pass, lint is clean, both sanity scripts (`smoke.py`, `langgraph_check.py`) stay green. Public API is feature-frozen for 0.1; further work is bug-fix-and-polish unless the user explicitly opens a new phase.

Key sources of truth:

- `README.md` — the authoritative technical spec (ТЗ). Every architectural decision is recorded there with rationale. **Treat it as load-bearing — do not drift from it silently.** If you want to deviate, surface the tradeoff in conversation first.
- `REASEARCH.md` (the typo is intentional, the filename stays) — prior-art / landscape research. Its API examples are outdated (the early `Annotated[T, Fact(...)]` shape, replaced long ago by `node(type=Entity, ...)`). Read for context, not as API reference.
- `TODO.md` — **not** a phase roadmap anymore (all three phases are done); a running list of known limitations, desired follow-ups, and explicit refactor candidates. Priority 1 right now is a `LangfuseHook` redesign — the current implementation works but produces disconnected traces and two spans per ingest, which is noisy in a real Langfuse UI.
- `docs/` — user-facing docs in **English** (`docs/en/`) and **Russian** (`docs/ru/`). Eight pages each: index, quickstart, concepts, extraction-rules, graph-tools, hooks-and-config, pitfalls, api-reference, examples. **Update both languages when changing user-facing behaviour.**
- The equivalent sanity checks now live inside the test suite: `tests/test_acceptance.py` covers the README §16 acceptance criteria end-to-end, and `tests/test_decorator.py` / `tests/test_tools.py` exercise real `ToolRuntime` injection via `ToolNode` (the assumptions that the deleted `langgraph_check.py` used to verify).

## Common commands

Project uses `uv` for dependency management and virtualenv. Target: Python 3.12+.

```bash
uv sync                                   # sync env to pyproject.toml
uv add <package>                          # add a runtime dep
uv add --optional <extra> <package>       # add to an optional extra (e.g. ws, langfuse)

uv run pytest -q --timeout=20             # full test suite (171 tests)
uv run pytest tests/test_acceptance.py -v # 8 Phase-1 acceptance tests
uv run pytest tests/test_<file>.py::TestClass::test_name -v   # single test

uv run ruff check pinboard/               # lint
uv run ruff check pinboard/ --fix         # auto-fix

uv run python examples/agent_demo.py      # full agent with mock LLM
uv run python examples/web/server_demo.py # WS server + agent (then open http://localhost:8765/)
```

**Pytest can hang on first import in some shells.** Always launch long-running test commands with `run_in_background: true` and use `TaskOutput` to wait — never `sleep` in the foreground. The first run in a fresh worker may take 5-10 seconds while imports warm up; subsequent runs are sub-second.

The acceptance tests (`tests/test_acceptance.py`) are the executable contract — they should stay green through any change. If a change breaks them, that's a signal, not an excuse to edit the checks.

## Architecture at a glance

PinBoard is a Python library for LLM-agent working memory as a fact graph scoped to one session (minutes-hours). Agent calls tools, tool results get auto-extracted into a `FactGraph`, the LLM reads the graph via pre-built read-tools. Design is **domain-neutral** (security examples are for clarity, not for scope).

**Four close-sounding concepts, not the same thing:**

| Term | What it is | Created by |
|---|---|---|
| `Entity` | Frozen value-object describing a *node type* (name, description, optional normalizer) | User, once per type |
| `node(...)` | Pydantic field factory that marks a field as a graph node | User, in their Pydantic tool-response models |
| `FactNode` | Runtime graph node for an extracted fact | Library, automatically |
| `EventNode` | Runtime graph node representing one tool invocation | Library, automatically |

**Core flow:**

1. User writes a Pydantic model for the tool's return type, marking nodable fields with `node(type=SomeEntity, description="...")`.
2. User decorates the tool: `@fact(model=X) @tool def f(...): ...` (`@fact` must be **above** `@tool` — reverse order raises `PinBoardConfigError` at decoration time).
3. On each tool call, the decorator validates the return against the model, creates one `EventNode`, walks the model by five extraction rules, upserts `FactNode`s with autolinking on `(node_type, canonical_value)`, and connects them to the EventNode via `FactEdge`s whose `edge_type = "{ModelClass}.{field_name}"` (where `ModelClass` is the class that **declares** the field — found via MRO walk, so inherited / reused models keep stable labels).
4. LLM reads the graph via `make_graph_tools()` — seven tools: `explore`, `find_path`, `timeline`, `graph_summary`, `search_nodes`, `get_evidence`, `what_have_i_done`.

**Storage model:**

Graph lives in LangGraph `Store` under namespace `("pinboard", thread_id, ...)` split across:

- `nodes/<id>` — one key per `FactNode` / `EventNode`
- `edges/<id>` — one key per `FactEdge`
- `entities` — the session entity registry (single blob)
- `tool_calls/<id>` — one key per `ToolCallRecord`
- `raw_events/<event_id>` — only when `@fact(store_raw=True)`

In-process cache per `thread_id`. `threading.RLock` (reentrant — `@fact` and `get_or_load_session` both acquire it) protects only the ingestion read-modify-write block (step 4 of the `@fact` flow), not the whole tool body, so LangGraph's parallel tool execution is preserved. Async loads use a separate `asyncio.Lock` per `thread_id` to avoid TOCTOU between concurrent awaiters. `thread_id` comes from `runtime.config.configurable.thread_id`; absent → fresh UUID4 with a warning (parallel anonymous calls never silently merge).

**Topology:**

`star around EventNode` — FactNodes never link directly to each other, only through the Event. `explore` / `find_path` default to `skip_events=True`, treating EventNodes as transparent connectors so the LLM sees direct fact-to-fact relations.

## What is explicitly out of scope

README §16 lists ~14 capabilities rejected with rationale and references to REASEARCH §E.2 risks. The ones to push back on hardest:

- Bi-temporal model, confidence scoring, fuzzy entity resolution, state replacement, async deep-enrichment — deliberate tradeoffs, not oversights.
- LLM in the runtime extraction path — violates the core principle (deterministic, free extraction).
- Built-in domain normalizers, OCSF / STIX models — user responsibility; library provides the interface (`Entity.normalizer`), not implementations.
- Per-object `schema_version` — YAGNI; semver on the whole dump (`pinboard_version`) is enough.
- `rationale` / `interpretation` in tool log — library can't read the LLM's mind; that's an agent-level concern.
- `max_turns` / Exit-tool guardrails — agent-level (`create_agent` has these), not library-level.

If a change "while you're here" tries to add any of these, surface it as a real proposal — these were rejected after explicit discussion.

## Coordinates for finding things

**Spec (README.md):**
- Data model — §5
- Five extraction rules — §4.1, implemented via `match` statement
- `@fact` pipeline (7 steps, lock on step 4 only) — §6.1, §9.1
- Exception hierarchy (`PinBoardError` + 4 subclasses) — §6.7
- Enums (`Direction`, `OnDuplicate` — `StrEnum`) — §10, §6.3
- Sharded Store schema — §9.1
- Phase 1 acceptance criteria (the 8 tests) — §16

**Implementation (`pinboard/`):**
- `entity.py`, `fields.py`, `enums.py` — public markers / factories
- `models.py`, `graph.py` — runtime data classes + `FactGraph` (in-memory `MultiDiGraph` + sidecar indices + `dump_to_dict`/`load_from_dict`)
- `extract.py` — 5 extraction rules via `match`
- `decorator.py` — `@fact` pipeline (sync + async wrappers share pure helpers; sync/async paths differ only in the I/O calls)
- `store.py` — sharded sync + async I/O over LangGraph `BaseStore`
- `session.py` — per-session cache, `RLock`, `asyncio.Lock` for async loads, `thread_id_from(runtime)`
- `tools.py` — seven graph tools (`make_graph_tools`)
- `hooks.py`, `config.py`, `registry.py` — supporting machinery
- `integrations/langfuse_hook.py` — `LangfuseHook` + `render_mermaid` (optional `pinboard[langfuse]`)
- `integrations/websocket_hook.py` — `WebSocketHook` + `serve_websocket` (optional `pinboard[ws]`)

**Tests (`tests/`):**
- One file per module (`test_<name>.py`) plus `test_acceptance.py` for the §16 ACs and `test_review_fixes.py` for regression coverage of every reviewer-found bug.
- `tests/conftest.py` provides the autouse `reset_pinboard_state` fixture that wipes process-global state between tests.
- `tests/_helpers.py` — shared `make_runner` / `call` for driving `@fact`-tools through a tiny `ToolNode`-driven graph.

**Examples (`examples/`):**
- `agent_demo.py` — minimal end-to-end agent with `MockChatModel` walking a 6-step plan.
- `web/server_demo.py` + `web/index.html` — same agent, but with `WebSocketHook` + Cytoscape.js live visualisation.

**User docs (`docs/`):**
- English: `docs/en/{quickstart,concepts,extraction-rules,graph-tools,hooks-and-config,pitfalls,api-reference,examples}.md`
- Russian: same names under `docs/ru/`.

## Conventions worth noting

- `ToolRuntime` is imported from **`langgraph.prebuilt`**, not from `langchain_core.tools`. The `runtime: ToolRuntime` parameter on a `@tool` is auto-detected as injected — it appears in `args_schema` (for validation) but NOT in `tool_call_schema` (what the LLM sees). `langgraph_check.py` verifies this.
- `from langchain.agents import create_agent` — `langgraph.prebuilt.create_react_agent` is deprecated in V1, use the `langchain.agents` re-export.
- `StrEnum` (not `class X(str, Enum)`) for user-facing enums.
- `type NodeId = str`, `type EventId = str` — PEP 695 aliases; used in signatures for intent.
- `@dataclass(slots=True)` on all graph models; `frozen=True` where immutability matters (`Entity`, `FactEdge`, `ToolCallRecord`).
- `@override` from `typing` on hook subclasses (README §11.1) — typechecker catches typos in overridden method names.
- `match`-statement is the implementation of the 5 extraction rules (`pinboard/extract.py::_walk`). Adding a rule = adding a `case`, not amending an `if` chain.
- Per-session locks are `threading.RLock` (sync, reentrant) + `asyncio.Lock` (async loads). The spec originally said `anyio.Lock` but that doesn't work synchronously; the deviation is documented inline in `pinboard/session.py`.
- Hook callbacks are wrapped in `try/except` by `pinboard.hooks.fire`; a hook that raises **never breaks ingestion**, the failure is logged at ERROR. Preserve this contract when adding new hooks or hook implementations.
- Process-global state lives in `pinboard/{registry,session,config}.py`. Each module exposes a `_reset()` function used by the autouse `reset_pinboard_state` fixture in `tests/conftest.py`. **New modules with process-global state must expose `_reset()` and register it in that fixture.**
- Sync ↔ async parity: every async public function should mirror its sync counterpart (`load_graph` / `aload_graph`, `persist_delta` / `apersist_delta`, etc.). Decorator wraps both in the same `@fact` based on whether the underlying tool is sync or async (detected via `asyncio.iscoroutinefunction`).
- Optional integrations live under `pinboard/integrations/`. Each module imports its dependency lazily and raises `ImportError(_DEPENDENCY_HINT)` with a friendly install command. Add new integrations the same way; keep them out of the top-level public API.
