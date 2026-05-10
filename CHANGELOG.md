# Changelog

All notable changes to AgentPinBoard are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.1.0] — 2026-05-11

Initial public release.

### Added

#### Core API

- **`pin(*, model, many=False, on_duplicate=ALWAYS, mask_args=None, response_transform=None, store_raw=False)`** — decorator that wraps a LangChain `@tool` so its returns are validated against a Pydantic model and the extracted facts are merged into the session graph. Supports sync and async tools (detected via `asyncio.iscoroutinefunction`). Must be placed above `@tool`; reverse order raises `AgentPinBoardConfigError` at decoration time.
- **`node(*, type, description, ...)`** — Pydantic field factory that marks a field as a graph node. Accepts any `Field` kwarg in addition to `type: Entity` and `description: str` (required).
- **`Entity(name, description, normalizer=None)`** — frozen value-object describing a node type. The optional `normalizer: Callable[[Any], str]` decides canonical equivalence (e.g. lowercase / `ipaddress.ip_address().compressed`).
- **`INGEST_EVENT = "agent_pinboard:ingest"`** — custom-event name dispatched after every successful ingest; subscribed to via `BaseCallbackHandler.on_custom_event`.
- **`make_graph_tools() -> list[BaseTool]`** — factory returning the seven graph-read tools.
- **`configure(*, tool_log_soft_limit=500)`** — process-global settings; for now only the soft-limit on the per-session tool log.

#### Graph model

- **`FactNode`** — semantic entity in the graph. Stored representation is the immutable subset (`id`, `node_type`, `value`, `canonical_value`); provenance fields (`source_events`, `source_tools`, `first_seen`, `last_seen`) are derived from edges + EventNodes by `FactGraph.from_snapshot` at load time, making the storage layer idempotent under cross-process upserts.
- **`EventNode`** — one per tool invocation, always created. Carries `source_tool`, `timestamp`, and non-node fields from the tool's return as `properties`.
- **`FactEdge`** — append-only `EventNode → FactNode` edge with deterministic id `{event_id}|{edge_type}|{target_id}`. `edge_type` is `"{ModelClass}.{field_name}"` where `ModelClass` is the class that declares the field (MRO-walked).
- **`FactGraph`** — runtime container over a `networkx.MultiDiGraph` plus sidecar indices (`nodes_by_key`, `nodes_by_type`). Exposes `upsert_fact`, `add_event`, `add_edge`, query helpers (`get`, `find_by_value`, `search_by_type`, `all_facts`, `all_events`, `edges_for_event`), `to_mermaid(max_facts=30)`, and JSON round-trip via `dump_to_dict` / `load_from_dict`.
- **`IngestResult`** — summary dataclass passed in the ingest event payload: `event_ids`, `new_nodes`, `linked_nodes`, `new_edges`, `warnings`.
- **`ToolCallRecord`** — per-call log entry written under namespace `("agent_pinboard", thread_id, "tool_calls", record_id)`.

#### Extraction

- Five extraction rules in `agent_pinboard/extract.py::_walk`, implemented as a `match` statement: scalar `node`, `list[primitive]` `node`, nested `BaseModel`, `list[BaseModel]`, plain field → `EventNode.properties`.
- Eager-scan: `@pin(model=X)` walks the model at decoration time to populate the session-level "declared entities" registry, so `graph_summary()` works **before** the first tool call.
- Recursion guard via `seen: set[type[BaseModel]]` — `Process(parent: Process | None)` does not loop.
- Fail-loud on Pydantic validation failure (raises `AgentPinBoardValidationError`); graph is not mutated, `IngestResult` is not produced. Same contract for normalizer exceptions (`AgentPinBoardNormalizerError`).

#### Read-side graph tools

- **`explore(node_type, value, depth=2, direction=BOTH, skip_events=True, max_nodes=30)`** — neighbourhood around a fact. `skip_events=True` treats EventNodes as transparent connectors so the LLM sees fact-to-fact edges directly.
- **`find_path(from_type, from_value, to_type, to_value, top=1, max_depth=6, skip_events=True)`** — shortest paths (undirected BFS).
- **`timeline(node_type, value, limit=50)`** — chronological event timeline for an entity, ranked AriGraph-style.
- **`graph_summary(top_per_type=5)`** — types with counts plus top-N facts per type. Works on empty graph too (returns known types from the registry).
- **`search_nodes(node_type=None, value_pattern=None, include_events=False, limit=50)`** — listing / glob filter; EventNodes hidden by default.
- **`get_evidence(event_id)`** — returns the raw tool return for an event when the tool used `@pin(store_raw=True)`.
- **`what_have_i_done(tool_name=None, node_type=None, value=None, limit=50)`** — tool-call log filter. `value` is matched against the **canonical** form via the registered `Entity.normalizer`.

#### Storage

- Sharded LangGraph `Store` layout under namespace `("agent_pinboard", thread_id, ...)`: `nodes/<id>`, `edges/<id>`, `entities`, `tool_calls/<id>`, `raw_events/<event_id>`.
- **No process-local graph cache.** Every `@pin` ingest and every read tool performs a fresh `load_graph` from the Store. Combined with the mergeable `FactNode` storage, this gives cross-process correctness without a distributed lock: two workers upserting the same canonical fact write byte-identical node dicts and append distinct edges with unique ids; provenance is derived from those edges on the next load.
- Per-`thread_id` `threading.RLock` still serialises the read-modify-write window within one process.
- `session_id` resolution from `runtime.config["configurable"]["thread_id"]`; absent → fresh UUID4 with a WARN-level log line so two anonymous parallel runs are guaranteed to be isolated.
- Tested against in-memory `InMemoryStore` for the test suite; works against `langgraph.store.postgres.AsyncPostgresStore` out of the box for production deployments.

#### Observability

Wired through the standard LangChain callback chain — any `BaseCallbackHandler` registered via `config={"callbacks": [...]}` on `agent.invoke` / `ainvoke` receives the dispatched `agent_pinboard:ingest` custom event with payload:

| Key | Type | Notes |
| --- | --- | --- |
| `thread_id` | `str` | session id |
| `tool_name` | `str` | decorated tool's name |
| `result` | `IngestResult` | per-call delta summary |
| `events` | `list[EventNode]` | one per call (or per item if `many=True`) |
| `new_facts` | `list[FactNode]` | brand-new facts |
| `linked_facts` | `list[FactNode]` | existing facts re-linked this ingest |
| `new_edges` | `list[FactEdge]` | one per fact occurrence in the model |
| `graph` | `FactGraph` | post-ingest graph (in-memory view) |

- Dispatch is guarded by `try/except (RuntimeError, LookupError)` so calling a `@pin` tool's underlying `func` directly (e.g. in a unit test outside a runnable context) does not log spurious errors. Other exceptions are caught and logged at ERROR — observability never breaks ingestion.
- **`LangfuseHook`** (optional, `pip install 'agent-pinboard[langfuse]'`) — `BaseCallbackHandler` that emits one `agent_pinboard.ingest` span per ingest plus an optional `agent_pinboard.graph_snapshot` span with a Mermaid render of the post-ingest graph in metadata. Auto-nesting under the LangChain tool span requires `langfuse.langchain.CallbackHandler` to be in the same `callbacks` list (documented).
- **`WebSocketHook`** + **`serve_websocket`** (optional, `pip install 'agent-pinboard[ws]'`) — `BaseCallbackHandler` that fans each ingest into per-node / per-edge / per-link / `ingest_complete` JSON deltas on a bounded thread-safe queue; `serve_websocket(hook, ...)` runs an asyncio WebSocket server that broadcasts them to connected clients and optionally serves a static HTML page on the same port.

#### Configuration & exceptions

- `configure(tool_log_soft_limit: int = 500)` — process-global, mutable. When the per-session tool-call log exceeds the limit, a WARN is logged; no hard cap.
- Exception hierarchy: `AgentPinBoardError` and four subclasses (`AgentPinBoardConfigError`, `AgentPinBoardValidationError`, `AgentPinBoardNormalizerError`, `AgentPinBoardExtractionError`).

#### Examples

Three Jupyter notebooks under `examples/` (render inline on GitHub):

- **`examples/agent_demo.ipynb`** — full end-to-end agent with a deterministic `MockChatModel` walking a 6-step plan; registers a `PrintIngest` `BaseCallbackHandler` so the observability story is visible from the start.
- **`examples/web/server_demo.ipynb`** + **`examples/web/index.html`** — same agent with `WebSocketHook` and Cytoscape.js live visualisation served on `http://localhost:8765/`.
- **`examples/langfuse_demo.ipynb`** — minimal `LangfuseHook` setup with a friendly guard if the `[langfuse]` extra is not installed.

#### Documentation

- English: `docs/en/{index,quickstart,concepts,extraction-rules,graph-tools,hooks-and-config,pitfalls,examples,api-reference}.md` — nine pages.
- Russian: `docs/ru/` mirrors the same nine pages.
- Detailed Russian technical spec / motivation / comparison-with-alternatives lives in `README.ru.md`.
- mkdocs configuration (`mkdocs.yml`) with **mkdocs-material**, **mkdocs-static-i18n** (folder mode → `docs/{en,ru}/` map directly), and **mkdocstrings** (configured but not yet used — `api-reference.md` is hand-written).

#### Tooling

- `pyproject.toml` with `agent-pinboard` distribution name, Apache-2.0 classifier, `>=Python 3.12`, optional extras `[langfuse]` and `[ws]`, and a `docs` dependency group for `mkdocs` + `mike`.
- **CI** (`.github/workflows/ci.yml`) — ruff + pytest on Python 3.12 and 3.13, with both optional extras installed (full 172-test suite). Cancels in-flight runs on the same ref. Exposes a `workflow_call` trigger.
- **Release** (`.github/workflows/release.yml`) — triggered on `v*` tags. Reuses the CI matrix as a green-tag gate; verifies the tag matches `[project].version` in `pyproject.toml`; builds with `uv build`; publishes to PyPI via `pypa/gh-action-pypi-publish` gated by the `pypi` GitHub Environment.
- **Docs** (`.github/workflows/docs.yml`) — `main` pushes deploy as `dev`; `v*` tags deploy as the version + `latest` alias via `mike`. Targets the `gh-pages` branch.

### Performance

Soft targets (not blockers, observed locally on a single InMemoryStore):

- `explore(depth=2)` on a 10 000-node graph — under 50 ms.
- `@pin` overhead (ingestion block for an event with 5 facts) — under 10 ms.
- Session load (sharded read of all nodes + edges) — under 500 ms for 10 000 nodes.

### Tested

172 tests pass with both optional extras installed (`uv sync --extra langfuse --extra ws && uv run pytest -q --timeout=20`). Coverage of note:

- Eight README §16 acceptance criteria end-to-end (`tests/test_acceptance.py`).
- Cross-process concurrency on a shared `InMemoryStore` with two threads without an in-process lock — verifies the mergeable storage model preserves both workers' edges (`tests/test_concurrency.py`).
- Async dispatch through a real `BaseCallbackHandler` via `ainvoke` (`tests/test_decorator.py::TestAsyncTool::test_async_dispatched_event_carries_result`).
- No-context dispatch swallow (direct call to a wrapped tool's `func` outside a runnable) does not log any "dispatch failed" ERROR.
- `FactGraph.from_snapshot` edge cases: orphan facts (no incoming edges → empty provenance), edges pointing to events missing from the snapshot (silently dropped), deterministic `source_events` ordering when timestamps tie.

[Unreleased]: https://github.com/cop1cat/agent-pinboard/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cop1cat/agent-pinboard/releases/tag/v0.1.0
