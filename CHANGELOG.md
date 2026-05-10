# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-11

Initial public release.

### Added

- `pin` decorator that wraps a LangChain `@tool` and ingests its return into a session-scoped fact graph; supports sync and async tools, `many=True` for list returns, `on_duplicate` (`ALWAYS` / `SKIP` / `CACHE`), `mask_args`, `response_transform`, and `store_raw=True` for forensic raw-return capture.
- `node(*, type, description, ...)` Pydantic field factory that marks a model field as a graph node.
- `Entity(name, description, normalizer=None)` value-object describing a node type, with optional canonicalisation callable.
- `INGEST_EVENT` constant (`"agent_pinboard:ingest"`) — name of the LangChain custom event dispatched after each successful ingest.
- `make_graph_tools()` factory returning the seven graph-read tools: `explore`, `find_path`, `timeline`, `graph_summary`, `search_nodes`, `get_evidence`, `what_have_i_done`.
- `configure(tool_log_soft_limit=500)` for process-global settings.
- Graph model: `FactGraph`, `FactNode`, `EventNode`, `FactEdge`, `IngestResult`, `ToolCallRecord`, with `Direction` and `OnDuplicate` enums.
- `FactGraph.to_mermaid(max_facts=30)` for ad-hoc Mermaid renders.
- `FactGraph.dump_to_dict()` / `FactGraph.load_from_dict()` for JSON round-trip.
- Five extraction rules (`agent_pinboard/extract.py`): scalar `node`, `list[primitive]` `node`, nested `BaseModel`, `list[BaseModel]`, plain field → `EventNode.properties`.
- Eager-scan of `@pin(model=...)` at decoration time so `graph_summary()` sees declared entity types before the first tool call.
- Recursion guard for self-referential Pydantic models.
- Sharded LangGraph `Store` layout under `("agent_pinboard", thread_id, ...)`: `nodes/<id>`, `edges/<id>`, `entities`, `tool_calls/<id>`, `raw_events/<event_id>`.
- Mergeable `FactNode` storage: only the immutable subset (`id`, `node_type`, `value`, `canonical_value`) is persisted; provenance (`source_events`, `source_tools`, `first_seen`, `last_seen`) is derived from edges + EventNodes by `FactGraph.from_snapshot` at load time, so two processes upserting the same canonical fact never lose each other's links.
- Per-`thread_id` `threading.RLock` serialising the read-modify-write window of one ingest within a single process.
- Cross-process correctness without a distributed lock; verified against `langgraph.store.memory.InMemoryStore` and compatible with `langgraph.store.postgres.AsyncPostgresStore` out of the box.
- `thread_id` resolution from `runtime.config["configurable"]["thread_id"]`, with UUID4 fallback + WARN log when absent.
- Observability via the standard LangChain callback chain: every successful ingest dispatches `INGEST_EVENT` with payload `{thread_id, tool_name, result: IngestResult, events, new_facts, linked_facts, new_edges, graph}`. Subscribers register through `config={"callbacks": [...]}` on `agent.invoke` / `ainvoke`.
- Dispatch is guarded by `try/except (RuntimeError, LookupError)` so calling a `@pin` tool's underlying `func` directly (outside a runnable context) is silent; other handler exceptions are caught and logged at ERROR — observability never breaks ingestion.
- `LangfuseHook` (`pip install 'agent-pinboard[langfuse]'`) — `BaseCallbackHandler` that emits one `agent_pinboard.ingest` span per ingest plus an optional `agent_pinboard.graph_snapshot` span carrying a Mermaid render in metadata.
- `WebSocketHook` and `serve_websocket` (`pip install 'agent-pinboard[ws]'`) — `BaseCallbackHandler` that fans each ingest into per-node / per-edge / per-link / `ingest_complete` JSON deltas on a bounded thread-safe queue plus an asyncio WebSocket server that broadcasts them and optionally serves a static HTML page on the same port.
- Exception hierarchy: `AgentPinBoardError`, `AgentPinBoardConfigError`, `AgentPinBoardValidationError`, `AgentPinBoardNormalizerError`, `AgentPinBoardExtractionError`.
- Fail-loud behaviour on Pydantic validation failure and on `Entity.normalizer` exceptions; the graph is not mutated and `IngestResult` is not produced.
- Three Jupyter example notebooks: `examples/agent_demo.ipynb` (full agent with mock LLM and `PrintIngest` callback), `examples/web/server_demo.ipynb` + `examples/web/index.html` (live Cytoscape.js visualisation), `examples/langfuse_demo.ipynb`.
- English (`docs/en/`) and Russian (`docs/ru/`) user docs — nine pages each: index, quickstart, concepts, extraction-rules, graph-tools, hooks-and-config, pitfalls, examples, api-reference.
- Detailed Russian technical specification in `README.ru.md`.
- `mkdocs.yml` with `mkdocs-material`, `mkdocs-static-i18n` (folder mode → `docs/{en,ru}/`), `mkdocstrings`, and `mike` for versioned deploys; `[dependency-groups] docs` group.
- `pyproject.toml` with `agent-pinboard` distribution name, Apache-2.0 license, Python 3.12+ requirement, `[langfuse]` and `[ws]` optional extras, and PyPI-ready classifiers.
- GitHub Actions workflows: `ci.yml` (ruff + pytest on Python 3.12 / 3.13 with both extras installed; in-flight runs cancelled on the same ref; exposes a `workflow_call` trigger), `release.yml` (`v*` tags re-run CI as a green-tag gate, verify the tag matches `[project].version`, build with `uv build`, publish to PyPI gated by the `pypi` Environment), `docs.yml` (`main` pushes deploy as `dev`, `v*` tags deploy as the version + `latest` alias via `mike` to the `gh-pages` branch).
- 172-test suite covering the eight README §16 acceptance criteria, cross-process concurrency on a shared Store without an in-process lock, async dispatch through a real `BaseCallbackHandler`, the no-context dispatch swallow, and `FactGraph.from_snapshot` edge cases (orphan facts, ghost edges, deterministic ordering tiebreaker).

[Unreleased]: https://github.com/cop1cat/agent-pinboard/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cop1cat/agent-pinboard/releases/tag/v0.1.0
