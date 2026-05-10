# PinBoard — Documentation (English)

PinBoard is a Python library that gives an LLM agent a **fact graph** as
its working memory for the duration of one session.

The agent calls tools, the tools' responses are auto-extracted into
`FactNode` / `EventNode` graph nodes, and the LLM reads the graph through
a small set of prebuilt tools (`explore`, `timeline`, `graph_summary`,
`search_nodes`, `what_have_i_done`).

## Reading order

1. **[Quickstart](./quickstart.md)** — install, run a 30-line example.
2. **[Concepts](./concepts.md)** — `Entity`, `node()`, `@fact`, `FactNode`, `EventNode`.
3. **[Extraction rules](./extraction-rules.md)** — the five rules that
   decide how a Pydantic field becomes a graph node.
4. **[Graph tools](./graph-tools.md)** — what the agent sees when it
   calls `explore`, `timeline`, etc.
5. **[Hooks & config](./hooks-and-config.md)** — observability and
   process-global settings.
6. **[Common pitfalls](./pitfalls.md)** — decorator order, normalizer
   identity, secret masking, and other things that trip up first-time users.
7. **[API reference](./api-reference.md)** — every public symbol.
8. **[Examples](./examples.md)** — full agents (security investigation
   and a non-security walkthrough).

## Status

Phase 1 is complete: extraction, autolinking, sharded persistence,
sync + async tools, hooks, five graph-read tools, exception hierarchy,
and 127 passing tests.

`find_path` is Phase 2; `get_evidence`, AriGraph-style ranking,
WebSocket streaming and others are Phase 3 (see the spec's §16 for
the explicit out-of-scope list and rationale).
