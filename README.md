# AgentPinBoard

> LLM-agent working memory as a session-scoped fact graph.

[Documentation (English)](docs/en/index.md) · [Документация (русский)](docs/ru/index.md) · [Detailed spec (RU)](README.ru.md)

`AgentPinBoard` gives a LangChain / LangGraph agent a **fact graph** as
working memory for one session (minutes to hours). The agent calls
your tools; their structured returns are auto-extracted into the graph
by the `@pin` decorator. The agent then reads the graph through
ready-made graph tools (`explore`, `find_path`, `timeline`, ...) to
navigate what it has already learned — without burning context on raw
tool returns.

The graph is a side-effect of normal tool calls. No explicit
`memory.add(...)` API. No LLM in the extraction path: extraction is
deterministic, free, and fast.

## Install

```bash
pip install agent-pinboard

# optional integrations:
pip install 'agent-pinboard[langfuse]'   # LangfuseHook
pip install 'agent-pinboard[ws]'         # WebSocketHook + Cytoscape.js demo
```

Python 3.12+. LangChain ≥ 1.2, LangGraph ≥ 1.1, Pydantic ≥ 2.13.

## 60-second quickstart

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from agent_pinboard import Entity, make_graph_tools, node, pin

# 1. Declare your entity types.
IP = Entity(name="IP", description="ipv4/ipv6", normalizer=lambda v: str(v).lower())
User = Entity(name="User", description="acting principal")

# 2. Pydantic model for the tool's return; mark fields that should
#    become graph nodes with node(...).
class CloudTrailEvent(BaseModel):
    src_ip: str | None = node(type=IP, description="source IP", default=None)
    actor: str | None = node(type=User, description="who", default=None)

# 3. Decorate the tool: @pin above @tool.
@pin(model=CloudTrailEvent, many=True)
@tool
def fetch_cloudtrail(user_arn: str, runtime: ToolRuntime) -> list[dict]:
    """Fetch CloudTrail events."""
    return [
        {"src_ip": "185.220.101.42", "actor": user_arn},
        {"src_ip": "185.220.101.42", "actor": user_arn},
    ]

# 4. Hand the agent your tools + the read-side graph tools.
agent = create_agent(
    model=your_llm,
    tools=[fetch_cloudtrail, *make_graph_tools()],
    store=InMemoryStore(),  # use AsyncPostgresStore in production
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Investigate AssumeRole."}]},
    config={"configurable": {"thread_id": "investigation-001"}},
)
```

After a few tool calls the LLM can ask the graph:

- `explore(node_type="IP", value="185.220.101.42")` — neighbours.
- `find_path(from_type="IP", from_value="…", to_type="User", to_value="…")` — shortest path.
- `timeline(node_type="User", value="alice")` — chronological events.
- `graph_summary()` — types known + top-N facts per type.
- `what_have_i_done()` — tool-call log.

## Observability

Wire `LangfuseHook`, `WebSocketHook`, or any `BaseCallbackHandler`
through the standard LangChain callback chain. After every successful
ingest `@pin` dispatches an `agent_pinboard:ingest` custom event with
the per-call delta (`IngestResult`, new facts, linked facts, edges,
post-ingest graph reference).

```python
from langchain_core.callbacks import BaseCallbackHandler
from agent_pinboard import INGEST_EVENT

class PrintIngest(BaseCallbackHandler):
    def on_custom_event(self, name, data, *, run_id, **kw):
        if name != INGEST_EVENT:
            return
        r = data["result"]
        print(f"{data['tool_name']}: +{r.new_nodes} new, +{r.linked_nodes} linked")

agent.invoke(..., config={"callbacks": [PrintIngest()], "configurable": {...}})
```

See [`docs/en/hooks-and-config.md`](docs/en/hooks-and-config.md) for
the full payload schema, the bundled `LangfuseHook` /
`WebSocketHook`, and production-storage notes.

## Why a graph (and not just longer context)

- **Deduplication.** Two tools mention `8.8.8.8` → one canonical
  `FactNode`, both calls' provenance preserved.
- **Cross-tool linking.** A user the CloudTrail tool surfaced and a
  user a SAML tool surfaced collapse to the same node — the agent can
  see the connection without re-reasoning.
- **Provenance for every fact.** Every fact carries the EventNodes (=
  tool calls) it came from; `get_evidence(event_id)` returns the raw
  tool return when `@pin(store_raw=True)`.
- **Bounded memory under long agent loops.** The agent re-reads a
  compact graph view instead of an ever-growing message history.
- **Multi-process correct.** Stored `FactNode` is the immutable subset
  only; provenance is derived from edges + EventNodes at load time, so
  workers sharing one `PostgresStore` never lose each other's links.

## Examples

The `examples/` directory ships three Jupyter notebooks (they render
inline on GitHub):

- [`examples/agent_demo.ipynb`](examples/agent_demo.ipynb) — full
  end-to-end agent with a deterministic mock LLM and a `PrintIngest`
  callback handler. **Start here.**
- [`examples/web/server_demo.ipynb`](examples/web/server_demo.ipynb)
  — same agent, but with `WebSocketHook` + Cytoscape.js live
  visualisation served on `http://localhost:8765/`.
- [`examples/langfuse_demo.ipynb`](examples/langfuse_demo.ipynb) —
  minimal `LangfuseHook` setup.

## Documentation

- English: [`docs/en/`](docs/en/index.md) — quickstart, concepts,
  extraction rules, graph tools, hooks, pitfalls, API reference,
  examples.
- Русский: [`docs/ru/`](docs/ru/index.md) — те же 8 страниц.
- Detailed Russian technical spec (motivation, design tradeoffs,
  comparison with Graphiti / Mem0 / Letta / Cognee / AriGraph):
  [`README.ru.md`](README.ru.md).

## Status

0.1 — public API is feature-frozen for this release; further work is
bug-fix and polish unless explicitly opened. 167 tests pass with all
optional extras installed (`uv run pytest`); ruff clean
(`uv run ruff check agent_pinboard/`).

## License

Apache-2.0 — see [LICENSE](LICENSE).
