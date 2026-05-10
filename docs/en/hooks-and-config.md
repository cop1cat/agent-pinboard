# Hooks and configuration

## Hooks

`AgentPinBoardHooks` is a plain class — subclass and override only the
methods you care about. Every callback is wrapped in `try/except`:
**a hook that raises never breaks ingestion**, the failure is logged
at ERROR level and ingestion continues.

```python
from typing import override
from agent_pinboard import AgentPinBoardHooks
from agent_pinboard.models import EventId, FactNode, IngestResult

class MyHook(AgentPinBoardHooks):
    @override
    def on_node_added(self, node) -> None:
        print(f"new node: {node.node_type}")

    @override
    def on_link_found(self, existing: FactNode, event_id: EventId) -> None:
        print(f"linked existing: {existing.value}")

    @override
    def on_ingest_complete(self, result: IngestResult) -> None:
        print(f"+{result.new_nodes} nodes, {len(result.warnings)} warnings")
```

`@typing.override` is a Python 3.12 decorator — typecheckers will catch
typos in the method names you're overriding.

### Available callbacks

| Callback | Fires when |
|---|---|
| `on_node_added(node)` | Any new `FactNode` or `EventNode` is created |
| `on_edge_added(edge)` | Any new `FactEdge` is created |
| `on_link_found(existing, event_id)` | An existing `FactNode` is re-linked from a new event (one call per distinct linked fact per ingest) |
| `on_ingest_complete(result)` | One `@pin` invocation finished successfully |
| `on_graph_changed()` | Coarse "the graph mutated" signal, fires once per ingest |

### Built-in implementations

```python
from agent_pinboard import LoggingHook, CompositeHook
import logging

# Logs every callback at INFO.
log_hook = LoggingHook(level=logging.INFO)

# Fan out to several hooks; each isolated by try/except.
combined = CompositeHook([log_hook, MyHook()])
```

`LangfuseHook` and `WebSocketHook` ship as optional integrations — see
below for both.

### `LangfuseHook`

Optional dependency. Install with:

```bash
uv add 'agent_pinboard[langfuse]'        # or: pip install agent_pinboard[langfuse]
```

Then:

```python
from langfuse import Langfuse
from agent_pinboard.integrations.langfuse_hook import LangfuseHook

client = Langfuse(public_key=..., secret_key=..., host=...)
hooks = LangfuseHook(client)

@pin(model=CloudTrailEvent, many=True, hooks=hooks)
@tool
def fetch_cloudtrail(...): ...
```

What it emits:

* On every `on_ingest_complete` — a Langfuse span `agent_pinboard.ingest`
  with the per-call delta (`new_nodes`, `linked_nodes`, `new_edges`,
  warnings).
* On every `on_graph_changed` — a span `agent_pinboard.graph_snapshot`
  whose metadata carries a Mermaid flowchart of the current top facts
  and the events connecting them. Langfuse renders Mermaid in
  metadata, giving you a visual graph alongside the trace.

Constructor options:

* `max_facts_in_snapshot=30` — top-N facts (by event count) included
  in each Mermaid render.
* `emit_snapshots=False` — disable the per-change snapshot if you only
  want ingest spans (cheaper, less Langfuse traffic).

The hook never raises — failures are logged at ERROR (the
`AgentPinBoardHooks` log-and-continue contract is preserved).

### `WebSocketHook`

Optional dependency. Install with:

```bash
uv add 'agent_pinboard[ws]'        # or: pip install agent_pinboard[ws]
```

The hook collects every graph-change event into a thread-safe queue;
``serve_websocket(hook, ...)`` runs an asyncio WebSocket server that
broadcasts each delta (and a one-off snapshot on connect) to every
connected client.

```python
import asyncio
from agent_pinboard import pin, make_graph_tools
from agent_pinboard.integrations.websocket_hook import (
    WebSocketHook, serve_websocket,
)

hook = WebSocketHook(thread_id_label="investigation-001")

@pin(model=CloudTrailEvent, many=True, hooks=hook)
@tool
def fetch_cloudtrail(...): ...

async def main():
    server = asyncio.create_task(serve_websocket(hook, port=8765))
    # ... drive your agent (sync work goes through asyncio.to_thread) ...
    await server

asyncio.run(main())
```

Wire format (JSON, one message per line):

* `snapshot` — full graph dump on connect.
* `node_added` / `edge_added` — incremental changes.
* `link_found` — an existing fact was re-linked from a new event.
* `ingest_complete` — a `@pin` invocation finished successfully.

A ready-to-use Cytoscape.js frontend lives at
`examples/web/index.html` and the demo runner at
`examples/web/server_demo.py` ties everything together — run that and
open the HTML in a browser to watch the graph build live.

The hook never raises; like the others, exceptions in the WS layer are
logged and swallowed.

### Wiring hooks into a tool

Pass the hook to `@pin` (per-tool) and to `make_graph_tools` (for the
read tools, where they currently no-op):

```python
hooks = MyHook()

@pin(model=CloudTrailEvent, many=True, hooks=hooks)
@tool
def fetch_cloudtrail(...): ...

agent_tools = [fetch_cloudtrail, *make_graph_tools(hooks=hooks)]
```

## `configure()` — process-global settings

```python
from agent_pinboard import configure

configure(tool_log_soft_limit=200)
```

The only setting in Phase 1 is `tool_log_soft_limit` (default 500).
When the per-session tool-call log exceeds the limit, a warning is
logged; there's no hard cap. The warning is mostly a signal that
"the LLM is going in circles" — not a sign that storage is overloaded.

`configure()` is **process-global, mutable state**. Per-session
overrides are out of scope; for them, write a hook that drops records.

## Tool log

Every `@pin` invocation appends one `ToolCallRecord` to the per-session
log under namespace `("agent_pinboard", thread_id, "tool_calls", record_id)`:

```python
@dataclass(slots=True, frozen=True)
class ToolCallRecord:
    tool_name: str
    args_repr: str           # canonical JSON, deterministic for dedup
    timestamp: datetime
    event_id: EventId | None # None for duplicates that didn't ingest
    summary: str             # "+2 nodes, +1 linked, +3 edges" / "duplicate (skipped)" / "error: ..."
    duration_ms: int
```

The agent reads this through `what_have_i_done(...)`. Two ways the log
helps:

1. The LLM can ask "did I already query VirusTotal for this IP?" without
   re-running the call. Combined with `on_duplicate=OnDuplicate.SKIP`,
   the call simply returns a marker string.
2. After a long session you can post-mortem what tools ran, in what
   order, and what they produced.

### Args representation

`args_repr` is a stable JSON string built from the call's positional
and keyword arguments, with:

- `ToolRuntime` excluded (not serialisable, and per-session anyway),
- `kwargs` keys sorted (so `f(a=1, b=2)` and `f(b=2, a=1)` collide),
- Pydantic `BaseModel` instances dumped via `model_dump(mode="json")`,
- Anything else passed through `json.dumps(..., default=str)`.

This is what duplicate-detection (`on_duplicate`) compares against.

### Masking secrets

If your tool takes an API token or password, exclude it from the log
with `mask_args`:

```python
@pin(model=VTReport, mask_args=["api_key"])
@tool
def vt_lookup(value: str, api_key: str, runtime: ToolRuntime) -> dict:
    """."""
    ...
```

Masked arguments appear as `"***"` in `args_repr`. **Caveat**: a
rotating secret will look identical to a previous secret in the log,
so two calls with different real keys are dedup-equivalent. If you
rotate keys mid-session, either pass them through `runtime.config`
instead or use `on_duplicate=OnDuplicate.ALWAYS`.

## Multi-process / production storage

`InMemoryStore` is fine for tests and single-process demos, but it
loses everything when the process exits and cannot be shared across
workers. For production, use a shared backend — LangGraph ships an
async PostgreSQL store that AgentPinBoard works against without any
extra plumbing:

```python
from langgraph.store.postgres import AsyncPostgresStore
from langchain.agents import create_agent

async with AsyncPostgresStore.from_conn_string(
    "postgresql://user:pass@host:5432/db"
) as store:
    await store.setup()  # creates the tables once

    agent = create_agent(
        model=llm,
        tools=[*my_tools, *make_graph_tools()],
        store=store,
    )
    result = await agent.ainvoke(
        {"messages": [...]},
        config={"configurable": {"thread_id": "session-42"}},
    )
```

AgentPinBoard does **not** keep a process-local cache of the graph —
every `@pin` ingest and every read tool call performs a fresh
`load_graph` from the Store. Combined with the mergeable `FactNode`
storage (only the immutable subset is persisted; provenance is derived
from edges + EventNodes at load time), this means two workers on
different processes can ingest into the same `thread_id` concurrently
without losing each other's links.

A `threading.RLock` per `thread_id` still serializes the
read-modify-write window inside one process — preventing two threads
in the same worker from racing on their reload+persist cycle. There is
no cross-process distributed lock; you don't need one, because the
storage model is mergeable by construction.
