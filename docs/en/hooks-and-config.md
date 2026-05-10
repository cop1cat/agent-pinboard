# Callbacks and configuration

## Observability via LangChain callbacks

AgentPinBoard plugs into the **standard LangChain callback chain**.
After every successful `@pin` ingest the decorator dispatches a
`agent_pinboard:ingest` custom event into whatever
`BaseCallbackHandler`s the caller registered via
`config={"callbacks": [...]}` on `agent.invoke` / `ainvoke`.

This means the same dispatch mechanism that surfaces `on_tool_start` /
`on_tool_end` / `on_llm_start` / etc. surfaces AgentPinBoard ingest
events too — your handler gets one stream, parented under the
LangChain run that emitted it (so e.g. Langfuse spans nest naturally
under the tool span instead of floating as detached traces).

### A minimal handler

```python
from langchain_core.callbacks import BaseCallbackHandler
from agent_pinboard.decorator import INGEST_EVENT

class PrintIngest(BaseCallbackHandler):
    def on_custom_event(self, name, data, *, run_id, tags=None, metadata=None, **kw):
        if name != INGEST_EVENT:
            return
        result = data["result"]
        print(
            f"{data['tool_name']}: +{result.new_nodes} new, "
            f"+{result.linked_nodes} linked, +{result.new_edges} edges"
        )
```

Then attach it on every invocation:

```python
agent.invoke(
    {"messages": [...]},
    config={
        "configurable": {"thread_id": "session-42"},
        "callbacks": [PrintIngest()],
    },
)
```

### `agent_pinboard:ingest` payload

The custom event's `data` dict carries the post-ingest delta and a
reference to the freshly-loaded graph:

| Key | Type | Notes |
|---|---|---|
| `thread_id` | `str` | The session id this ingest landed in |
| `tool_name` | `str` | The decorated tool's name |
| `result` | `IngestResult` | `event_ids`, `new_nodes`, `linked_nodes`, `new_edges`, `warnings` |
| `events` | `list[EventNode]` | One per call (or one per item if `many=True`) |
| `new_facts` | `list[FactNode]` | Brand-new facts created in this ingest |
| `linked_facts` | `list[FactNode]` | Existing facts that this ingest re-linked |
| `new_edges` | `list[FactEdge]` | One per fact occurrence in the model |
| `graph` | `FactGraph` | The post-ingest graph (in-memory, this call's view) |

A handler that wants per-node granularity iterates `events` /
`new_facts` / `linked_facts` itself; one that just wants a coarse
"something changed" signal looks at `result`.

### Failure isolation

The decorator wraps `dispatch_custom_event` in `try/except`: a handler
that raises does **not** break ingestion — the exception is logged at
ERROR. The `agent_pinboard:ingest` payload always reflects the
successfully-persisted delta, even if a handler later crashes
processing it.

## `LangfuseHook`

Optional dependency. Install with:

```bash
uv add 'agent_pinboard[langfuse]'        # or: pip install agent_pinboard[langfuse]
```

```python
from langfuse import Langfuse
from agent_pinboard.integrations.langfuse_hook import LangfuseHook

client = Langfuse(public_key=..., secret_key=..., host=...)
handler = LangfuseHook(client)

result = await agent.ainvoke(
    {"messages": [...]},
    config={
        "callbacks": [handler],
        "configurable": {"thread_id": "session-42"},
    },
)
```

What it emits:

* one Langfuse span `agent_pinboard.ingest` per ingest, with the
  per-call delta (`new_nodes`, `linked_nodes`, `new_edges`, warnings).
* (optional, on by default) one span `agent_pinboard.graph_snapshot`
  per ingest, whose metadata carries a Mermaid flowchart of the
  current top-N facts and the events connecting them. Langfuse
  renders Mermaid in metadata, so you get a visual graph alongside
  the trace.

Both spans are parented under the current LangChain tool span — the
trace tree stays connected.

Constructor options:

* `max_facts_in_snapshot=30` — top-N facts (by event count) included
  in each Mermaid render.
* `emit_snapshots=False` — disable the per-change snapshot if you only
  want ingest spans (cheaper, less Langfuse traffic).

The handler swallows its own exceptions — failures are logged at
ERROR and the surrounding agent run is unaffected.

## `WebSocketHook`

Optional dependency. Install with:

```bash
uv add 'agent_pinboard[ws]'        # or: pip install agent_pinboard[ws]
```

The handler turns each `agent_pinboard:ingest` event into a stream of
JSON deltas (one per node, edge, link, plus a final
`ingest_complete`) into a thread-safe queue;
`serve_websocket(handler, ...)` runs an asyncio WebSocket server that
broadcasts each delta to every connected client.

```python
import asyncio
from langchain.agents import create_agent
from agent_pinboard import pin, make_graph_tools
from agent_pinboard.integrations.websocket_hook import (
    WebSocketHook, serve_websocket,
)

handler = WebSocketHook(thread_id_label="investigation-001")

async def main():
    server = asyncio.create_task(serve_websocket(handler, port=8765))
    agent = create_agent(...)
    await asyncio.to_thread(
        agent.invoke,
        {"messages": [...]},
        {
            "configurable": {"thread_id": "investigation-001"},
            "callbacks": [handler],
        },
    )
    await server

asyncio.run(main())
```

Wire format (JSON, one message per line):

* `snapshot` — full graph dump on connect.
* `node_added` / `edge_added` — incremental changes.
* `link_found` — an existing fact was re-linked from a new event.
* `ingest_complete` — a `@pin` invocation finished successfully.

A ready-to-use Cytoscape.js frontend lives at
`examples/web/index.html` and the live demo notebook
`examples/web/server_demo.ipynb` ties everything together — run that
and open `http://localhost:8765/` in a browser to watch the graph
build up live.

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
overrides are out of scope; for them, write a callback handler that
filters records.

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
