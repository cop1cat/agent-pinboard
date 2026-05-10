# Concepts

Four close-sounding terms — keep them straight or your code reads
ambiguous later.

| Term | Plain English | Created by |
|---|---|---|
| `Entity` | The *kind* of thing a node represents (its name, description, optional normalizer) | You — once per type, in your project |
| `node(...)` | Marker on a Pydantic field: "this value should become a graph node of the given Entity" | You — when defining a tool's response model |
| `FactNode` | The runtime graph node for a single extracted entity occurrence | The library, automatically |
| `EventNode` | The runtime graph node for one tool invocation | The library, automatically |

`Entity` and `node()` are write-time concepts — you read and write them
when designing your tools. `FactNode` and `EventNode` are runtime objects
you almost never touch directly; the agent observes them through graph
tools.

## `Entity` — node-type descriptor

```python
from agent_pinboard import Entity

IP = Entity(
    name="IP",
    description="IPv4 or IPv6 network address",
    normalizer=canonical_ip,           # optional
)
```

- `name` becomes `FactNode.node_type` on every node of this kind.
- `description` shows up in `graph_summary` so the LLM understands
  *what* this type is.
- `normalizer` canonicalises raw values for autolinking — without it,
  `192.168.001.001` and `192.168.1.1` would become two distinct nodes.

`Entity` is a frozen dataclass — pure value object, no side effects on
construction. Define each one **once** (typically in a project
`entities.py`) and import it everywhere you need it.

## `node()` — Pydantic field factory

```python
from agent_pinboard import node
from pydantic import BaseModel

class CloudTrailEvent(BaseModel):
    src_ip: str | None = node(
        type=IP,
        description="IP from which the API call was made",
        default=None,
    )
```

`node(...)` returns a regular Pydantic `FieldInfo` with AgentPinBoard
metadata attached. Pydantic still validates the field normally; the
extractor reads the metadata to decide what to do with the value.

Two descriptions, both required, both meaningful for the LLM:

- **`Entity.description`** — what this type *is* in general
  ("IPv4 or IPv6 network address"). Shown by `graph_summary`.
- **`node(description=...)`** — how this *specific field* relates to
  the event ("IP from which the API call was made"). Shown on edges
  by `explore` / `timeline`.

## When to use `node()` vs plain `Field()`

A heuristic that resolves 80% of cases:

- **Use `node(...)` if the value can recur across events and the LLM
  benefits from seeing connections between events that share it.**
  IPs, user IDs, file hashes, ARNs, ИНН, order IDs.
- **Use plain `Field(...)` if the value is one-shot and linking it makes
  no sense.** Timestamps, raw message bodies, latencies, status codes.

Plain-`Field` values land in `EventNode.properties` and are shown by
`timeline`.

## `@pin` — the decorator

```python
from agent_pinboard import pin
from langchain_core.tools import tool

@pin(model=CloudTrailEvent, many=True)
@tool
def fetch_cloudtrail(user: str, runtime: ToolRuntime) -> list[dict]:
    """Fetch CloudTrail logs for a user."""
    return aws_client.lookup_events(user)
```

The decorator is a side-effect. Every call:

1. validates the tool's return against `model`;
2. creates an `EventNode`;
3. extracts `FactNode`s and `FactEdge`s from the model;
4. merges them into the session graph (under a per-thread lock);
5. records a `ToolCallRecord`;
6. fires hooks;
7. optionally rewrites the return via `response_transform`.

`@pin` **must be above** `@tool`. Reverse order raises
`AgentPinBoardConfigError` at decoration time.

## Star topology

Topology is always **star around `EventNode`**: extracted facts connect
to their event, never directly to other facts.

```
              EventNode:fetch_cloudtrail @ 14:22:01
             /            |              \
       IP:185.220       User:admin       Action:AssumeRole
```

Why star: each event creates one new node + N edges instead of N²
edges. Two facts that "go together" share an event; the LLM finds them
via `explore(...)` (which by default treats events as transparent
connectors — `skip_events=True`).

## Session and Store

A **session** is identified by `thread_id` (read from
`runtime.config.configurable.thread_id`). All graph state for one
session lives in the LangGraph `Store` under a sharded namespace:

```
("agent_pinboard", thread_id, "nodes", node_id)        → FactNode | EventNode
("agent_pinboard", thread_id, "edges", edge_id)        → FactEdge
("agent_pinboard", thread_id, "entities")              → session entity registry
("agent_pinboard", thread_id, "tool_calls", record_id) → ToolCallRecord
```

In-memory caches accelerate the hot path; the store remains the source
of truth. Sessions in the same process are isolated by `thread_id`. If
no `thread_id` is supplied, AgentPinBoard generates a UUID4 and warns —
parallel "default" sessions never silently merge.

## Out of scope (deliberately)

The library does NOT do any of the following — see README §16 for the
rationale on each:

- bi-temporal validity (`valid_at` / `invalid_at`)
- confidence scoring on facts
- fuzzy entity resolution beyond exact-canonical match
- LLM-based extraction
- cross-session persistence as a primary scenario
- direct fact-to-fact edges (star topology only)

If you need any of these, build them on top — `Entity.properties` and
hooks give you the extension points.
