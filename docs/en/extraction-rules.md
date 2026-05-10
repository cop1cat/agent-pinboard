# Extraction rules

When `@pin` runs, it walks the validated Pydantic model and applies
five rules to every field. The rules are mutually exclusive — exactly
one fires per field.

For each field, let `value` be the actual value and `entity` be the
`Entity` attached via `node()` (or `None` if it's a plain `Field()`).

## Rule 1 — primitive with `node()`

```python
class Event(BaseModel):
    src_ip: str | None = node(type=IP, description="src", default=None)
```

→ creates `FactNode(type=IP.name, value=src_ip)` (subject to
autolinking and normalization), plus `FactEdge` from the EventNode
labelled `Event.src_ip`.

## Rule 2 — `None`

→ skipped silently. Use this for optional fields.

## Rule 3 — `list[primitive]` with `node()`

```python
class VTReport(BaseModel):
    related_ips: list[str] = node(
        type=IP, description="Related IPs from VT", default_factory=list,
    )
```

→ Rule 1 applied to every element. Each element becomes its own
FactNode and gets its own edge labelled `VTReport.related_ips`.

If an element is itself a `BaseModel`, `dict`, `tuple`, or `list`,
`AgentPinBoardExtractionError` is raised — `node()` on a list expects flat
primitives.

## Rule 4 — nested `BaseModel` (or `list[BaseModel]`) without `node()`

```python
class Actor(BaseModel):
    user_arn: str | None = node(type=User, description="who", default=None)

class CloudTrailEvent(BaseModel):
    actor: Actor | None = None        # no node() — Rule 4
```

→ recurse into the nested model, applying the rules to its fields.
Edges still come from the same outer `EventNode`. Their `edge_type`
uses the **declaring** class, so `Actor.user_arn` stays the same name
regardless of which event-model embeds `Actor`.

`list[BaseModel]` works the same way: each element is recursed.

## Rule 5 — plain `Field()` (no `node()` metadata)

→ value goes into `EventNode.properties`. Visible to the LLM via
`timeline(...)`. Used for things like `event_time`, `latency_ms`, raw
status codes — values that aren't worth turning into nodes.

## Unsupported shapes (raise `AgentPinBoardExtractionError`)

- `dict[str, BaseModel]` or any dict container in a node-marked field
- `Union[NodeA, NodeB]` (Union of distinct node types)
- `tuple` containers
- Lists with mixed-type elements
- `node(...)` on a `BaseModel`-typed field (raises at registration
  time, before runtime)

These limits are deliberate — they keep the extractor predictable and
the graph schema understandable. Most can be worked around by
flattening the model.

## `edge_type` derivation

```
edge_type = "{ModelClass}.{field_name}"
```

`ModelClass` is the class that **physically declares** the field —
found by walking the MRO in order. This means inheritance and reuse
keep edge labels stable:

```python
class Actor(BaseModel):
    user_arn: str | None = node(type=User, description="who", default=None)

class CloudTrailEvent(BaseModel):
    actor: Actor | None = None

class S3AccessLog(BaseModel):
    actor: Actor | None = None        # same Actor class

# Both produce edges of type "Actor.user_arn", regardless of the outer model.
```

Inheritance resolves the same way: a subclass that does *not*
re-declare the field still produces the base-class label.

## Recursion guard

The eager scan and the runtime extractor both protect against
recursive models like `Process(parent: Process | None)`. The eager
scan tracks visited model classes; the runtime scan uses Python's
own object graph, which is finite by construction.

## What `EventNode.properties` contains

`event_properties(model)` collects every field that:

- has no `node()` metadata,
- has a non-`None` value,
- is **not** a `BaseModel`, `list`, `dict`, or `tuple`.

So `event_time: datetime`, `action_name: str`, `latency_ms: int` all
land in properties. Nested objects do NOT — they're recursed into for
node extraction (Rule 4) but their scalar non-node fields end up in the
*event's* properties (because the event is the only node in the star
topology that carries properties).
