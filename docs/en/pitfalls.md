# Common pitfalls

Things that trip up first-time users. Bookmark this — most of these are
hour-long debug sessions if you don't know the answer in advance.

## 1. Decorator order

**`@pin` always above `@tool`.**

```python
# ✓ correct
@pin(model=X)
@tool
def f(...): ...

# ✗ raises AgentPinBoardConfigError at decoration time
@tool
@pin(model=X)
def f(...): ...
```

`@tool` produces a `BaseTool`; `@pin` then wraps it. Reverse the order
and `@pin` receives a function instead of a Tool — the spec rejects
this immediately so you don't get a confusing traceback later.

## 2. `node()` vs `Field()`

A field is a node if and only if you used `node(...)` to declare it.
Plain `Field(...)` (or no annotation at all) means "this scalar goes
into `EventNode.properties`".

Quick rule:

> **Recurs across events?** → `node(...)`.
> **One-shot scalar (timestamp, latency, raw status)?** → `Field(...)`.

## 3. Two `Entity` instances with the same name

AgentPinBoard's session registry is keyed by `Entity.name`. If two pieces
of code create `Entity(name="IP", ...)` independently — even with
identical attributes — you'll get a warning and the first registration
wins.

**Right way**: declare each `Entity` once in a project module
(commonly `entities.py`) and import it everywhere. Don't copy-paste
`Entity` declarations.

## 4. Normalizer identity (the "two `canonical_ip` files" trap)

`Entity` equality includes the `normalizer` callable, compared by
identity. If you import `canonical_ip` from one module in
`models_a.py` and copy-paste the same code into `models_b.py`, the
two functions are different Python objects → the two `Entity` objects
with the same `name` are not equal → registry warns about a collision.

**Right way**: import normalizers from a single shared module.

## 5. `node()` on a `BaseModel`-typed field

```python
class Inner(BaseModel):
    x: str

class Bad(BaseModel):
    inner: Inner = node(type=SomeEntity, description="...")  # rejected
```

This is rejected at `register_model` time with `AgentPinBoardConfigError`.
A `BaseModel` is a structured value, not a leaf — turning it into a
node would be semantically wrong (and would silently fall through to
Rule 4 in older versions). If you want `Inner.x` to be a node, mark
**that** field with `node()` and let Rule 4 recurse into `Inner`.

## 6. `skip_events` semantics in `explore`

`skip_events=True` (default) means **events are transparent connectors,
not hops**. `explore("IP", "1.2.3.4", depth=1, skip_events=True)`
returns every fact that shares **any event** with that IP — visually
zero hops, semantically "directly related".

`skip_events=False` walks the underlying `MultiDiGraph`, where each
EventNode consumes a hop. A FactNode → its EventNode → another FactNode
is two hops in this mode.

If you mean "what tool calls touched this IP", use `timeline(...)` or
`what_have_i_done(node_type="IP", value=...)` — those expose the events
directly, no graph traversal needed.

## 7. Graph not compiled with a Store

```python
graph = builder.compile()  # ✗ no store=
```

Every `@pin` invocation needs `runtime.store` to be set, which only
happens when you call `.compile(store=...)`. Without it the first tool
invocation raises `AgentPinBoardConfigError("graph must be compiled with .compile(store=...)")`.

```python
from langgraph.store.memory import InMemoryStore
graph = builder.compile(store=InMemoryStore())  # ✓
```

## 8. Missing `thread_id`

If `runtime.config.configurable.thread_id` is unset, AgentPinBoard generates
a fresh UUID4 per call and warns. Two parallel "anonymous" calls each
get their own session — they will not see each other's data.

This is intentional (silent-merge would be much worse), but the warning
is easy to miss in Jupyter logs. Always pass `thread_id` explicitly:

```python
graph.invoke(
    {...},
    config={"configurable": {"thread_id": "investigation-001"}},
)
```

## 9. Returning a Pydantic instance vs a dict from a tool

Both work:

```python
@pin(model=CloudTrailEvent, many=True)
@tool
def fetch(...) -> list[dict]:
    return [{"src_ip": "1.1.1.1"}, ...]    # validated via model_validate

@pin(model=CloudTrailEvent, many=True)
@tool
def fetch(...) -> list[CloudTrailEvent]:
    return [CloudTrailEvent(src_ip="1.1.1.1"), ...]  # validation skipped
```

But the type annotation in your `@tool` signature affects what
LangChain/LLM see — keep it accurate.

## 10. `fnmatchcase` is case-sensitive

`search_nodes(node_type="IP", value_pattern="abc*")` will NOT match a
node whose canonical value is `"ABC123"`. We use `fnmatch.fnmatchcase`
for cross-OS deterministic behaviour.

If you want case-insensitive search, ensure your normalizer
lower-cases the value, then compare in the canonical case:

```python
def canonical_email(v: str) -> str:
    return v.strip().lower()
```

## 11. `mask_args` and rotating secrets

Two calls with different real secrets but same `mask_args` produce the
same `args_repr` and dedup-collide. If your secret rotates within a
session, either:

- Inject the secret via `runtime.config` (out of `args_repr` entirely), or
- Use `on_duplicate=OnDuplicate.ALWAYS` so the dedup never fires.

## 12. `response_transform` async/sync mismatch

A sync tool with an async `response_transform` is rejected at
decoration time. The result is needed synchronously and there's no
event loop available to await the transform. Match the modes — sync
tool ↔ sync transform, async tool ↔ async-or-sync transform.

## 13. EventNode visibility in `search_nodes`

By default `search_nodes` hides EventNodes. If your agent is trying to
find tool-call records by glob, pass `include_events=True`:

```python
search_nodes(node_type="Event", value_pattern="fetch_*", include_events=True)
```

For non-glob queries (just "what tool calls happened"), prefer
`what_have_i_done(...)` — it filters the structured tool-call log and
returns more useful info.
