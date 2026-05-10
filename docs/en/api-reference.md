# API reference

Every public symbol re-exported from `agent_pinboard`.

## Markers and factories

### `Entity`

```python
@dataclass(slots=True, frozen=True)
class Entity:
    name: str
    description: str
    normalizer: Callable[[Any], str] | None = None
```

A frozen value object describing one node type. `name` becomes
`FactNode.node_type`, `description` is shown by `graph_summary`, and
`normalizer` canonicalises raw values for autolinking.

`__post_init__` rejects empty `name` or `description` with `ValueError`.

### `node(*, type, description, **field_kwargs) -> FieldInfo`

Factory that returns a Pydantic `FieldInfo` carrying the AgentPinBoard
node-marker metadata.

- `type: Entity` — the entity this field's values represent. String
  or other types raise `AgentPinBoardConfigError`.
- `description: str` — non-empty; describes the field's role in the
  event. Raises `AgentPinBoardConfigError` if empty.
- `**field_kwargs` — forwarded to `pydantic.Field` (e.g. `default`,
  `default_factory`, `alias`, `ge`).

The annotation of the field must be a primitive (or `list[primitive]`).
A `BaseModel`-typed field with `node()` raises `AgentPinBoardConfigError`
at `register_model` time.

## Decorator

### `pin(*, model, many=False, on_duplicate=ALWAYS, mask_args=None, response_transform=None, store_raw=False)`

Decorator factory for LangChain tools. **Apply above `@tool`.**

- `model: type[BaseModel]` — Pydantic model the tool's return is
  validated against.
- `many: bool` — set `True` for tools returning lists; each element is
  validated and ingested separately.
- `on_duplicate: OnDuplicate` — see enum below.
- `mask_args: list[str] | None` — parameter names to replace with
  `"***"` in `args_repr`.
- `response_transform: (raw, IngestResult) -> Any` — rewrite the
  tool's return for the LLM. Default keeps the original value.
- `store_raw: bool` — if `True`, also stash the tool's raw return
  under `("agent_pinboard", thread_id, "raw_events", event_id)` so
  `get_evidence` can replay it.

Observability is wired via standard LangChain callbacks — pass
handlers through `config={"callbacks": [...]}` on `agent.invoke` /
`ainvoke`. Each successful ingest dispatches an
`agent_pinboard:ingest` custom event; see
[hooks-and-config](./hooks-and-config.md).

## Enums

### `Direction(StrEnum)`

```python
class Direction(StrEnum):
    OUT = "out"
    IN = "in"
    BOTH = "both"
```

Used by `explore` (and, in Phase 2, `find_path`) to control traversal
direction. Only meaningful when `skip_events=False`.

### `OnDuplicate(StrEnum)`

```python
class OnDuplicate(StrEnum):
    ALWAYS = "always"   # default — execute on every call
    SKIP   = "skip"     # second identical call returns "duplicate call skipped"
    CACHE  = "cache"    # second identical call returns the prior raw return marker
```

## Models (runtime data classes)

All `@dataclass(slots=True)`; `FactEdge` and `ToolCallRecord` are
also `frozen=True`.

### `FactNode`

```python
class FactNode:
    id: NodeId                  # sha256("{type}|{canonical}")[:16]
    node_type: str              # entity.name
    value: str                  # display value
    canonical_value: str        # autolink key
    properties: dict[str, Any]
    first_seen: datetime
    last_seen: datetime
    source_events: list[EventId]
    source_tools: set[str]
```

### `EventNode`

```python
class EventNode:
    id: EventId                 # UUID4
    source_tool: str            # decorated tool name
    timestamp: datetime         # moment of invocation
    properties: dict[str, Any]  # non-node scalars from the model
    node_type: str = "Event"    # reserved type name
```

### `FactEdge`

```python
class FactEdge:               # frozen
    event_id: EventId
    target_id: NodeId
    edge_type: str            # "{ModelClass}.{field_name}"
    description: str          # from node(description=...)

    @property
    def id(self) -> str: ...  # f"{event_id}|{edge_type}|{target_id}"
```

### `IngestResult`

```python
class IngestResult:
    event_ids: list[EventId]
    new_nodes: int             # FactNodes created
    linked_nodes: int          # distinct existing FactNodes re-linked
    new_edges: int
    warnings: list[str]
```

### `ToolCallRecord`

```python
class ToolCallRecord:         # frozen
    tool_name: str
    args_repr: str             # canonical JSON
    timestamp: datetime
    event_id: EventId | None   # None for skipped duplicates / errors
    summary: str
    duration_ms: int
```

### Type aliases

```python
type NodeId = str
type EventId = str
```

## Graph

### `FactGraph`

```python
class FactGraph:
    g: nx.MultiDiGraph
    nodes_by_key: dict[(str, str), NodeId]
    nodes_by_type: dict[str, set[NodeId]]

    def add_event(event: EventNode) -> None
    def upsert_fact(entity: Entity, value, event_id, source_tool, *, warnings=None)
        -> tuple[NodeId | None, was_new: bool]
    def add_edge(edge: FactEdge) -> None

    def get(node_id) -> FactNode | EventNode | None
    def search_by_type(node_type) -> list[NodeId]
    def find_by_value(node_type, value) -> NodeId | None
    def edges_for_event(event_id) -> list[FactEdge]
    def all_events() -> list[EventNode]
    def all_facts() -> Iterable[FactNode]

    @classmethod
    def from_snapshot(nodes, edges) -> FactGraph
```

In normal use you don't construct this — `@pin` and the graph tools
manage it. It's exported so tests and advanced users can poke at the
in-memory representation.

## Read-tools factory

### `make_graph_tools() -> list[BaseTool]`

Returns the five Phase-1 graph tools: `explore`, `timeline`,
`graph_summary`, `search_nodes`, `what_have_i_done`. See
[graph-tools](./graph-tools.md) for signatures and behaviour.

## Observability

Observability is provided through the standard LangChain callback
chain. After every successful `@pin` ingest the decorator dispatches
a custom event:

```python
from agent_pinboard.decorator import INGEST_EVENT  # "agent_pinboard:ingest"
```

Pass any `BaseCallbackHandler` subclass via
`config={"callbacks": [...]}` on `agent.invoke` / `ainvoke` — its
`on_custom_event(name, data, *, run_id, ...)` receives the event with
`name == INGEST_EVENT` and a `data` dict containing `thread_id`,
`tool_name`, `result: IngestResult`, `events`, `new_facts`,
`linked_facts`, `new_edges`, `graph`. See
[hooks-and-config](./hooks-and-config.md) for the payload schema and
the bundled `LangfuseHook` / `WebSocketHook` integrations.

## Configuration

### `configure(*, tool_log_soft_limit: int | None = None) -> None`

Process-global settings. Currently only `tool_log_soft_limit` (default
500). When the per-session log exceeds it, a warning is emitted; no
hard cap.

## Exceptions

```python
class AgentPinBoardError(Exception): ...
class AgentPinBoardConfigError(AgentPinBoardError): ...        # decoration / setup error
class AgentPinBoardValidationError(AgentPinBoardError): ...    # Pydantic validation failed
class AgentPinBoardNormalizerError(AgentPinBoardError): ...    # Entity.normalizer raised
class AgentPinBoardExtractionError(AgentPinBoardError): ...    # unsupported field shape
```

Catch the base `AgentPinBoardError` to handle any library-raised failure.

## Internals (not public API, but accessible)

These exist and are sometimes useful for tests or extensions. Do not
rely on their signatures across versions.

```python
from agent_pinboard.registry import known_entities, register_model, _reset
from agent_pinboard.session import (
    get_or_load_session, aget_or_load_session,
    lock_for, thread_id_from, _reset as reset_sessions,
)
from agent_pinboard import store as store_io        # sync + async I/O
from agent_pinboard.extract import extract, event_properties
from agent_pinboard.fields import field_entity, META_KEY
from agent_pinboard.config import _reset as reset_config
```
