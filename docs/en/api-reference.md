# API reference

Every public symbol re-exported from `pinboard`.

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

Factory that returns a Pydantic `FieldInfo` carrying the PinBoard
node-marker metadata.

- `type: Entity` — the entity this field's values represent. String
  or other types raise `PinBoardConfigError`.
- `description: str` — non-empty; describes the field's role in the
  event. Raises `PinBoardConfigError` if empty.
- `**field_kwargs` — forwarded to `pydantic.Field` (e.g. `default`,
  `default_factory`, `alias`, `ge`).

The annotation of the field must be a primitive (or `list[primitive]`).
A `BaseModel`-typed field with `node()` raises `PinBoardConfigError`
at `register_model` time.

## Decorator

### `fact(*, model, many=False, on_duplicate=ALWAYS, mask_args=None, hooks=None, response_transform=None)`

Decorator factory for LangChain tools. **Apply above `@tool`.**

- `model: type[BaseModel]` — Pydantic model the tool's return is
  validated against.
- `many: bool` — set `True` for tools returning lists; each element is
  validated and ingested separately.
- `on_duplicate: OnDuplicate` — see enum below.
- `mask_args: list[str] | None` — parameter names to replace with
  `"***"` in `args_repr`.
- `hooks: PinBoardHooks | None` — observability sink.
- `response_transform: (raw, IngestResult) -> Any` — rewrite the
  tool's return for the LLM. Default keeps the original value.

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

In normal use you don't construct this — `@fact` and the graph tools
manage it. It's exported so tests and advanced users can poke at the
in-memory representation.

## Read-tools factory

### `make_graph_tools(hooks=None) -> list[BaseTool]`

Returns the five Phase-1 graph tools: `explore`, `timeline`,
`graph_summary`, `search_nodes`, `what_have_i_done`. See
[graph-tools](./graph-tools.md) for signatures and behaviour.

## Hooks

### `PinBoardHooks`

Base class with no-op methods — override what you need:

```python
class PinBoardHooks:
    def on_node_added(self, node: FactNode | EventNode) -> None: ...
    def on_edge_added(self, edge: FactEdge) -> None: ...
    def on_link_found(self, existing: FactNode, event_id: EventId) -> None: ...
    def on_ingest_complete(self, result: IngestResult) -> None: ...
    def on_graph_changed(self) -> None: ...
```

Every callback fires under `try/except`. Hook exceptions are logged
and swallowed; ingestion never fails because of a hook.

### `LoggingHook(level=logging.INFO)`

Logs each callback to the standard `logging` module.

### `CompositeHook(hooks: list[PinBoardHooks])`

Fans every callback out to the provided hooks in order.

## Configuration

### `configure(*, tool_log_soft_limit: int | None = None) -> None`

Process-global settings. Currently only `tool_log_soft_limit` (default
500). When the per-session log exceeds it, a warning is emitted; no
hard cap.

## Exceptions

```python
class PinBoardError(Exception): ...
class PinBoardConfigError(PinBoardError): ...        # decoration / setup error
class PinBoardValidationError(PinBoardError): ...    # Pydantic validation failed
class PinBoardNormalizerError(PinBoardError): ...    # Entity.normalizer raised
class PinBoardExtractionError(PinBoardError): ...    # unsupported field shape
```

Catch the base `PinBoardError` to handle any library-raised failure.

## Internals (not public API, but accessible)

These exist and are sometimes useful for tests or extensions. Do not
rely on their signatures across versions.

```python
from pinboard.registry import known_entities, register_model, _reset
from pinboard.session import (
    get_or_load_session, aget_or_load_session,
    lock_for, thread_id_from, _reset as reset_sessions,
)
from pinboard import store as store_io        # sync + async I/O
from pinboard.extract import extract, event_properties
from pinboard.fields import field_entity, META_KEY
from pinboard.config import _reset as reset_config
```
