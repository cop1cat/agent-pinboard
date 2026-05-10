# Справочник API

Каждый публичный символ, реэкспортированный из `agent_pinboard`.

## Маркеры и фабрики

### `Entity`

```python
@dataclass(slots=True, frozen=True)
class Entity:
    name: str
    description: str
    normalizer: Callable[[Any], str] | None = None
```

Frozen value-object, описывающий один тип ноды. `name` становится
`FactNode.node_type`, `description` показывается в `graph_summary`,
`normalizer` канонизирует сырые значения для автолинковки.

`__post_init__` реджектит пустой `name` или `description` через
`ValueError`.

### `node(*, type, description, **field_kwargs) -> FieldInfo`

Фабрика, возвращающая Pydantic `FieldInfo` с AgentPinBoard-меткой.

- `type: Entity` — сущность, которую представляет значение поля.
  Строка или другие типы → `AgentPinBoardConfigError`.
- `description: str` — непустой; описывает роль поля в событии.
  Пустой → `AgentPinBoardConfigError`.
- `**field_kwargs` — пробрасываются в `pydantic.Field` (например,
  `default`, `default_factory`, `alias`, `ge`).

Аннотация поля должна быть примитивом (или `list[primitive]`).
`BaseModel`-типизированное поле с `node()` кидает
`AgentPinBoardConfigError` на этапе `register_model`.

## Декоратор

### `pin(*, model, many=False, on_duplicate=ALWAYS, mask_args=None, response_transform=None, store_raw=False)`

Фабрика декоратора для LangChain-тулов. **Применяется выше `@tool`.**

- `model: type[BaseModel]` — модель, против которой валидируется
  return тула.
- `many: bool` — `True` для тулов, возвращающих списки; каждый
  элемент валидируется и ingest'ится отдельно.
- `on_duplicate: OnDuplicate` — см. enum ниже.
- `mask_args: list[str] | None` — имена параметров, заменяемых на
  `"***"` в `args_repr`.
- `response_transform: (raw, IngestResult) -> Any` — переписывает
  return тула для LLM. По дефолту оригинал.
- `store_raw: bool` — если `True`, raw return тула также кладётся под
  `("agent_pinboard", thread_id, "raw_events", event_id)` для
  `get_evidence`.

Observability — через стандартный LangChain callback-канал. Хендлеры
передаются через `config={"callbacks": [...]}` на `agent.invoke` /
`ainvoke`. После каждого успешного ингеста декоратор диспатчит
`agent_pinboard:ingest` custom-event; см.
[hooks-and-config](./hooks-and-config.md).

## Enums

### `Direction(StrEnum)`

```python
class Direction(StrEnum):
    OUT = "out"
    IN = "in"
    BOTH = "both"
```

Управляет направлением обхода в `explore` (и в Phase 2 — `find_path`).
Имеет смысл только при `skip_events=False`.

### `OnDuplicate(StrEnum)`

```python
class OnDuplicate(StrEnum):
    ALWAYS = "always"   # дефолт — исполнять каждый вызов
    SKIP   = "skip"     # повторный вызов с тем же args — "duplicate call skipped"
    CACHE  = "cache"    # повторный вызов — маркер с timestamp прошлого
```

## Модели (runtime data classes)

Все `@dataclass(slots=True)`; `FactEdge` и `ToolCallRecord` ещё и
`frozen=True`.

### `FactNode`

```python
class FactNode:
    id: NodeId                  # sha256("{type}|{canonical}")[:16]
    node_type: str              # entity.name
    value: str                  # отображаемое значение
    canonical_value: str        # ключ автолинковки
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
    source_tool: str            # имя декорированного тула
    timestamp: datetime         # момент вызова
    properties: dict[str, Any]  # не-node скаляры из модели
    node_type: str = "Event"    # зарезервированное имя типа
```

### `FactEdge`

```python
class FactEdge:               # frozen
    event_id: EventId
    target_id: NodeId
    edge_type: str            # "{ModelClass}.{field_name}"
    description: str          # из node(description=...)

    @property
    def id(self) -> str: ...  # f"{event_id}|{edge_type}|{target_id}"
```

### `IngestResult`

```python
class IngestResult:
    event_ids: list[EventId]
    new_nodes: int             # созданных FactNode
    linked_nodes: int          # уникальных существующих FactNode, перелинкованных
    new_edges: int
    warnings: list[str]
```

### `ToolCallRecord`

```python
class ToolCallRecord:         # frozen
    tool_name: str
    args_repr: str             # canonical JSON
    timestamp: datetime
    event_id: EventId | None   # None для пропущенных дубликатов / ошибок
    summary: str
    duration_ms: int
```

### Type aliases

```python
type NodeId = str
type EventId = str
```

## Граф

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

В обычном использовании не конструируется — `@pin` и graph-тулы
управляют им сами. Экспортирован, чтобы тесты и продвинутые
пользователи могли поковыряться с in-memory представлением.

## Фабрика read-тулов

### `make_graph_tools() -> list[BaseTool]`

Возвращает пять Phase-1 graph-тулов: `explore`, `timeline`,
`graph_summary`, `search_nodes`, `what_have_i_done`. Сигнатуры и
поведение — см. [graph-tools](./graph-tools.md).

## Observability

Observability обеспечивается через стандартный LangChain callback-канал.
После каждого успешного `@pin`-ингеста декоратор диспатчит custom-event:

```python
from agent_pinboard.decorator import INGEST_EVENT  # "agent_pinboard:ingest"
```

Любой `BaseCallbackHandler`-подкласс передаётся через
`config={"callbacks": [...]}` на `agent.invoke` / `ainvoke` — его
`on_custom_event(name, data, *, run_id, ...)` получает event с
`name == INGEST_EVENT` и `data` dict со следующими ключами:
`thread_id`, `tool_name`, `result: IngestResult`, `events`,
`new_facts`, `linked_facts`, `new_edges`, `graph`. Schema payload и
готовые интеграции (`LangfuseHook`, `WebSocketHook`) — см.
[hooks-and-config](./hooks-and-config.md).

## Конфигурация

### `configure(*, tool_log_soft_limit: int | None = None) -> None`

Process-global настройки. Сейчас только `tool_log_soft_limit`
(дефолт 500). При превышении per-session лога — warning, hard cap'a
нет.

## Исключения

```python
class AgentPinBoardError(Exception): ...
class AgentPinBoardConfigError(AgentPinBoardError): ...        # ошибка декорирования / setup'a
class AgentPinBoardValidationError(AgentPinBoardError): ...    # Pydantic-валидация провалена
class AgentPinBoardNormalizerError(AgentPinBoardError): ...    # Entity.normalizer кинул
class AgentPinBoardExtractionError(AgentPinBoardError): ...    # неподдерживаемая форма поля
```

Ловите базовый `AgentPinBoardError`, чтобы обработать любую ошибку
библиотеки.

## Internals (не публичный API, но доступны)

Существуют и иногда полезны для тестов или расширений. Не полагайтесь
на их сигнатуры между версиями.

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
