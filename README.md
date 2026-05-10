# PinBoard — Техническое задание

## 1. Обзор

### 1.1 Суть

PinBoard — Python-библиотека, реализующая рабочую память LLM-агента в виде графа фактов на время одной сессии. Метафора — детективная доска: агент вызывает тулы, полученные данные автоматически превращаются в ноды и рёбра графа, агент читает граф и принимает решения на его основе.

PinBoard — это примитивы: модель графа, декоратор для извлечения фактов из результатов тулов, готовые тулы чтения графа, система хуков. Сам агент (LangGraph StateGraph, промпты, выбор модели) — пользовательский код поверх этих примитивов.

Типичные сценарии: расследование security-инцидента, due diligence по компании, разбор биографии/кейса, threat intelligence, связывание сущностей между разными API. Везде, где агент многоразово дёргает тулы, накапливает сущности и должен удерживать связи между ними — лучше чем в линейной истории сообщений.

### 1.2 Ключевые принципы

- **Агент не работает с сырыми JSON-ами.** Из ответов тулов извлекаются факты-сущности, агент оперирует только графом фактов.
- **Факт — конкретное значение поля.** IP-адрес, email, ИНН, хэш файла, идентификатор заказа — это ноды графа.
- **Граф живёт в памяти процесса агента.** Одна сессия — один граф. По окончании сессии граф либо выбрасывается, либо дампится целиком как артефакт.
- **LLM подключается только там, где нужен интеллект.** Извлечение фактов и автолинковка — детерминированные операции. LLM принимает решения, интерпретирует и формирует отчёты.
- **PinBoard — невидимый side-effect.** Декоратор `@fact` не изменяет return тула. Граф строится в фоне. Пользователь может опционально настроить поведение через хуки и `response_transform`.
- **Пользователь знает свои источники.** Пользователь пишет Pydantic-модели результатов тулов и размечает нодовые поля через фабрику `node(...)` — никаких автогенераций.
- **Один граф = semantic + episodic подграфы.** Semantic — извлечённые сущности и связи между ними. Episodic — сами события (вызовы тулов) и их привязки к фактам. Один `FactGraph`, два логических слоя (терминология из AriGraph, IJCAI 2025).

### 1.3 Область применения

Одна сессия работы агента — минуты до часов. Агент запрашивает источники, обогащает контекст, ищет взаимосвязи между сущностями и отвечает на исходный вопрос. Примеры в этом документе используют security-домен (CloudTrail, VirusTotal) как наиболее наглядный, но библиотека полностью доменно-нейтральная.

---

## 2. Архитектура

```text
┌─────────────────────────────────────────────────────────────┐
│       Пользовательский код: агент на LangGraph              │
│       (промпты, StateGraph, выбор модели — вне PinBoard)    │
└──────────────────────┬──────────────────────────────────────┘
                       │ использует
┌──────────────────────▼──────────────────────────────────────┐
│                  PinBoard (библиотека)                      │
│                                                             │
│   @fact декоратор → извлечение фактов из return тула        │
│                     (поля, размеченные node(...))           │
│                                                             │
│   FactGraph (NetworkX) с автолинковкой по (type, value)     │
│                                                             │
│   Готовые graph-тулы (Фаза 1): explore, timeline,           │
│   graph_summary, search_nodes, what_have_i_done.            │
│   find_path (Фаза 2), get_evidence (Фаза 3).                │
│                                                             │
│   Хуки: on_node_added, on_edge_added, on_ingest_complete,   │
│   on_graph_changed, ...                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ хранится в
┌──────────────────────▼──────────────────────────────────────┐
│   LangGraph Store (InMemoryStore) — runtime-хранилище       │
│   графа на время сессии, доступен из тулов через ToolRuntime│
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Разметка моделей

Пользователь описывает результат каждого своего тула как Pydantic-модель. Типы нод графа объявляются как `Entity`-инстансы. Поля, значения которых становятся нодами графа, размечаются фабрикой `node(...)` со ссылкой на `Entity`.

### Глоссарий — четыре близких термина, не путать

| Термин | Что это | Кто создаёт |
|---|---|---|
| `Entity` | Описание типа ноды (имя, описание, нормализатор) | Пользователь (один раз на тип) |
| `node(...)` | Маркер «сделай это поле нодой графа» на Pydantic-поле | Пользователь (при написании модели) |
| `FactNode` | Извлечённая нода в runtime-графе | Библиотека (автоматически) |
| `EventNode` | Технический узел вызова тула | Библиотека (автоматически) |

`Entity` и `node()` — write-time концепции, пользовательские. `FactNode` и `EventNode` — runtime, наблюдаются только через graph-тулы.

**Когда `node(...)` vs обычный `Field(...)`:** значение может встречаться в разных событиях и полезно видеть связи → `node(...)` (станет FactNode, автолинкуется). Значение уникальное для события и линковать бессмысленно (timestamp, raw message, latency) → обычный `Field(...)` (попадёт в `EventNode.properties`).

### 3.1 `Entity` — тип ноды

`Entity` — frozen value-object, описывающий один тип сущности (IP, User, Domain, ИНН, Company, OrderID — что угодно в домене пользователя).

```python
from pinboard import Entity

IP = Entity(
    name="IP",
    description="IPv4 or IPv6 network address",
    normalizer=canonical_ip,
)

User = Entity(
    name="User",
    description="Identified person or service account (ARN, email, or user ID)",
)

Resource = Entity(
    name="Resource",
    description="Cloud resource identified by ARN",
)
```

Сигнатура:

```python
@dataclass(frozen=True)
class Entity:
    name: str                                     # имя типа в графе (FactNode.node_type)
    description: str                              # непустая строка, что это за тип
    normalizer: Callable[[Any], str] | None = None
```

`Entity` — чистый value-object, без side-effect при создании. Нет глобального регистра. Где хранить инстансы — на усмотрение пользователя (обычно — отдельный модуль `entities.py` по паттерну `constants.py`). Пользователь определяет `IP` один раз и импортирует там, где нужен.

**Session-registry, заполняется eager.** При применении `@fact(model=X)` к функции модель сразу сканируется, все встреченные `Entity` регистрируются в process-level «declared entities». В начале сессии (первый invoke агента) declared entities копируются в session-registry под `("pinboard", thread_id, "entities")`. Это означает: LLM при первом же `graph_summary()` (до любого ingestion) видит полный список типов, с которыми знакомы подключённые тулы, — это карта местности до первого tool-call.

**Защита от рекурсивных моделей.** Pydantic разрешает рекурсию: `Process.parent: Process | None`. Обход `model_fields` при eager-scan защищается set-ом уже посещённых типов (`seen: set[type[BaseModel]]`), чтобы не зациклиться. Runtime-экстракция (при `@fact`-вызове) рекурсию отрабатывает по реальным инстансам, где граф данных конечен по построению.

**Конфликты имён.** Session-registry индексируется по `Entity.name`. При встрече двух `Entity` с одинаковым `name` — warning в лог с указанием обоих определений, побеждает первый встреченный. Библиотека не полицейский: пользователь сам отвечает за то, чтобы один тип определялся один раз.

### 3.2 `node(...)` — разметка поля

```python
from pydantic import BaseModel, Field
from datetime import datetime
from pinboard import node

from .entities import IP, User, Resource

class CloudTrailEvent(BaseModel):
    src_ip: str | None = node(
        type=IP,
        description="IP from which the API call was made",
    )
    actor_arn: str | None = node(
        type=User,
        description="Who performed the action",
    )
    target_resource: str | None = node(
        type=Resource,
        description="AWS resource being accessed",
    )
    action_name: str = Field(description="API action name")
    event_time: datetime = Field(description="When the event occurred")
```

Сигнатура `node`:

```python
def node(
    *,
    type: Entity,
    description: str,                  # обязательный, не пустой
    default: Any = None,
    **field_kwargs,
) -> Any:
    ...
```

- `type` — инстанс `Entity`. Типобезопасно, без строковых опечаток, IDE подсказывает, refactor-rename работает.
- `description` — **обязательный**. Описывает *связь* (как это поле относится к событию), а не тип. Попадает в свойства ребра и **всегда** рендерится в выводе `explore`/`timeline` — LLM не должна догадываться о смысле поля по его имени.
- `default`, `**field_kwargs` — пробрасываются в Pydantic `Field(...)`.

`node(...)` проверяет при применении:

- `description` непустой — иначе `PinBoardConfigError`.
- `type` — инстанс `Entity`, не строка — иначе `PinBoardConfigError` с подсказкой «pass an Entity instance, not a string».
- тип поля — примитив или `list[primitive]`; `BaseModel`-поля с `node()` — запрещены, проверяется при `register_model` (`PinBoardConfigError`).

Возвращает обычный `FieldInfo` с PinBoard-метаданными в `json_schema_extra`. Pydantic работает с моделью как с обычной.

**Required vs optional.** Поле без `default` — required (Pydantic упадёт при отсутствии в сырых данных). Поле с `default=None` и типом `T | None` — optional, `None` пропускается правилом 2 экстрактора:

```python
class CloudTrailEvent(BaseModel):
    actor: str = node(type=User, description="...")                    # required, упадёт без значения
    src_ip: str | None = node(type=IP, description="...", default=None) # optional, None пропускается
```

Выбор между ними — решение пользователя в зависимости от контракта API-источника.

### 3.3 Два уровня описаний

| Уровень | Где живёт | Что описывает | Где видно |
|---|---|---|---|
| `Entity.description` | На `Entity`-инстансе | Что это за тип в принципе («IPv4/IPv6 address») | `graph_summary`, списки типов |
| `node(description=...)` | На поле модели | Как это конкретное поле связано с событием («IP from which the API call was made») | Рёбра графа, `explore`, `timeline` |

Оба обязательны, оба осмысленны для LLM.

---

## 4. Правила экстракции

### 4.1 Пять правил обхода Pydantic-модели

Декоратор рекурсивно обходит `model_fields` инстанса модели и применяет пять правил:

1. **Поле-примитив с `node(...)` меткой и значением не `None`** → создаётся `FactNode(type=meta.type.name, value=field_value)`, ребро `Event --[field_name]--> FactNode`.
2. **Поле со значением `None`** → пропускается.
3. **Поле `list[primitive]` с `node(...)` меткой** → для каждого элемента списка применяется правило 1. Имя поля используется как тип ребра для всех элементов.
4. **Поле — вложенная `BaseModel` (без `node(...)`) или `list[BaseModel]` без `node(...)`** → рекурсивный обход по этим же правилам. Рёбра продолжают идти от той же event-ноды.
5. **Поле — обычный `Field(...)` без `node(...)` метки** → значение попадает в `properties` event-ноды, нодой не становится.

**Явно не поддерживается в MVP (понятное исключение при попытке):** `dict[str, BaseModel]`, `Union[NodeA, NodeB]`, `tuple`, модели с произвольными `**extra` полями, списки с разнотипными элементами.

### 4.2 Имя поля как тип ребра

Внутренний `edge_type = "{ModelClass}.{field_name}"`, где `ModelClass` — **класс, в котором физически объявлено поле**. Если `Actor.user_arn` определено в классе `Actor`, а `Actor` используется как вложенная модель в `CloudTrailEvent` и в `S3AccessLog`, то edge_type всегда `Actor.user_arn` — переиспользуемость вложенных моделей сохраняется.

Резолв — через обход `__mro__` и `__annotations__` (самый базовый класс в MRO, у которого поле объявлено в собственных аннотациях). Для унаследованных полей без переопределения это родитель, для переопределённых — подкласс, который переопределяет.

Полная квалификация гарантирует уникальность между моделями: `CloudTrailEvent.src_ip` и `DNSLog.src_ip` — разные типы рёбер.

В пользовательском отображении (`explore`, `timeline`) ребро рендерится как короткое имя поля + `description` из `node(description=...)`, без verbose-префиксов. Полный qualified name доступен в `FactEdge.edge_type` для программной фильтрации.

Пользователь контролирует именование через имена полей: нужно ребро `source` вместо `src_ip` — переименовывает поле в модели.

### 4.3 Event-нода

На каждый вызов тула, обёрнутого `@fact`, создаётся ровно одна `EventNode`. При `many=True` — по одной на каждый элемент возвращённого списка.

```python
@dataclass(slots=True)
class EventNode:
    id: EventId                   # UUID4
    node_type: str                # всегда "Event"
    source_tool: str              # имя декорированного тула
    timestamp: datetime           # момент вызова
    properties: dict[str, Any]    # non-node поля исходной модели
    # args_repr не хранится в EventNode: он нужен только для dedup и tool-log,
    # его место — ToolCallRecord (см. 9.3).
```

`node_type="Event"` — зарезервированное имя, коллизия с пользовательскими типами проверяется при создании `Entity(name="Event", ...)` и кидает `ValueError`. EventNode индексируется в тех же структурах, что и FactNode (`nodes_by_id`, `nodes_by_type["Event"]`), но **не** в `nodes_by_key` (canonical_value у EventNode отсутствует).

Все извлечённые FactNode подключаются к EventNode рёбрами типа `{ModelClass}.{field_name}`. Прямых рёбер между FactNode нет — топология всегда `star around EventNode`. Семантические связи между фактами вычисляются запросом «какие ноды делят одну EventNode» через `explore` / `find_path` с `skip_events=True` (дефолт).

**EventNode создаётся всегда** при успешной валидации return, даже если все нодовые поля оказались `None` и ни один FactNode не извлёкся. Это нужно для `what_have_i_done` («вызов был, ничего полезного не пришло»).

**`EventNode.timestamp` = момент вызова тула.** Это не обязательно совпадает с «реальным временем события» (CloudTrail-запись может быть часовой давности). В MVP `timeline` сортирует по `EventNode.timestamp`. Если пользовательский сценарий чувствителен к реальному времени событий — это ограничение, которое снимается явной меткой timestamp в Фазе 2.

---

## 5. Граф фактов

### 5.1 Модели

```python
type NodeId = str
type EventId = str
type EdgeId = str

@dataclass(slots=True)
class FactNode:
    id: NodeId                        # sha256("{type}|{canonical}")[:16]
    node_type: str                    # entity.name
    value: str                        # отображаемое значение
    canonical_value: str              # ключ автолинковки (из entity.normalizer)
    properties: dict[str, Any]
    first_seen: datetime
    last_seen: datetime
    source_events: list[EventId]      # где встречалась
    source_tools: set[str]            # какие тулы её поднимали

@dataclass(slots=True, frozen=True)
class FactEdge:
    event_id: EventId                 # источник ребра (он же EventNode)
    target_id: NodeId                 # FactNode
    edge_type: str                    # "{ModelClass}.{field_name}"
    description: str                  # из node(description=...)
    # source_id / source_tool / created_at резолвятся через nodes[event_id] (денормализация убрана)
    # id производный: f"{event_id}|{edge_type}|{target_id}"

@dataclass(slots=True)
class IngestResult:
    event_ids: list[EventId]          # id созданных EventNode (несколько при many=True)
    new_nodes: int                    # сколько новых FactNode добавлено
    linked_nodes: int                 # сколько фактов слинковано с существующими
    new_edges: int
    warnings: list[str]               # non-fatal предупреждения
```

### 5.2 Автолинковка

Уникальный ключ факт-ноды — `(node_type, canonical_value)`. При добавлении:

1. Вычисляется `canonical_value = normalizer(raw_value)` (или `str(raw_value)` если normalizer не задан).
2. Lookup в `nodes_by_key: dict[(type, canonical), node_id]`.
3. Если нода существует — обновляется `last_seen`, в `source_events` добавляется id текущей EventNode, в `source_tools` — имя тула. Нода не дублируется.
4. Если не существует — создаётся новая.
5. Ребро от EventNode к факт-ноде создаётся всегда (уникальный ключ ребра включает `event_id`).

Автолинковка работает детерминированно, без LLM.

### 5.3 Runtime-структура `FactGraph`

Один граф + два sidecar-индекса:

```python
class FactGraph:
    g: nx.MultiDiGraph                              # source of truth, node/edge attrs
    nodes_by_key: dict[tuple[str, str], NodeId]     # O(1) автолинковка факт-нод
    nodes_by_type: dict[str, set[NodeId]]           # быстрые срезы по типу
```

- Ноды и рёбра — атрибуты в `g.nodes[id]` / `g.edges[(src, tgt, key)]`.
- Lookup по `id` — через `g.nodes[id]`. По `(type, canonical)` — через sidecar (для автолинковки). По type — через sidecar (для `search_nodes`, `graph_summary`).
- `timeline` и фильтрация по `event_id` — один проход по `g.edges(data=True)` с фильтром по `data["event_id"]`. O(E), приемлемо на session-scope.

Sidecar-индексы derived, строятся при загрузке из Store.

---

## 6. Декоратор `@fact`

### 6.1 Принцип

`@fact` оборачивает `@tool`. После вызова тула (sync или async):

1. Проверка дубликата по canonical args_repr (см. 6.3). Если `on_duplicate="skip"/"cache"` совпадает — ветка ниже.
2. Оригинальный return валидируется через заданную Pydantic-модель (если `many=True` — каждый элемент списка).
3. Создаётся EventNode, применяются правила экстракции 4.1, собираются FactNode/FactEdge **в локальной структуре** (без касания Store).
4. **[под lock'ом]** `FactGraph` загружается из кеша/Store, новые ноды и рёбра мержатся с автолинковкой, изменения персистятся обратно в Store (см. 9.1).
5. В лог действий (см. 9.2) пишется запись.
6. Вызываются хуки.
7. Если задан `response_transform` — его результат возвращается LLM; иначе возвращается оригинальный return без изменений.

Под lock'ом выполняется **только шаг 4** (read-modify-write графа). Тул-функция, валидация, экстракция и хуки — вне lock'а, параллелизм LangGraph сохраняется.

Один декоратор работает и с sync, и с async тулами (детекция через `asyncio.iscoroutinefunction`).

### 6.2 Использование

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pinboard import fact

@fact(model=CloudTrailEvent, many=True)
@tool
def fetch_cloudtrail(user_arn: str, hours: int, runtime: ToolRuntime) -> list[dict]:
    """Fetch CloudTrail logs for a user"""
    return boto3_client.lookup_events(UserArn=user_arn, Hours=hours)

@fact(model=VTReport)
@tool
async def vt_lookup(value: str, runtime: ToolRuntime) -> dict:
    """Check IP/domain/hash in VirusTotal"""
    return await vt_client.get_report(value)
```

**Порядок декораторов.** `@fact` всегда выше `@tool`. Python применяет декораторы снизу вверх: `@tool` создаёт `BaseTool` из функции, затем `@fact` оборачивает этот `BaseTool` — подменяет `invoke`/`ainvoke` на свою логику, сохраняя интерфейс LangChain-тула. Обратный порядок невалиден: `@tool` не ожидает видеть уже обёрнутую функцию.

### 6.3 Детекция повторных вызовов и args_repr

**Canonical args_repr.** Для стабильной дедупликации аргументы сериализуются в JSON с сортировкой ключей, а не через `repr()`. Pydantic-инстансы приводятся через `.model_dump(mode="json")`, остальное — `json.dumps(..., sort_keys=True, default=str)`. `ToolRuntime` исключается до сериализации. Это устраняет нестабильность `repr()` для datetime, Decimal, Pydantic-моделей и разницу `1 == 1.0`.

**Masking секретов.** Если тул принимает чувствительный аргумент (API-токен, пароль), он попадёт в args_repr, EventNode, tool-log и любые хуки-экспортеры (Langfuse). Для маскировки:

```python
@fact(model=VTReport, mask_args=["api_key", "token"])
@tool
def vt_lookup(value: str, api_key: str, runtime: ToolRuntime) -> dict: ...
```

Аргументы из `mask_args` в canonical-repr заменяются на `"***"` перед сериализацией.

**Known limitation.** Два вызова с разными ротируемыми секретами, но одинаковыми остальными аргументами будут считаться дубликатами (обе замаскированы одним `"***"`). Для ротации секретов внутри сессии — либо передавать токен не через kwargs (session context, config), либо использовать `on_duplicate="always"` (дедуп формально не сработает).

**Поведение `on_duplicate` (`OnDuplicate` StrEnum):**

- `OnDuplicate.ALWAYS` (по умолчанию) — исполняется заново, ingestion отрабатывает, автолинковка гарантирует отсутствие дубликатов нод. Хуки вызываются.
- `OnDuplicate.SKIP` — тул не исполняется, возвращается строка `"duplicate call skipped"`. Ingestion не происходит, хуки **не** вызываются, `IngestResult` не создаётся, запись в tool-log делается с `event_id=None` и `summary="duplicate (skipped)"`.
- `OnDuplicate.CACHE` — возвращается **сырой return** из предыдущего вызова (до `response_transform`). Ingestion не повторяется, хуки **не** вызываются. Запись в tool-log с пометкой `"duplicate (cached)"`.

### 6.4 Опциональные коллбэки

```python
@fact(
    model=CloudTrailEvent,
    many=True,
    hooks=hooks,
    response_transform=lambda raw, result: f"Loaded {len(raw)} events, {result.new_nodes} new entities",
)
@tool
def fetch_cloudtrail(...): ...
```

- `hooks` — `PinBoardHooks`-совместимый объект, слушает изменения графа. Сюда же пишется любая пользовательская логика-наблюдатель (см. `on_ingest_complete` в 11.1).
- `response_transform` — меняет то, что увидит LLM (по умолчанию — оригинальный return).

**Сигнатура `response_transform`:** `(raw: Any, result: IngestResult) -> Any`.

- `raw` — **оригинальный return тула целиком**. При `many=True` это весь `list[dict]`, не по одному элементу.
- `result` — **один** `IngestResult`, описывающий результат всего batch-а (его `event_ids` содержит id всех созданных EventNode).
- Возвращаемое значение — то, что получит LLM вместо оригинального return.

**Async.** Если тул async, а `response_transform` sync — вызывается напрямую в том же цикле. Если sync тул и async `response_transform` — PinBoard бросит `PinBoardConfigError` при регистрации (нельзя await-нуть из sync-контекста, не запуская loop). Проще: пишите коллбэки того же типа, что и тул.

**Хуки и `response_transform` — разные оси.** Хуки про наблюдение за графом. `response_transform` про управление контекстом LLM.

### 6.5 Ошибки в хуках и валидации

- **Ошибка валидации Pydantic** при `many=True` — `@fact` **кидает исключение, не маскируя его**. LangGraph получает error от тула, агент сам решает что делать. Граф не меняется, `response_transform` не вызывается, `IngestResult` не создаётся. Принцип: fail loud, не тихое деградирование.
- **Ошибка в хуке** — `try/except` вокруг каждого вызова, в `except` — `logger.error(...)`. Ingestion продолжается, граф не откатывается (в отличие от валидации — потому что граф уже в непротиворечивом состоянии).
- **Пустой canonical_value** — если `Entity.normalizer` вернул пустую строку или None, факт не извлекается (иначе все пустые значения всех типов слиплись бы в одну ноду), в лог пишется warning с указанием поля и тула. Остальные факты из того же события извлекаются нормально.
- **Exception в нормализаторе** — если `Entity.normalizer` кинул исключение (например, `canonical_ip("not-an-ip")` → `ValueError`), `@fact` пропускает исключение наверх, ingestion проваливается, граф не меняется. Нормализатор — пользовательский код, если он падает на данных, это сигнал, что либо нормализатор некорректный, либо данные кривые; fail-loud даёт явный диагноз вместо молчаливого warning.
- **`runtime.store is None`** — если граф собран без `compile(store=...)`, `@fact` кидает `PinBoardConfigError("graph must be compiled with .compile(store=...) to use @fact-decorated tools")` на первом же вызове. Защита от типичного забытого аргумента в пользовательской сборке агента.

### 6.6 Что возвращает декорированный тул

Return тула может быть любым из:

- `dict` или `list[dict]` — парсится через `model.model_validate(raw)`.
- `BaseModel` или `list[BaseModel]` — если тип совпадает с `model=`, парсинг пропускается.
- Любой другой тип — ошибка валидации.

### 6.7 Exception hierarchy

```python
class PinBoardError(Exception):
    """Базовый класс всех исключений библиотеки."""

class PinBoardConfigError(PinBoardError):
    """Ошибка конфигурации: невалидная Entity/node(), коллизия Event, неверный стек декораторов."""

class PinBoardValidationError(PinBoardError):
    """Pydantic-валидация return тула провалилась. Оборачивает pydantic.ValidationError."""

class PinBoardNormalizerError(PinBoardError):
    """Normalizer кинул исключение на входных данных. Оборачивает оригинальное исключение."""

class PinBoardExtractionError(PinBoardError):
    """Ошибка при обходе модели или записи в граф (не валидация и не normalizer)."""
```

Все fail-loud сценарии кидают подклассы `PinBoardError` — пользовательский агент может ловить одной точкой. `PinBoardConfigError` кидается при декорировании (ранний отказ), остальные — в runtime. Хуки ловят своё внутри (см. 11.1).

---

## 7. Нормализаторы значений

PinBoard **не поставляет** готовых нормализаторов. `Entity.normalizer` — это `Callable[[Any], str]`, пользователь передаёт любую функцию. Ядро остаётся доменно-нейтральным.

**Обоснование (расхождение с research).** REASEARCH §E.3 рекомендует шипить доменные normalizers (`canonical_ip`, `canonical_email`, `canonical_arn`) как differentiator библиотеки. Мы сознательно этого не делаем: нормализация — задача сервисного слоя пользователя, не библиотеки. Шипить `canonical_arn` = делать ядро AWS-specific, шипить `canonical_ip` = делать ядро network-specific. То же самое, от чего мы отказались с OCSF. Библиотека предоставляет *интерфейс* (поле `normalizer: Callable`), а конкретные реализации — у пользователя в 3 строках кода.

Нормализатор живёт на `Entity`, а не на поле. Это гарантирует, что все вхождения одного типа (независимо от того, в каком поле какой модели встретились) канонизируются одинаково и корректно автолинкуются.

Референс-примеры — три строки кода каждый, копипаст в проект:

```python
def canonical_ip(v: str) -> str:
    return ipaddress.ip_address(v).compressed

def canonical_email(v: str) -> str:
    return v.strip().lower()

def canonical_domain(v: str) -> str:
    return v.strip().rstrip(".").lower().encode("idna").decode("ascii")
```

**Паттерн использования.** Нормализаторы живут в одном общем модуле проекта и импортируются в файл определения `Entity`. Не копипастить одну и ту же функцию в разные модули — легче поддерживать и refactor консистентнее.

---

## 8. Общие словари между источниками

Если агент тянет данные из нескольких источников с пересекающейся семантикой, полезно привести их к общему словарю типов — тогда сущности автолинкуются между источниками. Примеры таких словарей: **OCSF** (event-ориентированный, authentication/DNS/network activity), **STIX 2.1 SDO/SCO** (entity-ориентированный: IPv4-Addr, Domain-Name, Threat-Actor, Malware), собственные внутренние онтологии компании.

PinBoard не поставляет готовых моделей ни по одному стандарту. Пользователь пишет свои Pydantic-модели с нужными `Entity` и согласует `Entity.name` между источниками. В документации даются референс-примеры, но это не часть ядра и не отдельные модули. Принцип тот же, что с нормализаторами: интерфейс — библиотека, конкретные реализации — пользователь.

---

## 9. Хранение

### 9.1 Schema в Store

Граф хранится **посегментно**, не одним blob. Это позволяет писать только дельту на каждой ingestion-операции и избегать O(всего графа) на каждое изменение.

```text
("pinboard", thread_id, "nodes", node_id)              → FactNode | EventNode (JSON)
("pinboard", thread_id, "edges", edge_id)              → FactEdge (JSON)
("pinboard", thread_id, "entities")                    → session-registry (один blob, редко меняется)
("pinboard", thread_id, "tool_calls", record_id)       → ToolCallRecord (JSON, один key на запись)
```

**Жизненный цикл в рамках сессии.**

1. При первом обращении из тула PinBoard загружает все ноды и рёбра через `store.search(("pinboard", thread_id))`, строит in-memory `FactGraph` с индексами (`nodes_by_key`, `nodes_by_type`, `edges_by_event`) и кладёт в process-level кеш по `thread_id`.
2. На каждую ingestion-операцию — из кеша получается текущий граф, добавляются дельты (обычно 1 EventNode + N FactNode + N FactEdge), в Store записываются **только затронутые keys**. Индексы обновляются in-memory.
3. На каждое чтение (`explore`, `search_nodes`, ...) — из in-memory кеша, O(1) для типовых операций.
4. Store — source of truth; in-memory кеш — hot-path для runtime.

**Session identity.** `thread_id` читается из `runtime.config.configurable.thread_id`. Если не задан — генерируется UUID4, в лог пишется warning (каждый запуск получает свой граф, два одновременных запуска без `thread_id` не смешаются).

**Concurrency.** LangGraph может параллельно звать несколько `@fact`-тулов в одном шаге — без защиты это даст classic lost-update race. PinBoard держит **`anyio.Lock`** per `thread_id` (единый примитив, корректно работающий и в sync, и в async контексте) и захватывает его **только вокруг шага 4 из 6.1** (load graph → apply delta → persist). Вызов тул-функции, валидация, экстракция, логирование, хуки — вне lock'а; параллелизм LangGraph сохраняется, сериализуется только запись в граф. В типовом сценарии шаг 4 — миллисекунды, lock contention минимальный.

Выбор `anyio.Lock`, а не `threading.Lock + asyncio.Lock` по ветвям — потому что в одной сессии могут мешать sync и async тулы, двух разных примитивов для них недостаточно. `anyio` даёт унифицированный лок, совместимый с обоими режимами.

**Multi-process.** In-memory кеш + lock работают только в рамках одного процесса. MVP scope — single-process агент. Multi-worker deployments (uvicorn workers, Celery) — вне Фазы 1.

### 9.2 Дамп и restore графа

Отдельного API `FactGraph.to_dict()/from_dict()` в MVP нет. Дамп/restore — паттерн через Store:

```python
def dump_session(store, thread_id: str) -> dict:
    return {
        "pinboard_version": "0.1",    # версия библиотеки, не per-object schema
        "nodes": [i.value for i in store.search(("pinboard", thread_id, "nodes"))],
        "edges": [i.value for i in store.search(("pinboard", thread_id, "edges"))],
        "entities": store.get(("pinboard", thread_id, "entities"), "registry"),
        "tool_calls": [i.value for i in store.search(("pinboard", thread_id, "tool_calls"))],
    }
```

Версионирование — одной строкой на весь дамп (semver библиотеки), не на каждом объекте. Если когда-нибудь потребуется миграция дампов между версиями — пишется отдельный скрипт-конвертер. Formal API (`FactGraph.dump()`/`load()`) — Фаза 2, если паттерн востребован.

### 9.3 Лог действий

Namespace `("pinboard", thread_id, "tool_calls")`. Каждый вызов `@fact`-декорированного тула добавляет запись:

```python
@dataclass(slots=True, frozen=True)
class ToolCallRecord:
    tool_name: str
    args_repr: str           # canonical JSON, см. 6.3
    timestamp: datetime
    event_id: EventId | None # id созданной EventNode или None при duplicate skip
    summary: str             # "+2 nodes, +3 edges" / "duplicate (skipped)" / "error: ..."
    duration_ms: int
```

Формат `args_repr` и правила masking — см. 6.3.

Используется тулом `what_have_i_done` и детекцией повторных вызовов.

**Рост лога.** Запись неограниченная — лог хранится целиком за сессию. Soft-limit задаётся глобальным конфигом PinBoard:

```python
from pinboard import configure

configure(tool_log_soft_limit=500)   # default
```

При пересечении порога пишется warning в лог. Жёсткого cap нет — пользователь сам решает, как обрезать, если хочет. На практике 500 вызовов за сессию — это много, лимит в основном сигнал «что-то не так с промптом, LLM ходит по кругу».

**`configure()` — process-global mutable state.** Применяется ко всем сессиям в пределах одного процесса. Это осознанный компромисс: для per-session override пришлось бы пробрасывать настройки через `ToolRuntime` в каждый тул — громоздко для ручки, которую на практике меняют раз в проекте. Если нужна per-session настройка (редкий случай) — пользователь пишет свой хук поверх `ToolCallRecord`-писателя.

---

## 10. Готовые тулы чтения графа

```python
from pinboard.tools import make_graph_tools

graph_tools = make_graph_tools(hooks=hooks)
```

| Тул | Фаза | Сигнатура | Возврат |
| --- | --- | --------- | ------- |
| `explore` | 1 | `(node_type, value, depth=2, direction=Direction.BOTH, skip_events=True, max_nodes=30)` | Подграф вокруг сущности |
| `timeline` | 1 | `(node_type, value, limit=50)` | Хронология событий для сущности |
| `graph_summary` | 1 | `(top_per_type=5)` | Типы с количествами + top-N сущностей |
| `search_nodes` | 1 | `(node_type=None, value_pattern=None, include_events=False, limit=50)` | Листинг/поиск FactNode |
| `what_have_i_done` | 1 | `(tool_name=None, node_type=None, value=None, limit=50)` | Лог вызовов тулов в сессии (фильтры — см. 10.6) |
| `find_path` | 2 | `(from_type, from_value, to_type, to_value, max_depth=6, skip_events=True)` | Кратчайший путь (undirected BFS) |
| `get_evidence` | 3 | `(event_id)` | Полный raw return тула (требует `@fact(store_raw=True)`) |

Enums (Python 3.11+ `StrEnum` — без multiple inheritance):

```python
from enum import StrEnum

class Direction(StrEnum):
    OUT = "out"
    IN = "in"
    BOTH = "both"

class OnDuplicate(StrEnum):
    ALWAYS = "always"
    SKIP = "skip"
    CACHE = "cache"
```

### 10.1 Политика лимитов

Все тулы возвращают компактное текстовое представление под ограниченный контекст локальных моделей. При превышении лимита ответ дополняется строкой `"... and N more (narrow by type / value_pattern / depth)"`. Дефолты подобраны под экран LLM в ~2K токенов.

### 10.2 Поведение при отсутствии сущности

`explore("IP", "1.2.3.4")` для несуществующей ноды — не исключение, а текстовый ответ:

```text
No node found: IP = 1.2.3.4
Try: search_nodes(node_type="IP") to list all IPs in the graph.
```

LLM нормально обрабатывает такое сообщение и корректно планирует следующий шаг.

### 10.3 fnmatch case-sensitivity

`search_nodes` всегда использует `fnmatch.fnmatchcase` — поведение одинаковое на Linux/macOS/Windows, регистр значимый. Пользователь явно контролирует чувствительность через normalizer (если хочет case-insensitive match — нормализует значения к lower).

### 10.4 `graph_summary` как discovery

Типы нод user-defined, агент изначально не знает, какие типы есть в графе. Первый вызов обычно — `graph_summary()`, он возвращает список типов с количествами. Далее агент вызывает `search_nodes(node_type=...)` / `explore(...)` уже со знанием схемы.

### 10.5 Видимость EventNode

EventNode — технические узлы (создаются декоратором, не пользователем). Видимость в разных тулах:

**В `graph_summary`** — EventNode не показываются как тип. LLM оперирует сущностями, не вызовами тулов.

**В `search_nodes`** — по умолчанию скрыты (`include_events=False`). Включаются явно, если LLM хочет найти конкретные вызовы тула — например, `search_nodes(node_type="Event", value_pattern="fetch_cloudtrail*", include_events=True)`.

**В `explore` и `find_path`** — поведение контролируется `skip_events`:

- `skip_events=True` (по умолчанию) — EventNode трактуются как прозрачные соединители. `explore("IP", "1.2.3.4", depth=1)` вернёт все FactNode, которые делят хотя бы одно событие с этим IP. Это натуральный «что связано с этой сущностью» — топология «через Event» для LLM невидима.
- `skip_events=False` — EventNode видны как первоклассные узлы графа, считаются как хоп. Полезно для forensic-анализа «через какое именно событие эти две сущности связаны».

### 10.6 Семантика фильтров `what_have_i_done`

- `tool_name=None, node_type=None, value=None` — весь лог вызовов сессии (с учётом `limit`).
- `tool_name="fetch_cloudtrail"` — только вызовы этого тула.
- `node_type="IP"` без `value` — вызовы тулов, в результате которых было извлечено хотя бы одно FactNode типа IP (через `edges_by_event` → FactNode).
- `node_type="IP", value="1.2.3.4"` — вызовы, в которых конкретная сущность `IP:canonical(1.2.3.4)` была извлечена. `value` нормализуется через `Entity.normalizer` перед поиском.
- `node_type=None, value="1.2.3.4"` — не валидно, `ValueError`: значение без типа неоднозначно.
- Все заданные фильтры применяются через AND.

### 10.7 Формат вывода `explore` / `timeline`

Компактное текстовое представление, всегда с описаниями рёбер. Пример:

```text
explore(type="IP", value="185.220.101.42", depth=1, skip_events=True):

IP: "IPv4 or IPv6 network address"
  value: 185.220.101.42  (first seen 2024-03-15 14:22:01, 3 events)

Related facts (via 3 events):
  User:arn:aws:iam::...:admin
    via fetch_cloudtrail @ 14:22:01
    "Who performed the action"
  Action:AssumeRole
    via fetch_cloudtrail @ 14:22:01
    "API action name"
  User:arn:aws:iam::...:admin (same as above)
    via fetch_cloudtrail @ 14:23:15
    "Who performed the action"
  ... and 4 more facts (use max_nodes=50 to see all)
```

Формат стабилен и документирован — от него зависят пользовательские промпты.

**Почему свой формат, а не RDF Turtle / TOON / JSON.** REASEARCH §C упоминает, что Turtle и TOON дают хорошие результаты на LLM-benchmarks. Мы выбрали custom plain-text потому что: а) локальным моделям не надо знать RDF-синтаксис, б) heterogeneous подграфы (разные типы нод и рёбер в одном выводе) плохо ложатся на TOON с его homogeneous-array оптимизацией, в) plain-text с indentation — наиболее предсказуемый шаблон под prompt engineering для небольших моделей.

---

## 11. Система хуков

### 11.1 Базовый класс

```python
class PinBoardHooks:
    def on_node_added(self, node: FactNode | EventNode) -> None: ...
    def on_edge_added(self, edge: FactEdge) -> None: ...
    def on_link_found(self, existing: FactNode, event_id: EventId) -> None: ...
    def on_ingest_complete(self, result: IngestResult) -> None: ...
    def on_graph_changed(self, graph: FactGraph) -> None: ...
```

Все методы — no-op по умолчанию. Каждый вызов обёрнут `try/except`, ошибка логируется, ingestion продолжается.

Пользовательский подкласс — с декоратором `@override` (Python 3.12+) для защиты от опечаток:

```python
from typing import override

class MyHook(PinBoardHooks):
    @override
    def on_ingest_complete(self, result: IngestResult) -> None:
        logger.info("+%d nodes", result.new_nodes)
```

### 11.2 Готовые реализации

```python
from pinboard.hooks import LoggingHook, LangfuseHook, WebSocketHook, CompositeHook
```

- `LoggingHook` — пишет в `logging`.
- `LangfuseHook` — отправляет span-ы в Langfuse.
- `WebSocketHook` — стримит дельты графа (для UI-визуализации).
- `CompositeHook` — объединяет несколько хуков.

### 11.3 Подключение

Хуки передаются в декоратор тула и в фабрику graph-тулов. Как правило — один и тот же объект:

```python
hooks = CompositeHook([LoggingHook(), LangfuseHook(client=langfuse)])

@fact(model=CloudTrailEvent, many=True, hooks=hooks)
@tool
def fetch_cloudtrail(...): ...

graph_tools = make_graph_tools(hooks=hooks)
```

### 11.4 WebSocket-визуализация

`WebSocketHook` шлёт дельты (не полный граф) на каждое изменение — фронт на Cytoscape.js анимирует граф в реалтайме, детективная доска «оживает».

---

## 12. Пример сборки агента

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

from pinboard import fact
from pinboard.tools import make_graph_tools
from pinboard.hooks import LangfuseHook

# my_project — гипотетический модуль пользователя:
#   entities.py — Entity-инстансы (IP, User, Domain, ...)
#   models.py — Pydantic-модели ответов тулов, размеченные node(...)
from my_project.models import CloudTrailEvent, OktaEvent, VTReport

hooks = LangfuseHook(client=langfuse)

@fact(model=CloudTrailEvent, many=True, hooks=hooks)
@tool
def fetch_cloudtrail(user_arn: str, hours: int, runtime: ToolRuntime) -> list[dict]:
    """Fetch CloudTrail logs for a user"""
    return boto3_client.lookup_events(UserArn=user_arn, Hours=hours)

@fact(model=OktaEvent, many=True, hooks=hooks)
@tool
def fetch_okta(user_email: str, runtime: ToolRuntime) -> list[dict]:
    """Fetch Okta logs for a user"""
    return okta_client.get_logs(filter=f'actor.alternateId eq "{user_email}"')

@fact(model=VTReport, hooks=hooks)
@tool
async def vt_lookup(value: str, runtime: ToolRuntime) -> dict:
    """Check IP/domain/hash in VirusTotal"""
    return await vt_client.get_report(value)

store = InMemoryStore()

agent = create_agent(
    model=ChatOpenAI(base_url="http://localhost:8000/v1", model="qwen2.5-72b"),
    tools=[
        fetch_cloudtrail,
        fetch_okta,
        vt_lookup,
        *make_graph_tools(hooks=hooks),
    ],
    store=store,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Investigate AssumeRole from 185.220.101.42"}]},
    config={"configurable": {"thread_id": "investigation-001"}},
)
```

---

## 13. Разделение ответственности LLM

**LLM делает:** принятие решений (куда копать, что обогащать), выбор тулов и параметров, интерпретацию графа, формирование гипотез, построение вердикта, решение о том, когда звать `graph_summary` / `search_nodes` / `what_have_i_done`.

**LLM не делает:** извлечение фактов из ответов тулов (детерминированные правила экстракции), автолинковку (exact match по `(type, canonical_value)`), контроль каскадного обогащения (естественно ограничивается через промпт + дедупликацию).

**Требования к LLM:** OpenAI-совместимый API, tool calling. Рекомендуемые модели: Qwen 2.5, Llama 3.1/3.3, Mistral.

---

## 14. Стек технологий

| Компонент | Технология |
| --------- | ---------- |
| Язык | Python 3.10+ |
| Граф | NetworkX |
| Runtime-хранилище | LangGraph Store (InMemoryStore) |
| Модели данных | Pydantic v2 |
| Интеграция с агентом | `@tool` из `langchain_core.tools`, `ToolRuntime` из `langgraph.prebuilt` |
| LLM-бэкенд | vLLM / SGLang (OpenAI-совместимый) |
| Наблюдаемость | Langfuse (опционально) |
| Тесты | pytest |

LangGraph — peer-dependency.

---

## 15. Публичный API

```python
from pinboard import (
    fact,                     # @fact декоратор
    node,                     # фабрика для нодовых полей Pydantic-модели
    Entity,                   # value-object для типа ноды
    Direction,                # StrEnum для explore: OUT / IN / BOTH
    OnDuplicate,              # StrEnum для @fact(on_duplicate=...): ALWAYS / SKIP / CACHE
    configure,                # глобальные настройки (tool_log_soft_limit и т.п.)
    FactGraph,
    FactNode, FactEdge, EventNode,
    IngestResult,
    PinBoardHooks,
    # Exceptions
    PinBoardError,
    PinBoardConfigError,
    PinBoardValidationError,
    PinBoardNormalizerError,
    PinBoardExtractionError,
)

from pinboard.tools import make_graph_tools

from pinboard.hooks import (
    LoggingHook,
    CompositeHook,
)
# LangfuseHook (Фаза 2) и WebSocketHook (Фаза 3) добавятся позже.
```

### 15.1 Что НЕ входит в ядро

- Конкретные Pydantic-модели источников (CloudTrail, Okta, GitHub Audit, VT, WHOIS, ...) — пишутся пользователем или поставляются как отдельные пакеты (`pinboard-aws`, `pinboard-enrichment-vt`).
- Сам агент и его StateGraph — это пользовательский код.
- Нормализаторы значений — всегда пользовательский код (см. 7).
- Готовые модели по доменным онтологиям (OCSF, STIX, внутренние схемы) — только в примерах документации (см. 8).

---

## 16. Этапы реализации

### Фаза 1 — Ядро

**Что делаем:**

- `Entity` value-object с валидацией `description` и session-registry (warning при duplicate name).
- `node(...)` фабрика с обязательным `description`, принимающая `Entity`.
- Экстрактор: пять правил обхода Pydantic-модели (4.1), отклонение пустого canonical_value, recursion guard через `seen: set[type]`.
- `FactNode`, `EventNode` (с `node_type="Event"` и `schema_version=1`), `FactEdge`, `FactGraph`, `IngestResult` с автолинковкой и индексами.
- `@fact` декоратор: sync + async, `many=False/True`, `on_duplicate`, `mask_args`, canonical JSON args_repr, fail-loud при валидации, поддержка dict/BaseModel return.
- Session identity через `thread_id` (UUID4 fallback), per-session `anyio.Lock` вокруг шага 4.
- Sharded хранение в Store (nodes/edges/entities/tool_calls как отдельные keys).
- Graph-тулы: `explore` (с `Direction`), `timeline`, `graph_summary`, `search_nodes`, `what_have_i_done`.
- EventNode скрыты по умолчанию в `search_nodes` / `graph_summary`.
- Exception hierarchy (`PinBoardError` и подклассы).
- `PinBoardHooks` + `LoggingHook` с log-and-continue поведением.
- `configure(tool_log_soft_limit=500)` глобальный конфиг.

**Acceptance criteria (что считается готовым):**

1. **Extraction на вложенной модели.** Тест: `CloudTrailEvent` с `Actor.user_arn` и `src_endpoint.ip` (2 уровня вложенности). После одного вызова тула в графе: 1 EventNode, ровно N FactNode (по числу non-None node-полей), рёбра типа `Actor.user_arn` и `Endpoint.ip`.
2. **Автолинковка и дедупликация.** Тест: два вызова с одинаковым IP в разных тулах. В графе — одна FactNode типа IP, две EventNode, `FactNode.source_tools` содержит оба имени, `source_events` содержит оба event_id.
3. **Recursion guard.** Тест: модель `Process(parent: Process | None)`, eager-scan не зависает.
4. **Concurrency.** Тест: 10 параллельных `@fact`-тулов в одном LangGraph-шаге, каждый добавляет ноду. После всех — в графе 10 нод, ни одного lost-update.
5. **Duplicate detection.** Тест: два вызова с одинаковыми args + `on_duplicate="skip"`. Второй вызов не исполняет тул-функцию (проверяется по счётчику вызовов mock), возвращает маркер-строку, хуки не зовутся.
6. **Fail-loud на валидации.** Тест: тул возвращает битый dict. `@fact` кидает `PinBoardValidationError`, граф не меняется.
7. **Session isolation.** Тест: два `thread_id` в одном процессе, события одной сессии не видны во второй.
8. **Discovery без ingestion.** Тест: агент с `@fact`-тулами, но без вызовов. `graph_summary()` возвращает list known types из eager-registry (counts = 0).

**Performance targets (soft, не блокеры):**

- `explore(depth=2)` на графе 10 000 нод — < 50 мс.
- `@fact` overhead (ingestion-блок на событие с 5 фактами) — < 10 мс.
- Load сессии (sharded read всех nodes/edges) — < 500 мс для 10 000 нод.

### Фаза 2 — Расширения

- `find_path` (undirected BFS на view MultiDiGraph).
- `response_transform` и `on_ingest` коллбэки.
- `LangfuseHook`.
- Пример полного агента с пользовательскими моделями и нормализаторами.

### Фаза 3 — Экосистема

- `WebSocketHook` + пример фронта на Cytoscape.js.
- `get_evidence(event_id)` + опция `@fact(store_raw=True)` для хранения полного return тула в отдельном namespace — для forensic-анализа.
- Ранжирование в `timeline` по формуле AriGraph (`n_i / max(N_i, 1) * log(max(N_i, 1))`).
- Примеры пакетов-плагинов для конкретных источников.
- Документация с референс-моделями (OCSF, STIX), нормализаторами (IP/email/domain/ARN/ИНН) и паттернами написания моделей.

### Явно вне scope (и почему)

Для каждого пункта — ссылка на REASEARCH §E.2 и обоснование tradeoff.

- **Bi-temporal модель** (`valid_at` / `invalid_at`) — Risk 1. Для session-scope (минуты) нет смысла invalidate'ить факты. Пользователь, если надо, хранит timestamps в `properties` FactNode.
- **Confidence scoring** — Risk 3. Research говорит «критично для security». Мы не согласны: единое `confidence: float` на рёбрах даёт ложное ощущение калиброванности (что значит 0.7? откуда? сравнимо ли между VT и WHOIS?). Пользователь, если надо, кладёт свои confidence/reliability в `properties` модели.
- **Противоречивые факты** — Risk 2. Star-топология сохраняет оба факта как отдельные события. LLM видит в `EventNode.properties` источник каждого и решает сама. Автоматический last-write-wins был бы опаснее.
- **Fuzzy entity resolution** (John Smith / J. Smith) — Risk 5. Доменно-специфичная задача. `Entity.normalizer` закрывает детерминированные случаи; fuzzy — пользовательский код с LLM/embeddings, не инфраструктура.
- **Guardrails каскадного обогащения** (Exit-tool, hard max_turns) — REASEARCH §C. Это agent-level concern, не library: `max_iterations` — параметр LangGraph `create_agent`, Exit-tool — пользовательский тул в агенте. PinBoard даёт repetition detection через `on_duplicate` и depth-limits в `explore`.
- **Salient node selection / PageRank** — Risk 4. В MVP жёсткие лимиты (`max_nodes`, `depth`) дают детерминированное обрезание. PageRank — Фаза 3.
- **Async enrichment в фоне** (Letta sleep-time, MAGMA dual-stream) — Risk 8. Для session-scope с окном в минуты — overengineering. При переходе на hours/days сессии имеет смысл вернуться.
- **`rationale` / `interpretation` в `ToolCallRecord`** — REASEARCH §C. Это содержимое LLM-ответа, библиотека его не знает. Пользователь обогащает лог через `on_ingest` хук, если нужно.
- **State replacement и вытеснение фактов** — в MVP граф только растёт. Если увидим, что мешает — добавим по конкретному кейсу.
- **Прямые рёбра между FactNode в обход EventNode.** Star-топология достаточна. `skip_events=True` в `explore`/`find_path` даёт иллюзию прямой связности для LLM.
- **Embedding-based retrieval в graph-тулах.** Для IP/hash/email semantic similarity бесполезна; для свободного текста — отдельный тул в user-space.
- **LLM в runtime-пути извлечения фактов.** Противоречит базовому принципу библиотеки (детерминизм, ноль стоимости extraction).
- **Кросс-сессионная персистентность.** Граф живёт одну сессию; дамп/restore — обвязка пользователя.
- **Per-object `schema_version`.** YAGNI. Одинаковая цифра на 10K нод — шум. Миграция дампов (если когда-нибудь потребуется) делается через версию библиотеки в semver + отдельный скрипт-конвертер.

### Сравнение с соседями

Уникальность ниши PinBoard — комбинация трёх архитектурных решений, которой нет ни у одной из существующих библиотек (REASEARCH §E.1):

1. **Декларативная разметка через `node(type=Entity, ...)`** на полях пользовательских Pydantic-моделей — ни один конкурент не использует аннотации на полях для определения того, что становится нодой графа.
2. **Side-effect декоратор `@fact`** — прозрачное извлечение фактов при вызове тула без изменения return value. Во всех альтернативах memory API вызывается явно.
3. **Session-scoped граф без LLM в runtime-извлечении** — детерминистично, бесплатно, быстро.

Позиционирование против конкретных соседей:

- **Graphiti** (Zep AI) — персистентный KG с LLM-extraction и bi-temporal моделью. Мощнее, но дороже на порядок (600K+ токенов на диалог) и требует LLM в runtime. PinBoard делает другой tradeoff: детерминизм и дешевизна в обмен на гибкость.
- **Mem0** — гибрид vector + optional graph, ориентирован на персонализацию, не на session-scoped working memory.
- **Letta / MemGPT** — hierarchical self-editing memory, не граф. Ортогонально и потенциально комплементарно.
- **Cognee** — document-oriented, ECL-пайплайн, LLM-extraction. Другая задача (ingestion документов).
- **AriGraph** (IJCAI 2025) — ближайший академический аналог: episodic + semantic подграфы, инкрементальное построение. PinBoard заимствует терминологию и формулу ранжирования эпизодов; отличие — извлечение декларативное, не LLM-based.
- **LangGraph Store** — фундамент, на котором PinBoard живёт. Store — плоский key-value, PinBoard добавляет графовую семантику поверх.
