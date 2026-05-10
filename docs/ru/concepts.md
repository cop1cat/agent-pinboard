# Концепты

Четыре близких по звучанию термина — путать нельзя, иначе через
неделю код станет двусмысленным.

| Термин | На пальцах | Кто создаёт |
|---|---|---|
| `Entity` | *Тип* того, что представляет нода (имя, описание, опциональный нормализатор) | Вы — один раз на тип, в своём проекте |
| `node(...)` | Маркер на Pydantic-поле: «значение этого поля становится нодой графа указанного типа» | Вы — при определении модели ответа тула |
| `FactNode` | Runtime-нода в графе для одного извлечённого факта | Библиотека, автоматически |
| `EventNode` | Runtime-нода для одного вызова тула | Библиотека, автоматически |

`Entity` и `node()` — write-time концепции, читаются и пишутся при
проектировании тулов. `FactNode` и `EventNode` — runtime-объекты,
напрямую с ними почти не работаете; агент видит их через graph-тулы.

## `Entity` — описание типа ноды

```python
from agent_pinboard import Entity

IP = Entity(
    name="IP",
    description="IPv4 or IPv6 network address",
    normalizer=canonical_ip,           # опционально
)
```

- `name` становится `FactNode.node_type` для всех нод этого типа.
- `description` показывается в `graph_summary` — LLM понимает, что
  это за тип.
- `normalizer` канонизирует значения для автолинковки. Без него
  `192.168.001.001` и `192.168.1.1` дадут две разных ноды.

`Entity` — frozen-dataclass, чистый value-object, без side-effect при
конструировании. Объявите каждый **один раз** (обычно в `entities.py`)
и импортируйте везде, где он нужен.

## `node()` — фабрика Pydantic-поля

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

`node(...)` возвращает обычный Pydantic `FieldInfo` с AgentPinBoard-метой.
Pydantic валидирует поле как обычно, экстрактор читает мету и решает,
что делать со значением.

Два описания, оба обязательные, оба полезны для LLM:

- **`Entity.description`** — что это за тип в принципе («IPv4/IPv6
  network address»). Видно в `graph_summary`.
- **`node(description=...)`** — как *это конкретное поле* связано
  с событием («IP from which the API call was made»). Видно на
  рёбрах в `explore` / `timeline`.

## Когда `node()` против обычного `Field()`

Эвристика, которая решает 80% случаев:

- **Используй `node(...)`, если значение может встречаться в разных
  событиях и LLM полезно видеть связи между событиями, делящими его.**
  IP, user ID, хэши файлов, ARN, ИНН, order ID.
- **Используй обычный `Field(...)`, если значение one-shot и линковать
  его бессмысленно.** Timestamp, raw message body, latency, status code.

Значения plain-`Field` попадают в `EventNode.properties` и видны
агенту через `timeline`.

## `@pin` — декоратор

```python
from agent_pinboard import pin
from langchain_core.tools import tool

@pin(model=CloudTrailEvent, many=True)
@tool
def fetch_cloudtrail(user: str, runtime: ToolRuntime) -> list[dict]:
    """Fetch CloudTrail logs for a user."""
    return aws_client.lookup_events(user)
```

Декоратор — side-effect. Каждый вызов:

1. валидирует return через `model`;
2. создаёт `EventNode`;
3. извлекает `FactNode`-ы и `FactEdge`-ы из модели;
4. мержит дельту в session-граф под per-thread локом;
5. пишет `ToolCallRecord`;
6. дёргает хуки;
7. опционально переписывает return через `response_transform`.

`@pin` **всегда выше** `@tool`. Обратный порядок кидает
`AgentPinBoardConfigError` сразу при декорировании.

## Star-топология

Топология всегда **звезда вокруг `EventNode`**: извлечённые факты
подключаются к своему событию, никогда напрямую к другим фактам.

```
              EventNode:fetch_cloudtrail @ 14:22:01
             /            |              \
       IP:185.220       User:admin       Action:AssumeRole
```

Зачем звезда: каждое событие создаёт одну новую ноду + N рёбер
вместо N² рёбер. Два факта, «которые вместе», делят событие; LLM
находит их через `explore(...)` (по дефолту с `skip_events=True` —
события трактуются как прозрачные коннекторы).

## Сессия и Store

**Сессия** идентифицируется `thread_id` (читается из
`runtime.config.configurable.thread_id`). Всё состояние графа сессии
живёт в LangGraph `Store` под sharded-namespace:

```
("agent_pinboard", thread_id, "nodes", node_id)        → FactNode | EventNode
("agent_pinboard", thread_id, "edges", edge_id)        → FactEdge
("agent_pinboard", thread_id, "entities")              → session entity registry
("agent_pinboard", thread_id, "tool_calls", record_id) → ToolCallRecord
```

In-memory кеши ускоряют hot-path; store остаётся source of truth.
Сессии в одном процессе изолированы по `thread_id`. Если `thread_id`
не передан — AgentPinBoard генерирует UUID4 и пишет warning, параллельные
«анонимные» сессии не сольются молча.

## Вне scope (намеренно)

Библиотека НЕ делает ничего из этого — обоснования каждого пункта
в README §16:

- bi-temporal валидность (`valid_at` / `invalid_at`)
- confidence scoring на фактах
- fuzzy entity resolution за пределами exact-canonical
- LLM-based извлечение
- кросс-сессионная персистентность как основной сценарий
- прямые fact-to-fact рёбра (только star-топология)

Если что-то из этого нужно — стройте поверх. `Entity.properties` и
хуки дают точки расширения.
