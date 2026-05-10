# Graph-тулы

`make_graph_tools()` возвращает пять LangChain-тулов, чтобы LLM-агент
читал граф. Они stateless и стабильные между вызовами — регистрируйте
рядом с вашими `@pin`-тулами и агент сам найдёт.

```python
from agent_pinboard import make_graph_tools

tools = [my_fetch_tool, *make_graph_tools()]
```

| Тул | Фаза | Назначение |
|---|---|---|
| `graph_summary` | 1 | Карта известных типов сущностей + счётчики + top-сущности |
| `search_nodes` | 1 | Листинг / glob-фильтр по факт-нодам |
| `explore` | 1 | Подграф вокруг сущности, настраиваемые depth + direction |
| `timeline` | 1 | Хронология событий, в которых участвовала сущность |
| `what_have_i_done` | 1 | Фильтр tool-call лога текущей сессии |
| `find_path` | 2 | Top-N кратчайших путей между двумя сущностями |
| `get_evidence` | 3 | Сырой return JSON для конкретного события (требует `@pin(store_raw=True)`) |

## Рекомендуемый workflow LLM

1. **Discover** — `graph_summary()` первым, до любого reasoning'a.
   Возвращает все типы, которые могут произвести зарегистрированные
   тулы, со счётчиками текущих инстансов в графе.
2. **Locate** — `search_nodes(node_type="IP", value_pattern="185.220.*")`
   найти кандидатов по глобу.
3. **Investigate** — `explore("IP", "185.220.101.42")` посмотреть, что
   связано. Дефолт `skip_events=True` показывает факты, делящие событие,
   без необходимости агенту думать о EventNode-хопах.
4. **Reconstruct** — `timeline("IP", "185.220.101.42")` для хронологии.
5. **Self-introspect** — `what_have_i_done(node_type="IP", value="...")`
   когда LLM не уверен, не запрашивал ли он уже это.

## `graph_summary(top_per_type=5)`

Списком все типы сущностей, известные сессии — и объявленные через
`@pin(model=...)`, и присутствующие в графе. Счётчики из живого
графа; типы с `0 in graph` означают «агент мог бы запросить, но пока
не делал».

```text
graph_summary:
  Action (2 in graph) — Performed API action
    AssumeRole  (in 1 events, via ['fetch'])
    ListBuckets  (in 1 events, via ['fetch'])
  IP (3 in graph) — IPv4 or IPv6 network address
    185.220.101.42  (in 3 events, via ['fetch', 'vt'])
    45.77.0.1  (in 1 events, via ['vt'])
  User (1 in graph) — Acting principal
    arn:aws:iam::123:user/admin  (in 2 events, via ['fetch'])
```

`top_per_type` — максимум top-фактов на тип. Top — по числу событий,
в которых факт встречался (по убыванию).

## `search_nodes`

```python
search_nodes(
    node_type=None,         # точное совпадение FactNode.node_type
    value_pattern=None,     # fnmatchcase glob над canonical_value
    include_events=False,   # True — ищем конкретные tool-вызовы
    limit=50,               # глобальный лимит на матчи
)
```

`fnmatch.fnmatchcase` означает, что паттерн **case-sensitive** на
любой ОС, и использует стандартные unix-globs (`*`, `?`, `[abc]`).
Для case-insensitive поиска настройте нормализатор `Entity` так,
чтобы канонизация уходила в lower-case, и передавайте паттерн в нижнем
регистре.

EventNode скрыты по дефолту. `include_events=True` — если агент
именно ищет конкретный вызов тула.

## `explore`

```python
explore(
    node_type, value,
    depth=2,
    direction=Direction.BOTH,
    skip_events=True,
    max_nodes=30,
)
```

Обходит подграф вокруг сущности до `depth` хопов. С `skip_events=True`
(дефолт) один хоп = «от FactNode через любое разделяемое событие к
другой FactNode» — события прозрачны. С `skip_events=False` обход идёт
по нативному `MultiDiGraph` напрямую, EventNode потребляют хопы.

`direction` имеет смысл только при `skip_events=False`, потому что
star-топология означает: у FactNode только **входящие** рёбра от
EventNode. Конкретно:

- `Direction.IN` + `skip_events=False` — найти EventNode, откуда
  пришёл факт.
- `Direction.OUT` + `skip_events=False` — для FactNode ничего не
  найдёт (исходящих рёбер нет).
- `Direction.BOTH` (дефолт) — оба направления.

## `find_path`

```python
find_path(
    from_type, from_value,
    to_type, to_value,
    max_depth=6,
    skip_events=True,
    top=1,
)
```

Возвращает top-N кратчайших простых путей между двумя факт-нодами
(алгоритм Йена через `networkx.shortest_simple_paths`). С
`skip_events=True` (дефолт) поиск идёт по fact-only проекции — два
факта смежны если делят хотя бы одно событие, путь длины N означает
«N хопов между фактами». С `skip_events=False` обход идёт по нативному
MultiDiGraph, EventNode потребляют хопы.

`top` дефолт 1 (только кратчайший). Поставьте 5, чтобы получить до 5
кратчайших путей в неубывающем порядке — полезно когда есть несколько
плausible цепочек и хочется, чтобы LLM сравнила.

`max_depth` обрезает: пути длиннее не возвращаются. Дефолт 6 —
разумный верхний лимит для session-scope графов.

Если пути нет в рамках лимитов — возвращается hint-сообщение, не
exception.

```text
find_path(IP='185.220.101.42' → IP='8.8.8.8', top=1, max_depth=6, skip_events=True): found 1 path(s)
  Path 1 (1 hop):
    IP: 185.220.101.42
      ↓  via vt_lookup@2026-04-15T12:00:00+00:00
    IP: 8.8.8.8
```

## `get_evidence`

```python
get_evidence(event_id: str)
```

Возвращает сырой return тула, породившего ``event_id`` — только если
тул был декорирован с ``@pin(store_raw=True)``. Иначе — текстовая
подсказка с указанием на структурированные ``properties`` EventNode
(не-node скалярные поля распарсенной модели).

```text
get_evidence('e-1234') — tool=fetch_cloudtrail, timestamp=2026-04-15T14:22:01+00:00:
{
  "src_ip": "185.220.101.42",
  "actor": {"user_arn": "arn:aws:iam::123:user/admin"},
  "action_name": "AssumeRole",
  ...
}
```

`store_raw` — opt-in: большинству агентов сырые payload не нужны, это
только forensic/compliance use case.

## `timeline`

Списком события, в которых участвовал конкретный факт. Дефолт —
хронологический (старые сначала). Передайте ``rank=True`` чтобы
сортировать по AriGraph relevance score: события, чьи другие факты
пересекаются с neighborhood'ом сущности, ранжируются выше, с
log-понижением для bulk-событий, тащащих несвязанные факты. Каждая
запись — timestamp, source tool, опциональный score, properties
события (не-node скалярные поля модели).

```text
timeline(IP='185.220.101.42', 3 events):
  2026-04-15T14:22:01+00:00 via fetch_cloudtrail
    properties: {'action': 'AssumeRole', 'event_time': '...'}
  2026-04-15T14:23:15+00:00 via fetch_cloudtrail
    properties: {'action': 'ListBuckets', 'event_time': '...'}
  2026-04-15T14:24:00+00:00 via vt_lookup
    properties: {}
```

## `what_have_i_done`

Фильтрует per-session tool-call лог. Фильтры комбинируются по AND.

```python
what_have_i_done(
    tool_name=None,     # только вызовы этого тула
    node_type=None,     # только вызовы, давшие ноду этого типа
    value=None,         # только вызовы, давшие именно эту сущность
    limit=50,
)
```

`value` требует `node_type` (значение без типа неоднозначно — `"1"`
может быть IP, user ID, order number). `value` нормализуется через
`Entity.normalizer` перед фильтром, поэтому можно искать как по
сырому, так и по каноническому виду.

```text
what_have_i_done (3 of 7 records):
  2026-04-15T14:22:01+00:00 fetch({"user": "admin"}) -> +2 nodes, +0 linked, +2 edges (5ms)
  2026-04-15T14:23:15+00:00 fetch({"user": "admin"}) -> duplicate (skipped) (0ms)
  2026-04-15T14:24:00+00:00 vt({"value": "185.220.101.42"}) -> +1 nodes, +1 linked, +2 edges (12ms)
```

## Формат вывода

Все тулы возвращают plain text, не JSON. Формат намеренно стабильный
и контрактный — промпты вашего агента могут на него полагаться.
Описание ребра из `node()` **всегда** рендерится рядом с фактом, имя
поля само по себе — не контракт для LLM.

## Пустые результаты — не исключения

Если тул ничего не нашёл, он возвращает text-подсказку, не exception.
Пример:

```text
No node found: IP = 9.9.9.9
Try: search_nodes(node_type='IP') to list all nodes of this type.
```

Это держит цепочку reasoning'a LLM непрерывной — отсутствие ноды это
просто данные, не ошибка.
