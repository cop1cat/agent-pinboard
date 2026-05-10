# Правила экстракции

Когда `@pin` отрабатывает, он обходит провалидированную Pydantic-модель
и применяет пять правил к каждому полю. Правила взаимоисключающие —
ровно одно срабатывает на поле.

Для каждого поля: `value` — фактическое значение, `entity` — `Entity`,
прикреплённая через `node()` (или `None`, если это обычный `Field()`).

## Правило 1 — примитив с `node()`

```python
class Event(BaseModel):
    src_ip: str | None = node(type=IP, description="src", default=None)
```

→ создаёт `FactNode(type=IP.name, value=src_ip)` (с автолинковкой и
нормализацией) + `FactEdge` от EventNode с типом `Event.src_ip`.

## Правило 2 — `None`

→ молча пропускается. Это для optional-полей.

## Правило 3 — `list[primitive]` с `node()`

```python
class VTReport(BaseModel):
    related_ips: list[str] = node(
        type=IP, description="Related IPs from VT", default_factory=list,
    )
```

→ Правило 1 применяется к каждому элементу. Каждый элемент — отдельная
FactNode, со своим ребром типа `VTReport.related_ips`.

Если элемент сам — `BaseModel`, `dict`, `tuple` или `list`, кидается
`AgentPinBoardExtractionError`. `node()` на списке ожидает плоских
примитивов.

## Правило 4 — вложенная `BaseModel` (или `list[BaseModel]`) без `node()`

```python
class Actor(BaseModel):
    user_arn: str | None = node(type=User, description="who", default=None)

class CloudTrailEvent(BaseModel):
    actor: Actor | None = None        # без node() — Правило 4
```

→ рекурсивно обходим вложенную модель, применяя правила к её полям.
Рёбра по-прежнему идут от той же внешней `EventNode`. Их `edge_type`
использует **класс, где поле объявлено**, поэтому `Actor.user_arn`
остаётся одинаковым именем независимо от того, в какую event-модель
встроена `Actor`.

`list[BaseModel]` обрабатывается так же: каждый элемент рекурсивно.

## Правило 5 — обычный `Field()` (без `node()` метаданных)

→ значение попадает в `EventNode.properties`. Видно агенту через
`timeline(...)`. Используется для `event_time`, `latency_ms`, raw
status кодов — значений, которые не стоит делать нодами.

## Неподдерживаемые формы (кидают `AgentPinBoardExtractionError`)

- `dict[str, BaseModel]` или любой dict-контейнер на node-поле
- `Union[NodeA, NodeB]` (Union разных нод-типов)
- `tuple` контейнеры
- Списки с разнотипными элементами
- `node(...)` на `BaseModel`-типизированном поле (кидается на этапе
  `register_model`, до runtime)

Эти ограничения намеренные — экстрактор остаётся предсказуемым,
схема графа — понятной. Большинство решается уплощением модели.

## Резолв `edge_type`

```
edge_type = "{ModelClass}.{field_name}"
```

`ModelClass` — класс, который **физически объявляет** поле (находится
обходом MRO). Это значит, что наследование и переиспользование держат
edge-метки стабильными:

```python
class Actor(BaseModel):
    user_arn: str | None = node(type=User, description="who", default=None)

class CloudTrailEvent(BaseModel):
    actor: Actor | None = None

class S3AccessLog(BaseModel):
    actor: Actor | None = None        # тот же Actor

# Оба дают рёбра типа "Actor.user_arn", независимо от внешней модели.
```

Наследование резолвится так же: если подкласс не переопределяет поле,
edge-метка использует имя базового класса.

## Защита от рекурсии

И eager-scan, и runtime-экстрактор защищены от рекурсивных моделей
вроде `Process(parent: Process | None)`. Eager-scan ведёт set
посещённых классов; runtime-обход идёт по реальным object-инстансам,
которые конечны по построению.

## Что попадает в `EventNode.properties`

`event_properties(model)` собирает каждое поле, у которого:

- нет `node()` метаданных,
- значение не `None`,
- значение **не** `BaseModel`, `list`, `dict` или `tuple`.

То есть `event_time: datetime`, `action_name: str`, `latency_ms: int`
все попадают в properties. Вложенные объекты — нет: они рекурсивно
обходятся для извлечения нод (Правило 4), а их скалярные не-node
поля попадают в properties **внешнего** события (event — единственная
нода star-топологии, несущая properties).
