# Типичные ловушки

То, на что натыкаются с первого раза. Запомните — каждая из них —
часовая дебаг-сессия, если не знаете ответ заранее.

## 1. Порядок декораторов

**`@pin` всегда выше `@tool`.**

```python
# ✓ правильно
@pin(model=X)
@tool
def f(...): ...

# ✗ кидает AgentPinBoardConfigError при декорировании
@tool
@pin(model=X)
def f(...): ...
```

`@tool` создаёт `BaseTool`; `@pin` его оборачивает. Обратный порядок
— и `@pin` получает функцию вместо Tool. Спека реджектит сразу,
чтобы не получить непонятный traceback позже.

## 2. `node()` против `Field()`

Поле — нода тогда и только тогда, когда вы объявили его через
`node(...)`. Plain `Field(...)` (или вообще без аннотации) означает
«это скалярное значение пойдёт в `EventNode.properties`».

Быстрое правило:

> **Может встречаться в разных событиях?** → `node(...)`.
> **One-shot скаляр (timestamp, latency, raw status)?** → `Field(...)`.

## 3. Два `Entity` с одинаковым именем

Session-registry индексируется по `Entity.name`. Если две части кода
независимо создают `Entity(name="IP", ...)` — даже с одинаковыми
атрибутами — будет warning и побеждает первая регистрация.

**Правильно**: объявите каждую `Entity` один раз в модуле проекта
(обычно `entities.py`) и импортируйте везде. Не копипастьте
объявления `Entity`.

## 4. Identity нормализатора (ловушка «два `canonical_ip` файла»)

Equality `Entity` включает `normalizer`-callable, сравнение по
identity. Если вы импортируете `canonical_ip` в `models_a.py` и
скопипастили тот же код в `models_b.py` — это два разных Python-объекта
→ две `Entity` с одинаковым `name` не равны → registry warns про
коллизию.

**Правильно**: импортируйте нормализаторы из одного общего модуля.

## 5. `node()` на `BaseModel`-типизированном поле

```python
class Inner(BaseModel):
    x: str

class Bad(BaseModel):
    inner: Inner = node(type=SomeEntity, description="...")  # реджектится
```

Реджектится в `register_model` через `AgentPinBoardConfigError`. `BaseModel`
— структурное значение, не лист. Делать его нодой семантически
неверно (и в старых версиях молча уходило в Правило 4). Если хотите,
чтобы `Inner.x` стало нодой — пометьте **то** поле через `node()`,
а Правило 4 пусть рекурсивно зайдёт в `Inner`.

## 6. Семантика `skip_events` в `explore`

`skip_events=True` (дефолт) означает **события — прозрачные
коннекторы, не хопы**. `explore("IP", "1.2.3.4", depth=1, skip_events=True)`
вернёт каждый факт, который делит **любое событие** с этим IP —
визуально ноль хопов, семантически «напрямую связано».

`skip_events=False` обходит нативный `MultiDiGraph`, где каждая
EventNode потребляет хоп. FactNode → её EventNode → другая FactNode
— это два хопа.

Если хочется «какие tool-вызовы коснулись этого IP» — берите
`timeline(...)` или `what_have_i_done(node_type="IP", value=...)`.
Они показывают события напрямую, без обхода графа.

## 7. Граф собран без Store

```python
graph = builder.compile()  # ✗ нет store=
```

Каждый `@pin`-вызов нуждается в `runtime.store`, который выставляется
только при `.compile(store=...)`. Без него первый же tool-вызов
кидает `AgentPinBoardConfigError("graph must be compiled with .compile(store=...)")`.

```python
from langgraph.store.memory import InMemoryStore
graph = builder.compile(store=InMemoryStore())  # ✓
```

## 8. Отсутствует `thread_id`

Если `runtime.config.configurable.thread_id` не задан — AgentPinBoard
генерирует UUID4 на каждый вызов и пишет warning. Два параллельных
«анонимных» вызова получают свои сессии — друг друга не видят.

Это намеренное поведение (silent merge был бы хуже), но warning
легко пропустить в Jupyter-логах. Всегда передавайте `thread_id`
явно:

```python
graph.invoke(
    {...},
    config={"configurable": {"thread_id": "investigation-001"}},
)
```

## 9. Тул возвращает Pydantic-инстанс или dict

Оба варианта работают:

```python
@pin(model=CloudTrailEvent, many=True)
@tool
def fetch(...) -> list[dict]:
    return [{"src_ip": "1.1.1.1"}, ...]    # валидация через model_validate

@pin(model=CloudTrailEvent, many=True)
@tool
def fetch(...) -> list[CloudTrailEvent]:
    return [CloudTrailEvent(src_ip="1.1.1.1"), ...]  # валидация пропускается
```

Но type annotation в сигнатуре `@tool` влияет на то, что видит
LangChain/LLM — держите её точной.

## 10. `fnmatchcase` case-sensitive

`search_nodes(node_type="IP", value_pattern="abc*")` НЕ найдёт ноду
с canonical value `"ABC123"`. Используем `fnmatch.fnmatchcase` для
кросс-OS детерминированного поведения.

Хотите case-insensitive — настройте нормализатор так, чтобы
канонизация уходила в lower-case, потом сравнивайте в lower:

```python
def canonical_email(v: str) -> str:
    return v.strip().lower()
```

## 11. `mask_args` и ротирующиеся секреты

Два вызова с разными настоящими секретами но одинаковым `mask_args`
дадут одинаковый `args_repr` и dedup-коллизию. Если секрет ротируется
внутри сессии — либо:

- передавайте секрет через `runtime.config` (вообще вне `args_repr`),
  либо
- используйте `on_duplicate=OnDuplicate.ALWAYS`, чтобы dedup не
  срабатывал.

## 12. `response_transform` async/sync mismatch

Sync-тул с async `response_transform` реджектится при декорировании.
Результат нужен синхронно, и нет event loop'a, чтобы await transform.
Согласуйте режимы — sync-тул ↔ sync transform, async-тул ↔
async-or-sync transform.

## 13. Видимость EventNode в `search_nodes`

По дефолту `search_nodes` скрывает EventNode. Если агент пытается
найти tool-call записи через glob — передавайте `include_events=True`:

```python
search_nodes(node_type="Event", value_pattern="fetch_*", include_events=True)
```

Для не-glob запросов (просто «какие tool-вызовы были») — лучше
`what_have_i_done(...)`. Он фильтрует структурированный лог и
возвращает больше полезной инфы.
