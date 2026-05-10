# Хуки и конфигурация

## Хуки

`PinBoardHooks` — обычный класс. Наследуйтесь и переопределяйте
только те методы, которые нужны. Каждый callback обёрнут в
`try/except`: **хук, который кинул, никогда не ломает ingestion** —
ошибка логируется на ERROR, ingestion продолжается.

```python
from typing import override
from pinboard import PinBoardHooks
from pinboard.models import EventId, FactNode, IngestResult

class MyHook(PinBoardHooks):
    @override
    def on_node_added(self, node) -> None:
        print(f"new node: {node.node_type}")

    @override
    def on_link_found(self, existing: FactNode, event_id: EventId) -> None:
        print(f"linked existing: {existing.value}")

    @override
    def on_ingest_complete(self, result: IngestResult) -> None:
        print(f"+{result.new_nodes} nodes, {len(result.warnings)} warnings")
```

`@typing.override` — декоратор Python 3.12. Typechecker поймает
опечатки в именах переопределяемых методов.

### Доступные коллбэки

| Callback | Когда вызывается |
|---|---|
| `on_node_added(node)` | Создан новый `FactNode` или `EventNode` |
| `on_edge_added(edge)` | Создан новый `FactEdge` |
| `on_link_found(existing, event_id)` | Существующий `FactNode` залинкован новым событием (один вызов на каждый distinct linked факт за ingest) |
| `on_ingest_complete(result)` | Один вызов `@fact` отработал успешно |
| `on_graph_changed()` | Грубый сигнал «граф изменился», один раз за ingest |

### Готовые реализации

```python
from pinboard import LoggingHook, CompositeHook
import logging

# Логирует каждый callback на INFO.
log_hook = LoggingHook(level=logging.INFO)

# Раскидывает на несколько хуков; каждый изолирован try/except.
combined = CompositeHook([log_hook, MyHook()])
```

`LangfuseHook` и `WebSocketHook` поставляются как опциональные
интеграции — см. ниже обе.

### `LangfuseHook`

Опциональная зависимость. Установка:

```bash
uv add 'pinboard[langfuse]'        # или: pip install pinboard[langfuse]
```

Использование:

```python
from langfuse import Langfuse
from pinboard.integrations.langfuse_hook import LangfuseHook

client = Langfuse(public_key=..., secret_key=..., host=...)
hooks = LangfuseHook(client)

@fact(model=CloudTrailEvent, many=True, hooks=hooks)
@tool
def fetch_cloudtrail(...): ...
```

Что эмитит:

* На каждый `on_ingest_complete` — Langfuse span `pinboard.ingest`
  с per-call дельтой (`new_nodes`, `linked_nodes`, `new_edges`,
  warnings).
* На каждый `on_graph_changed` — span `pinboard.graph_snapshot`,
  чья metadata содержит Mermaid-flowchart с top-фактами и связывающими
  их событиями. Langfuse рендерит Mermaid в metadata — получаете
  визуальный граф рядом с trace.

Параметры конструктора:

* `max_facts_in_snapshot=30` — сколько top-фактов (по числу событий)
  включать в каждый Mermaid-снимок.
* `emit_snapshots=False` — выключить снимки, оставить только ingest
  spans (дешевле, меньше traffic'a в Langfuse).

Хук никогда не падает — ошибки логируются на ERROR (контракт
`PinBoardHooks` log-and-continue сохранён).

### `WebSocketHook`

Опциональная зависимость. Установка:

```bash
uv add 'pinboard[ws]'        # или: pip install pinboard[ws]
```

Хук собирает каждое изменение графа в thread-safe очередь;
``serve_websocket(hook, ...)`` поднимает asyncio WebSocket-сервер,
который рассылает каждую дельту (и one-off snapshot на коннект)
всем подключённым клиентам.

```python
import asyncio
from pinboard import fact, make_graph_tools
from pinboard.integrations.websocket_hook import (
    WebSocketHook, serve_websocket,
)

hook = WebSocketHook(thread_id_label="investigation-001")

@fact(model=CloudTrailEvent, many=True, hooks=hook)
@tool
def fetch_cloudtrail(...): ...

async def main():
    server = asyncio.create_task(serve_websocket(hook, port=8765))
    # ... драйвите агента (sync-работа через asyncio.to_thread) ...
    await server

asyncio.run(main())
```

Wire-формат (JSON, одно сообщение на строку):

* `snapshot` — полный дамп графа на коннект.
* `node_added` / `edge_added` — инкрементальные изменения.
* `link_found` — существующий факт перелинкован новым событием.
* `ingest_complete` — `@fact`-вызов отработал успешно.

Готовый Cytoscape.js фронтенд — `examples/web/index.html`,
запускалка — `examples/web/server_demo.py`. Запустите и откройте
HTML в браузере, чтобы видеть как граф строится в реальном времени.

Хук не падает; как и остальные, исключения WS-слоя логируются и
проглатываются.

### Подключение хуков к тулу

Передавайте хук в `@fact` (per-tool) и в `make_graph_tools` (для
read-тулов, где они пока no-op):

```python
hooks = MyHook()

@fact(model=CloudTrailEvent, many=True, hooks=hooks)
@tool
def fetch_cloudtrail(...): ...

agent_tools = [fetch_cloudtrail, *make_graph_tools(hooks=hooks)]
```

## `configure()` — глобальные настройки процесса

```python
from pinboard import configure

configure(tool_log_soft_limit=200)
```

Единственная настройка в Phase 1 — `tool_log_soft_limit` (дефолт 500).
Когда per-session tool-call лог превышает порог — пишется warning,
жёсткого cap нет. Warning — в основном сигнал «LLM ходит по кругу»,
не «storage перегружен».

`configure()` — **process-global, mutable state**. Per-session
override вне scope; для них — пишите хук, который дропает записи.

## Tool log

Каждый `@fact`-вызов добавляет один `ToolCallRecord` в per-session
лог под namespace `("pinboard", thread_id, "tool_calls", record_id)`:

```python
@dataclass(slots=True, frozen=True)
class ToolCallRecord:
    tool_name: str
    args_repr: str           # canonical JSON, детерминированный для dedup
    timestamp: datetime
    event_id: EventId | None # None для дубликатов, не сделавших ingest
    summary: str             # "+2 nodes, +1 linked, +3 edges" / "duplicate (skipped)" / "error: ..."
    duration_ms: int
```

Агент читает лог через `what_have_i_done(...)`. Две причины пользы:

1. LLM может спросить «я уже запрашивал VirusTotal на этот IP?» без
   re-run'а вызова. Вместе с `on_duplicate=OnDuplicate.SKIP` второй
   вызов просто вернёт маркер-строку.
2. После долгой сессии можно сделать post-mortem: какие тулы шли,
   в каком порядке, что они дали.

### Представление аргументов

`args_repr` — стабильная JSON-строка из позиционных и keyword-аргументов
вызова, с:

- исключённым `ToolRuntime` (не сериализуемый, и per-session всё
  равно),
- отсортированными `kwargs` (`f(a=1, b=2)` и `f(b=2, a=1)` дают
  одинаковую строку),
- Pydantic `BaseModel` через `model_dump(mode="json")`,
- остальное — через `json.dumps(..., default=str)`.

Это то, против чего сравнивается duplicate-detection (`on_duplicate`).

### Маскировка секретов

Если ваш тул принимает API-токен или пароль, исключите его из лога
через `mask_args`:

```python
@fact(model=VTReport, mask_args=["api_key"])
@tool
def vt_lookup(value: str, api_key: str, runtime: ToolRuntime) -> dict:
    """."""
    ...
```

Замаскированные аргументы появляются как `"***"` в `args_repr`.
**Caveat**: ротирующийся секрет будет выглядеть как предыдущий
секрет в логе, два вызова с разными настоящими ключами окажутся
dedup-эквивалентными. Если ротируете ключи внутри сессии — либо
передавайте через `runtime.config` (не через args), либо ставьте
`on_duplicate=OnDuplicate.ALWAYS`.
