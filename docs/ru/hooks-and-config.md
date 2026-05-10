# Колбэки и конфигурация

## Observability через LangChain callbacks

AgentPinBoard включается в **стандартный callback-канал LangChain**.
После каждого успешного `@pin`-ингеста декоратор диспатчит custom
event `agent_pinboard:ingest` во все `BaseCallbackHandler`-ы, которые
caller зарегистрировал через `config={"callbacks": [...]}` на
`agent.invoke` / `ainvoke`.

Тот же механизм, который отдаёт `on_tool_start` / `on_tool_end` /
`on_llm_start`, отдаёт и события AgentPinBoard — handler получает
единый поток, и каждый event привязан к LangChain-run'у, в котором
произошёл (так что Langfuse-спаны естественно вкладываются в
tool-span, а не висят отдельным трейсом).

### Минимальный handler

```python
from langchain_core.callbacks import BaseCallbackHandler
from agent_pinboard.decorator import INGEST_EVENT

class PrintIngest(BaseCallbackHandler):
    def on_custom_event(self, name, data, *, run_id, tags=None, metadata=None, **kw):
        if name != INGEST_EVENT:
            return
        result = data["result"]
        print(
            f"{data['tool_name']}: +{result.new_nodes} new, "
            f"+{result.linked_nodes} linked, +{result.new_edges} edges"
        )
```

Подключение на каждом invoke:

```python
agent.invoke(
    {"messages": [...]},
    config={
        "configurable": {"thread_id": "session-42"},
        "callbacks": [PrintIngest()],
    },
)
```

### Payload `agent_pinboard:ingest`

`data` содержит дельту ингеста и ссылку на свежезагруженный граф:

| Ключ | Тип | Заметки |
|---|---|---|
| `thread_id` | `str` | Сессия, в которую упал ингест |
| `tool_name` | `str` | Имя задекорированного тула |
| `result` | `IngestResult` | `event_ids`, `new_nodes`, `linked_nodes`, `new_edges`, `warnings` |
| `events` | `list[EventNode]` | Один на вызов (или один на элемент при `many=True`) |
| `new_facts` | `list[FactNode]` | Свежесозданные факты |
| `linked_facts` | `list[FactNode]` | Существующие факты, на которые этот ингест навесил линк |
| `new_edges` | `list[FactEdge]` | По одному на каждое occurrence факта в модели |
| `graph` | `FactGraph` | Граф после ингеста (in-memory, view этого вызова) |

Handler, которому нужна гранулярность по нодам, проходит по `events`
/ `new_facts` / `linked_facts`; handler, которому нужен общий сигнал
"что-то изменилось" — смотрит на `result`.

### Изоляция ошибок

Декоратор оборачивает `dispatch_custom_event` в `try/except`: handler,
который кинул исключение, **не ломает ingestion** — exception
логируется на ERROR. Payload `agent_pinboard:ingest` всегда отражает
успешно записанную дельту, даже если handler потом упадёт.

## `LangfuseHook`

Опциональная зависимость:

```bash
uv add 'agent_pinboard[langfuse]'        # или: pip install agent_pinboard[langfuse]
```

```python
from langfuse import Langfuse
from agent_pinboard.integrations.langfuse_hook import LangfuseHook

client = Langfuse(public_key=..., secret_key=..., host=...)
handler = LangfuseHook(client)

result = await agent.ainvoke(
    {"messages": [...]},
    config={
        "callbacks": [handler],
        "configurable": {"thread_id": "session-42"},
    },
)
```

Что эмитит:

* span `agent_pinboard.ingest` на каждый ингест — с per-call дельтой
  (`new_nodes`, `linked_nodes`, `new_edges`, warnings).
* (опционально, по умолчанию включено) span
  `agent_pinboard.graph_snapshot` на каждый ингест с Mermaid-схемой
  топ-N фактов и связанных событий в metadata. Langfuse рендерит
  Mermaid в metadata — получаете визуальный граф рядом с трейсом.

Оба спана наследуют parent от текущего LangChain tool-span — дерево
трейса остаётся связным.

Параметры конструктора:

* `max_facts_in_snapshot=30` — top-N фактов в каждом Mermaid-рендере.
* `emit_snapshots=False` — отключить snapshot-span (дешевле, меньше
  трафика в Langfuse).

Handler глотает свои исключения — failure логируется на ERROR, agent
run продолжается.

## `WebSocketHook`

Опциональная зависимость:

```bash
uv add 'agent_pinboard[ws]'        # или: pip install agent_pinboard[ws]
```

Handler превращает каждый `agent_pinboard:ingest` в поток JSON-дельт
(одна на ноду, ребро, link, плюс финальный `ingest_complete`) и
кладёт в thread-safe очередь. `serve_websocket(handler, ...)` запускает
asyncio WebSocket-сервер, который раздаёт каждую дельту всем
подключённым клиентам.

```python
import asyncio
from langchain.agents import create_agent
from agent_pinboard import pin, make_graph_tools
from agent_pinboard.integrations.websocket_hook import (
    WebSocketHook, serve_websocket,
)

handler = WebSocketHook(thread_id_label="investigation-001")

async def main():
    server = asyncio.create_task(serve_websocket(handler, port=8765))
    agent = create_agent(...)
    await asyncio.to_thread(
        agent.invoke,
        {"messages": [...]},
        {
            "configurable": {"thread_id": "investigation-001"},
            "callbacks": [handler],
        },
    )
    await server

asyncio.run(main())
```

Wire-формат (JSON, по сообщению на строку):

* `snapshot` — полный дамп графа на connect.
* `node_added` / `edge_added` — инкрементальные изменения.
* `link_found` — линк к существующему факту из нового события.
* `ingest_complete` — `@pin`-вызов завершился успешно.

Готовый Cytoscape.js-фронтенд лежит в `examples/web/index.html`,
живой demo-ноутбук `examples/web/server_demo.ipynb` собирает всё
вместе — запустите его и откройте `http://localhost:8765/` в браузере,
чтобы видеть, как граф растёт в реальном времени.

## `configure()` — глобальные настройки процесса

```python
from agent_pinboard import configure

configure(tool_log_soft_limit=200)
```

Единственная настройка в Phase 1 — `tool_log_soft_limit` (по
умолчанию 500). Когда per-session tool-call log превышает порог,
пишется warning; жёсткого cap нет. Warning — сигнал что "LLM ходит
по кругу", а не что storage перегружен.

`configure()` — **process-global mutable state**. Per-session
override out of scope; для этого пишите callback-handler, фильтрующий
записи.

## Tool log

Каждый `@pin`-вызов добавляет один `ToolCallRecord` в per-session
log под namespace
`("agent_pinboard", thread_id, "tool_calls", record_id)`:

```python
@dataclass(slots=True, frozen=True)
class ToolCallRecord:
    tool_name: str
    args_repr: str           # canonical JSON, deterministic для дедупа
    timestamp: datetime
    event_id: EventId | None # None для duplicates без ingestion
    summary: str             # "+2 nodes, +1 linked, +3 edges" / "duplicate (skipped)" / "error: ..."
    duration_ms: int
```

Агент читает log через `what_have_i_done(...)`. Зачем log:

1. LLM может спросить "уже искал этот IP в VirusTotal?" не вызывая
   тул заново. С `on_duplicate=OnDuplicate.SKIP` дублирующий вызов
   возвращает marker-строку.
2. Post-mortem длинной сессии — какие тулы шли в каком порядке и что
   возвращали.

### Представление аргументов

`args_repr` — стабильная JSON-строка, построенная из позиционных и
keyword-аргументов вызова, с:

- `ToolRuntime` исключён (несериализуем, и per-session всё равно),
- ключи `kwargs` сортируются (`f(a=1, b=2)` и `f(b=2, a=1)`
  столкнутся при дедупликации),
- Pydantic `BaseModel` дампятся через `model_dump(mode="json")`,
- остальное — через `json.dumps(..., default=str)`.

Это то, против чего сравнивается duplicate-detection (`on_duplicate`).

### Маскировка секретов

Если ваш тул принимает API-токен или пароль, исключите его из лога
через `mask_args`:

```python
@pin(model=VTReport, mask_args=["api_key"])
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

## Multi-process / production storage

`InMemoryStore` подходит для тестов и однопроцессных демо, но он
теряет всё при выходе из процесса и не разделяется между воркерами.
В продакшене используйте shared backend — LangGraph поставляет async
PostgreSQL store, с которым AgentPinBoard работает без дополнительной
обвязки:

```python
from langgraph.store.postgres import AsyncPostgresStore
from langchain.agents import create_agent

async with AsyncPostgresStore.from_conn_string(
    "postgresql://user:pass@host:5432/db"
) as store:
    await store.setup()  # один раз создаёт таблицы

    agent = create_agent(
        model=llm,
        tools=[*my_tools, *make_graph_tools()],
        store=store,
    )
    result = await agent.ainvoke(
        {"messages": [...]},
        config={"configurable": {"thread_id": "session-42"}},
    )
```

AgentPinBoard **не** держит process-local кэш графа — каждый ингест
через `@pin` и каждый read-tool делают свежий `load_graph` из Store.
В сочетании с mergeable-сериализацией `FactNode` (на диске лежит
только иммутабельная подножка; провенанс выводится из рёбер + EventNode
при загрузке) это значит, что два воркера на разных процессах могут
параллельно ингестить в один `thread_id` без потери ссылок друг друга.

Per-`thread_id` `threading.RLock` всё ещё сериализует
read-modify-write окно ингеста внутри одного процесса — чтобы два
потока одного воркера не гонялись на своём reload+persist цикле.
Cross-process distributed lock не нужен: storage-модель mergeable
by construction.
