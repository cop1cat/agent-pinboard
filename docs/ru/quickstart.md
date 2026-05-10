# Quickstart

## Установка

PinBoard пока не на PyPI. Ставьте из исходников:

```bash
git clone <repo>
cd pinboard
uv sync
```

Требования: Python 3.12+, `pydantic>=2.13`, `langgraph>=1.1.6`,
`langchain>=1.2`.

## Hello world

30-строчный пример: объявляем сущность, Pydantic-модель ответа, тул,
агента с готовыми graph-тулами, и делаем один запрос.

```python
import ipaddress
from typing import Annotated
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from typing_extensions import TypedDict

from pinboard import Entity, fact, make_graph_tools, node

# 1. Объявляем тип сущности. Описывает, какого рода вещь — нода.
IP = Entity(
    name="IP",
    description="IPv4 or IPv6 network address",
    normalizer=lambda v: str(ipaddress.ip_address(v).compressed),
)

# 2. Pydantic-модель ответа с node()-полями.
class FetchResult(BaseModel):
    src_ip: str = node(type=IP, description="IP from which the call was made")

# 3. Декорируем тул. @fact ВСЕГДА выше @tool.
@fact(model=FetchResult)
@tool
def fetch(query: str, runtime: ToolRuntime) -> dict:
    """Pretend to call an upstream API."""
    return {"src_ip": "192.168.001.001"}  # канонизируется в 192.168.1.1

# 4. Минимальный LangGraph-агент.
class State(TypedDict):
    messages: Annotated[list, add_messages]

g = StateGraph(State)
g.add_node("seed", lambda s: {})
g.add_node("tools", ToolNode([fetch, *make_graph_tools()]))
g.add_edge(START, "seed")
g.add_edge("seed", "tools")
g.add_edge("tools", END)
graph = g.compile(store=InMemoryStore())

# 5. Запускаем тул, потом спрашиваем граф, что он нашёл.
def run(name: str, args: dict, call_id: str) -> str:
    out = graph.invoke(
        {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        )]},
        config={"configurable": {"thread_id": "demo"}},
    )
    return out["messages"][-1].content

run("fetch", {"query": "x"}, "1")
print(run("graph_summary", {}, "2"))
print(run("search_nodes", {"node_type": "IP"}, "3"))
```

Ожидаемый вывод (таймстемпы и id отличаются):

```
graph_summary:
  IP (1 in graph) — IPv4 or IPv6 network address

search_nodes(node_type='IP', pattern=None):
  IP: 192.168.001.001  (in 1 events, via ['fetch'])
```

IP сохранён под канонической формой (`192.168.1.1`), так что повторный
вызов с `192.168.1.1` слинкуется к этой же ноде, а не создаст новую.

## Что произошло

1. `@fact(model=FetchResult)` применён к тулу `fetch`. На этапе
   декорирования PinBoard просканировал модель и зарегистрировал
   `Entity` IP в session-registry.
2. Первый вызов `fetch` вернул `{"src_ip": "192.168.001.001"}`.
   PinBoard прогнал через `FetchResult.model_validate`, создал
   `EventNode` для вызова, прогнал значение через `IP.normalizer`
   (`canonical_ip`), создал одну `FactNode` типа `IP`.
3. `graph_summary` показывает все известные типы (из registry) +
   количество в живом графе.
4. `search_nodes(node_type="IP")` листинг IP-фактов.

## Куда дальше

- [Концепты](./concepts.md) — модель в голове: `Entity` vs `node()` vs
  `FactNode` vs `EventNode`.
- [Примеры](./examples.md) — полноценные агенты с несколькими тулами,
  хуками и изоляцией сессий.
