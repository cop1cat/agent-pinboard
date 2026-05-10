# Примеры

Два полноценных walkthrough'a: security-расследование (канонический
AgentPinBoard use case) и не-security сценарий (due-diligence-style company
lookup), чтобы доменная нейтральность стала конкретной.

Оба используют mock-тулы вместо настоящих API, чтобы можно было
скопировать и запустить как есть.

## Пример 1 — Security-расследование

Мини-агент incident-response, который тянет CloudTrail-подобные
записи, обогащает через VirusTotal-подобный lookup и даёт LLM граф,
по которому она может рассуждать.

### Сущности и нормализаторы

```python
import ipaddress

def canonical_ip(v: str) -> str:
    return str(ipaddress.ip_address(v).compressed)

def canonical_arn(v: str) -> str:
    return v.strip()

# entities.py
from agent_pinboard import Entity

IP = Entity(
    name="IP",
    description="IPv4 or IPv6 network address",
    normalizer=canonical_ip,
)
User = Entity(
    name="User",
    description="Identified user or service account (ARN, email, or user ID)",
    normalizer=canonical_arn,
)
Action = Entity(
    name="Action",
    description="API action performed by an actor",
)
Resource = Entity(
    name="Resource",
    description="Cloud resource identified by ARN",
    normalizer=canonical_arn,
)
```

### Модели

```python
from datetime import datetime
from pydantic import BaseModel, Field
from agent_pinboard import node

class Actor(BaseModel):
    user_arn: str | None = node(
        type=User, description="ARN of the user that performed the action",
        default=None,
    )

class CloudTrailEvent(BaseModel):
    src_ip: str | None = node(
        type=IP, description="IP from which the API call was made",
        default=None,
    )
    actor: Actor | None = None
    action_name: str | None = node(
        type=Action, description="API action performed",
        default=None,
    )
    target_resource: str | None = node(
        type=Resource, description="Resource accessed",
        default=None,
    )
    event_time: datetime | None = Field(default=None, description="When the event occurred")

class VTReport(BaseModel):
    queried: str = node(type=IP, description="IP that was queried")
    related_ips: list[str] = node(
        type=IP, description="Related IPs reported by VirusTotal",
        default_factory=list,
    )
    related_domains: list[str] = []   # не извлекается (нет Entity для Domain)
    score: int = Field(default=0, description="Risk score 0-100")
```

### Тулы

```python
from datetime import datetime, timezone
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from agent_pinboard import pin, OnDuplicate

@pin(model=CloudTrailEvent, many=True)
@tool
def fetch_cloudtrail(user_arn: str, runtime: ToolRuntime) -> list[dict]:
    """Fetch the user's recent CloudTrail events."""
    # В реальной жизни: boto3_client.lookup_events(UserArn=user_arn).
    return [
        {
            "src_ip": "185.220.101.42",
            "actor": {"user_arn": user_arn},
            "action_name": "AssumeRole",
            "target_resource": "arn:aws:iam::123456789012:role/Admin",
            "event_time": datetime.now(timezone.utc).isoformat(),
        },
        {
            "src_ip": "185.220.101.42",
            "actor": {"user_arn": user_arn},
            "action_name": "ListBuckets",
            "target_resource": "arn:aws:s3:::sensitive-bucket",
            "event_time": datetime.now(timezone.utc).isoformat(),
        },
    ]

@pin(model=VTReport, on_duplicate=OnDuplicate.SKIP)
@tool
def vt_lookup(value: str, runtime: ToolRuntime) -> dict:
    """Check an IP/domain in VirusTotal. SKIP duplicates: проверять дважды бессмысленно."""
    return {
        "queried": value,
        "related_ips": ["45.77.0.1", "8.8.8.8"],
        "related_domains": ["malicious.example"],
        "score": 87,
    }
```

### Сборка

```python
from typing import Annotated
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from agent_pinboard import make_graph_tools

class State(TypedDict):
    messages: Annotated[list, add_messages]

agent_tools = [
    fetch_cloudtrail,
    vt_lookup,
    *make_graph_tools(),
]

g = StateGraph(State)
g.add_node("seed", lambda s: {})
g.add_node("tools", ToolNode(agent_tools))
g.add_edge(START, "seed")
g.add_edge("seed", "tools")
g.add_edge("tools", END)
graph = g.compile(store=InMemoryStore())
```

Замените тривиальный `seed`-нод на настоящего LangGraph-агента
(`create_react_agent` с LLM) для полного цикла. Контракт для LLM
не меняется: вызывает `fetch_*` / `vt_lookup` чтобы наполнить граф,
потом `graph_summary` / `search_nodes` / `explore` / `timeline` /
`what_have_i_done` чтобы его читать.

### Прогон руками

```python
def call(name, args, call_id):
    out = graph.invoke(
        {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        )]},
        config={"configurable": {"thread_id": "investigation-001"}},
    )
    return out["messages"][-1].content

# Шаг 1: discovery — что вообще можно узнать.
print(call("graph_summary", {}, "1"))

# Шаг 2: тянем события.
call("fetch_cloudtrail", {"user_arn": "arn:aws:iam::123:user/admin"}, "2")

# Шаг 3: обогащаем подозрительный IP.
call("vt_lookup", {"value": "185.220.101.42"}, "3")

# Шаг 4: что связано с этим IP.
print(call("explore", {"node_type": "IP", "value": "185.220.101.42"}, "4"))

# Шаг 5: хронология всех событий с пользователем.
print(call("timeline", {
    "node_type": "User",
    "value": "arn:aws:iam::123:user/admin",
}, "5"))
```

## Пример 2 — Due-diligence (не security)

Та же библиотека, никакой security: «company lookup» агент.

### Сущности

```python
from agent_pinboard import Entity

Company = Entity(
    name="Company",
    description="Юрлицо, идентифицируемое регистрационным номером",
    normalizer=lambda v: str(v).strip(),
)
Person = Entity(
    name="Person",
    description="Директор, основатель или иной указанный principal",
    normalizer=lambda v: " ".join(v.split()).strip().lower(),
)
Address = Entity(
    name="Address",
    description="Почтовый адрес",
    normalizer=lambda v: " ".join(v.split()).strip(),
)
```

### Модели и тулы

```python
from pydantic import BaseModel, Field
from agent_pinboard import node, pin
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

class CompanyRecord(BaseModel):
    inn: str = node(type=Company, description="ИНН / регистрационный номер")
    name: str = Field(description="Наименование")
    director: str | None = node(
        type=Person, description="Текущий директор / CEO", default=None,
    )
    address: str | None = node(
        type=Address, description="Зарегистрированный адрес", default=None,
    )

class DirectorOtherCompanies(BaseModel):
    director: str = node(type=Person, description="Запрашиваемый директор")
    related_inns: list[str] = node(
        type=Company,
        description="Другие компании, где этот человек числится",
        default_factory=list,
    )

@pin(model=CompanyRecord)
@tool
def lookup_company(inn: str, runtime: ToolRuntime) -> dict:
    """Look up a company by its registry number."""
    return {
        "inn": inn,
        "name": "Acme Corp",
        "director": "  John   Doe ",
        "address": "  221B Baker Street, London  ",
    }

@pin(model=DirectorOtherCompanies)
@tool
def director_other_companies(name: str, runtime: ToolRuntime) -> dict:
    """List other companies where this director appears."""
    return {
        "director": name,
        "related_inns": ["7728168971", "5024140250"],
    }
```

Поток идентичен security-примеру: тулы наполняют граф, LLM использует
`graph_summary` / `explore` / etc. для навигации. Изменился только
домен — `Entity`-типы это произвольные строки, которые выбирает
пользователь.

## Пример 3 — Кастомный хук (alerting на подозрительные линки)

LangChain callback handler, который срабатывает, когда уже известный
«плохой» IP залинковывается, независимо от того, какой тул его поднял.

```python
from langchain_core.callbacks import BaseCallbackHandler
from agent_pinboard.decorator import INGEST_EVENT

KNOWN_BAD = {"185.220.101.42", "45.77.0.1"}

class BadIPAlerter(BaseCallbackHandler):
    def on_custom_event(self, name, data, *, run_id, tags=None, metadata=None, **kw):
        if name != INGEST_EVENT:
            return
        first_event = data["result"].event_ids[0] if data["result"].event_ids else ""
        for fact in data["linked_facts"]:
            if fact.node_type == "IP" and fact.value in KNOWN_BAD:
                print(f"!!! KNOWN BAD IP RE-OBSERVED: {fact.value} (event {first_event})")

agent.invoke(..., config={"callbacks": [BadIPAlerter()], "configurable": {...}})
```

`linked_facts` в payload-е custom-event'а перечисляет ровно те факты,
которые этот ингест перепривязал — то есть «мы снова видим этот IP в
новом контексте». Хорошо подходит для live-алертов без траты токенов
на LLM.

## Пример 4 — Async-тулы

`@pin` работает идентично с async-тулами.

```python
import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from agent_pinboard import pin, OnDuplicate

@pin(model=VTReport, on_duplicate=OnDuplicate.SKIP)
@tool
async def vt_lookup(value: str, runtime: ToolRuntime) -> dict:
    """Async VirusTotal lookup."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://api/v3/ip/{value}")
        return r.json()
```

Декоратор детектит async через `inspect.iscoroutinefunction` и
подключает соответствующий async-pipeline. Можно мешать sync и async
тулы в одном агенте — AgentPinBoard разруливает каждый правильно.

## Пример 5 — Полный LangGraph-агент с mock LLM

`examples/agent_demo.ipynb` — runnable end-to-end агент, использующий
`langchain.agents.create_agent` с детерминированной mock chat model.
Mock проходит фиксированный план (graph_summary → fetch_cloudtrail →
vt_lookup → explore → find_path → timeline) — демо работает без
API-ключей.

Открыть в Jupyter (или прочесть прямо на GitHub — он рендерит inline):

```bash
jupyter notebook examples/agent_demo.ipynb
```

Чтобы подключить реальную LLM — замените `MockChatModel` на любую
LangChain `BaseChatModel`. Например, `ChatOpenAI(base_url="http://localhost:11434/v1", model="qwen2.5:7b")`
для локального Ollama, или `ChatOpenAI(model="gpt-4o-mini")` для
OpenAI. AgentPinBoard-сторона не меняется.

## Пример 6 — LangfuseHook с Mermaid-визуализацией

```python
from langfuse import Langfuse
from agent_pinboard.integrations.langfuse_hook import LangfuseHook

client = Langfuse(public_key="pk-…", secret_key="sk-…", host="https://cloud.langfuse.com")

handler = LangfuseHook(client, max_facts_in_snapshot=20)

result = await agent.ainvoke(
    {"messages": [...]},
    config={
        "callbacks": [handler],
        "configurable": {"thread_id": "session-42"},
    },
)
```

Каждый вызов `fetch_cloudtrail` шлёт:

* `agent_pinboard.ingest` span — количественная сводка ingest'а.
* `agent_pinboard.graph_snapshot` span — текущий граф как Mermaid-flowchart
  в metadata. Langfuse рендерит Mermaid инлайн.

Mermaid-рендерер также экспортируется отдельно для ad-hoc дебага:

```python
from agent_pinboard.integrations.langfuse_hook import render_mermaid
print(render_mermaid(my_factgraph, max_facts=15))
```
