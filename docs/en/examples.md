# Examples

Two complete walkthroughs: a security investigation (the canonical
PinBoard use case) and a non-security one (due-diligence-style company
lookup) to make the domain-neutrality concrete.

Both examples use mock tools instead of real APIs, so you can copy and
run them as-is.

## Example 1 — Security investigation

A mini incident-response agent that pulls CloudTrail-like records,
enriches with a VirusTotal-like lookup, and gives the LLM a graph it
can reason over.

### Entities and normalizers

```python
import ipaddress

def canonical_ip(v: str) -> str:
    return str(ipaddress.ip_address(v).compressed)

def canonical_arn(v: str) -> str:
    return v.strip()

# entities.py
from pinboard import Entity

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

### Models

```python
from datetime import datetime
from pydantic import BaseModel, Field
from pinboard import node

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
    related_domains: list[str] = []   # not extracted (no Entity for Domain in this example)
    score: int = Field(default=0, description="Risk score 0-100")
```

### Tools

```python
from datetime import datetime, timezone
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pinboard import fact, OnDuplicate

@fact(model=CloudTrailEvent, many=True)
@tool
def fetch_cloudtrail(user_arn: str, runtime: ToolRuntime) -> list[dict]:
    """Fetch the user's recent CloudTrail events."""
    # In real life: boto3_client.lookup_events(UserArn=user_arn).
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

@fact(model=VTReport, on_duplicate=OnDuplicate.SKIP)
@tool
def vt_lookup(value: str, runtime: ToolRuntime) -> dict:
    """Check an IP/domain in VirusTotal. SKIP duplicates: no point checking twice."""
    return {
        "queried": value,
        "related_ips": ["45.77.0.1", "8.8.8.8"],
        "related_domains": ["malicious.example"],
        "score": 87,
    }
```

### Wire it up

```python
from typing import Annotated
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from pinboard import LoggingHook, make_graph_tools

class State(TypedDict):
    messages: Annotated[list, add_messages]

hooks = LoggingHook()
agent_tools = [
    fetch_cloudtrail,
    vt_lookup,
    *make_graph_tools(hooks=hooks),
]

g = StateGraph(State)
g.add_node("seed", lambda s: {})
g.add_node("tools", ToolNode(agent_tools))
g.add_edge(START, "seed")
g.add_edge("seed", "tools")
g.add_edge("tools", END)
graph = g.compile(store=InMemoryStore())
```

Replace the trivial `seed` node with a real LangGraph agent (e.g.
`create_react_agent` with an LLM) for a full investigative loop. The
contract for the LLM stays the same regardless: call `fetch_*` /
`vt_lookup` to populate the graph, then `graph_summary` /
`search_nodes` / `explore` / `timeline` / `what_have_i_done` to read it.

### Drive it manually

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

# Step 1: discover what we know.
print(call("graph_summary", {}, "1"))

# Step 2: pull events.
call("fetch_cloudtrail", {"user_arn": "arn:aws:iam::123:user/admin"}, "2")

# Step 3: enrich the suspicious IP.
call("vt_lookup", {"value": "185.220.101.42"}, "3")

# Step 4: see what's connected to that IP.
print(call("explore", {"node_type": "IP", "value": "185.220.101.42"}, "4"))

# Step 5: timeline of all events involving the user.
print(call("timeline", {
    "node_type": "User",
    "value": "arn:aws:iam::123:user/admin",
}, "5"))
```

## Example 2 — Due-diligence (non-security)

Same library, no security in sight: a "company lookup" agent.

### Entities

```python
from pinboard import Entity

Company = Entity(
    name="Company",
    description="Legal entity identified by registry number",
    normalizer=lambda v: str(v).strip(),
)
Person = Entity(
    name="Person",
    description="Director, founder, or other named principal",
    normalizer=lambda v: " ".join(v.split()).strip().lower(),
)
Address = Entity(
    name="Address",
    description="Postal address",
    normalizer=lambda v: " ".join(v.split()).strip(),
)
```

### Models and tools

```python
from pydantic import BaseModel, Field
from pinboard import node, fact
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

class CompanyRecord(BaseModel):
    inn: str = node(type=Company, description="Registry number")
    name: str = Field(description="Trading name")
    director: str | None = node(
        type=Person, description="Current director / CEO", default=None,
    )
    address: str | None = node(
        type=Address, description="Registered address", default=None,
    )

class DirectorOtherCompanies(BaseModel):
    director: str = node(type=Person, description="Director queried")
    related_inns: list[str] = node(
        type=Company,
        description="Other companies where this person is on record",
        default_factory=list,
    )

@fact(model=CompanyRecord)
@tool
def lookup_company(inn: str, runtime: ToolRuntime) -> dict:
    """Look up a company by its registry number."""
    return {
        "inn": inn,
        "name": "Acme Corp",
        "director": "  John   Doe ",
        "address": "  221B Baker Street, London  ",
    }

@fact(model=DirectorOtherCompanies)
@tool
def director_other_companies(name: str, runtime: ToolRuntime) -> dict:
    """List other companies where this director appears."""
    return {
        "director": name,
        "related_inns": ["7728168971", "5024140250"],
    }
```

The flow is identical to the security example: tools populate the
graph, the LLM uses `graph_summary` / `explore` / etc. to navigate it.
The only thing that changed is the domain — `Entity` types are
arbitrary strings the user picks.

## Example 3 — Custom hook (alerting on suspicious links)

A hook that fires whenever a known "bad" IP gets linked, regardless of
which tool surfaced it.

```python
from typing import override
from pinboard import PinBoardHooks
from pinboard.models import EventId, FactNode

KNOWN_BAD = {"185.220.101.42", "45.77.0.1"}

class BadIPAlerter(PinBoardHooks):
    @override
    def on_link_found(self, existing: FactNode, event_id: EventId) -> None:
        if existing.node_type == "IP" and existing.value in KNOWN_BAD:
            print(f"!!! KNOWN BAD IP RE-OBSERVED: {existing.value} (event {event_id})")

hooks = BadIPAlerter()
```

`on_link_found` fires when an existing fact gets a new edge from a new
event — i.e. exactly when "we've seen this IP again, in a new context."
Use it for live alerting without spending tokens on the LLM.

## Example 4 — Async tools

`@fact` works identically with async tools.

```python
import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pinboard import fact

@fact(model=VTReport, on_duplicate=OnDuplicate.SKIP)
@tool
async def vt_lookup(value: str, runtime: ToolRuntime) -> dict:
    """Async VirusTotal lookup."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://api/v3/ip/{value}")
        return r.json()
```

The decorator detects async via `inspect.iscoroutinefunction` and wires
the matching async pipeline. You can mix sync and async tools in one
agent — PinBoard handles each appropriately.

## Example 5 — Full LangGraph agent with a mock LLM

`examples/agent_demo.py` ships a runnable end-to-end agent that uses
`langchain.agents.create_agent` with a deterministic mock chat model.
The mock walks a fixed plan (graph_summary → fetch_cloudtrail →
vt_lookup → explore → find_path → timeline) so the demo runs without
any API key.

Run it:

```bash
uv run python examples/agent_demo.py
```

To swap in a real LLM, replace `MockChatModel` with any LangChain
`BaseChatModel` — e.g. `ChatOpenAI(base_url="http://localhost:11434/v1", model="qwen2.5:7b")`
for a local Ollama setup, or `ChatOpenAI(model="gpt-4o-mini")` for
OpenAI. The PinBoard side is identical.

## Example 6 — LangfuseHook with Mermaid visualization

```python
from langfuse import Langfuse
from pinboard.integrations.langfuse_hook import LangfuseHook

client = Langfuse(public_key="pk-…", secret_key="sk-…", host="https://cloud.langfuse.com")

hooks = LangfuseHook(client, max_facts_in_snapshot=20)

@fact(model=CloudTrailEvent, many=True, hooks=hooks)
@tool
def fetch_cloudtrail(...): ...
```

Each call to `fetch_cloudtrail` emits:

* `pinboard.ingest` span — quantitative summary of the ingest.
* `pinboard.graph_snapshot` span — current graph as a Mermaid
  flowchart in metadata. Langfuse renders the Mermaid inline.

The Mermaid renderer is also exported standalone for ad-hoc debugging:

```python
from pinboard.integrations.langfuse_hook import render_mermaid
print(render_mermaid(my_factgraph, max_facts=15))
```
