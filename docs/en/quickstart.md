# Quickstart

## Install

PinBoard is not on PyPI yet. Install from source:

```bash
git clone <repo>
cd pinboard
uv sync
```

Requirements: Python 3.12+, `pydantic>=2.13`, `langgraph>=1.1.6`,
`langchain>=1.2`.

## Hello world

A 30-line example that defines an entity, a Pydantic response model, a
tool, an agent built with the prebuilt graph tools, and runs one query.

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

# 1. Declare an entity type. Defines what kind of thing a node represents.
IP = Entity(
    name="IP",
    description="IPv4 or IPv6 network address",
    normalizer=lambda v: str(ipaddress.ip_address(v).compressed),
)

# 2. Pydantic response model with node()-marked fields.
class FetchResult(BaseModel):
    src_ip: str = node(type=IP, description="IP from which the call was made")

# 3. Decorate a tool. @fact must always be ABOVE @tool.
@fact(model=FetchResult)
@tool
def fetch(query: str, runtime: ToolRuntime) -> dict:
    """Pretend to call an upstream API."""
    return {"src_ip": "192.168.001.001"}  # canonicalises to 192.168.1.1

# 4. Wire a minimal LangGraph agent.
class State(TypedDict):
    messages: Annotated[list, add_messages]

g = StateGraph(State)
g.add_node("seed", lambda s: {})
g.add_node("tools", ToolNode([fetch, *make_graph_tools()]))
g.add_edge(START, "seed")
g.add_edge("seed", "tools")
g.add_edge("tools", END)
graph = g.compile(store=InMemoryStore())

# 5. Run the tool, then ask the graph what it found.
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

Expected output (timestamps and IDs vary):

```
graph_summary:
  IP (1 in graph) — IPv4 or IPv6 network address

search_nodes(node_type='IP', pattern=None):
  IP: 192.168.001.001  (in 1 events, via ['fetch'])
```

The IP was stored under its canonical form (`192.168.1.1`), so a second
call with `192.168.1.1` would link to the same node instead of creating
a new one.

## What just happened

1. `@fact(model=FetchResult)` was applied to the `fetch` tool. At
   decoration time PinBoard scanned the model and registered the `IP`
   entity in the session registry.
2. The first call to `fetch` returned `{"src_ip": "192.168.001.001"}`.
   PinBoard validated it against `FetchResult`, created an `EventNode`
   for the call, ran the `IP` field's normalizer (`canonical_ip`), and
   inserted a single `FactNode` of type `IP`.
3. `graph_summary` shows known types (from the registry) plus their
   counts in the live graph.
4. `search_nodes(node_type="IP")` lists the `IP` facts.

## Where to next

- [Concepts](./concepts.md) for the mental model of `Entity` vs `node()`
  vs `FactNode` vs `EventNode`.
- [Examples](./examples.md) for fuller agents with multiple tools, hooks,
  and per-session isolation.
