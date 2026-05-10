"""End-to-end AgentPinBoard agent demo with a deterministic mock LLM.

What this shows:
* Defining `Entity`-s, Pydantic response models, and `@pin`-decorated tools.
* Wiring everything into a LangGraph agent via `create_agent`.
* Driving the agent with a mock `BaseChatModel` so the demo runs without
  any LLM provider — handy for tests, CI, and offline development.

Replace ``MockChatModel`` with any OpenAI-compatible client (Ollama,
vLLM, OpenAI, Anthropic via langchain-anthropic, etc.) for a real agent.
The AgentPinBoard side is identical.

Run:
    uv run python examples/agent_demo.py
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from agent_pinboard import Entity, make_graph_tools, node, pin

# --------------------------------------------------------------------------- #
# 1. Domain model: entities + Pydantic schema                                  #
# --------------------------------------------------------------------------- #

def canonical_ip(v: str) -> str:
    return str(ipaddress.ip_address(v).compressed)


IP = Entity(
    name="IP",
    description="IPv4 or IPv6 network address",
    normalizer=canonical_ip,
)
User = Entity(
    name="User",
    description="Identified user or service account",
    normalizer=lambda v: str(v).strip(),
)
Action = Entity(
    name="Action",
    description="API action performed by an actor",
)


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
    event_time: datetime | None = Field(default=None, description="When event occurred")


class VTReport(BaseModel):
    queried: str = node(type=IP, description="IP queried in VirusTotal")
    related_ips: list[str] = node(
        type=IP, description="Related IPs reported by VT",
        default_factory=list,
    )
    score: int = Field(default=0, description="Risk score 0-100")


# --------------------------------------------------------------------------- #
# 2. Tools                                                                     #
# --------------------------------------------------------------------------- #

@pin(model=CloudTrailEvent, many=True)
@tool
def fetch_cloudtrail(user_arn: str, runtime: ToolRuntime) -> list[dict]:
    """Fetch the user's recent CloudTrail events."""
    return [
        {
            "src_ip": "185.220.101.42",
            "actor": {"user_arn": user_arn},
            "action_name": "AssumeRole",
            "event_time": datetime.now(UTC).isoformat(),
        },
        {
            "src_ip": "185.220.101.42",
            "actor": {"user_arn": user_arn},
            "action_name": "ListBuckets",
            "event_time": datetime.now(UTC).isoformat(),
        },
    ]


@pin(model=VTReport)
@tool
def vt_lookup(value: str, runtime: ToolRuntime) -> dict:
    """Check an IP, domain, or hash in VirusTotal."""
    return {
        "queried": value,
        "related_ips": ["45.77.0.1", "8.8.8.8"],
        "score": 87,
    }


# --------------------------------------------------------------------------- #
# 3. Mock LLM that walks a deterministic plan                                  #
# --------------------------------------------------------------------------- #

class MockChatModel(BaseChatModel):
    """Returns a hard-coded sequence of tool calls, then a final answer.

    Simulates a ReAct loop: each step inspects how many tool messages are
    in the conversation history and picks the next action.
    """

    plan: list[dict[str, Any]] = []

    def __init__(self, plan: list[dict[str, Any]]) -> None:
        super().__init__()
        # Field assignment after BaseModel __init__.
        object.__setattr__(self, "plan", plan)

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Count how many ToolMessage replies have come back so far —
        # that's our cursor into the plan.
        cursor = sum(1 for m in messages if isinstance(m, ToolMessage))
        if cursor < len(self.plan):
            step = self.plan[cursor]
            ai = AIMessage(
                content="",
                tool_calls=[{
                    "name": step["tool"],
                    "args": step["args"],
                    "id": f"call-{cursor}",
                    "type": "tool_call",
                }],
            )
        else:
            ai = AIMessage(content="Investigation complete.")
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def bind_tools(self, tools: list[Any], **_kwargs: Any) -> MockChatModel:
        # Real models need this to learn the tool schemas; the mock
        # already knows what to do, so it's a no-op.
        return self


# --------------------------------------------------------------------------- #
# 4. Wire and run                                                              #
# --------------------------------------------------------------------------- #

def build_agent(plan: list[dict[str, Any]]) -> Any:
    """Compose the agent. Swap MockChatModel for any real LangChain chat model."""
    llm = MockChatModel(plan=plan)
    tools = [fetch_cloudtrail, vt_lookup, *make_graph_tools()]
    return create_agent(llm, tools, store=InMemoryStore())


def main() -> None:
    plan = [
        {"tool": "graph_summary", "args": {}},
        {"tool": "fetch_cloudtrail", "args": {"user_arn": "arn:aws:iam::123:user/admin"}},
        {"tool": "vt_lookup", "args": {"value": "185.220.101.42"}},
        {"tool": "explore", "args": {"node_type": "IP", "value": "185.220.101.42"}},
        {"tool": "find_path", "args": {
            "from_type": "IP", "from_value": "185.220.101.42",
            "to_type": "IP", "to_value": "8.8.8.8",
        }},
        {"tool": "timeline", "args": {
            "node_type": "User", "value": "arn:aws:iam::123:user/admin",
        }},
    ]

    agent = build_agent(plan)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Investigate suspicious AssumeRole."}]},
        config={"configurable": {"thread_id": "investigation-001"}},
    )

    for msg in _interesting_messages(result["messages"]):
        print("─" * 70)
        print(msg)


def _interesting_messages(messages: list[BaseMessage]) -> Iterator[str]:
    """Skip the empty AIMessages that just dispatch tool calls; show the rest."""
    for m in messages:
        if isinstance(m, ToolMessage):
            yield f"[tool: {m.name}]\n{m.content}"
        elif isinstance(m, AIMessage) and m.content:
            yield f"[assistant]\n{m.content}"
        elif not isinstance(m, AIMessage):
            yield f"[{m.type}]\n{m.content}"


if __name__ == "__main__":
    main()
