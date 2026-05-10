"""Test helpers: drive @pin-decorated tools via a tiny LangGraph."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


def make_runner(tools: list[BaseTool], store: BaseStore, *, async_mode: bool = False):
    """Compile a graph that drives ToolNode with messages we hand-craft."""
    tool_node = ToolNode(tools)

    def seed(state: State, config: RunnableConfig) -> dict:
        return {}

    g = StateGraph(State)
    g.add_node("seed", seed)
    g.add_node("tools", tool_node)
    g.add_edge(START, "seed")
    g.add_edge("seed", "tools")
    g.add_edge("tools", END)
    return g.compile(store=store)


def call(
    graph,
    tool_name: str,
    args: dict,
    thread_id: str,
    *,
    call_id: str = "c-1",
    callbacks: list | None = None,
) -> str:
    config: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    out = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}
                    ],
                )
            ]
        },
        config=config,
    )
    return out["messages"][-1].content


async def acall(
    graph,
    tool_name: str,
    args: dict,
    thread_id: str,
    *,
    call_id: str = "c-1",
    callbacks: list | None = None,
) -> str:
    config: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    out = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}
                    ],
                )
            ]
        },
        config=config,
    )
    return out["messages"][-1].content
