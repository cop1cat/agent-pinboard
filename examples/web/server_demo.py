"""Live AgentPinBoard demo with WebSocket streaming.

Run::

    uv run python examples/web/server_demo.py

Then open ``examples/web/index.html`` in a browser. The page connects
to ``ws://localhost:8765`` and renders the graph in Cytoscape.js with
live deltas as the agent populates it.

Uses a deterministic ``MockChatModel`` so it works without an LLM
provider — replace it with any LangChain ``BaseChatModel`` to drive a
real agent.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow `from examples.agent_demo import MockChatModel` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.prebuilt import ToolRuntime  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from agent_pinboard import make_graph_tools, pin  # noqa: E402
from agent_pinboard.integrations.websocket_hook import (  # noqa: E402
    WebSocketHook,
    serve_websocket,
)
from examples.agent_demo import (  # noqa: E402
    CloudTrailEvent,
    MockChatModel,
    VTReport,
)

# Hook is shared between @pin tools and make_graph_tools — every change
# to the graph (extraction, link-found, summary) flows into the WS queue.
HOOK = WebSocketHook(thread_id_label="demo-live")


@pin(model=CloudTrailEvent, many=True, hooks=HOOK)
@tool
def fetch_cloudtrail(user_arn: str, runtime: ToolRuntime) -> list[dict]:
    """Mock CloudTrail fetch."""
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


@pin(model=VTReport, hooks=HOOK)
@tool
def vt_lookup(value: str, runtime: ToolRuntime) -> dict:
    """Mock VirusTotal lookup."""
    return {
        "queried": value,
        "related_ips": ["45.77.0.1", "8.8.8.8"],
        "score": 87,
    }


PLAN = [
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


async def drive_agent(*, settle_seconds: float = 1.0) -> None:
    llm = MockChatModel(plan=PLAN)
    tools = [fetch_cloudtrail, vt_lookup, *make_graph_tools(hooks=HOOK)]
    agent = create_agent(llm, tools, store=InMemoryStore())

    print(f"agent driving plan ({len(PLAN)} steps)…")
    # Run the agent in a worker thread so its sync tool invocations
    # don't block the asyncio event loop driving serve_websocket().
    await asyncio.to_thread(
        agent.invoke,
        {"messages": [{"role": "user", "content": "Investigate suspicious AssumeRole."}]},
        {"configurable": {"thread_id": "demo-live"}},
    )
    print("agent finished. WS server stays up — Ctrl-C to exit.")
    await asyncio.sleep(settle_seconds)


async def main() -> None:
    html_path = Path(__file__).resolve().parent / "index.html"
    print("Open in a browser:  http://localhost:8765/")
    print("(or connect a WebSocket client to ws://localhost:8765/)")
    print()

    server = asyncio.create_task(
        serve_websocket(
            HOOK,
            host="localhost",
            port=8765,
            poll_interval=0.05,
            html_path=str(html_path),
        )
    )
    # Give the browser a moment to open and connect before we ingest.
    await asyncio.sleep(2)
    try:
        await drive_agent()
        # Keep the server alive so the browser can inspect the final state.
        await server
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("bye")
