from __future__ import annotations

from typing import Annotated

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent_pinboard import (
    AgentPinBoardConfigError,
    AgentPinBoardValidationError,
    Direction,  # noqa: F401  re-export check
    Entity,
    OnDuplicate,
    node,
    pin,
)
from agent_pinboard import store as store_io
from agent_pinboard.session import get_or_load_session
from tests._helpers import call, make_runner

# Common entity definitions reused across tests.
IP = Entity(name="IP", description="ipv4/ipv6")
User = Entity(name="User", description="user")


class CloudTrailEvent(BaseModel):
    src_ip: str | None = node(type=IP, description="src ip", default=None)
    actor: str | None = node(type=User, description="who", default=None)
    action: str | None = Field(default=None, description="api action")


class TestStackOrderValidation:
    def test_fact_above_function_without_tool_raises(self) -> None:
        def plain(x: str) -> str:
            return x

        with pytest.raises(AgentPinBoardConfigError, match="ABOVE @tool"):
            pin(model=CloudTrailEvent)(plain)


class TestSyncBasicIngestion:
    def test_one_call_creates_event_and_facts(self, store: InMemoryStore) -> None:
        @pin(model=CloudTrailEvent)
        @tool
        def fetch(value: str, runtime: ToolRuntime) -> dict:
            """Returns one event."""
            return {"src_ip": "1.2.3.4", "actor": "admin", "action": "AssumeRole"}

        graph = make_runner([fetch], store)
        out = call(graph, "fetch", {"value": "anything"}, "tid")
        # Default behaviour returns the original dict; ToolNode stringifies it.
        assert "1.2.3.4" in out

        g = get_or_load_session(store, "tid")
        ips = list(g.search_by_type("IP"))
        users = list(g.search_by_type("User"))
        assert len(ips) == 1 and len(users) == 1


class TestManyTrue:
    def test_batch_extraction(self, store: InMemoryStore) -> None:
        @pin(model=CloudTrailEvent, many=True)
        @tool
        def fetch(value: str, runtime: ToolRuntime) -> list[dict]:
            """Batch."""
            return [
                {"src_ip": "1.1.1.1", "actor": "a", "action": "X"},
                {"src_ip": "2.2.2.2", "actor": "b", "action": "Y"},
            ]

        graph = make_runner([fetch], store)
        call(graph, "fetch", {"value": "x"}, "tid")
        g = get_or_load_session(store, "tid")
        assert len(list(g.search_by_type("IP"))) == 2


class TestFailLoudOnValidation:
    def test_bad_dict_raises_and_graph_unchanged(self, store: InMemoryStore) -> None:
        """README §16 AC6 — broken return → AgentPinBoardValidationError, graph empty."""

        class StrictModel(BaseModel):
            src_ip: str = node(type=IP, description="src")

        @pin(model=StrictModel)
        @tool
        def fetch(value: str, runtime: ToolRuntime) -> dict:
            """Returns broken payload."""
            return {"unrelated_field": 42}

        graph = make_runner([fetch], store)
        with pytest.raises(AgentPinBoardValidationError):
            call(graph, "fetch", {"value": "x"}, "tid")
        g = get_or_load_session(store, "tid")
        assert list(g.search_by_type("IP")) == []


class TestDuplicateDetection:
    def test_skip_does_not_invoke_tool_again(self, store: InMemoryStore) -> None:
        """README §16 AC5 — second call with same args is skipped."""
        invocations: list[int] = []

        @pin(model=CloudTrailEvent, on_duplicate=OnDuplicate.SKIP)
        @tool
        def fetch(value: str, runtime: ToolRuntime) -> dict:
            """Counted."""
            invocations.append(1)
            return {"src_ip": "1.2.3.4", "actor": "a", "action": "X"}

        graph = make_runner([fetch], store)
        first = call(graph, "fetch", {"value": "x"}, "tid", call_id="c-1")
        second = call(graph, "fetch", {"value": "x"}, "tid", call_id="c-2")
        third = call(graph, "fetch", {"value": "different"}, "tid", call_id="c-3")
        assert sum(invocations) == 2  # first + third only
        assert "skipped" in second

    def test_always_runs_each_time(self, store: InMemoryStore) -> None:
        invocations: list[int] = []

        @pin(model=CloudTrailEvent, on_duplicate=OnDuplicate.ALWAYS)
        @tool
        def fetch(value: str, runtime: ToolRuntime) -> dict:
            """Counted."""
            invocations.append(1)
            return {"src_ip": "1.2.3.4", "actor": "a", "action": "X"}

        graph = make_runner([fetch], store)
        call(graph, "fetch", {"value": "x"}, "tid", call_id="c-1")
        call(graph, "fetch", {"value": "x"}, "tid", call_id="c-2")
        assert sum(invocations) == 2
        # Autolink keeps node count at 1 anyway.
        g = get_or_load_session(store, "tid")
        assert len(list(g.search_by_type("IP"))) == 1


class TestMaskArgs:
    def test_secret_masked_in_log(self, store: InMemoryStore) -> None:
        @pin(model=CloudTrailEvent, mask_args=["api_key"])
        @tool
        def fetch(value: str, api_key: str, runtime: ToolRuntime) -> dict:
            """Sensitive arg."""
            return {"src_ip": "1.2.3.4", "actor": "a", "action": "X"}

        graph = make_runner([fetch], store)
        call(graph, "fetch", {"value": "v", "api_key": "SECRET"}, "tid")
        records = store_io.load_tool_calls(store, "tid")
        assert all("SECRET" not in r.args_repr for r in records)
        assert any("***" in r.args_repr for r in records)


class _S(TypedDict):
    messages: Annotated[list, add_messages]


class TestStoreMissing:
    def test_no_store_compile_raises(self) -> None:
        @pin(model=CloudTrailEvent)
        @tool
        def fetch(value: str, runtime: ToolRuntime) -> dict:
            """."""
            return {"src_ip": "1.2.3.4"}

        gb = StateGraph(_S)
        gb.add_node("seed", lambda s: {})
        gb.add_node("tools", ToolNode([fetch]))
        gb.add_edge(START, "seed")
        gb.add_edge("seed", "tools")
        gb.add_edge("tools", END)
        no_store_graph = gb.compile()  # NO store=

        with pytest.raises(AgentPinBoardConfigError, match="compile"):
            no_store_graph.invoke(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"name": "fetch", "args": {"value": "x"}, "id": "c", "type": "tool_call"}
                            ],
                        )
                    ]
                },
                config={"configurable": {"thread_id": "tid"}},
            )


class TestIngestEventDispatch:
    def test_dispatched_event_carries_result(self, store: InMemoryStore) -> None:
        from langchain_core.callbacks import BaseCallbackHandler

        from agent_pinboard.decorator import INGEST_EVENT

        seen: list[int] = []

        class Recorder(BaseCallbackHandler):
            def on_custom_event(self, name, data, *, run_id, tags=None, metadata=None, **kw):
                if name == INGEST_EVENT:
                    seen.append(data["result"].new_nodes)

        @pin(model=CloudTrailEvent)
        @tool
        def fetch(value: str, runtime: ToolRuntime) -> dict:
            """."""
            return {"src_ip": "1.1.1.1", "actor": "a", "action": "X"}

        graph = make_runner([fetch], store)
        call(graph, "fetch", {"value": "v"}, "tid", callbacks=[Recorder()])
        assert seen == [2]


class TestAsyncTool:
    @pytest.mark.asyncio
    async def test_async_ingestion(self, store: InMemoryStore) -> None:
        from tests._helpers import acall

        @pin(model=CloudTrailEvent)
        @tool
        async def afetch(value: str, runtime: ToolRuntime) -> dict:
            """Async."""
            return {"src_ip": "9.9.9.9", "actor": "a", "action": "X"}

        graph = make_runner([afetch], store, async_mode=True)
        await acall(graph, "afetch", {"value": "v"}, "tid-async")
        from agent_pinboard.session import aget_or_load_session

        g = await aget_or_load_session(store, "tid-async")
        assert g.find_by_value("IP", "9.9.9.9") is not None

    @pytest.mark.asyncio
    async def test_async_dispatched_event_carries_result(
        self, store: InMemoryStore
    ) -> None:
        """Async path through `adispatch_custom_event`: a BaseCallbackHandler
        attached via `config["callbacks"]` on `ainvoke` receives the event."""
        from langchain_core.callbacks import BaseCallbackHandler

        from agent_pinboard import INGEST_EVENT
        from tests._helpers import acall

        seen: list[int] = []

        class Recorder(BaseCallbackHandler):
            def on_custom_event(
                self, name, data, *, run_id, tags=None, metadata=None, **kw
            ):
                if name == INGEST_EVENT:
                    seen.append(data["result"].new_nodes)

        @pin(model=CloudTrailEvent)
        @tool
        async def afetch(value: str, runtime: ToolRuntime) -> dict:
            """Async."""
            return {"src_ip": "8.8.4.4", "actor": "a", "action": "X"}

        graph = make_runner([afetch], store, async_mode=True)
        await acall(
            graph, "afetch", {"value": "v"}, "tid-async-cb",
            callbacks=[Recorder()],
        )
        # Two new facts (IP + Action) plus the User-typed actor (string default
        # — not a node-marked field) → the count comes from extract.py's rules.
        assert seen and seen[0] >= 1


class TestDispatchOutsideRunnableContext:
    """`@pin` is also usable in plain unit-test code that calls the tool's
    underlying ``func`` directly, bypassing ``invoke``. There is no
    callback manager in scope; the decorator must swallow the no-context
    error and not log any ERROR."""

    def test_direct_func_call_dispatch_silent(
        self, store: InMemoryStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        from types import SimpleNamespace

        import logging as stdlogging

        @pin(model=CloudTrailEvent)
        @tool
        def fetch(value: str, runtime: ToolRuntime) -> dict:
            """."""
            return {"src_ip": "7.7.7.7", "actor": "a", "action": "X"}

        runtime_stub = SimpleNamespace(
            store=store,
            config={"configurable": {"thread_id": "tid-direct"}},
        )

        with caplog.at_level(stdlogging.ERROR, logger="agent_pinboard.decorator"):
            # Drive the wrapped function directly — no Runnable / no callback context.
            fetch.func("v", runtime=runtime_stub)  # type: ignore[arg-type]

        # Decorator must not have logged any dispatch failure.
        assert not [
            r for r in caplog.records
            if "dispatch failed" in r.message
        ]


class TestSyncToolWithAsyncTransform:
    def test_rejected_at_decoration_time(self) -> None:
        async def transform(raw, result):
            return raw

        with pytest.raises(AgentPinBoardConfigError, match="async response_transform"):
            @pin(model=CloudTrailEvent, response_transform=transform)
            @tool
            def fetch(value: str, runtime: ToolRuntime) -> dict:
                """."""
                return {"src_ip": "1.1.1.1"}


class TestResponseTransform:
    def test_overrides_return(self, store: InMemoryStore) -> None:
        @pin(
            model=CloudTrailEvent,
            response_transform=lambda raw, result: f"loaded {result.new_nodes} new",
        )
        @tool
        def fetch(value: str, runtime: ToolRuntime) -> dict:
            """."""
            return {"src_ip": "1.2.3.4", "actor": "a", "action": "X"}

        graph = make_runner([fetch], store)
        out = call(graph, "fetch", {"value": "x"}, "tid")
        assert "loaded" in out and "new" in out
