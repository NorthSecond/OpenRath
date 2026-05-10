"""Session loop: alternate LLM completions with sandbox tool execution (blocking)."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from rath.backend import (
    CodeResult,
    CommandResult,
    FileContent,
    FileEntries,
    FileWriteResult,
    ToolExecutionFailure,
    ToolResult,
)
from rath.flow.tool import (
    FlowToolCall,
    InlineToolResolution,
    SandboxToolResolution,
    build_loop_tool_table,
)
from rath.llm import (
    Provider,
    RathLLMChatResponse,
    RathLLMFunctionTool,
    RathLLMMessage,
    RathOpenAIChatClient,
)
from rath.session.chat_request_build import provider_into_chat_request
from rath.session.provider_builtin import DefaultSessionLoopExecutor
from rath.session.chunk import (
    ChunkTable,
    assistant_turn_chunk,
    chunk_table_to_messages,
    tool_feedback_chunk,
)
from rath.session.graph import LineageKind, LineageRecorder, SessionLineage
from rath.session.manager import session_registry
from rath.session.session import Session

logger = logging.getLogger(__name__)


@runtime_checkable
class SessionLoopExecutor(Protocol):
    """Runs completions and sandbox tool dispatch used by ``run_session_loop``."""

    def complete(self, req: RathLLMChatRequest) -> RathLLMChatResponse:
        """Run one chat completion."""

    def dispatch_tool(
        self,
        session: Session,
        call: FlowToolCall,
    ) -> ToolResult | bool:
        """Execute ``call`` on the session sandbox."""

    def tool_schemas(self) -> tuple[RathLLMFunctionTool, ...]:
        """Tool specs for OpenAI-style ``tools``. Empty means use the loop-local merged table."""


def _loop_tool_error_payload(
    kind: str, message: str, *, detail: str | None = None
) -> str:
    """JSON string for a tool roundtrip failure (scheme A) visible to the model."""

    payload: dict[str, Any] = {
        "ok": False,
        "error_kind": kind,
        "message": message,
    }
    if detail:
        payload["detail"] = detail[:4000]
    return json.dumps(payload)


def _summarize_inline_result(obj: Any) -> str:
    """JSON text for ``@tool`` return values."""

    try:
        if isinstance(obj, BaseModel):
            payload: Any = obj.model_dump(mode="json")
        else:
            payload = obj
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) > 48_000:
            text = text[:48_000] + "...(truncated)"
        return text
    except TypeError:
        return json.dumps({"repr": repr(obj), "type": type(obj).__name__})


def _summarize_tool_result(_call: FlowToolCall, raw: ToolResult | bool) -> str:
    """JSON text for the next ``role=tool`` message."""

    if isinstance(raw, ToolExecutionFailure):
        return json.dumps(
            {
                "ok": False,
                "error_kind": raw.kind,
                "message": raw.message,
                **({"detail": raw.detail} if raw.detail else {}),
            }
        )
    if isinstance(raw, bool):
        return json.dumps({"ok": raw})
    if isinstance(raw, CommandResult):
        return json.dumps(
            {
                "exit_code": raw.exit_code,
                "stdout": raw.stdout.decode("utf-8", errors="replace"),
                "stderr": raw.stderr.decode("utf-8", errors="replace"),
                "elapsed_ms": raw.elapsed_ms,
            }
        )
    if isinstance(raw, FileContent):
        data = raw.data
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        text = str(data)
        if len(text) > 12_000:
            text = text[:12_000] + "...(truncated)"
        return json.dumps({"data": text})
    if isinstance(raw, FileEntries):
        payload = [
            {"name": e.name, "path": e.path, "is_dir": e.is_dir}
            for e in raw.entries[:500]
        ]
        return json.dumps({"entries": payload})
    if isinstance(raw, FileWriteResult):
        return json.dumps({"bytes_written": raw.bytes_written})
    if isinstance(raw, CodeResult):
        stdout = raw.stdout.decode("utf-8", errors="replace")
        stderr = raw.stderr.decode("utf-8", errors="replace")
        return json.dumps(
            {
                "text": raw.text,
                "stdout": stdout,
                "stderr": stderr,
                "error": raw.error,
            }
        )
    return json.dumps({"type": type(raw).__name__, "note": "unserialised result"})


def run_session_loop(
    user_session: Session,
    agent_session: Session,
    *,
    agent_provider: Provider,
    tools: list[str] | None = None,
    executor: SessionLoopExecutor | None = None,
    max_tool_rounds: int = 16,
) -> Session:
    """Run one multi-turn assistant pass with optional tool rounds.

    Built-in sandbox tools come from :func:`~rath.flow.tool.global_system_tool_table`;
    ``tools`` lists names from :func:`~rath.flow.tool.global_user_tool_table` merged into
    the loop-scoped tool table for this call.

    Rebases ``BackendSandbox`` from ``user_session`` onto the returned session.
    LLM routing/sampling kwargs come from ``agent_provider``
    (:class:`~rath.llm.Provider`); completions and sandbox tool dispatch go through
    ``executor``. When ``executor`` is omitted, builds a fresh
    :class:`~rath.session.provider_builtin.DefaultSessionLoopExecutor`
    wrapping :class:`~rath.llm.client.RathOpenAIChatClient()` (``.env`` / env-based
    API settings).

    Inline tools registered via :func:`~rath.flow.tool.tool` execute **in-process**
    inside this loop (not on the sandbox backend).

    Message assembly concatenates ``agent_session.chunk_table`` ahead of user rows for
    the LLM; head rows stay out of ``out.chunk_table`` (assistant + tool-result only).

    When ``session_graph_mode()`` is true, stamps flat lineage on ``out`` and attaches
    legacy :class:`~rath.session.graph.SessionLineage`.
    """

    table = build_loop_tool_table(tools)

    if executor is None:
        executor = DefaultSessionLoopExecutor(RathOpenAIChatClient())

    prefs = agent_provider

    rows_list: list[Any] = list(user_session.chunk_table.rows)
    sb = user_session.take_sandbox()
    out = Session(
        chunk_table=ChunkTable(rows=tuple(rows_list)),
        sandbox=sb,
        sandbox_backend=user_session.sandbox_backend,
        _sandbox_open_spec=user_session._sandbox_open_spec,
        lineage=SessionLineage(
            producer_user_session_id=user_session.id,
            producer_system_session_id=agent_session.id,
        ),
    )
    LineageRecorder.stamp_new_session(
        out,
        parent_session_ids=(user_session.id, agent_session.id),
        lineage_operator="run_session_loop",
        lineage_kind=LineageKind.OP_SESSION_LOOP,
    )
    reg = session_registry()
    reg.register(user_session)
    reg.register(agent_session)
    reg.register(out)
    reg.set_active(out)

    tool_schemas = executor.tool_schemas()
    if not tool_schemas:
        tool_schemas = table.schemas()

    for _ in range(max_tool_rounds):
        head = chunk_table_to_messages(agent_session.chunk_table)
        tail = chunk_table_to_messages(ChunkTable(rows=tuple(rows_list)))
        messages = head + tail

        req = provider_into_chat_request(
            messages,
            tool_schemas,
            prefs,
            default_tool_choice="auto",
        )
        resp = executor.complete(req)
        choice = resp.primary_choice
        msg = choice.message
        tcalls = msg.tool_calls

        if tcalls:
            rows_list.append(
                assistant_turn_chunk(tool_calls=tcalls, content=msg.content)
            )
            for tc in tcalls:
                tool_name = tc.function.name
                if (
                    tc.function.arguments_parsed is None
                    or tc.function.arguments_parse_error
                ):
                    raw_dump = tc.function.arguments or ""
                    if len(raw_dump) > 2000:
                        raw_dump = raw_dump[:2000] + "...(truncated)"
                    body = _loop_tool_error_payload(
                        "invalid_tool_arguments",
                        (
                            f"tool {tool_name!r} returned non-JSON "
                            f"or unparseable arguments"
                        ),
                        detail=raw_dump,
                    )
                else:
                    try:
                        resolved = table.resolve(
                            tool_name, tc.function.arguments_parsed
                        )
                    except Exception as exc:
                        body = _loop_tool_error_payload(
                            "tool_resolve_failed",
                            str(exc),
                            detail=type(exc).__name__,
                        )
                    else:
                        try:
                            if isinstance(resolved, SandboxToolResolution):
                                raw = executor.dispatch_tool(out, resolved.call)
                                body = _summarize_tool_result(resolved.call, raw)
                            elif isinstance(resolved, InlineToolResolution):
                                result = resolved.fn(**resolved.kwargs)
                                body = _summarize_inline_result(result)
                        except Exception as exc:
                            logger.exception(
                                "tool invocation failed for tool=%s", tool_name
                            )
                            body = _loop_tool_error_payload(
                                "tool_execution_exception",
                                f"{type(exc).__name__}: {exc}",
                                detail=type(exc).__name__,
                            )
                rows_list.append(
                    tool_feedback_chunk(tc.id, tool_name, body)
                )
            continue

        rows_list.append(
            assistant_turn_chunk(tool_calls=None, content=msg.content)
        )
        if choice.finish_reason in ("stop", "length", "content_filter"):
            break

    out.chunk_table = ChunkTable(rows=tuple(rows_list))
    reg.set_active(out)
    return out


__all__ = [
    "SessionLoopExecutor",
    "run_session_loop",
]
