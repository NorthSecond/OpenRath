"""Default SessionLoopExecutor: blocking LLM completions plus sandbox tool dispatch."""

from __future__ import annotations

from rath.backend import ToolResult
from rath.flow.tool import FlowToolCall
from rath.llm import (
    RathLLMChatRequest,
    RathLLMChatResponse,
    RathLLMFunctionTool,
    RathOpenAIChatClient,
)
from rath.session.session import Session


class DefaultSessionLoopExecutor:
    """Default :class:`SessionLoopExecutor`: sync ``OpenAI`` client + sandbox ``dispatch``."""

    __slots__ = ("_client",)

    def __init__(self, client: RathOpenAIChatClient) -> None:
        self._client = client

    def tool_schemas(self) -> tuple[RathLLMFunctionTool, ...]:
        """Defer to :func:`~rath.session.loop.run_session_loop` loop-local tool table."""

        return ()

    def complete(self, req: RathLLMChatRequest) -> RathLLMChatResponse:
        """Call the synchronous :class:`~rath.llm.client.RathOpenAIChatClient`."""

        return self._client.complete(req)

    def dispatch_tool(
        self,
        session: Session,
        call: FlowToolCall,
    ) -> ToolResult | bool:
        """Run ``call`` on ``session``'s sandbox."""

        return session.require_sandbox().dispatch(call)


__all__ = ["DefaultSessionLoopExecutor"]
