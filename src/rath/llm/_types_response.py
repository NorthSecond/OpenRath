"""Frozen response types mirroring OpenAI ``chat.completion`` objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

__all__ = [
    "RathLLMTokenUsage",
    "RathLLMToolCallFunction",
    "RathLLMToolCallPart",
    "RathLLMAssistantMessage",
    "RathLLMChatChoice",
    "RathLLMChatResponse",
    "RathLLMFinishReason",
]

RathLLMFinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


@dataclass(frozen=True, slots=True)
class RathLLMTokenUsage:
    """Token counts from ``usage``; optional detail dicts stay JSON-shaped."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Mapping[str, Any] | None = None
    prompt_tokens_details: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RathLLMToolCallFunction:
    """``function`` payload inside a tool call (name + arguments string)."""

    name: str
    arguments: str
    arguments_parsed: dict[str, Any] | None
    arguments_parse_error: bool


@dataclass(frozen=True, slots=True)
class RathLLMToolCallPart:
    """One entry from ``message.tool_calls``."""

    id: str
    type: str
    function: RathLLMToolCallFunction


@dataclass(frozen=True, slots=True)
class RathLLMAssistantMessage:
    """Assistant message on a choice (content, optional tool calls, provider extras)."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    refusal: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[RathLLMToolCallPart, ...] | None = None
    function_call: Mapping[str, Any] | None = None
    annotations: tuple[Mapping[str, Any], ...] | None = None


@dataclass(frozen=True, slots=True)
class RathLLMChatChoice:
    """One element of ``choices``."""

    index: int
    finish_reason: RathLLMFinishReason
    message: RathLLMAssistantMessage
    logprobs: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RathLLMChatResponse:
    """Normalized non-streaming ``ChatCompletion``."""

    id: str
    choices: tuple[RathLLMChatChoice, ...]
    created: int
    model: str
    object_type: Literal["chat.completion"] = "chat.completion"
    service_tier: str | None = None
    system_fingerprint: str | None = None
    usage: RathLLMTokenUsage | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def primary_choice(self) -> RathLLMChatChoice:
        """The first choice (typical when ``n`` is 1)."""
        if not self.choices:
            raise IndexError("RathLLMChatResponse has no choices")
        return self.choices[0]
