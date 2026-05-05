"""Frozen response types mirroring OpenAI ``chat.completion`` objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

__all__ = [
    "LLMTokenUsage",
    "LLMToolCallFunction",
    "LLMToolCallPart",
    "LLMAssistantMessage",
    "LLMChatChoice",
    "LLMChatResponse",
]

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


@dataclass(frozen=True, slots=True)
class LLMTokenUsage:
    """Token counts from ``usage``; optional detail dicts stay JSON-shaped."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Mapping[str, Any] | None = None
    prompt_tokens_details: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LLMToolCallFunction:
    """``function`` payload inside a tool call (name + arguments string)."""

    name: str
    arguments: str
    arguments_parsed: dict[str, Any] | None
    arguments_parse_error: bool


@dataclass(frozen=True, slots=True)
class LLMToolCallPart:
    """One entry from ``message.tool_calls``."""

    id: str
    type: str
    function: LLMToolCallFunction


@dataclass(frozen=True, slots=True)
class LLMAssistantMessage:
    """Assistant message on a choice (content, optional tool calls, provider extras)."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    refusal: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[LLMToolCallPart, ...] | None = None
    function_call: Mapping[str, Any] | None = None
    annotations: tuple[Mapping[str, Any], ...] | None = None


@dataclass(frozen=True, slots=True)
class LLMChatChoice:
    """One element of ``choices``."""

    index: int
    finish_reason: FinishReason
    message: LLMAssistantMessage
    logprobs: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LLMChatResponse:
    """Normalized non-streaming ``ChatCompletion``."""

    id: str
    choices: tuple[LLMChatChoice, ...]
    created: int
    model: str
    object_type: Literal["chat.completion"] = "chat.completion"
    service_tier: str | None = None
    system_fingerprint: str | None = None
    usage: LLMTokenUsage | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def primary_choice(self) -> LLMChatChoice:
        """The first choice (typical when ``n`` is 1)."""
        if not self.choices:
            raise IndexError("LLMChatResponse has no choices")
        return self.choices[0]
