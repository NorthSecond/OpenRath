"""Fold :class:`~rath.llm.Provider` sampling prefs into :class:`~rath.llm.RathLLMChatRequest`."""

from __future__ import annotations

from typing import Any

from rath.llm import (
    Provider,
    RathLLMChatRequest,
    RathLLMFunctionTool,
    RathLLMMessage,
)


def provider_into_chat_request(
    messages: tuple[RathLLMMessage, ...],
    tools: tuple[RathLLMFunctionTool, ...] | None,
    prefs: Provider,
    *,
    default_tool_choice: Any,
) -> RathLLMChatRequest:
    """Build chat completion kwargs from messages + optional tools + provider defaults."""

    return RathLLMChatRequest(
        messages=messages,
        tools=tools,
        tool_choice=prefs.tool_choice
        if prefs.tool_choice is not None
        else default_tool_choice,
        parallel_tool_calls=prefs.parallel_tool_calls,
        model=prefs.model,
        temperature=prefs.temperature,
        top_p=prefs.top_p,
        max_completion_tokens=prefs.max_completion_tokens,
        max_tokens=prefs.max_tokens,
        stop=prefs.stop,
        n=prefs.n,
        seed=prefs.seed,
        frequency_penalty=prefs.frequency_penalty,
        presence_penalty=prefs.presence_penalty,
        response_format=prefs.response_format,
        logit_bias=prefs.logit_bias,
        logprobs=prefs.logprobs,
        top_logprobs=prefs.top_logprobs,
        reasoning_effort=prefs.reasoning_effort,
        verbosity=prefs.verbosity,
        metadata=prefs.metadata,
        user=prefs.user,
        store=prefs.store,
        service_tier=prefs.service_tier,
        extra_create_args=prefs.extra_create_args,
    )


__all__ = ["provider_into_chat_request"]
