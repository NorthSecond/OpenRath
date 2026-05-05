"""Integration tests against a live OpenAI-compatible endpoint (no mocks)."""

from __future__ import annotations

import pytest

from rath.llm import (
    LLMChatRequest,
    LLMFunctionTool,
    LLMMessage,
    OpenAIChatClient,
)


@pytest.fixture
def client() -> OpenAIChatClient:
    return OpenAIChatClient()


def test_complete_ping(client: OpenAIChatClient) -> None:
    req = LLMChatRequest(
        messages=(
            LLMMessage(role="user", content="Reply with exactly the single word: pong"),
        ),
        model=client.settings.default_model,
    )
    resp = client.complete(req)
    assert resp.id
    assert resp.model
    assert len(resp.choices) >= 1
    text = (resp.primary_choice.message.content or "").strip().lower()
    assert "pong" in text
    assert resp.usage is not None
    assert resp.usage.total_tokens > 0


def test_complete_uses_env_default_model_when_omitted(
    client: OpenAIChatClient,
) -> None:
    assert client.settings.default_model, (
        "OPENAI_DEFAULT_MODEL should be set for this test"
    )
    req = LLMChatRequest(
        messages=(LLMMessage(role="user", content="Say ok."),),
        model=None,
    )
    resp = client.complete(req)
    assert resp.model == client.settings.default_model


def test_function_tool_call_returns_add_arguments(client: OpenAIChatClient) -> None:
    tools = (
        LLMFunctionTool(
            name="add",
            description="Return the sum of two integers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First summand"},
                    "b": {"type": "integer", "description": "Second summand"},
                },
                "required": ["a", "b"],
            },
        ),
    )
    req = LLMChatRequest(
        messages=(
            LLMMessage(
                role="user",
                content=(
                    "Call the add tool with a=40 and b=2 only. "
                    "Do not answer with plain text for the sum."
                ),
            ),
        ),
        model=client.settings.default_model,
        tools=tools,
        tool_choice="auto",
    )
    resp = client.complete(req)
    choice = resp.primary_choice
    assert choice.finish_reason in ("tool_calls", "stop")
    tc_list = choice.message.tool_calls
    assert tc_list is not None and len(tc_list) >= 1
    add_call = next((t for t in tc_list if t.function.name == "add"), None)
    assert add_call is not None
    assert add_call.function.arguments
    assert add_call.function.arguments_parse_error is False
    assert add_call.function.arguments_parsed is not None
    assert add_call.function.arguments_parsed.get("a") in (40, 40.0)
    assert add_call.function.arguments_parsed.get("b") in (2, 2.0)
