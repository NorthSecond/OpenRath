"""Tests for :func:`~rath.llm.to_create_kwargs` (OpenAI SDK keyword arguments)."""

from __future__ import annotations

from rath.llm import (
    RathLLMChatRequest,
    RathLLMFunctionTool,
    RathLLMMessage,
    to_create_kwargs,
)


def test_to_create_kwargs_includes_assistant_tool_calls() -> None:
    """Assistant role messages serialize ``tool_calls`` for the OpenAI wire format."""
    wire_tc = (
        {
            "id": "tc1",
            "type": "function",
            "function": {"name": "run_shell_command", "arguments": '{"cmd":"noop"}'},
        },
    )
    msg = RathLLMMessage(
        role="assistant",
        content=None,
        tool_calls=tuple(dict(x) for x in wire_tc),
    )
    req = RathLLMChatRequest(messages=(msg,), model="gpt-test-wire")
    kwargs = to_create_kwargs(req, default_model=None)
    out_msgs = kwargs["messages"]
    assert len(out_msgs) == 1
    assert out_msgs[0]["role"] == "assistant"
    assert "tool_calls" in out_msgs[0]
    assert out_msgs[0]["tool_calls"][0]["function"]["name"] == "run_shell_command"


def test_to_create_kwargs_strips_additional_properties_on_tools() -> None:
    """Serialized tool ``parameters`` must not contain ``additionalProperties``."""
    msg = RathLLMMessage(role="user", content="hi")
    tool = RathLLMFunctionTool(
        name="demo",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        },
    )
    req = RathLLMChatRequest(
        messages=(msg,),
        model="glm-test-wire",
        tools=(tool,),
    )
    kwargs = to_create_kwargs(req, default_model=None)
    wire_params = kwargs["tools"][0]["function"]["parameters"]
    assert "additionalProperties" not in wire_params
    assert wire_params["type"] == "object"
    assert "properties" in wire_params
