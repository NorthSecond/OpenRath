"""Integration tests against a live OpenAI-compatible endpoint (no mocks)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rath.llm import (
    RathLLMChatRequest,
    RathLLMFunctionTool,
    RathLLMMessage,
    RathOpenAIChatAgent,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_dotenv_api_key(env_path: Path) -> str:
    """Read OPENAI_API_KEY from a dotenv file (no dependencies)."""
    text = env_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("OPENAI_API_KEY="):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    raise AssertionError(f"OPENAI_API_KEY not found in {env_path}")


@pytest.fixture
def agent() -> RathOpenAIChatAgent:
    return RathOpenAIChatAgent()


def test_openai_api_key_in_process_matches_project_dotenv() -> None:
    """Ensure pytest loaded the same key as on disk (real credential path)."""
    env_path = _project_root() / ".env"
    assert env_path.is_file(), "project .env must exist for live LLM tests"
    from_disk = _parse_dotenv_api_key(env_path)
    assert from_disk, ".env OPENAI_API_KEY must be non-empty"
    in_process = os.environ.get("OPENAI_API_KEY", "").strip()
    assert in_process == from_disk, (
        "process OPENAI_API_KEY must match .env after conftest load_dotenv"
    )
    assert len(in_process) >= 8, "API key from .env must have plausible length"


def test_rath_openai_chat_agent_uses_dotenv_credentials(
    agent: RathOpenAIChatAgent,
) -> None:
    assert agent.settings.api_key == os.environ["OPENAI_API_KEY"].strip()
    env_path = _project_root() / ".env"
    assert agent.settings.api_key == _parse_dotenv_api_key(env_path)


def test_complete_ping_hits_remote_model(agent: RathOpenAIChatAgent) -> None:
    req = RathLLMChatRequest(
        messages=(
            RathLLMMessage(
                role="user",
                content="Reply with exactly the single word: pong",
            ),
        ),
        model=agent.settings.default_model,
    )
    resp = agent.complete(req)
    assert resp.id, "remote completions must return an id"
    assert resp.model
    assert len(resp.choices) >= 1
    text = (resp.primary_choice.message.content or "").strip().lower()
    assert "pong" in text
    assert resp.usage is not None
    assert resp.usage.total_tokens > 0
    assert resp.raw is not None
    assert resp.raw.get("object") == "chat.completion"


def test_complete_uses_env_default_model_when_omitted(
    agent: RathOpenAIChatAgent,
) -> None:
    assert agent.settings.default_model, (
        "OPENAI_DEFAULT_MODEL should be set for this test"
    )
    req = RathLLMChatRequest(
        messages=(RathLLMMessage(role="user", content="Say ok."),),
        model=None,
    )
    resp = agent.complete(req)
    assert resp.model == agent.settings.default_model


def test_function_tool_call_returns_add_arguments(agent: RathOpenAIChatAgent) -> None:
    tools = (
        RathLLMFunctionTool(
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
    req = RathLLMChatRequest(
        messages=(
            RathLLMMessage(
                role="user",
                content=(
                    "Call the add tool with a=40 and b=2 only. "
                    "Do not answer with plain text for the sum."
                ),
            ),
        ),
        model=agent.settings.default_model,
        tools=tools,
        tool_choice="auto",
    )
    resp = agent.complete(req)
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
