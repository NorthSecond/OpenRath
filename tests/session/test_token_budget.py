"""Token usage accumulation and budget guard for the session loop."""

from __future__ import annotations

import logging

import pytest

from rath.backend import get
from rath.flow.agent_param import AgentParam, Provider
from rath.llm import (
    BudgetExceededError,
    RathLLMAssistantMessage,
    RathLLMChatChoice,
    RathLLMChatResponse,
    RathLLMTokenUsage,
)
from rath.session import Session, run_session_loop, session_registry
from tests.session.scripted_loop_executor import ScriptedSessionLoopExecutor


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    yield
    session_registry().set_active(None)


def _stop_response(*, prompt: int, completion: int) -> RathLLMChatResponse:
    return RathLLMChatResponse(
        id="usage-test",
        choices=(
            RathLLMChatChoice(
                index=0,
                finish_reason="stop",
                message=RathLLMAssistantMessage(content="ok"),
            ),
        ),
        created=1,
        model="scripted",
        usage=RathLLMTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def test_cumulative_usage_starts_none_and_accumulates() -> None:
    """One scripted completion with usage produces a Session.cumulative_usage."""
    executor = ScriptedSessionLoopExecutor([_stop_response(prompt=10, completion=20)])
    agent = AgentParam(Session.from_agent_prompt("sys"), Provider())
    backend = get("local")
    with backend.open() as sb:
        user = Session.from_user_message("hi").with_sandbox(sb)
        out = run_session_loop(
            user,
            agent.agent_session,
            agent_provider=agent.provider,
            executor=executor,
        )
    assert out.cumulative_usage is not None
    assert out.cumulative_usage.prompt_tokens == 10
    assert out.cumulative_usage.completion_tokens == 20
    assert out.cumulative_usage.total_tokens == 30


def test_usage_remains_none_when_provider_reports_no_usage() -> None:
    no_usage = RathLLMChatResponse(
        id="x",
        choices=(
            RathLLMChatChoice(
                index=0,
                finish_reason="stop",
                message=RathLLMAssistantMessage(content="ok"),
            ),
        ),
        created=1,
        model="scripted",
        usage=None,
    )
    executor = ScriptedSessionLoopExecutor([no_usage])
    agent = AgentParam(Session.from_agent_prompt("sys"), Provider())
    backend = get("local")
    with backend.open() as sb:
        user = Session.from_user_message("hi").with_sandbox(sb)
        out = run_session_loop(
            user,
            agent.agent_session,
            agent_provider=agent.provider,
            executor=executor,
        )
    assert out.cumulative_usage is None


def test_budget_exceeded_invokes_callback() -> None:
    """Provider.on_budget_exceeded is called once total_tokens crosses the cap."""
    seen: list[tuple[object, RathLLMTokenUsage]] = []

    def _cb(sess: object, usage: RathLLMTokenUsage) -> None:
        seen.append((sess, usage))

    executor = ScriptedSessionLoopExecutor([_stop_response(prompt=100, completion=50)])
    agent = AgentParam(
        Session.from_agent_prompt("sys"),
        Provider(budget_total_tokens=100, on_budget_exceeded=_cb),
    )
    backend = get("local")
    with backend.open() as sb:
        user = Session.from_user_message("hi").with_sandbox(sb)
        out = run_session_loop(
            user,
            agent.agent_session,
            agent_provider=agent.provider,
            executor=executor,
        )
    assert len(seen) == 1
    cb_session, cb_usage = seen[0]
    assert cb_session is out
    assert cb_usage.total_tokens == 150


def test_budget_callback_raising_aborts_loop() -> None:
    """Raising BudgetExceededError from the callback aborts run_session_loop."""

    def _hard_stop(sess: object, usage: RathLLMTokenUsage) -> None:
        raise BudgetExceededError(
            f"hit cap; cumulative={usage.total_tokens}",
        )

    # First response triggers the budget; a second response would otherwise
    # be consumed if the loop continued.
    executor = ScriptedSessionLoopExecutor(
        [
            _stop_response(prompt=200, completion=50),
            _stop_response(prompt=999, completion=999),
        ]
    )
    agent = AgentParam(
        Session.from_agent_prompt("sys"),
        Provider(budget_total_tokens=100, on_budget_exceeded=_hard_stop),
    )
    backend = get("local")
    with backend.open() as sb:
        user = Session.from_user_message("hi").with_sandbox(sb)
        with pytest.raises(BudgetExceededError, match="hit cap"):
            run_session_loop(
                user,
                agent.agent_session,
                agent_provider=agent.provider,
                executor=executor,
            )


def test_budget_without_callback_emits_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No callback set: still warn so the overrun is at least visible."""
    executor = ScriptedSessionLoopExecutor([_stop_response(prompt=200, completion=0)])
    agent = AgentParam(
        Session.from_agent_prompt("sys"),
        Provider(budget_total_tokens=100),
    )
    backend = get("local")
    with backend.open() as sb:
        user = Session.from_user_message("hi").with_sandbox(sb)
        with caplog.at_level(logging.WARNING, logger="rath.session.loop"):
            out = run_session_loop(
                user,
                agent.agent_session,
                agent_provider=agent.provider,
                executor=executor,
            )

    assert out.cumulative_usage is not None
    assert any(
        "budget_total_tokens" in rec.message for rec in caplog.records
    )
