"""Tests for :class:`~rath.flow.tool.tool_table.ToolTable` defaults."""

from __future__ import annotations

import pytest

from rath.backend import BackendToolCommandRun, BackendToolFilesWrite
from rath.flow.tool import FlowToolCall, flow_tool_command_run
from rath.flow.tool.tool_table import (
    ToolNameConflictError,
    ToolRegistration,
    ToolTable,
    global_tool_table,
    register_builtin_session_tools,
    register_global_tool,
)


def test_register_builtin_includes_expected_tools() -> None:
    table = ToolTable()
    register_builtin_session_tools(table)
    names = {s.name for s in table.schemas()}
    assert "run_shell_command" in names
    assert "write_workspace_file" in names


def test_build_run_shell_command() -> None:
    table = ToolTable()
    register_builtin_session_tools(table)
    call = table.build("run_shell_command", {"cmd": "echo x"})
    assert isinstance(call, BackendToolCommandRun)
    assert call.cmd == "echo x"


def test_build_write_workspace_file() -> None:
    table = ToolTable()
    register_builtin_session_tools(table)
    call = table.build(
        "write_workspace_file",
        {"path": "rel.txt", "content": "body"},
    )
    assert isinstance(call, BackendToolFilesWrite)
    assert call.path == "rel.txt"
    assert call.data == "body"


def test_reject_multiline_shell_command() -> None:
    table = ToolTable()
    register_builtin_session_tools(table)
    with pytest.raises(ValueError, match="multiline"):
        table.build("run_shell_command", {"cmd": "echo\nbad"})


def test_reject_oversized_shell_command() -> None:
    table = ToolTable()
    register_builtin_session_tools(table)
    with pytest.raises(ValueError, match="too long"):
        table.build("run_shell_command", {"cmd": "x" * 3000})


def test_missing_tool_raises() -> None:
    table = ToolTable()
    with pytest.raises(KeyError):
        table.build("nonexistent", {})


def test_register_accepts_tool_registration() -> None:
    table = ToolTable()

    def _echo(args: dict[str, object]) -> FlowToolCall:
        return flow_tool_command_run(cmd=f"echo {args['x']!r}")

    table.register(
        ToolRegistration(
            name="echo_x",
            builder=_echo,
            description="Echo x",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        )
    )
    call = table.build("echo_x", {"x": "hi"})
    assert isinstance(call, BackendToolCommandRun)
    assert call.cmd == "echo 'hi'"


def test_register_unique_rejects_duplicate() -> None:
    table = ToolTable()

    def _a(_args: dict[str, object]) -> FlowToolCall:
        return flow_tool_command_run(cmd="echo a")

    reg = ToolRegistration(
        name="dup_test",
        builder=_a,
        description="a",
        parameters={"type": "object", "properties": {}},
    )
    table.register_unique(reg)
    with pytest.raises(ToolNameConflictError, match="dup_test"):
        table.register_unique(reg)


def test_register_global_tool_and_conflict() -> None:
    name = "_rath_global_tool_probe"
    g = global_tool_table()
    g.unregister(name)
    try:

        def _b(_args: dict[str, object]) -> FlowToolCall:
            return flow_tool_command_run(cmd="echo b")

        register_global_tool(
            ToolRegistration(
                name=name,
                builder=_b,
                description="b",
                parameters={"type": "object", "properties": {}},
            )
        )
        with pytest.raises(ToolNameConflictError, match=name):
            register_global_tool(
                ToolRegistration(
                    name=name,
                    builder=_b,
                    description="b2",
                    parameters={"type": "object", "properties": {}},
                )
            )
    finally:
        g.unregister(name)
