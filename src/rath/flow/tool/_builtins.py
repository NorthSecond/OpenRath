"""Default sandbox tools registered on :func:`~rath.flow.tool.tool_table.global_system_tool_table`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rath.flow.tool.base import FlowToolCall
from rath.flow.tool.command_run import flow_tool_command_run
from rath.flow.tool.files_write import flow_tool_files_write
from rath.flow.tool.tool_table import ToolRegistration, ToolTable


def extend_builtin_sandbox_tools(table: ToolTable) -> None:
    """Install ``run_shell_command`` and ``write_workspace_file`` (idempotent ``.register``)."""

    def _shell_cmd(args: Mapping[str, Any]) -> FlowToolCall:
        cmd = args["cmd"]
        if not isinstance(cmd, str):
            cmd = str(cmd)
        if "\n" in cmd or "\r" in cmd:
            raise ValueError("multiline commands are rejected")
        if len(cmd) > 2048:
            raise ValueError("command too long")
        return flow_tool_command_run(cmd=cmd)

    table.register(
        ToolRegistration(
            name="run_shell_command",
            builder=_shell_cmd,
            description=(
                "Run one shell command inside the active sandbox workspace. "
                "Prefer short commands such as ``echo Hello``."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command string",
                    },
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
        )
    )

    def _write_file(args: Mapping[str, Any]) -> FlowToolCall:
        path = str(args["path"])
        raw = args["content"]
        if isinstance(raw, str):
            return flow_tool_files_write(path=path, data=raw)
        raise TypeError("content must be text for write_workspace_file")

    table.register(
        ToolRegistration(
            name="write_workspace_file",
            builder=_write_file,
            description="Write UTF-8 text to a path inside the sandbox workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        )
    )


__all__ = ["extend_builtin_sandbox_tools"]
