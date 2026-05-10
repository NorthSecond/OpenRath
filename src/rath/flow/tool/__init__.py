"""Tool payloads, factories, registration, and ``@tool`` decorator.

:class:`~rath.flow.tool.base.FlowToolCall` aliases
:class:`~rath.backend.tool_types.BackendTool`.
"""

from __future__ import annotations

from rath.backend.tool_types import (
    BackendTool,
    BackendToolCodeRun,
    BackendToolCommandRun,
    BackendToolFilesExists,
    BackendToolFilesList,
    BackendToolFilesRead,
    BackendToolFilesWrite,
)
from rath.flow.tool._builtins import extend_builtin_sandbox_tools
from rath.flow.tool.base import FlowToolCall
from rath.flow.tool.code_run import flow_tool_code_run
from rath.flow.tool.command_run import flow_tool_command_run
from rath.flow.tool.decorators import tool
from rath.flow.tool.files_exists import flow_tool_files_exists
from rath.flow.tool.files_list import flow_tool_files_list
from rath.flow.tool.files_read import flow_tool_files_read
from rath.flow.tool.files_write import flow_tool_files_write
from rath.flow.tool.tool_table import (
    InlineToolResolution,
    SandboxToolResolution,
    ToolNameConflictError,
    ToolRegistration,
    ToolTable,
    build_loop_tool_table,
    global_system_tool_table,
    global_tool_table,
    global_user_tool_table,
    register_global_tool,
)

global_system_tool_table()

__all__ = [
    "BackendTool",
    "BackendToolCodeRun",
    "BackendToolCommandRun",
    "BackendToolFilesExists",
    "BackendToolFilesList",
    "BackendToolFilesRead",
    "BackendToolFilesWrite",
    "FlowToolCall",
    "InlineToolResolution",
    "SandboxToolResolution",
    "ToolNameConflictError",
    "ToolRegistration",
    "ToolTable",
    "build_loop_tool_table",
    "extend_builtin_sandbox_tools",
    "flow_tool_code_run",
    "flow_tool_command_run",
    "flow_tool_files_exists",
    "flow_tool_files_list",
    "flow_tool_files_read",
    "flow_tool_files_write",
    "global_system_tool_table",
    "global_tool_table",
    "global_user_tool_table",
    "register_global_tool",
    "tool",
]
