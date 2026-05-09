"""Register tool names, JSON Schema parameters, and ``BackendTool`` builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

from rath.flow.tool.base import FlowToolCall
from rath.flow.tool.command_run import flow_tool_command_run
from rath.flow.tool.files_write import flow_tool_files_write
from rath.llm import RathLLMFunctionTool


class ToolNameConflictError(ValueError):
    """Raised when a tool name is already present in a :class:`ToolTable`."""


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """``name``, ``description``, ``parameters`` (JSON Schema), ``builder``."""

    name: str
    builder: Callable[[Mapping[str, Any]], FlowToolCall]
    description: str | None = None
    parameters: Mapping[str, Any] | None = None


class ToolTable:
    """Maps each tool name to an OpenAI-style schema and a backend dispatch builder."""

    __slots__ = ("_tools", "_lock")

    def __init__(self) -> None:
        self._tools: dict[str, ToolRegistration] = {}
        self._lock = Lock()

    def register(self, registration: ToolRegistration) -> None:
        """Register or replace one tool (idempotent overwrite)."""
        self._store(registration, replace=True)

    def register_unique(self, registration: ToolRegistration) -> None:
        """Register one tool; raise :exc:`ToolNameConflictError` if the name exists."""

        self._store(registration, replace=False)

    def _store(self, registration: ToolRegistration, *, replace: bool) -> None:
        schema = dict(
            registration.parameters or {"type": "object", "properties": {}}
        )
        normalized = ToolRegistration(
            name=registration.name,
            builder=registration.builder,
            description=registration.description,
            parameters=schema,
        )
        with self._lock:
            if not replace and registration.name in self._tools:
                raise ToolNameConflictError(
                    f"tool {registration.name!r} is already registered"
                )
            self._tools[registration.name] = normalized

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def schemas(self) -> tuple[RathLLMFunctionTool, ...]:
        with self._lock:
            specs = sorted(self._tools.values(), key=lambda s: s.name)
        return tuple(
            RathLLMFunctionTool(
                name=s.name,
                description=s.description,
                parameters=dict(s.parameters),
            )
            for s in specs
        )

    def build(self, name: str, arguments: Mapping[str, Any] | None) -> FlowToolCall:
        with self._lock:
            try:
                reg = self._tools[name]
            except KeyError as exc:
                raise KeyError(name) from exc
            builder = reg.builder
        merged: dict[str, Any] = dict(arguments or {})
        return builder(merged)


_GLOBAL = ToolTable()


def global_tool_table() -> ToolTable:
    """Singleton used by :func:`run_session_loop` unless a table is injected."""
    return _GLOBAL


def register_global_tool(registration: ToolRegistration) -> None:
    """Register ``registration`` on the process-wide table if the name is unused.

    This is a convenience for :func:`global_tool_table` plus
    :meth:`ToolTable.register_unique`.

    Raises:
        ToolNameConflictError: If ``registration.name`` is already registered.
    """
    global_tool_table().register_unique(registration)


def register_builtin_session_tools(table: ToolTable | None = None) -> ToolTable:
    """Register defaults for sandbox session loops (shell + workspace write)."""

    target = table if table is not None else _GLOBAL

    def _shell_cmd(args: Mapping[str, Any]) -> FlowToolCall:
        cmd = args["cmd"]
        if not isinstance(cmd, str):
            cmd = str(cmd)
        if "\n" in cmd or "\r" in cmd:
            raise ValueError("multiline commands are rejected")
        if len(cmd) > 2048:
            raise ValueError("command too long")
        return flow_tool_command_run(cmd=cmd)

    target.register(
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

    target.register(
        ToolRegistration(
            name="write_workspace_file",
            builder=_write_file,
            description=(
                "Write UTF-8 text to a path inside the sandbox workspace."
            ),
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

    return target


__all__ = [
    "ToolNameConflictError",
    "ToolRegistration",
    "ToolTable",
    "global_tool_table",
    "register_global_tool",
    "register_builtin_session_tools",
]
