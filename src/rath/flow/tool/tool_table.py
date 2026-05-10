"""Register tool names, JSON schemas, sandbox builders, or inline ``@tool`` callables."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any

from pydantic import BaseModel, ValidationError

from rath.flow.tool.base import FlowToolCall
from rath.llm import RathLLMFunctionTool


class ToolNameConflictError(ValueError):
    """Raised when a tool name is already present in a :class:`ToolTable`."""


@dataclass(frozen=True, slots=True)
class SandboxToolResolution:
    """Resolved sandbox dispatch payload."""

    call: FlowToolCall


@dataclass(frozen=True, slots=True)
class InlineToolResolution:
    """Resolved in-process :func:`tool` invocation."""

    fn: Callable[..., Any]
    kwargs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """Exactly one of ``builder`` (sandbox) or ``inline_fn`` (``@tool``) must be set."""

    name: str
    description: str | None = None
    parameters: Mapping[str, Any] | None = None
    builder: Callable[[Mapping[str, Any]], FlowToolCall] | None = None
    inline_fn: Callable[..., Any] | None = None
    args_schema: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        sb = self.builder is not None
        inl = self.inline_fn is not None
        if sb == inl:
            raise ValueError(
                "ToolRegistration requires exactly one of builder or inline_fn"
            )
        if inl and self.args_schema is None:
            raise ValueError("inline_fn requires args_schema")


class ToolTable:
    """Maps tool names to OpenAI-style schemas plus sandbox or inline handlers."""

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
            description=registration.description,
            parameters=schema,
            builder=registration.builder,
            inline_fn=registration.inline_fn,
            args_schema=registration.args_schema,
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
            parameters=dict(s.parameters or {}),
            )
            for s in specs
        )

    def resolve(
        self, name: str, arguments: Mapping[str, Any] | None
    ) -> SandboxToolResolution | InlineToolResolution:
        """Validate arguments and return either a sandbox call or inline kwargs."""

        with self._lock:
            try:
                reg = self._tools[name]
            except KeyError as exc:
                raise KeyError(name) from exc
            builder = reg.builder
            inline_fn = reg.inline_fn
            args_schema = reg.args_schema

        merged: dict[str, Any] = dict(arguments or {})
        if builder is not None:
            return SandboxToolResolution(call=builder(merged))
        assert inline_fn is not None and args_schema is not None
        try:
            model = args_schema.model_validate(merged)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        kwargs = model.model_dump()
        return InlineToolResolution(fn=inline_fn, kwargs=kwargs)

    def build(self, name: str, arguments: Mapping[str, Any] | None) -> FlowToolCall:
        """Resolve a sandbox tool only; inline tools raise :exc:`TypeError`."""

        resolved = self.resolve(name, arguments)
        if isinstance(resolved, SandboxToolResolution):
            return resolved.call
        raise TypeError(
            f"tool {name!r} is inline-only (@tool); sandbox build() is not supported"
        )

    def copy_all_from(self, source: ToolTable) -> None:
        """Copy every registration from ``source`` into this table (later wins on name clash)."""

        with source._lock:
            regs = list(source._tools.values())
        for reg in regs:
            self.register(reg)

    def copy_named_from(self, source: ToolTable, names: Sequence[str]) -> None:
        """Copy registrations for ``names`` from ``source``; raises :exc:`KeyError` if any name is missing."""

        with source._lock:
            missing = [n for n in names if n not in source._tools]
            if missing:
                raise KeyError(f"unknown tool name(s): {missing}")
            regs = [source._tools[n] for n in names]
        for reg in regs:
            self.register(reg)


_SYSTEM_GLOBAL = ToolTable()
_USER_GLOBAL = ToolTable()
_builtin_install_lock = Lock()
_builtin_installed_flag = False


def _ensure_builtins_on_system() -> None:
    """Populate default sandbox tools on the system table once."""

    global _builtin_installed_flag
    if _builtin_installed_flag:
        return
    with _builtin_install_lock:
        if _builtin_installed_flag:
            return
        from rath.flow.tool._builtins import extend_builtin_sandbox_tools

        extend_builtin_sandbox_tools(_SYSTEM_GLOBAL)
        _builtin_installed_flag = True


def global_system_tool_table() -> ToolTable:
    """Process-wide table for built-in (sandbox) tools, seeded on first access."""

    _ensure_builtins_on_system()
    return _SYSTEM_GLOBAL


def global_user_tool_table() -> ToolTable:
    """Process-wide table for :func:`~rath.flow.tool.decorators.tool` and :func:`register_global_tool`."""

    return _USER_GLOBAL


def global_tool_table() -> ToolTable:
    """Backward-compatible alias for :func:`global_user_tool_table`."""

    return global_user_tool_table()


def build_loop_tool_table(user_tool_names: list[str] | None = None) -> ToolTable:
    """System tools plus the named subset of user tools for one :func:`~rath.session.loop.run_session_loop` run."""

    loop = ToolTable()
    loop.copy_all_from(global_system_tool_table())
    names = list(user_tool_names or ())
    if names:
        loop.copy_named_from(global_user_tool_table(), names)
    return loop


def register_global_tool(registration: ToolRegistration) -> None:
    """Register ``registration`` on :func:`global_user_tool_table` if the name is unused.

    Raises:
        ToolNameConflictError: If ``registration.name`` is already registered.
    """

    global_user_tool_table().register_unique(registration)


__all__ = [
    "InlineToolResolution",
    "SandboxToolResolution",
    "ToolNameConflictError",
    "ToolRegistration",
    "ToolTable",
    "build_loop_tool_table",
    "global_system_tool_table",
    "global_tool_table",
    "global_user_tool_table",
    "register_global_tool",
]
