"""LangChain-style ``@tool`` decorator with Pydantic ``args_schema``."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from rath.flow.tool.tool_table import ToolRegistration, global_user_tool_table

F = TypeVar("F", bound=Callable[..., Any])


def tool(
    fn: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel],
    register: bool = True,
) -> F | Callable[[F], F]:
    """Register an inline Python tool on :func:`global_user_tool_table`.

    The decorated function runs **in-process** when the model selects this tool.
    Arguments are validated with ``args_schema`` (Pydantic v2).

    Usage::

        @tool(name=\"add\", description=\"Sum\", args_schema=AddIn)
        def add(a: int, b: int) -> int:
            return a + b
    """

    def decorator(f: F) -> F:
        tool_name = name if name is not None else f.__name__
        doc = inspect.getdoc(f)
        desc = description if description is not None else doc
        if desc is not None:
            desc = desc.strip() or None

        params = dict(args_schema.model_json_schema())
        reg = ToolRegistration(
            name=tool_name,
            description=desc,
            parameters=params,
            inline_fn=f,
            args_schema=args_schema,
        )
        if register:
            global_user_tool_table().register(reg)
        return f

    if fn is not None:
        return decorator(fn)
    return decorator


__all__ = ["tool"]
