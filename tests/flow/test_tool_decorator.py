"""Tests for :func:`~rath.flow.tool.decorators.tool`."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from rath.flow.tool import InlineToolResolution, global_tool_table, tool


class _AddIn(BaseModel):
    a: int = Field(description="left")
    b: int = Field(description="right")


def test_tool_registers_inline_on_global() -> None:
    g = global_tool_table()
    name = "_rath_probe_math_add"
    g.unregister(name)

    @tool(name=name, description="add two ints", args_schema=_AddIn)
    def _math_add(a: int, b: int) -> int:
        return a + b

    try:
        resolved = g.resolve(name, {"a": 2, "b": 3})
        assert isinstance(resolved, InlineToolResolution)
        assert resolved.fn(**resolved.kwargs) == 5
    finally:
        g.unregister(name)


def test_inline_only_build_raises() -> None:
    g = global_tool_table()
    name = "_rath_probe_inline_only"
    g.unregister(name)

    @tool(name=name, args_schema=_AddIn)
    def _add(a: int, b: int) -> int:
        return a + b

    try:
        with pytest.raises(TypeError, match="inline-only"):
            g.build(name, {"a": 1, "b": 1})
    finally:
        g.unregister(name)
