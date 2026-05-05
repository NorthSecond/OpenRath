"""Tool call value objects.

A :class:`ToolCall` is the first-class unit of work that a :class:`Backend`
executes. Every concrete subclass is a frozen dataclass with ``slots=True`` so
that calls have value semantics: they are immutable, comparable, picklable and
(for hashable field values) hashable. This is the same property that makes
PyTorch operators safe to use as nodes in ``torch.fx`` graphs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class ToolCall:
    """Marker base class for all tool calls.

    Concrete subclasses are dataclasses; this base only exists so that
    :meth:`Backend.supported_calls` can describe its support set as a
    ``frozenset[type[ToolCall]]`` and so that ``match`` statements have a
    single discriminator type.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class CommandRun(ToolCall):
    """Run a shell command inside the sandbox."""

    cmd: str | Sequence[str]
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    stdin: bytes | None = None
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class FilesRead(ToolCall):
    """Read a file from the sandbox.

    With ``encoding`` set (default ``"utf-8"``) the result is text; pass
    ``encoding=None`` to read raw bytes.
    """

    path: str
    encoding: str | None = "utf-8"


@dataclass(frozen=True, slots=True)
class FilesWrite(ToolCall):
    """Write a file inside the sandbox with the given Unix mode."""

    path: str
    data: bytes | str
    mode: int = 0o644


@dataclass(frozen=True, slots=True)
class FilesList(ToolCall):
    """List entries (non-recursive) under a sandbox directory."""

    path: str


@dataclass(frozen=True, slots=True)
class FilesExists(ToolCall):
    """Check whether a path exists inside the sandbox."""

    path: str


@dataclass(frozen=True, slots=True)
class CodeRun(ToolCall):
    """Execute a code snippet inside the sandbox in the given language."""

    code: str
    language: str = "python"
    timeout: float | None = None
