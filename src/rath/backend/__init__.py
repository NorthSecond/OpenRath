"""Public API of :mod:`rath.backend`.

Re-exports the ABCs, value types, registry helpers and built-in backends. The
``local`` backend is imported eagerly (it is dependency-free) so that
``rath.backend.is_available("local")`` works without prior side-effects from
the caller.
"""

from __future__ import annotations

from rath.backend.flow_tool_shim import (
    FlowToolCall,
    FlowToolCodeRun,
    FlowToolCommandRun,
    FlowToolFilesExists,
    FlowToolFilesList,
    FlowToolFilesRead,
    FlowToolFilesWrite,
)
from rath.backend.abc import Backend, BackendSandbox, BackendSandboxSpec
from rath.backend.capabilities import Capabilities, IsolationLevel
from rath.backend.errors import (
    BackendError,
    BackendNotFound,
    BackendSandboxClosed,
    UnsupportedFlowToolCall,
)
from rath.backend.registry import (
    current,
    get,
    get_class,
    is_available,
    list_names,
    preferred,
    register,
    set_default,
)
from rath.backend.results import (
    CodeResult,
    CommandResult,
    FileContent,
    FileEntries,
    FileEntry,
    FileWriteResult,
    ToolResult,
)
from rath.backend.stream import Event, Future, Stream

# Eagerly register built-in backends. ``local`` has no extra deps; ``opensandbox``
# is guarded so the package stays importable without the optional extra.
from rath.backend import local as _local  # noqa: F401

try:
    from rath.backend import opensandbox as _opensandbox  # noqa: F401
except ImportError:  # pragma: no cover - exercised when extra is missing
    pass

__all__ = [
    # ABC + handles
    "Backend",
    "BackendSandbox",
    "BackendSandboxSpec",
    # Flow tool calls
    "FlowToolCall",
    "FlowToolCommandRun",
    "FlowToolFilesRead",
    "FlowToolFilesWrite",
    "FlowToolFilesList",
    "FlowToolFilesExists",
    "FlowToolCodeRun",
    # Tool results
    "ToolResult",
    "CommandResult",
    "FileContent",
    "FileEntry",
    "FileEntries",
    "FileWriteResult",
    "CodeResult",
    # Capabilities
    "Capabilities",
    "IsolationLevel",
    # Errors
    "BackendError",
    "BackendNotFound",
    "BackendSandboxClosed",
    "UnsupportedFlowToolCall",
    # Concurrency primitives
    "Stream",
    "Event",
    "Future",
    # Registry
    "register",
    "list_names",
    "get",
    "get_class",
    "is_available",
    "preferred",
    "set_default",
    "current",
]
