# rath.backend design

This document is the repository-facing design for the `rath.backend` module: a
torch-inspired abstraction over sandbox **compute backends** (`local` host
processes vs container runtimes such as OpenSandbox). It complements the
working plan under the maintainer’s `.claude/` tree without duplicating every
revision.

## Scope

- **In scope**: `rath.backend` — backends, sandbox handles, tool calls/results,
  registry, `Stream` / `Event`, `LocalBackend`, `OpenSandboxBackend`.
- **Out of scope**: `rath.Session`, `rath.Agent`, and any LLM-loop code.
  Session is conceptualized elsewhere as the “graph” that **records**
  `ToolCall` nodes; this layer only defines immutable value objects suitable
  for that future use.

## Core concepts

| Concept | Role |
|--------|------|
| `ToolCall` / `ToolResult` | Frozen, slotted dataclasses — the minimum unit of work and its outcome. |
| `Backend` | Unified `dispatch(sandbox, call)` entry point plus lifecycle and metadata. |
| `Sandbox` | Opaque handle issued by a backend — **no** conversational / LLM context. |
| `Stream` | Per-sandbox FIFO queue of tool calls, one async worker; several streams on one sandbox run concurrently. |
| `Event` | Cross-stream “happens-after” barrier between submission timelines. |

## Why Stream and Event

- **Stream**: Mirrors the need for ordered submission with optional overlap.
  One stream preserves FIFO order for dependent tool chains (write file → run
  command → read file). Multiple streams on the same sandbox allow concurrent
  independent work (e.g. overlapping I/O-bound RPCs) without baking threading
  into each backend.
- **Event**: Expresses synchronization **between** streams without coupling
  them into a single queue — e.g. “stream B may proceed only after stream A
  has finished everything up to point *E*.”

Both are implemented with **anyio** so every backend that implements async
`dispatch` gets the same semantics.

## Backend contract

- **Static**: `is_available()`, `capabilities()`, `supported_calls()`.
- **Instance**: `sandbox_count()`, `open(spec)`, `close(sandbox)`,
  `dispatch(sandbox, call) -> ToolResult | bool` (`FilesExists` returns `bool`).
- **Errors**: `UnsupportedToolCall`, `SandboxClosed`, `BackendNotFound` under
  `BackendError`.

There is **no** “device” abstraction; pool size is reflected in
`sandbox_count()` and optional `Capabilities` hints.

## OpenSandbox adapter

- Requires optional dependencies: `opensandbox`, `opensandbox-code-interpreter`
  (`pip install rath[opensandbox]`).
- Expects a reachable **opensandbox-server** (e.g. via `OPENSANDBOX_DOMAIN` and
  `OPENSANDBOX_PROTOCOL`).
- Relative paths and default command cwd use sandbox-internal `/workspace`.
- `CommandRun` with `stdin` raises `UnsupportedToolCall` (SDK limitation).
- Client-side `anyio.fail_after` enforces `timeout=` on commands and code runs
  so callers see `TimeoutError` even when server behaviour varies.
- Default container image entrypoint targets `opensandbox/code-interpreter`
  images so the code interpreter service starts reliably.

## Local backend

- Always available: subprocesses via `anyio.run_process`, filesystem via
  `anyio.Path`, one temp directory per sandbox handle.

## Availability checks

- `Backend.is_available()` is static and **non-invasive** (imports + config
  keys such as `OPENSANDBOX_DOMAIN` / `~/.sandbox.toml` for OpenSandbox).
- Integration tests use a **TCP probe** to `localhost:8080` (`opensandbox_real`
  marker); they are skipped when no server listens, not “mocked green”.

## Testing layout

| Layer | Purpose |
|-------|---------|
| `tests/unit/` | Value semantics, registry, streams, errors. |
| `tests/backends/` | Backend-specific behaviour. |
| `tests/conformance/` | Same scenarios parametrized over `local` and
  `opensandbox` (when live). |

Conformance uses a `python_cmd` fixture: host interpreter for `local`,
`python3` in-container for OpenSandbox (portable from Windows hosts).

## Further reading

- [CHANGELOG](../../CHANGELOG.md) — release-oriented notes.
- PyTorch analogy: backends resemble `torch.cuda` / `torch.mps`; `Stream` /
  `Event` resemble CUDA stream/event ordering, but **without** a device type.
