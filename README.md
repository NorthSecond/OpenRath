# OpenRath

OpenRath is a small Python framework for **session-centered, tool-using agent workflows**.
It keeps agent state in `Session`, runs tools through sandbox `Backend`s, and composes
agents with a PyTorch-like `Workflow` / `AgentParam` API.

## Install

```bash
git clone https://github.com/Rath-Team/OpenRath.git
cd OpenRath
pip install -e .
```

For development with `uv`:

```bash
uv sync --dev
```

For OpenSandbox support:

```bash
pip install -e ".[opensandbox]"
```

Create `.env` from `.env.example` and set:

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_DEFAULT_MODEL=...
```

## Minimal Session Loop

```python
import rath.flow as flow
from rath.session import Session, run_session_loop

agent_session = Session.from_agent_prompt("You are a concise assistant.")
user_session = Session.from_user_message("List this workspace and summarize it.")
user_session = user_session.to("local", spec="./")

out = run_session_loop(
    user_session=user_session,
    agent_session=agent_session,
    agent_provider=flow.Provider(model="gpt-5.5"),
)

print(out)
```

`run_session_loop` requires the user session to carry a sandbox target or handle.
Call `Session.to("local")`, `Session.to("opensandbox")`, or `Session.with_sandbox(...)`
before entering the loop.

## Core Ideas

| Concept | Code | What it does |
| --- | --- | --- |
| Session | `rath.session.Session` | Stores ordered chunks and optional sandbox placement. |
| Backend | `rath.backend.Backend` | Opens sandboxes and dispatches typed backend tool payloads. |
| Tool | `rath.flow.tool.FlowToolCall` | Exposes model-visible function schemas and executes against a `Session`. |
| Workflow | `rath.flow.Workflow` | Composes `AgentParam` objects and transforms sessions. |
| Provider | `rath.llm.Provider` | Carries OpenAI-style model and sampling options. |

## Documentation

Build the local docs with:

```bash
bash scripts/build_docs.sh
```

The Chinese documentation starts at [`docs/source/index.md`](docs/source/index.md).

Runnable examples live under [`example/`](example/):

| Example | What it demonstrates |
| --- | --- |
| `session_usage.py` | Session loop and compression. |
| `custom_tool_usage.py` | User-defined `FlowToolCall`. |
| `sandbox_backend_local.py` | Local backend workspace binding. |
| `sandbox_backend_opensandbox.py` | OpenSandbox backend binding. |
| `trading_agents/` | Sequential multi-agent research workflow with a market-data tool. |
| `engineering_agents/` | Nested engineering workflow with lead, squad, backend pair, and QA roles. |

## Current Scope

OpenRath currently provides:

- local process sandbox backend;
- optional OpenSandbox backend;
- typed backend tool payloads and results;
- OpenAI-compatible synchronous chat client;
- blocking session loop with tool rounds;
- `Workflow`, `AgentParam`, `Agent`, and `SessionCompressor`;
- session lineage helpers for debugging and provenance.

The API is still early. Prefer reading the user guide and examples before treating
any internal module as stable.
