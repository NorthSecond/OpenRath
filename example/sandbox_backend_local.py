"""Example: session loop with the local subprocess sandbox (LLM keys as needed)."""

import rath.flow as flow
import rath.backend as backend
from rath.session import Session, run_session_loop


SANDBOX_BACKEND = "local"

Agent = flow.Agent(
    agent_session=Session.from_agent_prompt("You are a helpful assistant."),
    provider=flow.Provider(model="glm-5.1"),
)


def main() -> None:
    user_session = Session.from_user_message(
        "List all files in the current directory. And summarize the result."
    )
    print(user_session.sandbox_backend)

    # No working directory: ephemeral empty workspace.
    user_session = user_session.to(SANDBOX_BACKEND, spec=None)
    out_session = run_session_loop(
        user_session=user_session,
        agent_session=Agent.agent_session,
        agent_provider=Agent.provider,
    )
    print(out_session.chunk_table.rows[-1].payload["content"])

    # Project root on the host.
    user_session = user_session.to(SANDBOX_BACKEND, spec=".")
    out_session = run_session_loop(
        user_session=user_session,
        agent_session=Agent.agent_session,
        agent_provider=Agent.provider,
    )
    print(out_session.chunk_table.rows[-1].payload["content"])


if __name__ == "__main__":
    if backend.get(SANDBOX_BACKEND).is_available():
        main()
    else:
        print(f"Sandbox backend {SANDBOX_BACKEND} is not available.")
