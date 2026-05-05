"""Conformance: cancellation of in-flight dispatches.

Cancelling a slow dispatch must not corrupt the sandbox; subsequent dispatches
on the same handle should still succeed.
"""

from __future__ import annotations

import sys

import anyio
import pytest

from rath.backend import Backend, CommandResult, CommandRun

pytestmark = pytest.mark.anyio


async def test_cancellation_leaves_sandbox_usable(backend: Backend) -> None:
    async with await backend.open() as sb:
        with anyio.move_on_after(0.3):
            await sb.dispatch(
                CommandRun(
                    cmd=[
                        sys.executable,
                        "-c",
                        "import time; time.sleep(10)",
                    ]
                )
            )
        # The cancel scope exited normally; the sandbox must still be usable.
        result = await sb.dispatch(
            CommandRun(cmd=[sys.executable, "-c", "print('after')"])
        )
        assert isinstance(result, CommandResult)
        assert result.exit_code == 0
        assert b"after" in result.stdout
