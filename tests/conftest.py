"""Top-level pytest configuration shared by all test layers."""

from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Pin async tests to asyncio; trio coverage is out of scope for now."""
    return "asyncio"
