"""Conformance fixture: parametrize over every available backend.

Phase 1 only includes ``local``. Phase 3 will add ``opensandbox`` parametrized
with a skip marker that activates when a real ``opensandbox-server`` is
running on localhost.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from rath.backend import Backend, get

_BACKEND_PARAMS = [
    pytest.param("local", id="local"),
]


@pytest.fixture(params=_BACKEND_PARAMS)
async def backend(request: pytest.FixtureRequest) -> AsyncIterator[Backend]:
    """Yield a fresh backend instance for each test."""
    bk = get(request.param)
    yield bk
