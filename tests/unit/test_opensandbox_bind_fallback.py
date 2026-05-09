"""Unit tests for OpenSandbox workspace bind rejection heuristics."""

from __future__ import annotations

import pytest

pytest.importorskip("opensandbox")

from opensandbox.exceptions import SandboxApiException  # noqa: E402

from rath.backend import opensandbox as osh  # noqa: E402


def test_likely_reject_uses_status_code() -> None:
    exc = SandboxApiException("Create sandbox failed: HTTP 422", status_code=422)
    assert osh._likely_workspace_bind_rejected(exc) is True
    exc2 = SandboxApiException("bad", status_code=500)
    assert osh._likely_workspace_bind_rejected(exc2) is False


def test_likely_reject_falls_back_to_message() -> None:
    err = RuntimeError("HTTP 422 whatever")
    assert osh._likely_workspace_bind_rejected(err) is True


def test_likely_reject_opensandbox_storage_allowlist_message() -> None:
    """Same classification when the HTTP status is not 422."""

    msg = (
        r"Create sandbox failed: Host path 'T:\OpenRath\OpenRath' is not under any "
        r"allowed prefix. Allowed prefixes: []"
    )
    exc = SandboxApiException(msg, status_code=400)
    assert osh._likely_workspace_bind_rejected(exc) is True

    exc_none = SandboxApiException(msg, status_code=None)
    assert osh._likely_workspace_bind_rejected(exc_none) is True
