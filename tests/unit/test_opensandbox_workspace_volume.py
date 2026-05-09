"""Bind-mount helper tests for ``BackendSandboxSpec.working_dir``."""

from __future__ import annotations

from pathlib import Path

import pytest

from rath.backend.abc import BackendSandboxSpec


def test_bind_workspace_returns_none_without_working_dir() -> None:
    from rath.backend.opensandbox import bind_workspace_volumes_from_spec

    assert bind_workspace_volumes_from_spec(None, "/workspace") is None
    assert bind_workspace_volumes_from_spec(BackendSandboxSpec(), "/workspace") is None


def test_bind_workspace_creates_dir_and_sets_host_mount(tmp_path: Path) -> None:
    pytest.importorskip("opensandbox")
    from rath.backend.opensandbox import bind_workspace_volumes_from_spec

    sub = tmp_path / "nested" / "ws"
    assert not sub.exists()
    spec = BackendSandboxSpec(working_dir=str(sub))
    vols = bind_workspace_volumes_from_spec(spec, "/workspace")
    assert vols is not None
    assert len(vols) == 1
    v0 = vols[0]
    assert v0.name == "rath-workspace"
    assert v0.mount_path == "/workspace"
    assert v0.host is not None
    assert Path(v0.host.path) == sub.resolve()
    assert sub.is_dir()


def test_bind_workspace_resolves_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("opensandbox")
    from rath.backend.opensandbox import bind_workspace_volumes_from_spec

    monkeypatch.chdir(tmp_path)
    spec = BackendSandboxSpec(working_dir=".")
    vols = bind_workspace_volumes_from_spec(spec, "/workspace")
    assert vols is not None
    assert Path(vols[0].host.path) == tmp_path.resolve()
