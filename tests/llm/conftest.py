"""Load .env before session checks; fail fast if credentials are missing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _project_root() -> Path:
    """OpenRath package root (directory containing pyproject.toml)."""
    return Path(__file__).resolve().parents[2]


def pytest_configure(config: pytest.Config) -> None:
    # Real integration tests: load local .env without overriding exported env vars.
    from dotenv import load_dotenv

    env_path = _project_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        pytest.exit(
            "OPENAI_API_KEY must be set (e.g. in .env) for tests/llm integration suite",
            returncode=2,
        )
