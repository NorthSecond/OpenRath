"""Load credentials and defaults from ``.env`` / process environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

__all__ = ["LLMSettings", "default_dotenv_path", "load_llm_settings"]


def default_dotenv_path() -> Path:
    """``OpenRath/OpenRath/.env`` next to ``pyproject.toml``."""
    # src/rath/llm/_settings.py -> parents[3] = project root containing pyproject.toml
    return Path(__file__).resolve().parents[3] / ".env"


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Values used to construct ``openai.OpenAI`` and default chat ``model``."""

    api_key: str
    base_url: str | None = None
    default_model: str | None = None


def load_llm_settings(dotenv_path: Path | None = None) -> LLMSettings:
    """Load ``OPENAI_*`` from ``.env`` (if present) then read environment.

    Existing environment variables are not overwritten by ``.env``
    (``load_dotenv(..., override=False)``).
    """
    path = dotenv_path if dotenv_path is not None else default_dotenv_path()
    if path.is_file():
        load_dotenv(path, override=False)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_raw = os.environ.get("OPENAI_BASE_URL", "").strip()
    model_raw = os.environ.get("OPENAI_DEFAULT_MODEL", "").strip()

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is empty: set it in .env or the environment",
        )

    return LLMSettings(
        api_key=api_key,
        base_url=base_raw or None,
        default_model=model_raw or None,
    )
