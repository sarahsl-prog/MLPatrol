"""Centralized configuration helpers for MLPatrol."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore


@dataclass(frozen=True)
class Settings:
    """Typed container for application configuration."""

    anthropic_api_key: Optional[str]
    openai_api_key: Optional[str]
    nvd_api_key: Optional[str]
    use_local_llm: bool
    local_llm_model: str
    local_llm_url: str
    log_level: str = "INFO"

    @property
    def has_cloud_llm(self) -> bool:
        """True if either Anthropic or OpenAI credentials are present."""
        return bool(self.anthropic_api_key or self.openai_api_key)


def _load_env_file(env_file: Optional[str]) -> None:
    """Load a .env file when python-dotenv is available."""
    if not env_file or load_dotenv is None:
        return
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)


@lru_cache(maxsize=1)
def get_settings(env_file: Optional[str] = None) -> Settings:
    """Return cached application settings."""
    _load_env_file(env_file)

    use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        nvd_api_key=os.getenv("NVD_API_KEY"),
        use_local_llm=use_local_llm,
        local_llm_model=os.getenv("LOCAL_LLM_MODEL", "ollama/llama3.1:8b"),
        local_llm_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434"),
        log_level=os.getenv("MLPATROL_LOG_LEVEL", "INFO").upper(),
    )


def refresh_settings(env_file: Optional[str] = None) -> Settings:
    """Clear cached settings and reload them (useful for tests)."""
    get_settings.cache_clear()
    return get_settings(env_file)