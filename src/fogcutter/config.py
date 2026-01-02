from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class FogcutterSettings(BaseSettings):
    """
    Fogcutter configuration.

    Load from:
    1. [tool.fogcutter] in pyproject.toml (project default)
    2. ~/.config/fogcutter.toml (user config)
    3. FOGCUTTER_* environment variables
    4. Defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="FOGCUTTER_",
        env_ignore_empty=True,
        extra="ignore",
        env_file="~/.config/fogcutter.toml",
    )

    # Provider defaults
    default_provider: str = "openai"  # "openai", "gemini"
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_project_id: Optional[str] = None
    gemini_vertex_ai: bool = False

    # Metric defaults
    n_samples: int = 5  # For self-consistency, semantic entropy
    temperature: float = 0.7
    logprobs_top_k: int = 5


# Global singleton
settings = FogcutterSettings()
