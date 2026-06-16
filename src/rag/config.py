from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    docs_dir: Path  # required — no default
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    profile: Literal["dev", "test"] = "dev"


def load_config() -> Settings:
    return Settings()  # type: ignore[call-arg]
