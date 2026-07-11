from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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
    chunk_hard_max: int = Field(default=768, gt=0)
    chunk_output_path: Path = Path("data/processed/chunks/chunks.jsonl")
    chunk_write_mode: Literal["overwrite", "safe_append"] = "overwrite"
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_max_chars: int = Field(default=4096, gt=0)
    vector_index_path: Path = Path("data/index/vector_store.sqlite")
    vector_collection_name: str = Field(default="default", min_length=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    profile: Literal["dev", "test"] = "dev"

    @model_validator(mode="after")
    def validate_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.chunk_size > self.chunk_hard_max:
            raise ValueError("chunk_size must be less than or equal to chunk_hard_max")
        return self


def load_config() -> Settings:
    return Settings()  # type: ignore[call-arg]
