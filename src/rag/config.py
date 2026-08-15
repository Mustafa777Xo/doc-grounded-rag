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
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
    )
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_max_chars: int = Field(default=4096, gt=0)
    vector_index_path: Path = Path("data/index/vector_store.sqlite")
    vector_collection_name: str = Field(default="default", min_length=1)
    dense_top_k: int = Field(default=20, gt=0)
    keyword_top_k: int = Field(default=20, gt=0)
    fusion_rrf_k: int = Field(default=60, gt=0)
    rerank_candidate_count: int = Field(default=20, gt=0)
    final_result_count: int = Field(default=5, gt=0)
    reranker_batch_size: int = Field(default=16, gt=0)
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        min_length=1,
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    profile: Literal["dev", "test"] = "dev"

    @model_validator(mode="after")
    def validate_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.chunk_size > self.chunk_hard_max:
            raise ValueError("chunk_size must be less than or equal to chunk_hard_max")
        if self.final_result_count > self.rerank_candidate_count:
            raise ValueError(
                "final_result_count must be less than or equal to "
                "rerank_candidate_count"
            )
        if self.rerank_candidate_count > self.dense_top_k + self.keyword_top_k:
            raise ValueError(
                "rerank_candidate_count must be less than or equal to "
                "dense_top_k + keyword_top_k"
            )
        return self


def load_config() -> Settings:
    return Settings()  # type: ignore[call-arg]
