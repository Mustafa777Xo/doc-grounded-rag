from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import Settings


def test_config_loads_with_required_field() -> None:
    settings = Settings(docs_dir=Path("data/pdfs"))
    assert settings.docs_dir.name == "pdfs"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.chunk_hard_max == 768
    assert settings.chunk_output_path == Path("data/processed/chunks/chunks.jsonl")
    assert settings.chunk_write_mode == "overwrite"
    assert settings.profile == "dev"


def test_config_missing_docs_dir_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remove from env and bypass .env file so docs_dir is truly absent
    monkeypatch.delenv("DOCS_DIR", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_config_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), chunk_size=0)


def test_config_invalid_chunk_overlap_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), chunk_size=128, chunk_overlap=128)


def test_config_invalid_chunk_hard_max_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), chunk_size=512, chunk_hard_max=256)


def test_config_invalid_profile_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), profile="production")  # type: ignore[arg-type]


def test_config_test_profile() -> None:
    settings = Settings(docs_dir=Path("data/pdfs"), profile="test")
    assert settings.profile == "test"


def test_config_docs_dir_is_path_type() -> None:
    settings = Settings(docs_dir=Path("some/path"))
    assert isinstance(settings.docs_dir, Path)


def test_config_chunk_storage_settings() -> None:
    settings = Settings(
        docs_dir=Path("data/pdfs"),
        chunk_output_path=Path("artifacts/chunks.jsonl"),
        chunk_write_mode="safe_append",
    )

    assert settings.chunk_output_path == Path("artifacts/chunks.jsonl")
    assert settings.chunk_write_mode == "safe_append"


def test_config_invalid_chunk_write_mode_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(
            docs_dir=Path("data/pdfs"),
            chunk_write_mode="append",  # type: ignore[arg-type]
        )
