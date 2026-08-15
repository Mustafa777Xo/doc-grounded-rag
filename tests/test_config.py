from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import Settings, load_config


def test_config_loads_with_required_field() -> None:
    settings = Settings(  # type: ignore[call-arg]
        docs_dir=Path("data/pdfs"),
        _env_file=None,
    )
    assert settings.docs_dir.name == "pdfs"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.chunk_hard_max == 768
    assert settings.chunk_output_path == Path("data/processed/chunks/chunks.jsonl")
    assert settings.chunk_write_mode == "overwrite"
    assert settings.embedding_batch_size == 32
    assert settings.embedding_max_chars == 4096
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.vector_index_path == Path("data/index/vector_store.sqlite")
    assert settings.vector_collection_name == "default"
    assert settings.dense_top_k == 20
    assert settings.keyword_top_k == 20
    assert settings.fusion_rrf_k == 60
    assert settings.rerank_candidate_count == 20
    assert settings.final_result_count == 5
    assert settings.reranker_batch_size == 16
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
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


def test_config_embedding_batch_settings() -> None:
    settings = Settings(
        docs_dir=Path("data/pdfs"),
        embedding_batch_size=8,
        embedding_max_chars=1024,
    )

    assert settings.embedding_batch_size == 8
    assert settings.embedding_max_chars == 1024


def test_config_invalid_embedding_batch_settings_raise() -> None:
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), embedding_batch_size=0)
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), embedding_max_chars=0)


def test_config_vector_index_settings() -> None:
    settings = Settings(
        docs_dir=Path("data/pdfs"),
        vector_index_path=Path("artifacts/vector.sqlite"),
        vector_collection_name="policies",
    )

    assert settings.vector_index_path == Path("artifacts/vector.sqlite")
    assert settings.vector_collection_name == "policies"


def test_config_invalid_vector_collection_name_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), vector_collection_name="")


def test_config_retrieval_settings() -> None:
    settings = Settings(
        docs_dir=Path("data/pdfs"),
        dense_top_k=30,
        keyword_top_k=10,
        fusion_rrf_k=40,
        rerank_candidate_count=25,
        final_result_count=8,
        reranker_batch_size=4,
        embedding_model="org/embedder",
        reranker_model="org/reranker",
    )

    assert settings.dense_top_k == 30
    assert settings.keyword_top_k == 10
    assert settings.fusion_rrf_k == 40
    assert settings.rerank_candidate_count == 25
    assert settings.final_result_count == 8
    assert settings.reranker_batch_size == 4
    assert settings.embedding_model == "org/embedder"
    assert settings.reranker_model == "org/reranker"


def test_config_rejects_non_positive_retrieval_settings() -> None:
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), dense_top_k=0)
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), keyword_top_k=0)
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), fusion_rrf_k=0)
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), rerank_candidate_count=0)
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), final_result_count=0)
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), reranker_batch_size=0)


def test_config_rejects_empty_retrieval_model_names() -> None:
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), embedding_model="")
    with pytest.raises(ValidationError):
        Settings(docs_dir=Path("data/pdfs"), reranker_model="")


def test_config_rejects_final_count_above_rerank_count() -> None:
    with pytest.raises(
        ValidationError,
        match="final_result_count.*rerank_candidate_count",
    ):
        Settings(
            docs_dir=Path("data/pdfs"),
            rerank_candidate_count=5,
            final_result_count=6,
        )


def test_config_rejects_rerank_count_above_retrieval_budget() -> None:
    with pytest.raises(
        ValidationError,
        match="rerank_candidate_count.*dense_top_k.*keyword_top_k",
    ):
        Settings(
            docs_dir=Path("data/pdfs"),
            dense_top_k=5,
            keyword_top_k=5,
            rerank_candidate_count=11,
        )


def test_load_config_fails_clearly_for_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOCS_DIR", "data/pdfs")
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "4")
    monkeypatch.setenv("FINAL_RESULT_COUNT", "5")

    with pytest.raises(
        ValidationError,
        match="final_result_count.*rerank_candidate_count",
    ):
        load_config()
