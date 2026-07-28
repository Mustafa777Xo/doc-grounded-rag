from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
EMBEDDING_DIMENSION = 384
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
DIRECT_PACKAGE_VERSIONS = {
    "bm25s": "0.3.10",
    "sentence-transformers": "5.6.0",
    "torch": "2.7.1",
    "transformers": "4.57.6",
}
TRANSITIVE_PACKAGES = ("numpy", "scipy")
DEFAULT_QUERY = "Who is eligible for policy coverage?"
WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 20
EMBEDDING_MODEL_FILE_PATTERNS = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
RERANKER_MODEL_FILE_PATTERNS = (
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _assert_selected_versions() -> None:
    mismatches = [
        f"{name}={_package_version(name)} (expected {expected})"
        for name, expected in DIRECT_PACKAGE_VERSIONS.items()
        if _package_version(name) != expected
    ]
    if mismatches:
        details = ", ".join(mismatches)
        raise RuntimeError(f"selected package version mismatch: {details}")


def _load_chunk_texts(path: Path) -> tuple[str, ...]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"invalid chunk text at line {line_number}")
            texts.append(text)
    if not texts:
        raise ValueError("chunk artifact must contain at least one chunk")
    return tuple(texts)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(
    samples_ms: Sequence[float],
    items_per_call: int,
) -> dict[str, float]:
    mean_ms = statistics.fmean(samples_ms)
    return {
        "iterations": float(len(samples_ms)),
        "items_per_call": float(items_per_call),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "mean_ms": mean_ms,
        "p50_ms": _percentile(samples_ms, 0.50),
        "p95_ms": _percentile(samples_ms, 0.95),
        "throughput_items_per_second": items_per_call / (mean_ms / 1000.0),
    }


def _measure(call: Any, *, items_per_call: int) -> dict[str, float]:
    for _ in range(WARMUP_ITERATIONS):
        call()
    samples_ms: list[float] = []
    for _ in range(MEASURED_ITERATIONS):
        started = time.perf_counter()
        call()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    return _timing_summary(samples_ms, items_per_call)


def _validate_vectors(vectors: Any, expected_count: int) -> None:
    if tuple(vectors.shape) != (expected_count, EMBEDDING_DIMENSION):
        raise RuntimeError(
            f"unexpected embedding shape {tuple(vectors.shape)}, "
            f"expected {(expected_count, EMBEDDING_DIMENSION)}"
        )
    for vector in vectors:
        values = tuple(float(value) for value in vector)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("embedding contains non-finite values")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise RuntimeError(f"embedding norm {norm} is not approximately 1.0")


def _validate_scores(scores: Any, expected_count: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in scores)
    if len(values) != expected_count:
        raise RuntimeError(
            f"unexpected reranker score count {len(values)}, expected {expected_count}"
        )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("reranker returned non-finite scores")
    return values


def _bm25_smoke(texts: Sequence[str]) -> dict[str, object]:
    import bm25s

    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(
        ["claims filed within days"],
        stopwords="en",
        show_progress=False,
    )
    result_indexes, scores = retriever.retrieve(
        query_tokens,
        k=min(2, len(texts)),
        show_progress=False,
    )
    first_index = int(result_indexes[0, 0])
    first_score = float(scores[0, 0])
    if "claims" not in texts[first_index].lower():
        raise RuntimeError("BM25 smoke query did not rank the claims chunk first")
    if not math.isfinite(first_score):
        raise RuntimeError("BM25 returned a non-finite score")
    return {
        "query": "claims filed within days",
        "top_text": texts[first_index],
        "top_score": first_score,
    }


def _environment() -> dict[str, object]:
    package_names = tuple(DIRECT_PACKAGE_VERSIONS) + TRANSITIVE_PACKAGES
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "device": "cpu",
        "hf_home": os.environ.get("HF_HOME", "default-user-cache"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "packages": {name: _package_version(name) for name in package_names},
    }


def _compatibility_summary(result: dict[str, Any]) -> dict[str, object]:
    environment = result["environment"]
    validation = result["validation"]
    if not isinstance(environment, dict) or not isinstance(validation, dict):
        raise ValueError("compatibility result has an invalid shape")
    return {
        "python": environment["python"],
        "architecture": environment["architecture"],
        "packages": environment["packages"],
        "validation": validation,
    }


def _run_benchmark(chunks_path: Path) -> dict[str, object]:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    _assert_selected_versions()
    environment = _environment()
    if not environment["hf_hub_offline"] or not environment["transformers_offline"]:
        raise RuntimeError(
            "benchmark requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1"
        )

    texts = _load_chunk_texts(chunks_path)

    embedding_load_started = time.perf_counter()
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_ID,
        revision=EMBEDDING_MODEL_REVISION,
        device="cpu",
        local_files_only=True,
    )
    embedding_load_ms = (time.perf_counter() - embedding_load_started) * 1000.0

    def embed_query() -> Any:
        return embedding_model.encode(
            [DEFAULT_QUERY],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_chunks() -> Any:
        return embedding_model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    _validate_vectors(embed_query(), 1)
    _validate_vectors(embed_chunks(), len(texts))

    reranker_load_started = time.perf_counter()
    reranker = CrossEncoder(
        RERANKER_MODEL_ID,
        revision=RERANKER_MODEL_REVISION,
        device="cpu",
        local_files_only=True,
        model_kwargs={"use_safetensors": False},
    )
    reranker_load_ms = (time.perf_counter() - reranker_load_started) * 1000.0

    relevant_text = next(
        (text for text in texts if "eligibility" in text.lower()),
        texts[0],
    )
    irrelevant_text = next(
        (text for text in texts if text != relevant_text),
        "Claims must be filed within 30 days.",
    )
    relevance_pairs = (
        (DEFAULT_QUERY, relevant_text),
        (DEFAULT_QUERY, irrelevant_text),
    )
    relevance_scores = _validate_scores(
        reranker.predict(list(relevance_pairs), show_progress_bar=False),
        len(relevance_pairs),
    )
    if relevance_scores[0] <= relevance_scores[1]:
        raise RuntimeError("reranker did not rank the eligibility passage first")

    single_pair = [relevance_pairs[0]]
    batch_pairs = [relevance_pairs[index % 2] for index in range(10)]

    def rerank_one() -> Any:
        return reranker.predict(single_pair, show_progress_bar=False)

    def rerank_ten() -> Any:
        return reranker.predict(batch_pairs, show_progress_bar=False)

    _validate_scores(rerank_one(), 1)
    _validate_scores(rerank_ten(), 10)

    return {
        "schema_version": "s3-model-selection-benchmark.v1",
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment,
        "models": {
            "embedding": {
                "id": EMBEDDING_MODEL_ID,
                "revision": EMBEDDING_MODEL_REVISION,
                "license": "Apache-2.0",
                "dimension": EMBEDDING_DIMENSION,
                "normalized": True,
                "load_ms": embedding_load_ms,
            },
            "reranker": {
                "id": RERANKER_MODEL_ID,
                "revision": RERANKER_MODEL_REVISION,
                "license": "Apache-2.0",
                "hidden_dimension": 384,
                "output_scores_per_pair": 1,
                "load_ms": reranker_load_ms,
            },
        },
        "bm25": {
            "package": "bm25s",
            "version": DIRECT_PACKAGE_VERSIONS["bm25s"],
            "license": "MIT",
            "stemming": False,
            "smoke": _bm25_smoke(texts),
        },
        "workload": {
            "chunk_count": len(texts),
            "warmup_iterations": WARMUP_ITERATIONS,
            "measured_iterations": MEASURED_ITERATIONS,
        },
        "latency": {
            "embedding_query": _measure(embed_query, items_per_call=1),
            "embedding_chunk_batch": _measure(
                embed_chunks,
                items_per_call=len(texts),
            ),
            "reranker_single_pair": _measure(rerank_one, items_per_call=1),
            "reranker_ten_pairs": _measure(rerank_ten, items_per_call=10),
        },
        "validation": {
            "embedding_shape": [len(texts), EMBEDDING_DIMENSION],
            "embedding_vectors_finite_and_normalized": True,
            "reranker_scores_finite": True,
            "reranker_relevant_passage_ranked_first": True,
            "bm25_exact_term_passage_ranked_first": True,
            "offline_execution": True,
            "torch_threads": torch.get_num_threads(),
        },
    }


def _prefetch() -> None:
    from huggingface_hub import snapshot_download

    _assert_selected_versions()
    for model_id, revision, allow_patterns in (
        (
            EMBEDDING_MODEL_ID,
            EMBEDDING_MODEL_REVISION,
            EMBEDDING_MODEL_FILE_PATTERNS,
        ),
        (
            RERANKER_MODEL_ID,
            RERANKER_MODEL_REVISION,
            RERANKER_MODEL_FILE_PATTERNS,
        ),
    ):
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=allow_patterns,
        )
        print(json.dumps({"model_id": model_id, "revision": revision, "path": path}))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch and benchmark the pinned Sprint 3 retrieval stack."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prefetch", help="Download pinned model snapshots.")
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the benchmark from the populated cache in offline mode.",
    )
    benchmark_parser.add_argument("--chunks", type=Path, required=True)
    benchmark_parser.add_argument(
        "--compatibility-result",
        action="append",
        default=[],
        type=Path,
        help="Prior benchmark result to summarize in the compatibility matrix.",
    )
    benchmark_parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "prefetch":
        _prefetch()
        return 0

    result = _run_benchmark(args.chunks)
    compatibility_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.compatibility_result
    ]
    compatibility_results.append(result)
    result["compatibility"] = sorted(
        (_compatibility_summary(item) for item in compatibility_results),
        key=lambda item: str(item["python"]),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
