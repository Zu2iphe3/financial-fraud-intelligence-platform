"""Load reproducible, local-only settings for the fictional policy assistant."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config" / "rag" / "policy_assistant.json"


class RagConfigurationError(ValueError):
    """Raised when the local RAG configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class RagSettings:
    """Settings shared by indexing, retrieval, and local answer generation."""

    config_version: str
    corpus_path: Path
    manifest_path: Path
    vector_store_path: Path
    collection_name: str
    embedding_model: str
    ollama_base_url: str
    ollama_chat_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    minimum_relevance_score: float
    insufficient_evidence_message: str


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_rag_settings(
    path: str | Path = DEFAULT_CONFIG_PATH,
    environ: Mapping[str, str] | None = None,
) -> RagSettings:
    """Load committed defaults while allowing local services and caches to vary."""

    environment = os.environ if environ is None else environ
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        corpus_path = _repository_path(str(document["corpus_path"]))
        settings = RagSettings(
            config_version=str(document["config_version"]),
            corpus_path=corpus_path,
            manifest_path=_repository_path(str(document["manifest_path"])),
            vector_store_path=_repository_path(
                environment.get("RAG_CHROMA_PATH", str(document["vector_store_path"]))
            ),
            collection_name=str(document["collection_name"]),
            embedding_model=environment.get(
                "RAG_EMBEDDING_MODEL", str(document["embedding_model"])
            ),
            ollama_base_url=environment.get(
                "OLLAMA_BASE_URL", str(document["ollama_base_url"])
            ).rstrip("/"),
            ollama_chat_model=environment.get(
                "OLLAMA_CHAT_MODEL", str(document["ollama_chat_model"])
            ),
            chunk_size=int(document["chunk_size"]),
            chunk_overlap=int(document["chunk_overlap"]),
            retrieval_top_k=int(document["retrieval_top_k"]),
            minimum_relevance_score=float(document["minimum_relevance_score"]),
            insufficient_evidence_message=str(document["insufficient_evidence_message"]),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RagConfigurationError("RAG configuration is invalid") from error

    expected_manifest = settings.corpus_path / "manifest.json"
    if settings.manifest_path.resolve() != expected_manifest.resolve():
        raise RagConfigurationError("Manifest must be the PH9-01 corpus manifest")
    if not settings.collection_name.strip():
        raise RagConfigurationError("Chroma collection name cannot be empty")
    if settings.chunk_size <= 0 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise RagConfigurationError("Chunk overlap must be non-negative and smaller than chunk size")
    if settings.retrieval_top_k <= 0:
        raise RagConfigurationError("Retrieval top-k must be positive")
    if not 0 <= settings.minimum_relevance_score <= 1:
        raise RagConfigurationError("Minimum relevance score must be between zero and one")
    if not settings.embedding_model.strip() or not settings.ollama_chat_model.strip():
        raise RagConfigurationError("Local embedding and Ollama model names are required")
    ollama_host = urlparse(settings.ollama_base_url).hostname
    if ollama_host not in {"localhost", "127.0.0.1", "::1"}:
        raise RagConfigurationError("Ollama must use a local host")
    if not settings.insufficient_evidence_message.strip():
        raise RagConfigurationError("An insufficient-evidence message is required")
    return settings
