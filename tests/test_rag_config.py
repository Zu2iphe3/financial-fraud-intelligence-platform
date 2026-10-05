"""Tests for local-only RAG settings."""

from __future__ import annotations

import json

import pytest

from src.rag.config import DEFAULT_CONFIG_PATH, RagConfigurationError, load_rag_settings


def test_load_rag_settings_uses_repository_defaults() -> None:
    settings = load_rag_settings(environ={})

    assert settings.corpus_path.name == "knowledge_base"
    assert settings.manifest_path == settings.corpus_path / "manifest.json"
    assert settings.vector_store_path.name == "chroma_db"
    assert settings.minimum_relevance_score == 0.40


def test_load_rag_settings_accepts_local_environment_overrides(tmp_path) -> None:
    settings = load_rag_settings(
        environ={
            "RAG_CHROMA_PATH": str(tmp_path / "index"),
            "RAG_EMBEDDING_MODEL": "local-test-embedding",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "OLLAMA_CHAT_MODEL": "local-test-chat",
        }
    )

    assert settings.vector_store_path == tmp_path / "index"
    assert settings.embedding_model == "local-test-embedding"
    assert settings.ollama_chat_model == "local-test-chat"


def test_load_rag_settings_rejects_a_non_local_ollama_url(tmp_path) -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["ollama_base_url"] = "https://hosted-model.example.com"
    path = tmp_path / "rag.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RagConfigurationError, match="local host"):
        load_rag_settings(path, environ={})
