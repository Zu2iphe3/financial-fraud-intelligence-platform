"""Opt-in validation with the real local embedding model and Ollama service."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from src.rag.assistant import PolicyAssistant
from src.rag.config import load_rag_settings
from src.rag.ingestion import rebuild_index
from src.rag.retrieval import PolicyRetriever, evaluate_retrieval


@pytest.mark.rag
@pytest.mark.skipif(
    os.getenv("RUN_RAG_MODEL_TESTS") != "1",
    reason="requires RUN_RAG_MODEL_TESTS=1 and the local embedding model",
)
def test_real_embedding_model_meets_retrieval_evaluation(tmp_path) -> None:
    settings = replace(load_rag_settings(), vector_store_path=tmp_path / "chroma")
    rebuild_index(settings)

    result = evaluate_retrieval(PolicyRetriever.from_settings(settings))

    assert result.covered_with_expected_source == result.covered_total
    assert result.unsupported_rejected == result.unsupported_total


@pytest.mark.rag
@pytest.mark.ollama
@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="requires RUN_OLLAMA_TESTS=1, local embeddings, and a running Ollama model",
)
def test_real_ollama_returns_a_grounded_answer_with_sources(tmp_path) -> None:
    settings = replace(load_rag_settings(), vector_store_path=tmp_path / "chroma")
    rebuild_index(settings)
    retriever = PolicyRetriever.from_settings(settings)

    response = PolicyAssistant(settings, retriever).answer(
        "What should an analyst investigate for a transfer to a new destination?"
    )

    assert response.status == "answered"
    assert response.answer
    assert response.sources
    assert all(source.source_id and source.section_id for source in response.sources)
