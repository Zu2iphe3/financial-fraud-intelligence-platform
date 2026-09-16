"""Tests for deterministic, section-aware policy indexing."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from langchain_core.embeddings import Embeddings

from src.rag.config import load_rag_settings
from src.rag.ingestion import create_policy_chunks, open_vector_store, rebuild_index


class DeterministicEmbeddings(Embeddings):
    """Fast local vectors keep index tests independent from model downloads."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def test_policy_chunks_preserve_document_and_section_metadata() -> None:
    settings = load_rag_settings(environ={})
    chunks, chunk_ids = create_policy_chunks(settings)

    assert len(chunks) == len(chunk_ids) == len(set(chunk_ids))
    assert {item.metadata["source_id"] for item in chunks} == {
        "fraud-investigation-policy-v1",
        "fraud-escalation-policy-v1",
        "analyst-investigation-guide-v1",
        "risk-alert-interpretation-guide-v1",
    }
    assert len({item.metadata["section_id"] for item in chunks}) == 25
    assert all(item.metadata["fictional"] is True for item in chunks)
    assert all(item.metadata["chunk_id"] in chunk_ids for item in chunks)


def test_rebuilding_an_unchanged_index_is_idempotent(tmp_path) -> None:
    settings = replace(load_rag_settings(environ={}), vector_store_path=tmp_path / "chroma")
    embedding = DeterministicEmbeddings()

    first = rebuild_index(settings, embedding)
    first_store = open_vector_store(settings, embedding)
    first_results = first_store.similarity_search("investigation actions", k=4)
    second = rebuild_index(settings, embedding)
    store = open_vector_store(settings, embedding)
    stored = store.get(include=["metadatas"])
    second_results = store.similarity_search("investigation actions", k=4)

    assert first.chunk_ids == second.chunk_ids
    assert first.chunk_count == second.chunk_count == len(stored["ids"])
    assert len(stored["ids"]) == len(set(stored["ids"]))
    assert all(metadata["section_id"] for metadata in stored["metadatas"])
    assert [item.metadata["chunk_id"] for item in first_results] == [
        item.metadata["chunk_id"] for item in second_results
    ]
