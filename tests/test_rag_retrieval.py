"""Tests for evidence thresholds and stable source attribution."""

from __future__ import annotations

from langchain_core.documents import Document
import pytest

from src.rag.config import load_rag_settings
from src.rag.retrieval import PolicyRetriever, RetrievalError


def _document(section_id: str, *, complete: bool = True) -> Document:
    metadata = {
        "source_id": "guide-v1",
        "title": "Guide",
        "source_path": "guides/guide.md",
        "section_id": section_id,
        "section_heading": "Investigation actions",
        "chunk_id": f"chunk-{section_id}",
        "fictional": True,
    }
    if not complete:
        metadata.pop("source_id")
    return Document(page_content="Review the earlier transaction activity.", metadata=metadata)


class ResultStore:
    def __init__(self, results) -> None:
        self.results = results

    def similarity_search_with_relevance_scores(self, question: str, k: int):
        return self.results[:k]


def test_retrieval_keeps_only_sufficient_unique_evidence() -> None:
    store = ResultStore(
        [
            (_document("GUIDE-02"), 0.82),
            (_document("GUIDE-02"), 0.80),
            (_document("RISK-02"), 0.20),
        ]
    )
    retriever = PolicyRetriever(load_rag_settings(environ={}), store)

    evidence = retriever.retrieve("What should I review?")

    assert len(evidence) == 1
    assert evidence[0].citation == "guide-v1 / GUIDE-02"
    assert evidence[0].relevance_score == 0.82


def test_retrieval_returns_no_evidence_below_threshold() -> None:
    retriever = PolicyRetriever(
        load_rag_settings(environ={}),
        ResultStore([(_document("GUIDE-02"), 0.34)]),
    )

    assert retriever.retrieve("How do I bake bread?") == ()


def test_retrieval_rejects_untrustworthy_metadata() -> None:
    retriever = PolicyRetriever(
        load_rag_settings(environ={}),
        ResultStore([(_document("GUIDE-02", complete=False), 0.90)]),
    )

    with pytest.raises(RetrievalError, match="invalid source metadata"):
        retriever.retrieve("What should I review?")
