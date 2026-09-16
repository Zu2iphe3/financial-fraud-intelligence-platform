"""Retrieve source-aware evidence from the local fictional-policy index."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.rag.config import REPOSITORY_ROOT, RagSettings
from src.rag.ingestion import create_embedding_model, open_vector_store


DEFAULT_EVALUATION_PATH = REPOSITORY_ROOT / "config" / "rag" / "retrieval_evaluation.json"
REQUIRED_METADATA = {
    "source_id",
    "title",
    "source_path",
    "section_id",
    "section_heading",
    "chunk_id",
    "fictional",
}


class RetrievalError(RuntimeError):
    """Raised when the generated index cannot provide trustworthy evidence."""


@dataclass(frozen=True)
class PolicyEvidence:
    """One relevant fictional-policy passage with its stable citation metadata."""

    content: str
    relevance_score: float
    source_id: str
    title: str
    source_path: str
    section_id: str
    section_heading: str
    chunk_id: str

    @property
    def citation(self) -> str:
        return f"{self.source_id} / {self.section_id}"


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Results from the intentionally small committed relevance check."""

    covered_total: int
    covered_with_expected_source: int
    unsupported_total: int
    unsupported_rejected: int
    details: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyRetriever:
    """Apply one explicit relevance boundary to local Chroma search results."""

    def __init__(self, settings: RagSettings, store: Any) -> None:
        self.settings = settings
        self.store = store

    @classmethod
    def from_settings(
        cls,
        settings: RagSettings,
        embedding_function: Any | None = None,
    ) -> "PolicyRetriever":
        try:
            embedding = embedding_function or create_embedding_model(settings)
            store = open_vector_store(settings, embedding)
            if not store.get(include=[]).get("ids"):
                raise RetrievalError("Policy index is empty; run the index command first")
        except RetrievalError:
            raise
        except Exception as error:
            raise RetrievalError("The local policy index could not be opened") from error
        return cls(settings, store)

    def retrieve(
        self,
        question: str,
        *,
        minimum_score: float | None = None,
    ) -> tuple[PolicyEvidence, ...]:
        """Return only evidence meeting the configured or requested score."""

        if not question.strip():
            raise ValueError("Question cannot be empty")
        threshold = (
            self.settings.minimum_relevance_score
            if minimum_score is None
            else minimum_score
        )
        if not 0 <= threshold <= 1:
            raise ValueError("Minimum relevance score must be between zero and one")

        try:
            results = self.store.similarity_search_with_relevance_scores(
                question,
                k=self.settings.retrieval_top_k,
            )
        except Exception as error:
            raise RetrievalError("Local policy retrieval is unavailable") from error
        evidence: list[PolicyEvidence] = []
        seen_chunk_ids: set[str] = set()
        for document, score in results:
            if float(score) < threshold:
                continue
            missing = REQUIRED_METADATA - document.metadata.keys()
            if missing or document.metadata.get("fictional") is not True:
                raise RetrievalError("Retrieved evidence has invalid source metadata")
            chunk_id = str(document.metadata["chunk_id"])
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            evidence.append(
                PolicyEvidence(
                    content=document.page_content,
                    relevance_score=float(score),
                    source_id=str(document.metadata["source_id"]),
                    title=str(document.metadata["title"]),
                    source_path=str(document.metadata["source_path"]),
                    section_id=str(document.metadata["section_id"]),
                    section_heading=str(document.metadata["section_heading"]),
                    chunk_id=chunk_id,
                )
            )
        return tuple(evidence)


def evaluate_retrieval(
    retriever: PolicyRetriever,
    evaluation_path: str | Path = DEFAULT_EVALUATION_PATH,
) -> RetrievalEvaluation:
    """Run the small documented set used to review the relevance threshold."""

    questions = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    details: list[dict[str, Any]] = []
    covered_hits = 0
    for item in questions["covered_questions"]:
        raw_evidence = retriever.retrieve(item["question"], minimum_score=0.0)
        evidence = tuple(
            result
            for result in raw_evidence
            if result.relevance_score >= retriever.settings.minimum_relevance_score
        )
        returned = {result.section_id for result in evidence}
        expected = set(item["expected_section_ids"])
        hit = bool(returned & expected)
        covered_hits += int(hit)
        details.append(
            {
                "category": "covered",
                "question": item["question"],
                "passed": hit,
                "returned_sections": sorted(returned),
                "raw_top_score": raw_evidence[0].relevance_score if raw_evidence else None,
                "accepted_count": len(evidence),
            }
        )

    unsupported_rejected = 0
    for question in questions["unsupported_questions"]:
        raw_evidence = retriever.retrieve(question, minimum_score=0.0)
        evidence = tuple(
            result
            for result in raw_evidence
            if result.relevance_score >= retriever.settings.minimum_relevance_score
        )
        rejected = not evidence
        unsupported_rejected += int(rejected)
        details.append(
            {
                "category": "unsupported",
                "question": question,
                "passed": rejected,
                "returned_sections": sorted({item.section_id for item in evidence}),
                "raw_top_score": raw_evidence[0].relevance_score if raw_evidence else None,
                "accepted_count": len(evidence),
            }
        )

    return RetrievalEvaluation(
        covered_total=len(questions["covered_questions"]),
        covered_with_expected_source=covered_hits,
        unsupported_total=len(questions["unsupported_questions"]),
        unsupported_rejected=unsupported_rejected,
        details=tuple(details),
    )
