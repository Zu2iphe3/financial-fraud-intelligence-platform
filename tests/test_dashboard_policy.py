"""Tests for the dashboard adapter around the existing PH9 assistant."""

from __future__ import annotations

import pytest

from src.dashboard.policy_service import (
    DashboardPolicyService,
    PolicyGenerationUnavailable,
    PolicyResponseRejected,
    PolicySearchUnavailable,
)
from src.rag.assistant import (
    GroundingError,
    OllamaUnavailableError,
    PolicyAnswer,
    SourceAttribution,
)
from src.rag.retrieval import RetrievalError


class FakeAssistant:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.response


class FakeRetriever:
    def __init__(self, evidence=(), error=None) -> None:
        self.evidence = evidence
        self.error = error
        self.questions = []

    def retrieve(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.evidence


def test_policy_adapter_preserves_grounded_answer_and_structured_sources() -> None:
    source = SourceAttribution(
        source_id="analyst-guide-v1",
        title="Analyst Guide",
        section_id="GUIDE-02",
        section_heading="New destination",
        source_path="guides/analyst-guide.md",
    )
    expected = PolicyAnswer("answered", "Review earlier destination activity.", (source,))
    assistant = FakeAssistant(response=expected)

    result = DashboardPolicyService(assistant).ask("What should be reviewed?")

    assert result == expected
    assert result.sources[0].section_id == "GUIDE-02"
    assert assistant.questions == ["What should be reviewed?"]


def test_policy_adapter_preserves_insufficient_evidence_without_sources() -> None:
    expected = PolicyAnswer("insufficient_evidence", "Not enough evidence.", ())

    result = DashboardPolicyService(FakeAssistant(response=expected)).ask("Annual leave?")

    assert result.status == "insufficient_evidence"
    assert result.sources == ()


def test_policy_search_readiness_requires_real_semantic_evidence() -> None:
    retriever = FakeRetriever(evidence=(object(), object()))
    service = DashboardPolicyService(FakeAssistant(), retriever)

    count = service.check_search_readiness()

    assert count == 2
    assert retriever.questions == [
        "What should an analyst investigate when a transfer goes to a new destination?"
    ]


def test_policy_search_readiness_rejects_an_empty_or_failed_query() -> None:
    empty = DashboardPolicyService(FakeAssistant(), FakeRetriever())
    failed = DashboardPolicyService(
        FakeAssistant(), FakeRetriever(error=RetrievalError("private index detail"))
    )

    with pytest.raises(PolicySearchUnavailable):
        empty.check_search_readiness()
    with pytest.raises(PolicySearchUnavailable) as captured:
        failed.check_search_readiness()

    assert "private" not in str(captured.value)


def test_policy_adapter_controls_ollama_failure() -> None:
    service = DashboardPolicyService(
        FakeAssistant(error=OllamaUnavailableError("private backend details"))
    )

    with pytest.raises(PolicyGenerationUnavailable) as captured:
        service.ask("What should be reviewed?")

    assert str(captured.value) == "Answer generation is unavailable"
    assert "private" not in str(captured.value)


def test_policy_adapter_distinguishes_search_and_grounding_failures() -> None:
    search = DashboardPolicyService(FakeAssistant(error=RetrievalError("private index")))
    rejected = DashboardPolicyService(FakeAssistant(error=GroundingError("private output")))

    with pytest.raises(PolicySearchUnavailable, match="Policy search is unavailable"):
        search.ask("What should be reviewed?")
    with pytest.raises(PolicyResponseRejected, match="generated answer was rejected"):
        rejected.ask("What should be reviewed?")
