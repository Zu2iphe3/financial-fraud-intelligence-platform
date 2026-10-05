"""Thin dashboard adapter around the existing grounded policy assistant."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.rag.assistant import GroundingError, OllamaUnavailableError, PolicyAnswer, PolicyAssistant
from src.rag.config import RagSettings, load_rag_settings
from src.rag.retrieval import PolicyRetriever, RetrievalError


class PolicyServiceUnavailable(RuntimeError):
    """Raised when local retrieval or answer generation is unavailable."""


class PolicySearchUnavailable(PolicyServiceUnavailable):
    """Raised when the local embedding model or policy index cannot be searched."""


class PolicyGenerationUnavailable(PolicyServiceUnavailable):
    """Raised when relevant evidence exists but local answer generation fails."""


class PolicyResponseRejected(PolicyServiceUnavailable):
    """Raised when a generated answer does not satisfy grounding safeguards."""


POLICY_READINESS_QUESTION = (
    "What should an analyst investigate when a transfer goes to a new destination?"
)


class DashboardPolicyService:
    """Reuse retrieval and grounding without recreating policy logic in Streamlit."""

    def __init__(self, assistant: Any, retriever: Any | None = None) -> None:
        self.assistant = assistant
        self.retriever = retriever

    @classmethod
    def from_settings(cls, settings: RagSettings | None = None) -> "DashboardPolicyService":
        try:
            resolved = settings or load_rag_settings()
            retriever = PolicyRetriever.from_settings(resolved)
            return cls(PolicyAssistant(resolved, retriever), retriever)
        except (OSError, RetrievalError, ValueError) as error:
            raise PolicySearchUnavailable("Policy search is unavailable") from error

    def check_search_readiness(self) -> int:
        """Prove the embedding model and Chroma query path work without calling Ollama."""

        if self.retriever is None:
            raise PolicySearchUnavailable("Policy search is unavailable")
        try:
            evidence = self.retriever.retrieve(POLICY_READINESS_QUESTION)
        except (RetrievalError, ValueError) as error:
            raise PolicySearchUnavailable("Policy search is unavailable") from error
        if not evidence:
            raise PolicySearchUnavailable("Policy search returned no readiness evidence")
        return len(evidence)

    def ask(self, question: str) -> PolicyAnswer:
        if not question.strip():
            raise ValueError("Enter a policy question")
        try:
            return self.assistant.answer(question)
        except RetrievalError as error:
            raise PolicySearchUnavailable("Policy search is unavailable") from error
        except OllamaUnavailableError as error:
            raise PolicyGenerationUnavailable("Answer generation is unavailable") from error
        except GroundingError as error:
            raise PolicyResponseRejected("The generated answer was rejected") from error


@lru_cache(maxsize=1)
def get_dashboard_policy_service() -> DashboardPolicyService:
    """Share one initialized embedding model and vector store across health and queries."""

    return DashboardPolicyService.from_settings()
