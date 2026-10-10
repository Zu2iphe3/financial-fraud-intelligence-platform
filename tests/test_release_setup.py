"""Release checks for the documented clean dashboard environment."""

from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _example_environment() -> dict[str, str]:
    """Read only committed non-secret defaults from the environment example."""

    values: dict[str, str] = {}
    for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split("=", maxsplit=1)
        values[key] = value
    return values


def test_requirements_declare_a_compatible_streamlit_version() -> None:
    """Prevent a clean install from silently omitting the dashboard runtime."""

    requirements = [
        Requirement(line)
        for line in (REPOSITORY_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    streamlit = next(
        requirement for requirement in requirements if requirement.name.lower() == "streamlit"
    )

    assert Version("1.40") in streamlit.specifier
    assert Version("1.99") in streamlit.specifier
    assert Version("2.0") not in streamlit.specifier


def test_environment_example_documents_dashboard_service_defaults() -> None:
    """Keep clean-machine dashboard endpoints and the safe port visible together."""

    environment = _example_environment()

    assert environment["STREAMLIT_PORT"] == "8501"
    assert environment["FRAUD_API_BASE_URL"] == "http://127.0.0.1:8000"
    assert environment["POSTGRES_HOST"] == "localhost"
    assert environment["POSTGRES_PORT"] == "5432"
    assert environment["OLLAMA_BASE_URL"] == "http://localhost:11434"
    assert environment["OLLAMA_CHAT_MODEL"] == "llama3.2:3b"
    assert environment["RAG_CHROMA_PATH"] == "./chroma_db"
