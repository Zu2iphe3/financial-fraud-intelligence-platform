"""Lightweight contract checks for the fictional policy corpus.

PH9-02 will rely on stable source and section identifiers when it creates
chunks and citations. These checks protect that small boundary without trying
to implement document ingestion or a general validation framework.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_DOMAINS = {
    "fraud_investigation",
    "escalation",
    "analyst_guidance",
    "risk_interpretation",
}
FICTIONAL_NOTICE = "FICTIONAL PORTFOLIO DOCUMENT"
PLACEHOLDERS = ("TODO", "TBD", "LOREM IPSUM", "YOUR_ORGANIZATION", "<PLACEHOLDER>")


def load_manifest(corpus_root: Path) -> dict[str, Any]:
    """Load the stable metadata that PH9-02 will use for source citations."""

    manifest_path = corpus_root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _section_text(document: str, section_id: str, heading: str) -> str:
    """Return one Markdown section so scenario checks stay source-specific."""

    marker = f"## {section_id} - {heading}"
    if marker not in document:
        return ""
    section = document.split(marker, maxsplit=1)[1]
    return section.split("\n## ", maxsplit=1)[0]


def validate_corpus(corpus_root: Path) -> list[str]:
    """Return contract violations for the PH9-01 knowledge base.

    The validator intentionally checks only the promises made by Issue #13:
    stable metadata, meaningful structure, fictional status, domain coverage,
    and complete guidance for each declared scenario.
    """

    errors: list[str] = []
    try:
        manifest = load_manifest(corpus_root)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return [f"Manifest cannot be loaded: {exc}"]

    if manifest.get("fictional") is not True:
        errors.append("Corpus manifest must declare fictional=true.")

    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        return errors + ["Manifest must declare at least one document."]

    source_ids: set[str] = set()
    document_paths: set[Path] = set()
    section_ids: set[str] = set()
    covered_domains: set[str] = set()
    loaded_documents: dict[str, tuple[dict[str, Any], str]] = {}

    for metadata in documents:
        required = {"source_id", "title", "path", "version", "domains", "sections"}
        missing = sorted(required - metadata.keys())
        if missing:
            errors.append(f"Document metadata is missing fields: {', '.join(missing)}.")
            continue

        source_id = metadata["source_id"]
        if source_id in source_ids:
            errors.append(f"Duplicate source_id: {source_id}.")
        source_ids.add(source_id)

        relative_path = Path(metadata["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"Document path must stay inside the corpus: {relative_path}.")
            continue
        if relative_path in document_paths:
            errors.append(f"Duplicate document path: {relative_path.as_posix()}.")
        document_paths.add(relative_path)

        document_path = corpus_root / relative_path
        if not document_path.is_file():
            errors.append(f"Document does not exist: {relative_path.as_posix()}.")
            continue

        document = document_path.read_text(encoding="utf-8")
        loaded_documents[source_id] = (metadata, document)
        covered_domains.update(metadata["domains"])

        if FICTIONAL_NOTICE not in document:
            errors.append(f"{source_id} is missing the fictional-use notice.")
        if not document.startswith(f"# {metadata['title']}\n"):
            errors.append(f"{source_id} does not display its stable title.")
        if f"**Source ID:** `{source_id}`" not in document:
            errors.append(f"{source_id} does not display its stable source ID.")
        if any(placeholder in document.upper() for placeholder in PLACEHOLDERS):
            errors.append(f"{source_id} contains unresolved placeholder content.")

        for section in metadata["sections"]:
            section_id = section.get("section_id")
            heading = section.get("heading")
            if not section_id or not heading:
                errors.append(f"{source_id} has incomplete section metadata.")
                continue
            if section_id in section_ids:
                errors.append(f"Duplicate section_id: {section_id}.")
            section_ids.add(section_id)
            if not _section_text(document, section_id, heading).strip():
                errors.append(f"{source_id} is missing section {section_id} - {heading}.")

    missing_domains = sorted(REQUIRED_DOMAINS - covered_domains)
    if missing_domains:
        errors.append(f"Corpus is missing domains: {', '.join(missing_domains)}.")

    scenarios = manifest.get("supported_scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        errors.append("Manifest must declare at least one supported scenario.")
        return errors

    scenario_ids: set[str] = set()
    for scenario in scenarios:
        required = {"scenario_id", "source_id", "section_id", "responsible_role"}
        missing = sorted(required - scenario.keys())
        if missing:
            errors.append(f"Scenario metadata is missing fields: {', '.join(missing)}.")
            continue

        scenario_id = scenario["scenario_id"]
        if scenario_id in scenario_ids:
            errors.append(f"Duplicate scenario_id: {scenario_id}.")
        scenario_ids.add(scenario_id)

        loaded = loaded_documents.get(scenario["source_id"])
        if loaded is None:
            errors.append(f"{scenario_id} references an unknown source document.")
            continue
        metadata, document = loaded
        section = next(
            (item for item in metadata["sections"] if item["section_id"] == scenario["section_id"]),
            None,
        )
        if section is None:
            errors.append(f"{scenario_id} references an unknown section.")
            continue

        content = _section_text(document, section["section_id"], section["heading"])
        expected_role = f"**Responsible fictional role:** {scenario['responsible_role']}"
        if expected_role not in content:
            errors.append(f"{scenario_id} does not identify its responsible fictional role.")
        if "### Investigation actions" not in content:
            errors.append(f"{scenario_id} does not identify investigation actions.")
        if "### Escalation conditions" not in content:
            errors.append(f"{scenario_id} does not identify escalation conditions.")

    return errors
