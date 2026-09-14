"""Contract tests for the original fictional fraud-policy corpus."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.rag.corpus_validation import load_manifest, validate_corpus


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPOSITORY_ROOT / "knowledge_base"


def _copy_corpus(tmp_path: Path) -> Path:
    # Keep the nested fixture path short enough for Windows/OneDrive checkouts.
    target = tmp_path / "kb"
    shutil.copytree(CORPUS_ROOT, target)
    return target


def _write_manifest(corpus_root: Path, manifest: dict) -> None:
    (corpus_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def test_repository_corpus_satisfies_issue_contract() -> None:
    assert validate_corpus(CORPUS_ROOT) == []


def test_validator_rejects_duplicate_source_identifiers(tmp_path: Path) -> None:
    corpus_root = _copy_corpus(tmp_path)
    manifest = load_manifest(corpus_root)
    manifest["documents"][1]["source_id"] = manifest["documents"][0]["source_id"]
    _write_manifest(corpus_root, manifest)

    assert any("Duplicate source_id" in error for error in validate_corpus(corpus_root))


def test_validator_rejects_a_title_that_does_not_match_the_document(tmp_path: Path) -> None:
    corpus_root = _copy_corpus(tmp_path)
    manifest = load_manifest(corpus_root)
    manifest["documents"][0]["title"] = "Changed title"
    _write_manifest(corpus_root, manifest)

    assert any("stable title" in error for error in validate_corpus(corpus_root))


def test_validator_rejects_a_missing_document(tmp_path: Path) -> None:
    corpus_root = _copy_corpus(tmp_path)
    (corpus_root / "policies" / "fraud-investigation-policy.md").unlink()

    assert any("Document does not exist" in error for error in validate_corpus(corpus_root))


def test_validator_rejects_incomplete_scenario_guidance(tmp_path: Path) -> None:
    corpus_root = _copy_corpus(tmp_path)
    guide = corpus_root / "guides" / "analyst-investigation-guide.md"
    guide.write_text(
        guide.read_text(encoding="utf-8").replace(
            "### Investigation actions",
            "### Review steps",
            1,
        ),
        encoding="utf-8",
    )

    assert any("does not identify investigation actions" in error for error in validate_corpus(corpus_root))


def test_validator_rejects_unresolved_placeholder_content(tmp_path: Path) -> None:
    corpus_root = _copy_corpus(tmp_path)
    guide = corpus_root / "guides" / "risk-and-alert-interpretation.md"
    guide.write_text(guide.read_text(encoding="utf-8") + "\nTODO: replace me\n", encoding="utf-8")

    assert any("placeholder content" in error for error in validate_corpus(corpus_root))
