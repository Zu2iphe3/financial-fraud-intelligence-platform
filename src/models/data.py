"""Load PH7-01 Parquet partitions without weakening their leakage contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.contract import FeatureContract


class ModelDataError(ValueError):
    """Raised when modelling inputs do not reconcile with the PH7-01 contract."""


@dataclass(frozen=True)
class PartitionData:
    """Aligned predictive features, labels, and evaluation-only metadata."""

    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame


def contract_sha256(path: str | Path) -> str:
    """Identify the exact reviewed feature contract used by a model run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset_manifest(dataset_root: str | Path) -> dict[str, Any]:
    """Read the generated dataset identity and reconciliation evidence."""
    path = Path(dataset_root) / "manifest.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelDataError(f"Cannot load modelling manifest {path}: {error}") from error
    if not isinstance(document, dict):
        raise ModelDataError("Modelling manifest root must be a JSON object")
    return document


def validate_dataset_manifest(manifest: dict[str, Any], contract: FeatureContract) -> None:
    """Reject stale, incomplete, or differently shaped generated datasets."""
    problems: list[str] = []
    if manifest.get("contract_version") != contract.contract_version:
        problems.append("manifest contract version does not match the feature contract")
    expected_schema = [
        {"name": feature.name, "dtype": feature.dtype} for feature in contract.features
    ]
    if manifest.get("feature_schema") != expected_schema:
        problems.append("manifest feature schema does not match the feature contract")
    if manifest.get("target") != contract.target_column:
        problems.append("manifest target does not match the feature contract")
    if not manifest.get("source_dataset_id"):
        problems.append("manifest must identify one source dataset fingerprint")

    observed_splits = manifest.get("splits", {})
    for split in contract.splits:
        observed = observed_splits.get(split.name, {})
        expected = (
            split.first_step,
            split.last_step,
            split.expected_rows,
            split.expected_fraud_rows,
        )
        actual = (
            observed.get("first_step"),
            observed.get("last_step"),
            observed.get("rows"),
            observed.get("fraud_rows"),
        )
        if actual != expected:
            problems.append(f"manifest {split.name} reconciliation does not match the contract")

    final_policy = manifest.get("final_day_policy", {})
    expected_final_policy = (
        contract.final_day.simulation_day,
        contract.final_day.first_step,
        contract.final_day.last_step,
        contract.final_day.evaluation_cohort,
        False,
    )
    actual_final_policy = (
        final_policy.get("simulation_day"),
        final_policy.get("first_step"),
        final_policy.get("last_step"),
        final_policy.get("evaluation_cohort"),
        final_policy.get("predictive_feature"),
    )
    if actual_final_policy != expected_final_policy:
        problems.append("manifest final-day policy does not match the feature contract")
    if problems:
        raise ModelDataError("Invalid modelling manifest: " + "; ".join(problems))


def _matching_part_paths(partition_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Require matching part files so rows cannot silently lose target alignment."""
    groups = [
        sorted((partition_root / group).glob("part-*.parquet"))
        for group in ("features", "target", "metadata")
    ]
    names = [[path.name for path in group] for group in groups]
    if not names[0] or names[0] != names[1] or names[0] != names[2]:
        raise ModelDataError(
            f"Feature, target, and metadata parts do not align under {partition_root}"
        )
    return groups[0], groups[1], groups[2]


def _read_parts(paths: list[Path]) -> pd.DataFrame:
    """Read deterministic part order while preserving PH7-01 row order."""
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def load_partition(
    dataset_root: str | Path,
    partition: str,
    contract: FeatureContract,
    *,
    validate_expected_counts: bool = True,
) -> PartitionData:
    """Load one partition and prove features, targets, and metadata still align."""
    split_by_name = {split.name: split for split in contract.splits}
    if partition not in split_by_name:
        raise ModelDataError(f"Unknown partition {partition!r}")
    feature_paths, target_paths, metadata_paths = _matching_part_paths(
        Path(dataset_root) / partition
    )
    features = _read_parts(feature_paths)
    target_frame = _read_parts(target_paths)
    metadata = _read_parts(metadata_paths)

    if tuple(features.columns) != contract.feature_names:
        raise ModelDataError(f"{partition} feature order does not match the contract")
    prohibited = set(features.columns) & contract.prohibited_predictive_sources
    if prohibited:
        raise ModelDataError(
            f"{partition} contains prohibited predictive fields: {sorted(prohibited)}"
        )
    if tuple(target_frame.columns) != (contract.target_column,):
        raise ModelDataError(f"{partition} target schema is invalid")
    if not target_frame[contract.target_column].isin((0, 1)).all():
        raise ModelDataError(f"{partition} target must contain only zero and one")
    if not (len(features) == len(target_frame) == len(metadata)):
        raise ModelDataError(f"{partition} feature, target, and metadata rows do not align")
    if not metadata["partition"].eq(partition).all():
        raise ModelDataError(f"{partition} metadata contains another partition")

    target = target_frame[contract.target_column].astype("int8")
    if validate_expected_counts:
        expected = split_by_name[partition]
        actual = (len(features), int(target.sum()))
        if actual != (expected.expected_rows, expected.expected_fraud_rows):
            raise ModelDataError(
                f"{partition} counts {actual} do not match "
                f"{(expected.expected_rows, expected.expected_fraud_rows)}"
            )
    return PartitionData(features=features, target=target, metadata=metadata)


def training_class_ratio(target: pd.Series) -> float:
    """Calculate imbalance from training labels only, never later partitions."""
    fraud = int(target.sum())
    legitimate = int(len(target) - fraud)
    if fraud <= 0 or legitimate <= 0:
        raise ModelDataError("Training data must contain both legitimate and fraud rows")
    return legitimate / fraud
