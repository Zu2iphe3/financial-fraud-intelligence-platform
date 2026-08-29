"""Load and validate the versioned PaySim predictive-feature contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = REPOSITORY_ROOT / "config" / "features" / "paysim_v1.json"
_RELATION_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")
_SUPPORTED_TRANSFORMATIONS = {
    "log1p",
    "one_hot",
    "precomputed_history",
}
_SUPPORTED_DTYPES = {"float32", "int8"}


class FeatureContractError(ValueError):
    """Raised when a feature definition could leak or cannot be reproduced safely."""


@dataclass(frozen=True)
class FeatureDefinition:
    """One output feature and the source data used to calculate it."""

    name: str
    source_columns: tuple[str, ...]
    transformation: str
    dtype: str
    value: str | None = None


@dataclass(frozen=True)
class SplitDefinition:
    """Inclusive PaySim step range for one chronological partition."""

    name: str
    first_step: int
    last_step: int
    expected_rows: int
    expected_fraud_rows: int


@dataclass(frozen=True)
class FinalDayPolicy:
    """Evaluation-only treatment of PaySim's abnormal final simulated day."""

    simulation_day: int
    first_step: int
    last_step: int
    evaluation_cohort: str


@dataclass(frozen=True)
class ExpectedDataset:
    """Known full-file totals used to reconcile a production dataset build."""

    total_rows: int
    fraud_rows: int
    final_day_rows: int
    final_day_fraud_rows: int


@dataclass(frozen=True)
class FeatureContract:
    """Complete, versioned rules for modelling inputs and time partitions."""

    contract_version: str
    source_relation: str
    predictive_source_columns: tuple[str, ...]
    internal_grouping_columns: tuple[str, ...]
    target_column: str
    metadata_columns: tuple[str, ...]
    transaction_types: tuple[str, ...]
    features: tuple[FeatureDefinition, ...]
    prohibited_predictive_sources: frozenset[str]
    splits: tuple[SplitDefinition, ...]
    final_day: FinalDayPolicy
    expected_dataset: ExpectedDataset

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the stable model-matrix column order."""
        return tuple(feature.name for feature in self.features)

    @property
    def required_source_columns(self) -> tuple[str, ...]:
        """Return only columns required for features, target, splitting, and audit."""
        return (
            *self.metadata_columns,
            *self.predictive_source_columns,
            *self.internal_grouping_columns,
            self.target_column,
        )


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    """Turn an unclear JSON KeyError into a learner-friendly contract error."""
    try:
        return mapping[key]
    except KeyError as error:
        raise FeatureContractError(f"Feature contract is missing {key!r}") from error


def parse_feature_contract(document: Mapping[str, Any]) -> FeatureContract:
    """Convert a JSON-compatible mapping into a validated immutable contract."""
    try:
        features = tuple(
            FeatureDefinition(
                name=item["name"],
                source_columns=tuple(item["source_columns"]),
                transformation=item["transformation"],
                dtype=item["dtype"],
                value=item.get("value"),
            )
            for item in _required(document, "features")
        )
        splits = tuple(
            SplitDefinition(
                name=item["name"],
                first_step=int(item["first_step"]),
                last_step=int(item["last_step"]),
                expected_rows=int(item["expected_rows"]),
                expected_fraud_rows=int(item["expected_fraud_rows"]),
            )
            for item in _required(document, "splits")
        )
        final_day_document = _required(document, "final_day")
        expected_document = _required(document, "expected_dataset")
        contract = FeatureContract(
            contract_version=str(_required(document, "contract_version")),
            source_relation=str(_required(document, "source_relation")),
            predictive_source_columns=tuple(
                _required(document, "predictive_source_columns")
            ),
            internal_grouping_columns=tuple(document.get("internal_grouping_columns", [])),
            target_column=str(_required(document, "target_column")),
            metadata_columns=tuple(_required(document, "metadata_columns")),
            transaction_types=tuple(_required(document, "transaction_types")),
            features=features,
            prohibited_predictive_sources=frozenset(
                _required(document, "prohibited_predictive_sources")
            ),
            splits=splits,
            final_day=FinalDayPolicy(
                simulation_day=int(final_day_document["simulation_day"]),
                first_step=int(final_day_document["first_step"]),
                last_step=int(final_day_document["last_step"]),
                evaluation_cohort=str(final_day_document["evaluation_cohort"]),
            ),
            expected_dataset=ExpectedDataset(
                total_rows=int(expected_document["total_rows"]),
                fraud_rows=int(expected_document["fraud_rows"]),
                final_day_rows=int(expected_document["final_day_rows"]),
                final_day_fraud_rows=int(expected_document["final_day_fraud_rows"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, FeatureContractError):
            raise
        raise FeatureContractError(f"Feature contract has invalid structure: {error}") from error

    validate_feature_contract(contract)
    return contract


def validate_feature_contract(contract: FeatureContract) -> None:
    """Fail before training when features, dependencies, or split rules are unsafe."""
    problems: list[str] = []
    if not contract.contract_version:
        problems.append("contract_version must not be empty")
    if not _RELATION_PATTERN.fullmatch(contract.source_relation):
        problems.append("source_relation must be a schema-qualified PostgreSQL name")

    feature_names = contract.feature_names
    if len(feature_names) != len(set(feature_names)):
        problems.append("feature names must be unique")
    if set(contract.predictive_source_columns) & contract.prohibited_predictive_sources:
        problems.append("predictive source allowlist contains a prohibited field")

    allowed_sources = set(contract.predictive_source_columns)
    internal_sources = set(contract.internal_grouping_columns)
    if internal_sources - contract.prohibited_predictive_sources:
        problems.append("internal grouping columns must remain prohibited predictive outputs")
    for feature in contract.features:
        dependencies = set(feature.source_columns)
        prohibited = sorted(dependencies & contract.prohibited_predictive_sources)
        history_feature = feature.transformation == "precomputed_history"
        permitted_internal = internal_sources if history_feature else set()
        if set(prohibited) - permitted_internal:
            problems.append(
                f"feature {feature.name!r} depends on prohibited source(s): "
                + ", ".join(sorted(set(prohibited) - permitted_internal))
            )
        unexpected = sorted(dependencies - allowed_sources - permitted_internal)
        if unexpected:
            problems.append(
                f"feature {feature.name!r} depends on source(s) outside the allowlist: "
                + ", ".join(unexpected)
            )
        if feature.transformation not in _SUPPORTED_TRANSFORMATIONS:
            problems.append(
                f"feature {feature.name!r} uses unsupported transformation "
                f"{feature.transformation!r}"
            )
        if feature.dtype not in _SUPPORTED_DTYPES:
            problems.append(f"feature {feature.name!r} uses unsupported dtype {feature.dtype!r}")
        if feature.transformation == "log1p" and len(feature.source_columns) != 1:
            problems.append(f"feature {feature.name!r} log1p requires exactly one source")
        if feature.transformation == "one_hot":
            if feature.source_columns != ("transaction_type",):
                problems.append(
                    f"feature {feature.name!r} one_hot must depend on transaction_type"
                )
            if feature.value not in contract.transaction_types:
                problems.append(
                    f"feature {feature.name!r} has unsupported category {feature.value!r}"
                )
        if history_feature and not (dependencies & internal_sources):
            problems.append(
                f"feature {feature.name!r} past-only history requires an internal grouping key"
            )

    expected_split_names = ("train", "validation", "test")
    if tuple(split.name for split in contract.splits) != expected_split_names:
        problems.append("splits must be ordered as train, validation, test")
    for split in contract.splits:
        if split.first_step <= 0 or split.first_step > split.last_step:
            problems.append(f"split {split.name!r} has an invalid step range")
        if split.expected_rows <= 0:
            problems.append(f"split {split.name!r} expected_rows must be positive")
        if not 0 <= split.expected_fraud_rows <= split.expected_rows:
            problems.append(f"split {split.name!r} has invalid expected fraud rows")
    for earlier, later in zip(contract.splits, contract.splits[1:]):
        if later.first_step != earlier.last_step + 1:
            problems.append(
                f"split {later.name!r} must begin immediately after {earlier.name!r}"
            )

    if contract.splits:
        test_split = contract.splits[-1]
        if not (
            test_split.first_step <= contract.final_day.first_step
            <= contract.final_day.last_step <= test_split.last_step
        ):
            problems.append("final simulated day must be contained within the test split")
        if sum(split.expected_rows for split in contract.splits) != (
            contract.expected_dataset.total_rows
        ):
            problems.append("split expected rows must reconcile to expected total rows")
        if sum(split.expected_fraud_rows for split in contract.splits) != (
            contract.expected_dataset.fraud_rows
        ):
            problems.append("split expected fraud rows must reconcile to expected fraud rows")

    if problems:
        raise FeatureContractError("Invalid feature contract: " + "; ".join(problems))


def load_feature_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> FeatureContract:
    """Load the committed contract without relying on a machine-specific path."""
    contract_path = Path(path)
    if not contract_path.is_absolute():
        contract_path = REPOSITORY_ROOT / contract_path
    try:
        document = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"Cannot load feature contract {contract_path}: {error}"
        raise FeatureContractError(message) from error
    if not isinstance(document, dict):
        raise FeatureContractError("Feature contract root must be a JSON object")
    return parse_feature_contract(document)
