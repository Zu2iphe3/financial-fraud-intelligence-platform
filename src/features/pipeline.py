"""Transform trusted PaySim rows into leakage-safe features, targets, and metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.contract import FeatureContract


class FeaturePipelineError(ValueError):
    """Raised when source values cannot be transformed without ambiguity or leakage."""


@dataclass(frozen=True)
class ModellingChunk:
    """Separated outputs for one bounded source chunk."""

    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Fail before transformation when the trusted-source contract is incomplete."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeaturePipelineError("Missing required source columns: " + ", ".join(missing))


def _validated_amounts(frame: pd.DataFrame) -> pd.Series:
    """Reject values that make the deterministic amount transform unsafe."""
    try:
        amounts = pd.to_numeric(frame["amount"], errors="raise").astype("float64")
    except (TypeError, ValueError) as error:
        raise FeaturePipelineError("amount must contain numeric values") from error
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise FeaturePipelineError("amount must contain finite non-negative values")
    return amounts


def _validate_transaction_types(frame: pd.DataFrame, contract: FeatureContract) -> None:
    """Keep one-hot output stable by rejecting undocumented PaySim categories."""
    actual = set(frame["transaction_type"].dropna().astype(str).unique())
    unsupported = sorted(actual - set(contract.transaction_types))
    if frame["transaction_type"].isna().any() or unsupported:
        details = ", ".join(unsupported) if unsupported else "null"
        raise FeaturePipelineError(f"Unsupported transaction type values: {details}")


def transform_features(frame: pd.DataFrame, contract: FeatureContract) -> pd.DataFrame:
    """Create the exact predictive matrix used later by training and FastAPI."""
    _require_columns(frame, contract.predictive_source_columns)
    amounts = _validated_amounts(frame)
    _validate_transaction_types(frame, contract)

    transformed: dict[str, pd.Series] = {}
    for definition in contract.features:
        if definition.transformation == "log1p":
            # Log scaling limits the influence of PaySim's extreme amount tail while
            # remaining deterministic and requiring no statistics from later periods.
            values = np.log1p(amounts).astype(definition.dtype)
        elif definition.transformation == "one_hot":
            values = (frame[definition.source_columns[0]] == definition.value).astype(
                definition.dtype
            )
        else:  # The contract validator prevents this defensive branch in normal use.
            raise FeaturePipelineError(
                f"Unsupported transformation {definition.transformation!r}"
            )
        transformed[definition.name] = pd.Series(values, index=frame.index)

    features = pd.DataFrame(transformed, index=frame.index)
    if tuple(features.columns) != contract.feature_names:
        raise FeaturePipelineError("Generated feature order does not match the contract")
    return features


def assign_partitions(steps: pd.Series, contract: FeatureContract) -> pd.Series:
    """Assign each row once using non-overlapping, forward-only PaySim step ranges."""
    try:
        numeric_steps = pd.to_numeric(steps, errors="raise").astype("int64")
    except (TypeError, ValueError) as error:
        raise FeaturePipelineError("step must contain integer values") from error
    if not (numeric_steps == pd.to_numeric(steps, errors="coerce")).all():
        raise FeaturePipelineError("step must contain integer values")

    partitions = pd.Series(pd.NA, index=steps.index, dtype="string")
    for split in contract.splits:
        mask = numeric_steps.between(split.first_step, split.last_step)
        partitions.loc[mask] = split.name
    if partitions.isna().any():
        invalid = sorted(numeric_steps.loc[partitions.isna()].unique().tolist())
        raise FeaturePipelineError(f"step values fall outside configured splits: {invalid}")
    return partitions


def _validated_target(frame: pd.DataFrame, contract: FeatureContract) -> pd.Series:
    """Keep fraud labels binary and separate from every predictive feature."""
    try:
        target = pd.to_numeric(frame[contract.target_column], errors="raise")
    except (TypeError, ValueError) as error:
        raise FeaturePipelineError(f"{contract.target_column} must contain 0 or 1") from error
    if not target.isin((0, 1)).all():
        raise FeaturePipelineError(f"{contract.target_column} must contain only 0 or 1")
    return target.astype("int8").rename(contract.target_column)


def prepare_modelling_chunk(frame: pd.DataFrame, contract: FeatureContract) -> ModellingChunk:
    """Separate model inputs from target and evaluation-only transaction metadata."""
    _require_columns(frame, contract.required_source_columns)
    features = transform_features(frame, contract)
    target = _validated_target(frame, contract)
    metadata = frame.loc[:, contract.metadata_columns].copy()

    try:
        metadata["source_row_number"] = pd.to_numeric(
            metadata["source_row_number"], errors="raise"
        ).astype("int64")
        steps = pd.to_numeric(metadata["step"], errors="raise").astype("int64")
        days = pd.to_numeric(metadata["simulation_day"], errors="raise").astype("int64")
    except (TypeError, ValueError) as error:
        raise FeaturePipelineError(
            "source_row_number, step, and simulation_day must contain integers"
        ) from error
    if metadata[["transaction_id", "source_dataset_id"]].isna().any().any():
        raise FeaturePipelineError("transaction and source dataset identities are required")
    # Psycopg returns transaction_id as UUID. Store audit identities as strings so
    # Parquet has a portable representation without letting identifiers enter features.
    metadata["transaction_id"] = metadata["transaction_id"].astype("string")
    metadata["source_dataset_id"] = metadata["source_dataset_id"].astype("string")
    expected_days = ((steps - 1) // 24) + 1
    if not expected_days.equals(days):
        raise FeaturePipelineError("simulation_day does not match the documented step calculation")

    metadata["step"] = steps
    metadata["simulation_day"] = days
    metadata["partition"] = assign_partitions(steps, contract)
    final_day_mask = days.eq(contract.final_day.simulation_day)
    configured_final_steps = steps.between(
        contract.final_day.first_step, contract.final_day.last_step
    )
    if not final_day_mask.equals(configured_final_steps):
        raise FeaturePipelineError("final-day rows do not match the configured final-day steps")
    metadata["evaluation_cohort"] = "standard_period"
    metadata.loc[final_day_mask, "evaluation_cohort"] = (
        contract.final_day.evaluation_cohort
    )

    prohibited_outputs = set(features.columns) & contract.prohibited_predictive_sources
    if prohibited_outputs:
        raise FeaturePipelineError(
            "Prohibited predictive outputs were generated: "
            + ", ".join(sorted(prohibited_outputs))
        )
    return ModellingChunk(features=features, target=target, metadata=metadata)
