"""Small, reusable helpers for chunked PaySim exploratory analysis."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

ANALYSIS_ONLY_BALANCE_COLUMNS = (
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)


def validate_schema(csv_path: str | Path) -> list[str]:
    """Return the CSV columns or raise when they differ from the PaySim schema."""
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if columns != list(EXPECTED_COLUMNS):
        raise ValueError(
            "Unexpected PaySim schema. "
            f"Expected {list(EXPECTED_COLUMNS)}, received {columns}."
        )
    return columns


def iter_paysim_chunks(
    csv_path: str | Path,
    *,
    chunk_size: int = 500_000,
    usecols: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield validated PaySim rows in memory-conscious chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    validate_schema(csv_path)
    yield from pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)


def simulated_day(step: pd.Series) -> pd.Series:
    """Convert one-based hourly PaySim steps to one-based simulated days."""
    if (step < 1).any():
        raise ValueError("PaySim steps must be positive integers.")
    return ((step - 1) // 24 + 1).astype("int64")


def balance_discrepancies(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate analysis-only balance discrepancies.

    PaySim documents that fraud transactions are cancelled. Consequently, all
    source balance columns and values derived from them are prohibited as
    predictive features. This helper exists only to demonstrate that leakage.
    Destination discrepancies are calculated only for customer destinations;
    PaySim does not maintain balances for merchant identifiers beginning `M`.
    """
    required = {
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Balance analysis requires columns: {sorted(missing)}")

    cash_in = frame["type"].eq("CASH_IN")
    expected_origin = np.where(
        cash_in,
        frame["oldbalanceOrg"] + frame["amount"],
        frame["oldbalanceOrg"] - frame["amount"],
    )
    expected_destination = np.where(
        cash_in,
        frame["oldbalanceDest"] - frame["amount"],
        frame["oldbalanceDest"] + frame["amount"],
    )

    result = pd.DataFrame(index=frame.index)
    result["origin_abs_discrepancy"] = (
        frame["newbalanceOrig"] - expected_origin
    ).abs().round(2)
    result["destination_abs_discrepancy"] = (
        frame["newbalanceDest"] - expected_destination
    ).abs().round(2)
    result.loc[
        ~frame["nameDest"].str.startswith("C", na=False),
        "destination_abs_discrepancy",
    ] = np.nan
    return result
