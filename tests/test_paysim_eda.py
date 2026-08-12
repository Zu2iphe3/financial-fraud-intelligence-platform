from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.paysim_eda import (
    EXPECTED_COLUMNS,
    balance_discrepancies,
    iter_paysim_chunks,
    simulated_day,
    validate_schema,
)


def sample_frame() -> pd.DataFrame:
    """Return a minimal valid PaySim-like dataset for unit tests."""
    rows = [
        [1, "PAYMENT", 25.0, "C1", 100.0, 75.0, "M1", 0.0, 0.0, 0, 0],
        [24, "TRANSFER", 40.0, "C2", 100.0, 60.0, "C3", 10.0, 50.0, 1, 0],
        [25, "CASH_IN", 20.0, "C4", 30.0, 50.0, "C5", 100.0, 80.0, 0, 0],
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def write_sample_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "paysim.csv"
    sample_frame().to_csv(csv_path, index=False)
    return csv_path


def test_validate_schema_accepts_expected_columns(tmp_path: Path) -> None:
    csv_path = write_sample_csv(tmp_path)
    assert validate_schema(csv_path) == list(EXPECTED_COLUMNS)


def test_validate_schema_rejects_changed_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame({"unexpected": [1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Unexpected PaySim schema"):
        validate_schema(csv_path)


def test_iter_paysim_chunks_respects_chunk_size(tmp_path: Path) -> None:
    chunks = list(iter_paysim_chunks(write_sample_csv(tmp_path), chunk_size=2))
    assert [len(chunk) for chunk in chunks] == [2, 1]


def test_iter_paysim_chunks_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    iterator = iter_paysim_chunks(write_sample_csv(tmp_path), chunk_size=0)
    with pytest.raises(ValueError, match="greater than zero"):
        next(iterator)


def test_simulated_day_converts_hourly_steps() -> None:
    result = simulated_day(pd.Series([1, 24, 25, 48, 49]))
    assert result.tolist() == [1, 1, 2, 2, 3]


def test_balance_discrepancies_reconcile_expected_movements() -> None:
    result = balance_discrepancies(sample_frame())

    assert result["origin_abs_discrepancy"].tolist() == [0.0, 0.0, 0.0]
    assert np.isnan(result.loc[0, "destination_abs_discrepancy"])
    assert result.loc[1:, "destination_abs_discrepancy"].tolist() == [0.0, 0.0]
