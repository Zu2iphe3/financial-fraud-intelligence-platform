from pathlib import Path

import pandas as pd

from src.data.explore_paysim import EXPECTED_COLUMNS, profile_dataset


def test_profile_dataset_summarizes_expected_schema(tmp_path: Path) -> None:
    rows = [
        [1, "PAYMENT", 100.0, "C1", 200.0, 100.0, "M1", 0.0, 0.0, 0, 0],
        [2, "TRANSFER", 500.0, "C2", 500.0, 0.0, "C3", 0.0, 500.0, 1, 1],
    ]
    csv_path = tmp_path / "paysim.csv"
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)

    summary = profile_dataset(csv_path, chunk_size=1)

    assert summary["rows"] == 2
    assert summary["fraud"]["transactions"] == 1
    assert summary["fraud"]["rate"] == 0.5
    assert summary["transaction_types"]["TRANSFER"]["fraud_transactions"] == 1
    assert all(value == 0 for value in summary["missing_values"].values())


def test_profile_dataset_rejects_unexpected_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame({"unexpected": [1]}).to_csv(csv_path, index=False)

    try:
        profile_dataset(csv_path, chunk_size=1)
    except ValueError as error:
        assert "Unexpected columns" in str(error)
    else:
        raise AssertionError("Expected an unexpected-schema error")
