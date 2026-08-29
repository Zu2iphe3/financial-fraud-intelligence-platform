"""Tests for leakage-safe destination history."""

import numpy as np
import pandas as pd
import pytest

from src.features.behavioral import calculate_bucket_history, calculate_destination_history
from src.features.contract import load_feature_contract


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_number": [1, 2, 3, 4],
            "step": [1, 2, 2, 26],
            "name_orig": ["C1", "C1", "C2", "C3"],
            "amount": [100.0, 200.0, 300.0, 50.0],
            "transaction_type": ["TRANSFER", "CASH_OUT", "TRANSFER", "PAYMENT"],
        }
    )


def test_same_step_rows_see_only_strictly_earlier_steps():
    result = calculate_destination_history(_rows()).set_index("source_row_number")

    assert result.loc[1, "dest_history_available"] == 0
    assert result.loc[2, "dest_tx_count_log1p_24h"] == pytest.approx(np.log1p(1))
    assert result.loc[3, "dest_tx_count_log1p_24h"] == pytest.approx(np.log1p(1))


def test_window_boundaries_and_cold_start_defaults_are_exact():
    result = calculate_destination_history(_rows()).set_index("source_row_number")

    # At step 26, step 2 remains inside the previous 24 steps and step 1 has expired.
    assert result.loc[4, "dest_tx_count_log1p_24h"] == pytest.approx(np.log1p(2))
    assert result.loc[4, "dest_tx_count_log1p_168h"] == pytest.approx(np.log1p(3))
    assert result.loc[4, "dest_hours_since_previous_activity"] == 24
    assert result.loc[1, "dest_hours_since_previous_activity"] == 744


def test_v2_contract_keeps_grouping_keys_prohibited_from_outputs():
    contract = load_feature_contract("config/features/paysim_v2.json")

    assert set(contract.internal_grouping_columns) == {"name_orig", "name_dest", "step"}
    assert not set(contract.internal_grouping_columns) & set(contract.feature_names)
    assert set(contract.internal_grouping_columns) <= contract.prohibited_predictive_sources


def test_bucket_scan_matches_single_destination_calculation():
    rows = _rows().assign(name_dest="C9")
    expected = calculate_destination_history(rows)
    actual = calculate_bucket_history(rows)

    pd.testing.assert_frame_equal(actual, expected)


def test_bucket_calculation_is_independent_of_input_chunk_order():
    rows = pd.concat(
        [_rows().assign(name_dest="C9"), _rows().assign(name_dest="C10", source_row_number=lambda x: x["source_row_number"] + 10)],
        ignore_index=True,
    )

    ordered = calculate_bucket_history(rows)
    shuffled = calculate_bucket_history(rows.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(ordered, shuffled)
