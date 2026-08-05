"""Create a reproducible profile of the PaySim transaction dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
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
]

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def profile_dataset(csv_path: Path, chunk_size: int = 250_000) -> dict[str, object]:
    """Profile a large PaySim CSV without loading the entire file into memory."""
    row_count = 0
    missing_values: Counter[str] = Counter()
    transaction_counts: Counter[str] = Counter()
    fraud_counts: Counter[str] = Counter()
    flagged_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    zero_counts: Counter[str] = Counter()
    fraud_label_counts: Counter[int] = Counter()
    flagged_label_counts: Counter[int] = Counter()
    amount_sum = 0.0
    fraud_amount_sum = 0.0
    fraud_count = 0
    flagged_count = 0
    amount_min = float("inf")
    amount_max = float("-inf")
    fraud_amount_min = float("inf")
    fraud_amount_max = float("-inf")
    step_min: int | None = None
    step_max: int | None = None
    columns: list[str] | None = None
    dtypes: dict[str, str] = {}

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if columns is None:
            columns = chunk.columns.tolist()
            if columns != EXPECTED_COLUMNS:
                raise ValueError(
                    f"Unexpected columns. Expected {EXPECTED_COLUMNS}, received {columns}."
                )
            dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}

        row_count += len(chunk)
        missing_values.update(chunk.isna().sum().astype(int).to_dict())
        transaction_counts.update(chunk["type"].value_counts().to_dict())

        fraud_rows = chunk[chunk["isFraud"] == 1]
        flagged_rows = chunk[chunk["isFlaggedFraud"] == 1]
        fraud_label_counts.update(chunk["isFraud"].value_counts().to_dict())
        flagged_label_counts.update(chunk["isFlaggedFraud"].value_counts().to_dict())
        fraud_counts.update(fraud_rows["type"].value_counts().to_dict())
        flagged_counts.update(flagged_rows["type"].value_counts().to_dict())

        fraud_count += len(fraud_rows)
        flagged_count += len(flagged_rows)
        amount_sum += float(chunk["amount"].sum())
        fraud_amount_sum += float(fraud_rows["amount"].sum())
        amount_min = min(amount_min, float(chunk["amount"].min()))
        amount_max = max(amount_max, float(chunk["amount"].max()))
        if not fraud_rows.empty:
            fraud_amount_min = min(fraud_amount_min, float(fraud_rows["amount"].min()))
            fraud_amount_max = max(fraud_amount_max, float(fraud_rows["amount"].max()))

        current_step_min = int(chunk["step"].min())
        current_step_max = int(chunk["step"].max())
        step_min = current_step_min if step_min is None else min(step_min, current_step_min)
        step_max = current_step_max if step_max is None else max(step_max, current_step_max)

        for column in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
            negative_counts[column] += int((chunk[column] < 0).sum())
            zero_counts[column] += int((chunk[column] == 0).sum())

    if not row_count or columns is None:
        raise ValueError("The dataset is empty.")

    type_summary = {}
    for transaction_type in sorted(transaction_counts):
        count = int(transaction_counts[transaction_type])
        type_fraud_count = int(fraud_counts[transaction_type])
        type_summary[transaction_type] = {
            "transactions": count,
            "fraud_transactions": type_fraud_count,
            "fraud_rate": type_fraud_count / count,
            "flagged_transactions": int(flagged_counts[transaction_type]),
        }

    return {
        "source_file": csv_path.name,
        "file_size_bytes": csv_path.stat().st_size,
        "rows": row_count,
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": {column: int(missing_values[column]) for column in columns},
        "step_range": {"minimum": step_min, "maximum": step_max},
        "amount": {
            "minimum": amount_min,
            "maximum": amount_max,
            "mean": amount_sum / row_count,
        },
        "fraud": {
            "transactions": fraud_count,
            "rate": fraud_count / row_count,
            "amount_minimum": fraud_amount_min,
            "amount_maximum": fraud_amount_max,
            "amount_mean": fraud_amount_sum / fraud_count if fraud_count else None,
        },
        "flagged_fraud": {
            "transactions": flagged_count,
            "rate": flagged_count / row_count,
        },
        "negative_value_counts": {column: int(negative_counts[column]) for column in negative_counts},
        "zero_value_counts": {column: int(zero_counts[column]) for column in zero_counts},
        "label_values": {
            "isFraud": {str(label): int(count) for label, count in sorted(fraud_label_counts.items())},
            "isFlaggedFraud": {
                str(label): int(count) for label, count in sorted(flagged_label_counts.items())
            },
        },
        "transaction_types": type_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to the raw PaySim CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/data_exploration_summary.json"),
        help="Path for the generated JSON summary",
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = profile_dataset(args.csv_path, args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Profiled {summary['rows']:,} rows. Summary written to {args.output}.")


if __name__ == "__main__":
    main()
