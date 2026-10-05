"""Reconstruct the reviewed PaySim v2 vector for one stored transaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
from uuid import UUID

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from src.data.postgres import PostgresSettings
from src.features.behavioral import HISTORY_FEATURE_NAMES, calculate_destination_history
from src.features.contract import FeatureContract, load_feature_contract
from src.features.pipeline import transform_features


SELECTED_TRANSACTION_QUERY = """
SELECT transaction_id, source_dataset_id, source_row_number, step,
       transaction_type, amount, name_orig, name_dest
FROM analytics_marts.fct_transactions
WHERE transaction_id = %s
"""

DESTINATION_HISTORY_QUERY = """
SELECT source_dataset_id, source_row_number, step, transaction_type,
       amount, name_orig, name_dest
FROM analytics_marts.fct_transactions
WHERE source_dataset_id = %s
  AND name_dest = %s
  AND step < %s
ORDER BY step, source_row_number
"""

REVIEW_TRANSACTIONS_QUERY = """
SELECT transaction_id, source_row_number, step, transaction_type,
       amount, name_orig, name_dest
FROM analytics_marts.fct_transactions
ORDER BY transaction_id DESC
LIMIT %s
"""


class PointInTimeFeatureError(RuntimeError):
    """Raised when a trustworthy model vector cannot be reconstructed."""


class TransactionNotFoundError(PointInTimeFeatureError):
    """Raised when the requested logical transaction does not exist."""


class FeatureHistoryUnavailableError(PointInTimeFeatureError):
    """Raised when PostgreSQL cannot provide the required transaction history."""


@dataclass(frozen=True)
class PreparedTransactionFeatures:
    """One audited transaction identity and its exact ordered model vector."""

    transaction_id: str
    source_dataset_id: str
    source_row_number: int
    step: int
    transaction_type: str
    amount: float
    name_orig: str
    name_dest: str
    features: pd.DataFrame

    @property
    def simulation_day(self) -> int:
        """Return PaySim's one-based day for clear investigation context."""
        return ((self.step - 1) // 24) + 1

    @property
    def simulation_hour_of_day(self) -> int:
        """Return PaySim's one-based hour within its simulated day."""
        return ((self.step - 1) % 24) + 1

    def prediction_payload(self) -> dict[str, float | int]:
        """Return JSON-safe scalars while preserving the contract column order."""
        row = self.features.iloc[0]
        return {
            name: int(value) if str(self.features[name].dtype) == "int8" else float(value)
            for name, value in row.items()
        }


@dataclass(frozen=True)
class ReviewTransaction:
    """Human-readable stored transaction offered for investigation."""

    transaction_id: str
    source_row_number: int
    step: int
    transaction_type: str
    amount: float
    name_orig: str
    name_dest: str

    @property
    def simulation_day(self) -> int:
        """Return PaySim's one-based simulated day."""
        return ((self.step - 1) // 24) + 1

    @property
    def simulation_hour_of_day(self) -> int:
        """Return the one-based hour within the simulated day."""
        return ((self.step - 1) % 24) + 1

    @property
    def short_transaction_id(self) -> str:
        """Keep queue labels compact without changing the real lookup identifier."""
        return f"{self.transaction_id[:8]}…{self.transaction_id[-4:]}"


def _validated_transaction_id(transaction_id: str | UUID) -> UUID:
    """Reject malformed identifiers before opening a database connection."""
    try:
        return transaction_id if isinstance(transaction_id, UUID) else UUID(str(transaction_id))
    except (TypeError, ValueError, AttributeError) as error:
        raise TransactionNotFoundError("transaction_id must be a valid UUID") from error


def reconstruct_feature_frame(
    selected: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    contract: FeatureContract,
) -> pd.DataFrame:
    """Build one vector with the same past-only functions used by offline v2.

    The database query already excludes same-step and future rows. These checks remain
    here as a second boundary so a changed repository/query cannot silently leak them.
    """
    selected_dataset = str(selected["source_dataset_id"])
    selected_destination = str(selected["name_dest"])
    selected_step = int(selected["step"])
    selected_row = int(selected["source_row_number"])

    for row in history:
        if str(row["source_dataset_id"]) != selected_dataset:
            raise PointInTimeFeatureError("History crossed the selected source dataset")
        if str(row["name_dest"]) != selected_destination:
            raise PointInTimeFeatureError("History crossed the selected destination")
        if int(row["step"]) >= selected_step:
            raise PointInTimeFeatureError("History contains same-step or future activity")
        if int(row["source_row_number"]) == selected_row:
            raise PointInTimeFeatureError("History contains the selected transaction")

    history_rows = [
        {
            "source_row_number": int(row["source_row_number"]),
            "step": int(row["step"]),
            "name_orig": str(row["name_orig"]),
            "amount": float(row["amount"]),
            "transaction_type": str(row["transaction_type"]),
        }
        for row in history
    ]
    history_rows.append(
        {
            "source_row_number": selected_row,
            "step": selected_step,
            "name_orig": str(selected["name_orig"]),
            "amount": float(selected["amount"]),
            "transaction_type": str(selected["transaction_type"]),
        }
    )

    behavioural = calculate_destination_history(pd.DataFrame(history_rows)).set_index(
        "source_row_number"
    )
    selected_history = behavioural.loc[selected_row, list(HISTORY_FEATURE_NAMES)]
    source = {
        "amount": float(selected["amount"]),
        "transaction_type": str(selected["transaction_type"]),
        **selected_history.to_dict(),
    }
    features = transform_features(pd.DataFrame([source]), contract)
    if tuple(features.columns) != contract.feature_names:
        raise PointInTimeFeatureError("Reconstructed features do not match the v2 contract")
    return features


def reconstruct_from_connection(
    connection: psycopg.Connection,
    transaction_id: str | UUID,
    contract: FeatureContract,
) -> PreparedTransactionFeatures:
    """Load safe history and reconstruct one existing transaction."""
    logical_id = _validated_transaction_id(transaction_id)
    selected = connection.execute(
        SELECTED_TRANSACTION_QUERY, (logical_id,)
    ).fetchone()
    if selected is None:
        raise TransactionNotFoundError("Transaction was not found")

    history = connection.execute(
        DESTINATION_HISTORY_QUERY,
        (
            selected["source_dataset_id"],
            selected["name_dest"],
            selected["step"],
        ),
    ).fetchall()
    features = reconstruct_feature_frame(selected, list(history), contract)
    return PreparedTransactionFeatures(
        transaction_id=str(selected["transaction_id"]),
        source_dataset_id=str(selected["source_dataset_id"]),
        source_row_number=int(selected["source_row_number"]),
        step=int(selected["step"]),
        transaction_type=str(selected["transaction_type"]),
        amount=float(selected["amount"]),
        name_orig=str(selected["name_orig"]),
        name_dest=str(selected["name_dest"]),
        features=features,
    )


ConnectFunction = Callable[..., psycopg.Connection]


class PointInTimeFeatureService:
    """Open short-lived PostgreSQL sessions for transaction feature preparation."""

    def __init__(
        self,
        settings: PostgresSettings,
        *,
        contract: FeatureContract | None = None,
        connect: ConnectFunction = psycopg.connect,
    ) -> None:
        self.settings = settings
        self.contract = contract or load_feature_contract("config/features/paysim_v2.json")
        self._connect = connect

    def reconstruct(self, transaction_id: str | UUID) -> PreparedTransactionFeatures:
        """Return a real feature vector or fail without fabricating defaults."""
        logical_id = _validated_transaction_id(transaction_id)
        try:
            with self._connect(
                **self.settings.connection_kwargs(), row_factory=dict_row
            ) as connection:
                return reconstruct_from_connection(connection, logical_id, self.contract)
        except TransactionNotFoundError:
            raise
        except (psycopg.Error, OSError) as error:
            raise FeatureHistoryUnavailableError(
                "PostgreSQL transaction history is unavailable"
            ) from error

    def list_review_transactions(self, limit: int = 20) -> tuple[ReviewTransaction, ...]:
        """Return real stored transactions without using fraud labels to rank them."""
        if isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("Review transaction limit must be between 1 and 100")
        try:
            with self._connect(
                **self.settings.connection_kwargs(), row_factory=dict_row
            ) as connection:
                rows = connection.execute(REVIEW_TRANSACTIONS_QUERY, (limit,)).fetchall()
        except (psycopg.Error, OSError) as error:
            raise FeatureHistoryUnavailableError(
                "PostgreSQL transaction history is unavailable"
            ) from error
        return tuple(
            ReviewTransaction(
                transaction_id=str(row["transaction_id"]),
                source_row_number=int(row["source_row_number"]),
                step=int(row["step"]),
                transaction_type=str(row["transaction_type"]),
                amount=float(row["amount"]),
                name_orig=str(row["name_orig"]),
                name_dest=str(row["name_dest"]),
            )
            for row in rows
        )
