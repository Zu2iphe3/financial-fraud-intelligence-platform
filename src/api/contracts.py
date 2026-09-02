"""Typed public contracts for the 20-feature model-scoring boundary."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


FiniteFeature = float
BinaryFeature = Literal[0, 1]


class PredictionRequest(BaseModel):
    """Leakage-safe feature vector prepared by an upstream feature component.

    Raw identities, balances, fraud labels, and simulator timing are intentionally
    absent. This service validates and scores features; it does not build history.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    amount_log1p: FiniteFeature = Field(ge=0)
    transaction_type_cash_in: BinaryFeature
    transaction_type_cash_out: BinaryFeature
    transaction_type_debit: BinaryFeature
    transaction_type_payment: BinaryFeature
    transaction_type_transfer: BinaryFeature
    dest_tx_count_log1p_24h: FiniteFeature = Field(ge=0)
    dest_tx_count_log1p_168h: FiniteFeature = Field(ge=0)
    # PH7 float32 aggregation can produce tiny values just below zero (about 1e-8),
    # so retain a narrow tolerance while rejecting impossible negative log sums.
    dest_amount_sum_log1p_24h: FiniteFeature = Field(ge=-1e-5)
    dest_amount_sum_log1p_168h: FiniteFeature = Field(ge=-1e-5)
    dest_unique_origins_log1p_24h: FiniteFeature = Field(ge=0)
    dest_unique_origins_log1p_168h: FiniteFeature = Field(ge=0)
    dest_hours_since_previous_activity: FiniteFeature = Field(ge=0, le=744)
    dest_history_available: BinaryFeature
    amount_log_ratio_to_dest_mean_168h: FiniteFeature
    dest_cash_in_count_log1p_24h: FiniteFeature = Field(ge=0)
    dest_cash_out_count_log1p_24h: FiniteFeature = Field(ge=0)
    dest_debit_count_log1p_24h: FiniteFeature = Field(ge=0)
    dest_payment_count_log1p_24h: FiniteFeature = Field(ge=0)
    dest_transfer_count_log1p_24h: FiniteFeature = Field(ge=0)

    @model_validator(mode="after")
    def require_one_transaction_type(self) -> "PredictionRequest":
        """Preserve the one-hot invariant produced by upstream feature preparation."""
        indicators = (
            self.transaction_type_cash_in,
            self.transaction_type_cash_out,
            self.transaction_type_debit,
            self.transaction_type_payment,
            self.transaction_type_transfer,
        )
        if sum(indicators) != 1:
            raise ValueError("exactly one transaction_type indicator must equal 1")
        return self

    def to_frame(self, feature_names: tuple[str, ...]) -> pd.DataFrame:
        """Create one ordered row and reject drift from the persisted model contract."""
        values = self.model_dump()
        if tuple(values) != feature_names:
            raise ValueError("Request schema does not match the persisted feature contract")
        return pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)


class PredictionResponse(BaseModel):
    """Stable output consumed later by the Streamlit application."""

    fraud_probability: float = Field(ge=0, le=1)
    risk_classification: Literal["low", "moderate", "high"]
    model_version: str
    risk_policy_version: str
    risk_policy_status: Literal["portfolio_review_only_not_production_approved"]


class ReadinessResponse(BaseModel):
    """State whether the process can return real model predictions."""

    status: Literal["ready", "unavailable"]
    prediction_available: bool
    model_version: str | None
    feature_contract_version: str
