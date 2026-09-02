"""Load traceable model and risk-policy settings for the prediction API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.features.contract import load_feature_contract
from src.models.data import contract_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config/api/fraud_prediction.json"


class ApiConfigurationError(ValueError):
    """Raised when scoring configuration is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class RiskPolicy:
    """Validation-derived portfolio risk bands, not a production fraud decision."""

    version: str
    status: str
    moderate_min_probability: float
    high_min_probability: float
    evidence: str

    def classify(self, probability: float) -> str:
        """Map one valid fraud score into its configured review band."""
        if probability >= self.high_min_probability:
            return "high"
        if probability >= self.moderate_min_probability:
            return "moderate"
        return "low"


@dataclass(frozen=True)
class ApiSettings:
    """Resolved service settings and the exact expected model-vector identity."""

    service_name: str
    service_version: str
    model_artifact_path: Path
    model_version: str
    selected_model: str
    selected_run_id: str
    feature_contract_version: str
    feature_contract_sha256: str
    feature_names: tuple[str, ...]
    risk_policy: RiskPolicy


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_api_settings(
    path: str | Path = DEFAULT_CONFIG_PATH,
    environ: Mapping[str, str] | None = None,
) -> ApiSettings:
    """Load committed policy while allowing only the local artifact path to vary."""
    environment = os.environ if environ is None else environ
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        model = document["model"]
        risk = document["risk_policy"]
        artifact_value = environment.get(
            "FRAUD_MODEL_ARTIFACT_PATH", str(model["artifact_path"])
        )
        contract_path = _resolve_path(str(model["feature_contract_path"]))
        contract = load_feature_contract(contract_path)
        policy = RiskPolicy(
            version=str(risk["policy_version"]),
            status=str(risk["status"]),
            moderate_min_probability=float(risk["moderate_min_probability"]),
            high_min_probability=float(risk["high_min_probability"]),
            evidence=str(risk["evidence"]),
        )
        settings = ApiSettings(
            service_name=str(document["service_name"]),
            service_version=str(document["service_version"]),
            model_artifact_path=_resolve_path(artifact_value),
            model_version=str(model["model_version"]),
            selected_model=str(model["selected_model"]),
            selected_run_id=str(model["mlflow_run_id"]),
            feature_contract_version=contract.contract_version,
            feature_contract_sha256=contract_sha256(contract_path),
            feature_names=contract.feature_names,
            risk_policy=policy,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ApiConfigurationError):
            raise
        raise ApiConfigurationError("Prediction API configuration is invalid") from error

    if len(settings.feature_names) != 20:
        raise ApiConfigurationError("PH8-01 requires the reviewed 20-feature v2 contract")
    if not (
        0 < policy.moderate_min_probability < policy.high_min_probability < 1
    ):
        raise ApiConfigurationError("Risk thresholds must be ordered strictly within 0 and 1")
    if policy.status != "portfolio_review_only_not_production_approved":
        raise ApiConfigurationError("Risk policy must remain explicitly non-production")
    return settings
