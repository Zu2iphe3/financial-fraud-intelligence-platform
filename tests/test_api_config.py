"""Tests for traceable, non-production PH8-01 service configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.config import ApiConfigurationError, load_api_settings


CONFIG_PATH = Path("config/api/fraud_prediction.json")


def test_api_configuration_matches_twenty_feature_contract():
    settings = load_api_settings()

    assert len(settings.feature_names) == 20
    assert settings.feature_contract_version == "paysim_features_v2"
    assert settings.risk_policy.status == "portfolio_review_only_not_production_approved"
    assert settings.risk_policy.classify(0.0) == "low"
    assert settings.risk_policy.classify(0.2) == "moderate"
    assert settings.risk_policy.classify(0.8) == "high"


def test_local_artifact_path_can_be_overridden_without_changing_policy(tmp_path):
    settings = load_api_settings(environ={"FRAUD_MODEL_ARTIFACT_PATH": str(tmp_path)})
    assert settings.model_artifact_path == tmp_path


def test_risk_thresholds_must_be_ordered(tmp_path):
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    document["risk_policy"]["moderate_min_probability"] = 0.9
    document["risk_policy"]["high_min_probability"] = 0.2
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ApiConfigurationError, match="ordered"):
        load_api_settings(path)
