"""Tests for leakage-safe model candidate construction."""

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.candidates import train_candidate
from src.models.config import load_model_config


def test_logistic_scaler_is_fitted_from_training_features_only():
    """A shifted validation distribution must not influence fitted scaling state."""
    train = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0], "type": [0, 1, 0, 1]})
    validation = pd.DataFrame(
        {"amount": [1_000.0, 2_000.0], "type": [0, 1]}
    )
    config = replace(
        load_model_config(),
        logistic_regression={
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 100,
            "solver": "lbfgs",
        },
    )

    candidate = train_candidate(
        "logistic_regression",
        train,
        pd.Series([0, 0, 1, 1]),
        validation,
        pd.Series([0, 1]),
        config,
        scale_pos_weight=1.0,
    )

    assert isinstance(candidate.estimator, Pipeline)
    np.testing.assert_allclose(
        candidate.estimator.named_steps["scale"].mean_, train.mean().to_numpy()
    )
    assert candidate.tracked_parameters["preprocessing"].startswith("StandardScaler")
    assert candidate.tracked_parameters["converged"] is True
    assert candidate.tracked_parameters["n_iter"] < 100
