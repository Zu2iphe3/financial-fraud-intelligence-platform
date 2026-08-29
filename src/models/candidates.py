"""Build the three comparable PH7-02 candidate fraud classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.models.config import ModelConfig


@dataclass(frozen=True)
class TrainedCandidate:
    """A fitted estimator and the parameters that must be tracked with it."""

    name: str
    estimator: ClassifierMixin
    tracked_parameters: dict[str, Any]


def _matrix(features: pd.DataFrame) -> np.ndarray:
    """Use one compact numeric representation for every candidate model."""
    return features.to_numpy(dtype=np.float32, copy=False)


def train_candidate(
    name: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    config: ModelConfig,
    scale_pos_weight: float,
) -> TrainedCandidate:
    """Fit one configured candidate using training labels and validation only."""
    train_matrix = _matrix(train_features)
    validation_matrix = _matrix(validation_features)
    if name == "no_skill":
        parameters: dict[str, Any] = {"strategy": "prior"}
        estimator: ClassifierMixin = DummyClassifier(**parameters)
        estimator.fit(train_matrix, train_target)
    elif name == "logistic_regression":
        logistic_parameters = {
            **dict(config.logistic_regression),
            "random_state": config.random_seed,
        }
        parameters = {
            "preprocessing": "StandardScaler fitted on training partition only",
            **logistic_parameters,
        }
        # Scaling is inside the fitted estimator so validation/test data can only be
        # transformed with means and variances learned from the training partition.
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(**logistic_parameters)),
            ]
        )
        estimator.fit(train_matrix, train_target)
        iterations = int(estimator.named_steps["classifier"].n_iter_[0])
        parameters["n_iter"] = iterations
        parameters["converged"] = iterations < int(logistic_parameters["max_iter"])
    elif name == "xgboost":
        parameters = {
            **dict(config.xgboost),
            "random_state": config.random_seed,
            "scale_pos_weight": scale_pos_weight,
        }
        estimator = XGBClassifier(**parameters)
        # Early stopping is allowed to use validation because this partition exists
        # specifically for tuning; the final test partition is not loaded yet.
        estimator.fit(
            train_matrix,
            train_target,
            eval_set=[(validation_matrix, validation_target)],
            verbose=False,
        )
        if hasattr(estimator, "best_iteration"):
            parameters["best_iteration"] = int(estimator.best_iteration)
    else:
        raise ValueError(f"Unsupported candidate {name!r}")
    return TrainedCandidate(name=name, estimator=estimator, tracked_parameters=parameters)


def fraud_probabilities(
    candidate: TrainedCandidate, features: pd.DataFrame
) -> np.ndarray:
    """Return fraud-class probabilities in source row order."""
    probabilities = candidate.estimator.predict_proba(_matrix(features))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(f"Candidate {candidate.name!r} did not return binary probabilities")
    return np.asarray(probabilities[:, 1], dtype=np.float64)


def model_explanation(candidate: TrainedCandidate, feature_names: tuple[str, ...]) -> pd.DataFrame:
    """Expose baseline coefficients or tree importance without claiming causality."""
    if candidate.name == "logistic_regression":
        classifier = (
            candidate.estimator.named_steps["classifier"]
            if isinstance(candidate.estimator, Pipeline)
            else candidate.estimator
        )
        values = np.asarray(classifier.coef_)[0]
        measure = "log_odds_coefficient"
    elif candidate.name == "xgboost":
        values = np.asarray(candidate.estimator.feature_importances_)
        measure = "feature_importance"
    else:
        return pd.DataFrame(columns=("feature", "measure", "value"))
    return pd.DataFrame(
        {"feature": feature_names, "measure": measure, "value": values}
    ).sort_values("value", ascending=False, ignore_index=True)
