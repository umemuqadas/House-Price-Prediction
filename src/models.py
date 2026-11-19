"""Model registry: baselines, linear models and tree ensembles.

Each entry bundles
* a *fresh* estimator factory (never share estimator instances between runs),
* whether the target is log-transformed (`log1p`) before fitting,
* the hyper-parameter grid searched with 5-fold CV,
* whether the linear pipeline needs feature scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .preprocess import make_preprocessor, make_variance_filter


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], Any]
    log_target: bool
    scale: bool
    grid: dict[str, list[Any]] = field(default_factory=dict)
    description: str = ""


def _ttr(estimator, log_target: bool):
    """Wrap an estimator so it trains on log1p(price) and predicts dollars.

    The right-skewed target (skew = 1.88) makes squared-error loss dominated
    by the most expensive houses; modelling log-price reduces that leverage
    and stabilises the linear models. Metrics are always reported in dollars
    because `TransformedTargetRegressor` inverse-transforms on predict.
    """
    if not log_target:
        return estimator
    return TransformedTargetRegressor(
        regressor=estimator, func=np.log1p, inverse_func=np.expm1
    )


# --------------------------------------------------------------------------- #
# Registry (order defines the order of the comparison report)
# --------------------------------------------------------------------------- #
MODEL_SPECS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec(
            name="baseline_mean",
            factory=lambda: DummyRegressor(strategy="mean"),
            log_target=False,
            scale=False,
            description="Predicts the training-set mean price (minimises RMSE).",
        ),
        ModelSpec(
            name="baseline_median",
            factory=lambda: DummyRegressor(strategy="median"),
            log_target=False,
            scale=False,
            description="Predicts the training-set median price (minimises MAE).",
        ),
        ModelSpec(
            name="linear_regression",
            factory=lambda: LinearRegression(),
            log_target=True,
            scale=True,
            description="Ordinary least squares on log-price, one-hot features.",
        ),
        ModelSpec(
            name="ridge",
            factory=lambda: Ridge(),
            log_target=True,
            scale=True,
            grid={"model__regressor__alpha": [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]},
            description="L2-regularised linear model; alpha tuned by CV.",
        ),
        ModelSpec(
            name="decision_tree",
            factory=lambda: DecisionTreeRegressor(random_state=42),
            log_target=True,
            scale=False,
            grid={
                "model__regressor__max_depth": [4, 6, 8, 10, 12, None],
                "model__regressor__min_samples_leaf": [1, 2, 5, 10],
            },
            description="Single CART tree; depth/leaf size tuned by CV.",
        ),
        ModelSpec(
            name="random_forest",
            factory=lambda: RandomForestRegressor(
                n_estimators=400, random_state=42, n_jobs=1
            ),
            log_target=True,
            scale=False,
            grid={
                "model__regressor__max_features": [0.5, "sqrt"],
                "model__regressor__min_samples_leaf": [1, 2, 4],
            },
            description="Bagged trees; feature sampling and leaf size tuned by CV.",
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            factory=lambda: HistGradientBoostingRegressor(
                max_iter=300, random_state=42
            ),
            log_target=True,
            scale=False,
            grid={
                "model__regressor__learning_rate": [0.05, 0.1],
                "model__regressor__max_leaf_nodes": [15, 31],
                "model__regressor__l2_regularization": [0.0, 1.0],
            },
            description=(
                "Histogram gradient boosting (bin-based splits, Newton steps); "
                "learning rate, leaf count and L2 tuned by CV."
            ),
        ),
    ]
}


def build_pipeline(
    spec: ModelSpec, numeric_cols: list[str], categorical_cols: list[str]
) -> tuple[Pipeline, list[dict[str, list[Any]]]]:
    """Return (unfitted pipeline, list of param grids) for a model spec.

    Pipeline stages: preprocessing -> near-zero-variance filter -> regressor
    (optionally wrapped in a log-target transform).  Every stage that learns
    from data is refit on each training fold by `GridSearchCV`.
    """
    preprocessor = make_preprocessor(
        numeric_cols, categorical_cols, scale=spec.scale
    )
    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("var", make_variance_filter()),
            ("model", _ttr(spec.factory(), spec.log_target)),
        ]
    )
    # `GridSearchCV` needs a list of dicts; `[{}]` means "no tuning".
    grids: list[dict[str, list[Any]]] = [spec.grid] if spec.grid else [{}]
    return pipeline, grids
