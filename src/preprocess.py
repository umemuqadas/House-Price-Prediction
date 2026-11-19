"""Preprocessing built as scikit-learn objects so it can live inside a Pipeline.

Design rules that guarantee *zero target leakage*:

* every estimator that learns from data (imputer medians, most-frequent
  categories, the one-hot vocabulary, the scaler's mean/std) is fit inside
  the pipeline, i.e. on training rows only;
* within model selection, `GridSearchCV` refits the whole pipeline on each
  training fold, so the held-out fold never influences its own preprocessing;
* the holdout test set is touched exactly once, for the final evaluation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg


def split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) of a feature frame."""
    numeric = [
        c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
    ]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    scale: bool,
) -> ColumnTransformer:
    """Median-impute + (optionally) scale numerics; impute + one-hot cats.

    Parameters
    ----------
    scale:
        Trees are invariant to feature scaling, so their pipeline skips the
        `StandardScaler`. Linear models need it for regularisation to behave.
    """
    if scale:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical_pipe = Pipeline(
        steps=[
            # Genuine remaining missingness (e.g. `Electrical`, 1 row).
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",          # unseen test categories
                    min_frequency=cfg.ONEHOT_MIN_FREQUENCY,  # rare-level bucket
                    drop="first",                     # avoid the dummy trap
                    sparse_output=False,              # small dense matrix
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_variance_filter() -> VarianceThreshold:
    """Drop encoded features that are non-zero in <1% of the training rows.

    Fit *inside* the pipeline, so the threshold is evaluated on each training
    fold only.  Without it, a level seen in ~9 training rows (PoolQC='Ex',
    PoolArea>0) receives a coefficient of +16 to +42 log-price units and the
    model extrapolates to $10^15 on any holdout row carrying that level.
    """
    return VarianceThreshold(threshold=cfg.VARIANCE_THRESHOLD)


def map_to_original_feature(
    transformed_names: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    """Map pipeline output names (`num__LotArea`, `cat__Neighborhood_NAmes`)
    back to the original feature they came from.

    Used to aggregate one-hot importance scores per source column.
    """
    numeric_set = set(numeric_cols)
    # Longest-first so that `BsmtQual` is matched before any shorter prefix.
    cats_by_length = sorted(categorical_cols, key=len, reverse=True)

    original: list[str] = []
    for name in transformed_names:
        body = name.split("__", 1)[-1]
        if body in numeric_set:
            original.append(body)
            continue
        match = next((c for c in cats_by_length if body.startswith(c + "_")), None)
        original.append(match if match is not None else body)
    return original
