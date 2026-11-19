"""Data loading, documented cleaning rules and leakage-free feature engineering.

Every transformation implemented here is *row-wise*: it only ever looks at the
row being transformed (or at the official Ames data dictionary). No statistic
is estimated from other rows and nothing derived from the target enters the
feature set, so these steps are safe to run before the train/test split.

Statistical preprocessing (imputation medians, one-hot vocabularies, scaling)
is deliberately **not** done here -- it lives inside the scikit-learn pipeline
in `src/preprocess.py` so that it is refit on every training fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg

# Columns that never appear in the raw CSV: they are created by the cleaning
# rules and by `add_derived_features` below.  `src.predict` subtracts this set
# from the model's feature list to work out which columns an inference file
# must actually contain.
DERIVED_COLUMNS = (
    "TotalSF",
    "TotalBath",
    "HouseAge",
    "YearsSinceRemod",
    "TotalPorchSF",
    "LogLotArea",
    "LogGrLivArea",
    *(f"{col}_is_missing" for col in cfg.MISSING_INDICATORS),
)


def load_dataset(path: Path | str = cfg.DATA_PATH) -> pd.DataFrame:
    """Load the Ames Housing CSV and apply the documented cleaning rules."""
    return clean_frame(pd.read_csv(path))


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning rules to an already-loaded raw frame.

    Kept separate from `load_dataset` so that inference (`src.predict`) can
    clean a frame it read itself instead of re-reading the file.
    """
    # Identifier columns carry no signal and would only add noise.
    df = df.drop(columns=[c for c in cfg.ID_COLS if c in df.columns])

    # "NA" == feature absent, per the Ames data dictionary.
    for col in cfg.NA_IS_NONE:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # No masonry veneer recorded -> 0 sq ft (8 rows have both fields missing).
    for col in cfg.NUMERIC_NA_TO_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Missingness indicators must be created *before* any imputation happens.
    for col in cfg.MISSING_INDICATORS:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(np.int8)

    # `MSSubClass` is a categorical code (20, 30, 60 ...), not a continuous
    # quantity, so treating it as numeric would invent a fake ordering.
    for col in cfg.CATEGORICAL_AS_CODE:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df = add_derived_features(df)

    # Backstop against drift between the raw schema and the derived features.
    produced = [c for c in DERIVED_COLUMNS if c not in df.columns]
    if produced:
        raise RuntimeError(
            "expected derived feature(s) missing after cleaning: "
            f"{produced} - the input lacks their source column(s)"
        )
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven features (all computed from a single row).

    These encode widely used housing-price relationships: total finished
    area, bathroom count, property age, time since remodel and porch area.
    """
    out = df.copy()

    out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]
    out["TotalBath"] = (
        out["FullBath"]
        + 0.5 * out["HalfBath"]
        + out["BsmtFullBath"]
        + 0.5 * out["BsmtHalfBath"]
    )
    out["HouseAge"] = out["YrSold"] - out["YearBuilt"]
    out["YearsSinceRemod"] = out["YrSold"] - out["YearRemodAdd"]
    out["TotalPorchSF"] = (
        out["OpenPorchSF"] + out["EnclosedPorch"] + out["3SsnPorch"] + out["ScreenPorch"]
    )
    # Right-skewed size predictors are log-compressed so that their
    # relationship with the (log) target is closer to linear.
    out["LogLotArea"] = np.log1p(out["LotArea"])
    out["LogGrLivArea"] = np.log1p(out["GrLivArea"])

    return out


def prepare_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the cleaned frame into features X and the price target y."""
    y = df[cfg.TARGET_COL].astype(float)
    X = df.drop(columns=[cfg.TARGET_COL])
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 holdout split, stratified on price deciles.

    Stratifying on price deciles keeps the price distribution of the small
    holdout set representative of the full dataset, which makes the single
    holdout estimate less noisy than a plain random split (the holdout price
    standard deviation ends up within 1% of the population value).
    """
    strata = pd.qcut(y, q=cfg.STRATA_BINS, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=strata,
    )
    return X_train, X_test, y_train, y_test
