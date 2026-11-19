"""Score new houses with the trained pipeline.

Run from the project root:

    python -m src.predict                      # demo: 5 held-out rows
    python -m src.predict --input houses.csv   # every row of a raw Ames CSV
    python -m src.predict --input houses.csv --output predictions.csv

`reports/best_model.joblib` is written by `python -m src.train` and holds the
*fitted* pipeline (imputation -> one-hot -> variance filter -> model) together
with the training feature order.  Scoring therefore reuses exactly the
preprocessing learned on the training split: nothing is refitted and no
statistic is estimated from the scoring data.

Input format: a raw Ames Housing CSV with the same columns as
`data/raw/ames_housing.csv`.  `Id` and `SalePrice` are optional; any extra
column is ignored, and a missing required column is reported by name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from . import data as data_mod


def load_artifact(path: Path) -> dict:
    """Load the fitted pipeline written by `src.train`."""
    if not Path(path).exists():
        raise SystemExit(
            f"Model artifact not found: {path}\n"
            "Train first with:  python -m src.train"
        )
    return joblib.load(path)


def missing_source_columns(raw_cols: list[str], expected: list[str]) -> list[str]:
    """Required raw columns the file does not have (derived ones are optional)."""
    present = set(raw_cols)
    required = [c for c in expected if c not in data_mod.DERIVED_COLUMNS]
    return [c for c in required if c not in present]


def score_file(path: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    """Read, clean and align a raw CSV; return (features, actual prices|None)."""
    raw = pd.read_csv(path)
    missing = missing_source_columns(list(raw.columns), feature_columns)
    if missing:
        raise SystemExit(
            f"{path} is missing {len(missing)} required column(s): "
            f"{', '.join(missing)}"
        )

    try:
        cleaned = data_mod.clean_frame(raw)
    except RuntimeError as exc:  # a derived feature could not be built
        raise SystemExit(f"{path}: {exc}") from None

    actual = (
        cleaned[cfg.TARGET_COL].astype(float)
        if cfg.TARGET_COL in cleaned.columns
        else None
    )
    # Aligns to the training feature order and drops everything else
    # (`Id`, `SalePrice`, user-supplied extra columns).
    features = cleaned.reindex(columns=feature_columns)
    return features, actual


def demo_rows(rows: int, start: int) -> tuple[pd.DataFrame, pd.Series]:
    """Take rows from the untouched holdout so the demo predicts unseen houses."""
    if not cfg.DATA_PATH.exists():
        raise SystemExit(
            f"{cfg.DATA_PATH} not found.\n"
            "Fetch it with:  python scripts/download_data.py"
        )
    df = data_mod.clean_frame(pd.read_csv(cfg.DATA_PATH))
    X, y = data_mod.prepare_features_target(df)
    _, X_test, _, y_test = data_mod.split_train_test(X, y)

    idx = X_test.index[start : start + rows]
    if len(idx) == 0:
        raise SystemExit(f"--start {start} is beyond the {len(X_test)} holdout rows.")
    return X_test.loc[idx], y_test.loc[idx]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Score houses with the fitted pipeline from reports/best_model.joblib",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="raw Ames CSV to score (default: rows from the held-out split)",
    )
    parser.add_argument("--model", type=Path, default=cfg.MODEL_PATH, help="artifact path")
    parser.add_argument("--output", type=Path, default=None, help="write predictions to this CSV")
    parser.add_argument("--rows", type=int, default=5, help="number of demo rows to score")
    parser.add_argument("--start", type=int, default=0, help="first demo row offset")
    args = parser.parse_args(argv)

    payload = load_artifact(args.model)
    pipeline = payload["pipeline"]
    feature_columns = payload["feature_columns"]

    if args.input is None:
        features, actual = demo_rows(args.rows, args.start)
        source = f"{len(features)} held-out row(s) from {cfg.DATA_PATH.name}"
    else:
        features, actual = score_file(args.input, feature_columns)
        source = f"{len(features)} row(s) from {args.input}"

    features = features.reindex(columns=feature_columns)
    predicted = np.clip(np.asarray(pipeline.predict(features), dtype=float), 0.0, None)

    print(
        f"model: {payload['model_name']}  ({cfg.MODEL_PATH if args.model == cfg.MODEL_PATH else args.model})\n"
        f"       CV RMSE ${payload['cv_rmse']:,.0f} | "
        f"{payload['training']['rows']} training rows | "
        f"{len(feature_columns)} features\n"
        f"scored: {source}\n"
    )

    header = f"{'row':>5}{'actual':>12}{'predicted':>12}{'error %':>10}"
    print(header)
    print("-" * len(header))

    errors: list[float] = []
    for position, i in enumerate(features.index):
        row_no = position + 1
        if actual is None:
            print(f"{row_no:>5}{'-':>12}{predicted[position]:>12,.0f}{'-':>10}")
            continue
        truth = float(actual.loc[i])
        error = 100.0 * (predicted[position] - truth) / truth
        errors.append(error)
        print(f"{row_no:>5}{truth:>12,.0f}{predicted[position]:>12,.0f}{error:>+10.1f}")

    if actual is None:
        print("\nInput has no `SalePrice` column: predicted values only.")

    if args.output is not None:
        out = pd.DataFrame(
            {"row": np.arange(1, len(predicted) + 1), "predicted_price": np.round(predicted, 2)}
        )
        if actual is not None:
            out.insert(1, "actual_price", actual.to_numpy(dtype=float))
            out["abs_error"] = (out["predicted_price"] - out["actual_price"]).abs()
        out.to_csv(args.output, index=False)
        print(f"\nWrote {len(out)} prediction(s) -> {args.output}")

    if errors:
        print(f"\nmean error {np.mean(errors):+.1f}% over {len(errors)} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
