"""End-to-end training, model selection and evaluation.

Run from the project root:

    python -m src.train

Outputs
-------
reports/metrics.json            all metrics, CV scores, best params, ablation
reports/cv_results.csv          one row per model (CV + holdout)
reports/test_predictions.csv    holdout predictions of every model
reports/best_model.joblib       fitted winning pipeline (used by src.predict)
reports/figures/*.png           every chart referenced by README.md
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, learning_curve

from . import config as cfg
from . import data as data_mod
from . import evaluate as ev
from .models import MODEL_SPECS, build_pipeline
from .preprocess import map_to_original_feature, split_column_types


def _cv_splitter() -> KFold:
    return KFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)


def train_and_evaluate() -> None:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("Loading and cleaning data ...")
    df = data_mod.load_dataset()
    X, y = data_mod.prepare_features_target(df)
    X_train, X_test, y_train, y_test = data_mod.split_train_test(X, y)

    print(
        f"  rows={len(X)}  features={X.shape[1]}  "
        f"train={len(X_train)}  test={len(X_test)}  "
        f"target skew={y.skew():.2f}"
    )

    ev.plot_target_distribution(y, cfg.FIGURES_DIR / "target_distribution.png")
    ev.plot_missing_values(X, cfg.FIGURES_DIR / "missing_values.png")

    # ------------------------------------------------------- model selection
    results: list[dict] = []
    predictions = pd.DataFrame({"actual": y_test.to_numpy()})
    searches: dict[str, GridSearchCV] = {}
    test_preds: dict[str, np.ndarray] = {}

    for name, spec in MODEL_SPECS.items():
        print(f"Training {name} ...", flush=True)
        numeric_cols, categorical_cols = split_column_types(X_train)
        pipeline, grids = build_pipeline(spec, numeric_cols, categorical_cols)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grids,
            cv=_cv_splitter(),
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            refit="rmse",
            n_jobs=-1,
            return_train_score=False,
        )
        started = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - started

        idx = search.best_index_
        cv = {
            "rmse": float(search.cv_results_["mean_test_rmse"][idx]) * -1,
            "rmse_std": float(search.cv_results_["std_test_rmse"][idx]),
            "mae": float(search.cv_results_["mean_test_mae"][idx]) * -1,
            "r2": float(search.cv_results_["mean_test_r2"][idx]),
            "fit_seconds": round(elapsed, 2),
        }

        y_pred = np.asarray(search.best_estimator_.predict(X_test), dtype=float)
        test = ev.regression_metrics(y_test, y_pred)

        searches[name] = search
        test_preds[name] = y_pred
        predictions[name] = np.clip(y_pred, 0.0, None)
        results.append(
            {
                "model": name,
                "description": spec.description,
                "cv_rmse": cv["rmse"],
                "cv_rmse_std": cv["rmse_std"],
                "cv_mae": cv["mae"],
                "cv_r2": cv["r2"],
                "fit_seconds": cv["fit_seconds"],
                "test_rmse": test["rmse"],
                "test_mae": test["mae"],
                "test_r2": test["r2"],
                "test_rmsle": test["rmsle"],
                "test_p90_abs_error": test["p90_abs_error"],
                "best_params": {
                    k.replace("model__regressor__", ""): v
                    for k, v in search.best_params_.items()
                },
            }
        )
        print(
            f"   CV RMSE ${cv['rmse']:>9,.0f} ± ${cv['rmse_std']:>7,.0f} | "
            f"test RMSE ${test['rmse']:>9,.0f} | test R² {test['r2']:.3f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    results_df = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    results_df.to_csv(cfg.REPORTS_DIR / "cv_results.csv", index=False)
    predictions = predictions[["actual"] + list(results_df["model"])]
    predictions.to_csv(cfg.REPORTS_DIR / "test_predictions.csv", index=False)

    # -------------------------------------------------------------- best model
    best_name = str(results_df.loc[0, "model"])
    best_search = searches[best_name]
    best_spec = MODEL_SPECS[best_name]
    best_pred = test_preds[best_name]
    best_test = ev.regression_metrics(y_test, best_pred)
    print(f"\nBest model by CV RMSE: {best_name}")

    baseline_row = results_df[results_df["model"] == "baseline_mean"].iloc[0]
    improvement = {
        "best_model": best_name,
        "rmse_improvement_pct": float(
            100 * (baseline_row["test_rmse"] - best_test["rmse"]) / baseline_row["test_rmse"]
        ),
        "mae_improvement_pct": float(
            100 * (baseline_row["test_mae"] - best_test["mae"]) / baseline_row["test_mae"]
        ),
        "baseline_test_rmse": float(baseline_row["test_rmse"]),
        "baseline_test_mae": float(baseline_row["test_mae"]),
        "baseline_test_r2": float(baseline_row["test_r2"]),
        "best_test_rmse": best_test["rmse"],
        "best_test_mae": best_test["mae"],
        "best_test_r2": best_test["r2"],
    }

    # ------------------------------------------------------------- ablation
    # Leakage-sensitivity check: are the sale-context features (sale month/
    # year / type / condition) carrying the performance? They are known at
    # sale time but not before it, so a model that relies on them would fail
    # in a "list the house before it sells" setting.
    ablation = _run_sale_context_ablation(best_spec, best_search, X_train, X_test, y_train, y_test)

    # ------------------------------------------- out-of-time (temporal) check
    # The headline split mixes 2006-2010 sales at random; a deployment predicts
    # *later* sales.  Refit the winner on 2006-2007 and test on 2008-2010.
    temporal = _run_temporal_evaluation(
        best_spec, best_search.best_params_, X, y, best_test["rmse"]
    )

    # ------------------------------------------------------- model artifact
    artifact_path = _save_best_artifact(
        name=best_name,
        pipeline=best_search.best_estimator_,
        feature_columns=X_train.columns,
        n_train=len(X_train),
        cv_rmse=float(results_df.loc[0, "cv_rmse"]),
        test_metrics=best_test,
        log_target=best_spec.log_target,
    )

    # ---------------------------------------------------------------- charts
    print("\nWriting charts ...")
    metrics_by_model = {
        r["model"]: ev.regression_metrics(y_test, predictions[r["model"]]) for r in results
    }

    ev.plot_model_comparison(results_df, cfg.FIGURES_DIR / "model_comparison.png")
    ev.plot_actual_vs_predicted(
        predictions, metrics_by_model, cfg.FIGURES_DIR / "actual_vs_predicted.png"
    )
    ev.plot_actual_vs_predicted_best(
        y_test, best_pred, best_test, best_name,
        cfg.FIGURES_DIR / "actual_vs_predicted_best_model.png",
    )
    ev.plot_residuals(
        y_test, best_pred, best_test, best_name, cfg.FIGURES_DIR / "residuals_best_model.png"
    )
    ev.plot_error_by_price_segment(
        _error_by_segment(y_test, best_pred), best_name,
        cfg.FIGURES_DIR / "error_by_price_segment.png",
    )

    # ------------------------------------------------- feature importance
    print("Permutation importance (holdout) ...")
    perm = permutation_importance(
        best_search.best_estimator_,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=cfg.N_PERM_REPEATS,
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1,
    )
    perm_series = (
        pd.Series(perm.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
    )
    ev.plot_feature_importance(
        perm_series,
        title=f"{best_name}: permutation importance (holdout RMSE increase)",
        xlabel="Increase in RMSE when the feature is permuted (USD)",
        path=cfg.FIGURES_DIR / "feature_importance_permutation.png",
    )

    gini_name, gini = _impurity_importance(searches, X_train, best_name)
    if gini is not None:
        ev.plot_feature_importance(
            gini,
            title=f"{gini_name}: impurity-based importance (aggregated over one-hot columns)",
            xlabel="Mean decrease in impurity (normalised)",
            path=cfg.FIGURES_DIR / "feature_importance_impurity.png",
        )

    ridge_coefs = _ridge_coefficients(searches)
    if ridge_coefs is not None:
        ev.plot_ridge_coefficients(
            ridge_coefs, cfg.FIGURES_DIR / "ridge_coefficients.png"
        )

    # ------------------------------------------------------- learning curve
    # (only meaningful for a learned model; the dummies have log_target=False)
    if best_spec.log_target:
        print("Learning curve ...")
        _plot_learning_curve(best_spec, best_search, X_train, y_train)

    # --------------------------------------------------------------- report
    report = {
        "dataset": {
            "name": "Ames Housing (Kaggle House Prices / OpenML 42193)",
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "target": cfg.TARGET_COL,
            "target_skew": float(y.skew()),
            "missing_columns": int((X.isna().sum() > 0).sum()),
        },
        "protocol": {
            "test_size": cfg.TEST_SIZE,
            "random_state": cfg.RANDOM_STATE,
            "cv_folds": cfg.CV_FOLDS,
            "cv_strategy": "shuffled KFold on the training split",
            "target_transform": "log1p (inverted before scoring) for learned models",
            "scoring": ["RMSE (USD)", "MAE (USD)", "R-squared", "RMSLE", "p90 |error|"],
        },
        "models": {
            r["model"]: {
                "description": r["description"],
                "cv": {
                    "rmse": r["cv_rmse"],
                    "rmse_std": r["cv_rmse_std"],
                    "mae": r["cv_mae"],
                    "r2": r["cv_r2"],
                    "fit_seconds": r["fit_seconds"],
                },
                "test": {
                    "rmse": r["test_rmse"],
                    "mae": r["test_mae"],
                    "r2": r["test_r2"],
                    "rmsle": r["test_rmsle"],
                    "p90_abs_error": r["test_p90_abs_error"],
                },
                "best_params": r["best_params"],
            }
            for r in results
        },
        "baseline_comparison": improvement,
        "ablation_no_sale_context": ablation,
        "temporal_split": temporal,
        "artifact": {
            "path": artifact_path.name,
            "pipeline_stages": [
                "impute (median / most-frequent)",
                "one-hot (handle_unknown='ignore', min_frequency=5, drop='first')",
                "VarianceThreshold(0.01)",
                "model",
            ],
            "feature_columns": [str(c) for c in X_train.columns],
            "consumed_by": "python -m src.predict",
        },
        "top_features_permutation": {
            k: float(v) for k, v in perm_series.head(15).items()
        },
    }
    with open(cfg.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    _print_summary(results_df, improvement, ablation, temporal)
    print(f"\nSaved metrics -> {cfg.REPORTS_DIR / 'metrics.json'}")
    print(f"Saved charts   -> {cfg.FIGURES_DIR}")
    print(f"Saved model    -> {artifact_path}  (score new rows: python -m src.predict)")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _error_by_segment(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """RMSE/MAE per actual-price quartile (failure-case analysis)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    bins = pd.qcut(y_true_arr, q=4)
    codes = pd.qcut(y_true_arr, q=4, labels=False)

    rows = []
    for code, interval in enumerate(bins.categories):
        idx = np.flatnonzero(codes == code)
        m = ev.regression_metrics(y_true_arr[idx], y_pred[idx])
        rows.append(
            {
                "segment": f"${interval.left/1000:.0f}k–${interval.right/1000:.0f}k",
                "n": int(len(idx)),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
            }
        )
    return pd.DataFrame(rows)


def _transformed_feature_names(fitted_pipeline) -> list[str]:
    """Names of the encoded columns *after* the variance filter."""
    names = list(fitted_pipeline.named_steps["pre"].get_feature_names_out())
    variance_step = fitted_pipeline.named_steps.get("var")
    if variance_step is not None:
        names = list(variance_step.get_feature_names_out(names))
    return names


def _impurity_importance(
    searches: dict[str, GridSearchCV], X_train: pd.DataFrame, best_name: str
) -> tuple[str | None, pd.Series | None]:
    """Impurity-based importance of the best *tree* model available.

    Several estimators do not expose `feature_importances_` (linear models and
    `HistGradientBoostingRegressor`), so when the winning model has none we
    fall back to another tree model that was trained in the same run and name
    the chart after the model that actually produced the scores.
    """
    numeric_cols, categorical_cols = split_column_types(X_train)

    for name in (best_name, "random_forest", "decision_tree"):
        search = searches.get(name)
        if search is None:
            continue
        fitted = search.best_estimator_
        model = fitted.named_steps["model"]
        regressor = getattr(model, "regressor_", model)
        if not hasattr(regressor, "feature_importances_"):
            continue

        names = _transformed_feature_names(fitted)
        original = map_to_original_feature(names, numeric_cols, categorical_cols)
        series = (
            pd.Series(regressor.feature_importances_, index=original)
            .groupby(level=0)
            .sum()
            .sort_values(ascending=False)
        )
        return name, series
    return None, None


def _ridge_coefficients(searches: dict[str, GridSearchCV]):
    """Standardised Ridge coefficients (model must have been selected)."""
    if "ridge" not in searches:
        return None
    fitted = searches["ridge"].best_estimator_
    model = fitted.named_steps["model"]
    regressor = getattr(model, "regressor_", model)
    if not hasattr(regressor, "coef_"):
        return None

    names = _transformed_feature_names(fitted)
    return pd.Series(regressor.coef_, index=[n.split("__", 1)[-1] for n in names])


def _plot_learning_curve(spec, search, X_train, y_train) -> None:
    numeric_cols, categorical_cols = split_column_types(X_train)
    pipeline, _ = build_pipeline(spec, numeric_cols, categorical_cols)
    pipeline.set_params(**search.best_params_)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.3, 1.0, 4),
        cv=_cv_splitter(),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    ev.plot_learning_curve(
        train_sizes, train_scores, val_scores, spec.name,
        cfg.FIGURES_DIR / "learning_curve_best_model.png",
    )


def _run_sale_context_ablation(spec, search, X_train, X_test, y_train, y_test) -> dict:
    """Re-fit the winning model without sale-context features.

    Answers: how much of the headline score depends on features that are only
    known once the sale happens (month/year sold, sale type, sale condition)?
    """
    drop = [c for c in cfg.SALE_CONTEXT if c in X_train.columns]
    Xtr = X_train.drop(columns=drop)
    Xte = X_test.drop(columns=drop)

    numeric_cols, categorical_cols = split_column_types(Xtr)
    pipeline, _ = build_pipeline(spec, numeric_cols, categorical_cols)
    pipeline.set_params(**search.best_params_)
    pipeline.fit(Xtr, y_train)
    metrics = ev.regression_metrics(y_test, pipeline.predict(Xte))

    full_rmse = ev.regression_metrics(y_test, search.best_estimator_.predict(X_test))["rmse"]
    return {
        "dropped_features": drop,
        "full_model_test_rmse": float(full_rmse),
        "test": metrics,
        "rmse_change_pct": float(100 * (metrics["rmse"] - full_rmse) / full_rmse),
    }


def _run_temporal_evaluation(
    spec, best_params: dict, X: pd.DataFrame, y: pd.Series, random_split_rmse: float
) -> dict:
    """Train on sales through 2007, test on every later sale (2008-2010).

    This is the deployment-shaped question the random split cannot answer:
    how does the model behave on genuinely *newer* houses?  It is strictly
    harder by construction -- different price regime, a `YrSold` value that is
    outside the training range, and sale types never seen in training.
    """
    year = X[cfg.TEMPORAL_YEAR_COL]
    train_mask = year <= cfg.TEMPORAL_TRAIN_MAX_YEAR
    test_mask = ~train_mask
    X_tr, X_te = X.loc[train_mask], X.loc[test_mask]
    y_tr, y_te = y.loc[train_mask], y.loc[test_mask]

    def _fit(X_tr_: pd.DataFrame, y_tr_: pd.Series, drop: tuple[str, ...] = ()) -> np.ndarray:
        cols = [c for c in X_tr_.columns if c not in drop]
        numeric_cols, categorical_cols = split_column_types(X_tr_[cols])
        pipeline, _ = build_pipeline(spec, numeric_cols, categorical_cols)
        pipeline.set_params(**best_params)
        pipeline.fit(X_tr_[cols], y_tr_)
        pred = np.asarray(pipeline.predict(X_te[cols]), dtype=float)
        return np.clip(pred, 0.0, None)

    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    predictions = {
        "baseline_mean": np.clip(np.asarray(dummy.predict(X_te), dtype=float), 0.0, None),
        "best_full_features": _fit(X_tr, y_tr),
        "best_without_sale_context": _fit(X_tr, y_tr, drop=cfg.SALE_CONTEXT),
    }
    metrics = {k: ev.regression_metrics(y_te, v) for k, v in predictions.items()}

    return {
        "train_period": f"sales <= {cfg.TEMPORAL_TRAIN_MAX_YEAR}",
        "test_period": (
            f"sales {cfg.TEMPORAL_TRAIN_MAX_YEAR + 1}-{int(year.max())}"
        ),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_price_mean": float(y_tr.mean()),
        "test_price_mean": float(y_te.mean()),
        # Reference point: the headline (random-split) score of the same model.
        "random_split_test_rmse": float(random_split_rmse),
        "rmse_vs_random_split_pct": {
            k: float(100 * (m["rmse"] - random_split_rmse) / random_split_rmse)
            for k, m in metrics.items()
        },
        "models": metrics,
    }


def _save_best_artifact(
    *,
    name: str,
    pipeline,
    feature_columns,
    n_train: int,
    cv_rmse: float,
    test_metrics: dict,
    log_target: bool,
):
    """Persist the fitted winning pipeline for `python -m src.predict`.

    The *whole* pipeline is stored (imputation -> one-hot -> variance filter
    -> model), so scoring a new row reuses the exact preprocessing learned on
    the training split and never refits anything on the scoring data.
    """
    payload = {
        "model_name": name,
        "pipeline": pipeline,
        "feature_columns": [str(c) for c in feature_columns],
        "cv_rmse": float(cv_rmse),
        "holdout_test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "training": {
            "dataset": cfg.DATA_PATH.name,
            "rows": int(n_train),
            "split": f"80% train split, stratified on price deciles (random_state={cfg.RANDOM_STATE})",
            "target_transform": "log1p" if log_target else "none",
        },
    }
    joblib.dump(payload, cfg.MODEL_PATH, compress=3)
    return cfg.MODEL_PATH


def _print_summary(
    results_df: pd.DataFrame, improvement: dict, ablation: dict, temporal: dict
) -> None:
    print("\n" + "=" * 96)
    print("MODEL COMPARISON (5-fold CV on train / single 20% holdout)")
    print("=" * 96)
    header = f"{'model':<20}{'CV RMSE':>12}{'CV MAE':>12}{'CV R²':>9}{'test RMSE':>12}{'test MAE':>12}{'test R²':>9}"
    print(header)
    print("-" * 96)
    for r in results_df.itertuples(index=False):
        print(
            f"{r.model:<20}{r.cv_rmse:>12,.0f}{r.cv_mae:>12,.0f}{r.cv_r2:>9.3f}"
            f"{r.test_rmse:>12,.0f}{r.test_mae:>12,.0f}{r.test_r2:>9.3f}"
        )
    print("-" * 96)
    print(
        f"Baseline (mean) test RMSE ${improvement['baseline_test_rmse']:,.0f} -> "
        f"{improvement['best_model']} ${improvement['best_test_rmse']:,.0f} "
        f"({improvement['rmse_improvement_pct']:.1f}% lower RMSE, "
        f"{improvement['mae_improvement_pct']:.1f}% lower MAE)"
    )
    print(
        f"Ablation without {', '.join(ablation['dropped_features'])}: "
        f"test RMSE ${ablation['test']['rmse']:,.0f} "
        f"({ablation['rmse_change_pct']:+.1f}%)"
    )
    full = temporal["models"]["best_full_features"]
    no_ctx = temporal["models"]["best_without_sale_context"]
    print(
        f"Temporal split ({temporal['train_period']} -> {temporal['test_period']}, "
        f"{temporal['n_train']}/{temporal['n_test']} rows): "
        f"baseline ${temporal['models']['baseline_mean']['rmse']:,.0f} | "
        f"{improvement['best_model']} ${full['rmse']:,.0f} "
        f"({temporal['rmse_vs_random_split_pct']['best_full_features']:+.1f}% vs the "
        f"${temporal['random_split_test_rmse']:,.0f} random-split score) | "
        f"without sale context ${no_ctx['rmse']:,.0f}"
    )


if __name__ == "__main__":
    train_and_evaluate()
