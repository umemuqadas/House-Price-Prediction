"""Domain metrics and every chart produced by the project.

All metrics are computed in dollars on the *original* price scale (models are
trained on log-price and inverse-transformed before scoring), so the baseline
and the learned models are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
    root_mean_squared_error,
)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GREY = "#7f7f7f"


def _usd_thousands(axis) -> None:
    """Format an axis in thousands of dollars (avoids overlapping tick labels)."""
    from matplotlib.ticker import FuncFormatter

    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))



# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE / MAE / R-squared plus two supporting diagnostics.

    `rmsle` measures relative error (useful for a skewed price target) and
    `p90_abs_error` captures the tail of the error distribution, which the
    mean-based metrics hide.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    residuals = y_true - y_pred

    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmsle": float(np.sqrt(mean_squared_log_error(y_true, y_pred))),
        "p90_abs_error": float(np.percentile(np.abs(residuals), 90)),
    }


# --------------------------------------------------------------------------- #
# EDA charts
# --------------------------------------------------------------------------- #
def plot_target_distribution(y: pd.Series, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(y, bins=40, color=BLUE, edgecolor="white")
    axes[0].axvline(y.mean(), color=RED, ls="--", lw=1.5,
                    label=f"mean = \\${y.mean():,.0f}")
    axes[0].set_title("SalePrice distribution")
    axes[0].set_xlabel("Sale price (USD)")
    axes[0].set_ylabel("Houses")
    axes[0].legend()

    axes[1].hist(np.log1p(y), bins=40, color=GREEN, edgecolor="white")
    axes[1].set_title(f"log1p(SalePrice)  (skew {y.skew():.2f} → {np.log1p(y).skew():.2f})")
    axes[1].set_xlabel("log(1 + sale price)")
    axes[1].set_ylabel("Houses")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_missing_values(X: pd.DataFrame, path) -> None:
    missing = X.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    pct = 100 * missing / len(X)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(pct.index, pct.values, color=ORANGE, edgecolor="white")
    for i, v in enumerate(pct.values):
        ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% of rows missing")
    ax.set_title("Missing values after documented NA-semantics cleaning\n"
                 "(remaining NaN is imputed inside each CV fold)")
    ax.set_xlim(0, pct.max() * 1.18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
def plot_model_comparison(results: pd.DataFrame, path) -> None:
    """Grouped bars for RMSE/MAE and R-squared across every model."""
    df = results.sort_values("cv_rmse")
    x = np.arange(len(df))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].bar(x - width / 2, df["cv_rmse"], width, label="CV RMSE", color=BLUE)
    axes[0].bar(x + width / 2, df["cv_mae"], width, label="CV MAE", color=ORANGE)
    for i, (r, m) in enumerate(zip(df["cv_rmse"], df["cv_mae"])):
        axes[0].text(i - width / 2, r, f"{r/1000:.0f}k", ha="center", va="bottom", fontsize=8)
        axes[0].text(i + width / 2, m, f"{m/1000:.0f}k", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, df["model"], rotation=30, ha="right")
    axes[0].set_ylabel("USD (lower is better)")
    axes[0].set_title("Cross-validated error vs. baseline")
    axes[0].legend()

    colors = [GREEN if v > 0 else GREY for v in df["cv_r2"]]
    axes[1].bar(x, df["cv_r2"], color=colors, edgecolor="white")
    for i, v in enumerate(df["cv_r2"]):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xticks(x, df["model"], rotation=30, ha="right")
    axes[1].set_ylabel("$R^2$ (higher is better)")
    axes[1].set_title("Cross-validated $R^2$")
    axes[1].set_ylim(min(0, df["cv_r2"].min() - 0.05), max(df["cv_r2"].max() + 0.1, 0.1))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(
    predictions: pd.DataFrame, metrics_by_model: dict[str, dict], path
) -> None:
    """Actual-vs-predicted scatter for every model (grid)."""
    models = list(predictions.columns[1:])
    n_cols = 3
    n_rows = int(np.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    lo = min(predictions.min().min(), 0)
    hi = predictions.max().max()

    for ax, name in zip(axes, models):
        m = metrics_by_model[name]
        ax.scatter(predictions[name], predictions["actual"], s=14, alpha=0.55,
                   color=BLUE, edgecolor="none")
        ax.plot([lo, hi], [lo, hi], ls="--", color=RED, lw=1.2, label="perfect prediction")
        ax.set_title(f"{name}\nRMSE \\${m['rmse']:,.0f}   $R^2$ {m['r2']:.3f}", fontsize=10)
        ax.set_xlabel("Predicted price (USD)")
        ax.set_ylabel("Actual price (USD)")
        _usd_thousands(ax.xaxis)
        _usd_thousands(ax.yaxis)
        ax.legend(fontsize=8, loc="lower right")
    for ax in axes[len(models):]:
        ax.axis("off")

    fig.suptitle("Actual vs. predicted sale price (held-out test set)", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted_best(
    y_true: pd.Series, y_pred: np.ndarray, metrics: dict[str, float], model_name: str, path
) -> None:
    """Single detailed chart: points coloured by absolute residual."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    resid = np.abs(y_true - y_pred)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    sc = ax.scatter(y_pred, y_true, c=resid, cmap="viridis", s=26,
                    alpha=0.85, edgecolor="white", linewidth=0.4)
    lims = [min(y_true.min(), y_pred.min()) * 0.95, max(y_true.max(), y_pred.max()) * 1.03]
    ax.plot(lims, lims, ls="--", color=RED, lw=1.4, label="perfect prediction")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("|residual| (USD)")
    _usd_thousands(cb.ax.yaxis)
    ax.set_xlabel("Predicted price (USD)")
    ax.set_ylabel("Actual price (USD)")
    _usd_thousands(ax.xaxis)
    _usd_thousands(ax.yaxis)
    ax.set_title(
        f"{model_name}: actual vs. predicted (test set)\n"
        f"RMSE \\${metrics['rmse']:,.0f} | MAE \\${metrics['mae']:,.0f} | $R^2$ {metrics['r2']:.3f}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    y_true, y_pred, metrics: dict[str, float], model_name: str, path
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    resid = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].scatter(y_pred, resid, s=16, alpha=0.6, color=BLUE, edgecolor="none")
    axes[0].axhline(0, color=RED, ls="--", lw=1.2)
    span = np.abs(resid).max()
    axes[0].set_ylim(-span * 1.05, span * 1.05)
    axes[0].axhline(1.96 * resid.std(), color=GREY, ls=":", lw=1)
    axes[0].axhline(-1.96 * resid.std(), color=GREY, ls=":", lw=1)
    axes[0].set_xlabel("Predicted price (USD)")
    axes[0].set_ylabel("Residual = actual − predicted (USD)")
    _usd_thousands(axes[0].xaxis)
    _usd_thousands(axes[0].yaxis)
    axes[0].set_title(f"{model_name}: residuals vs. fitted values")

    axes[1].hist(resid, bins=40, color=GREEN, edgecolor="white")
    axes[1].axvline(0, color=RED, ls="--", lw=1.2)
    axes[1].axvline(resid.mean(), color="black", ls=":", lw=1.2,
                    label=f"mean {resid.mean():,.0f}")
    axes[1].set_xlabel("Residual (USD)")
    axes[1].set_ylabel("Houses")
    _usd_thousands(axes[1].xaxis)
    axes[1].set_title(
        f"Residual distribution (std \\${resid.std():,.0f}, "
        f"p90 |err| \\${metrics['p90_abs_error']:,.0f})"
    )
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def plot_feature_importance(series: pd.Series, title: str, xlabel: str, path) -> None:
    top = series.head(20).iloc[::-1]
    # Values are USD for permutation importance and ~0-1 for impurity scores.
    label = (lambda v: f"{v:,.0f}") if top.abs().max() >= 1 else (lambda v: f"{v:,.3f}")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(top.index, top.values, color=BLUE, edgecolor="white")
    for i, v in enumerate(top.values):
        ax.text(v, i, f"  {label(v)}", va="center", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, top.max() * 1.15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_coefficients(coefs: pd.Series, path) -> None:
    """Two-sided bar chart of the most influential standardised coefficients."""
    top_pos = coefs[coefs > 0].nlargest(15)
    top_neg = coefs[coefs < 0].nsmallest(15)
    combined = pd.concat([top_neg, top_pos]).iloc[::-1]

    colors = [GREEN if v > 0 else RED for v in combined.values]
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    ax.barh(combined.index, combined.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Standardised coefficient on log(price)")
    ax.set_title("Ridge: strongest positive / negative coefficients")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_price_segment(segment_df: pd.DataFrame, model_name: str, path) -> None:
    x = np.arange(len(segment_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, segment_df["rmse"], width, label="RMSE", color=BLUE)
    ax.bar(x + width / 2, segment_df["mae"], width, label="MAE", color=ORANGE)
    for i, (r, m) in enumerate(zip(segment_df["rmse"], segment_df["mae"])):
        ax.text(i - width / 2, r, f"\\${r/1000:.0f}k", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, m, f"\\${m/1000:.0f}k", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, segment_df["segment"], rotation=15, ha="right")
    ax.set_ylabel("Error (USD)")
    ax.set_title(f"{model_name}: error by actual price segment (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    model_name: str,
    path,
) -> None:
    # Scoring is `neg_root_mean_squared_error`, so flip the sign for display.
    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, (train_mean - train_std) / 1000,
                    (train_mean + train_std) / 1000, alpha=0.15, color=BLUE)
    ax.fill_between(train_sizes, (val_mean - val_std) / 1000,
                    (val_mean + val_std) / 1000, alpha=0.15, color=ORANGE)
    ax.plot(train_sizes, train_mean / 1000, "o-", color=BLUE, label="Training CV RMSE")
    ax.plot(train_sizes, val_mean / 1000, "o-", color=ORANGE, label="Validation CV RMSE")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("RMSE (thousands of USD)")
    ax.set_title(f"{model_name}: learning curve (5-fold CV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
