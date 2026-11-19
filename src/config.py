"""Central configuration for the house-price regression project.

Everything that a reader might want to change (paths, seeds, split ratios,
documented missing-value semantics) lives here so that the modelling code
stays free of magic numbers.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ames_housing.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Fitted winning pipeline (preprocessing + model + feature order) written by
# `python -m src.train` and consumed by `python -m src.predict`.
MODEL_PATH = REPORTS_DIR / "best_model.joblib"

# --------------------------------------------------------------------------- #
# Reproducibility / validation protocol
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80/20 train/test holdout
CV_FOLDS = 5              # shuffled K-fold on the *training* split only
N_PERM_REPEATS = 20       # repeats for permutation importance

TARGET_COL = "SalePrice"
ID_COLS = ("Id",)

# Holdout stratification bins: price deciles keep the small holdout's price
# spread representative of the population (quartile bins left the holdout with
# a 21% wider spread because 5 of the 10 most expensive houses landed in it).
STRATA_BINS = 10

# Encoded features that are non-zero in fewer than ~1% of training rows are
# dropped inside the pipeline.  Rare indicators (PoolArea, PoolQC='Ex', ...)
# are the source of catastrophic extrapolation in unregularised regression:
# they received coefficients of +16 to +42 log-price units from ~9 examples.
VARIANCE_THRESHOLD = 0.01

# --------------------------------------------------------------------------- #
# Documented missing-value semantics (Ames Housing data dictionary)
#
# These are *row-wise* rules taken from the official data description: they
# never look at other rows or at the target, so applying them before the
# train/test split cannot leak information.
# --------------------------------------------------------------------------- #
# "NA" in these columns means "this feature is absent on the house".
NA_IS_NONE = (
    "Alley",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "BsmtQual",
    "Fence",
    "FireplaceQu",
    "GarageCond",
    "GarageFinish",
    "GarageQual",
    "GarageType",
    "MiscFeature",
    "PoolQC",
    "MasVnrType",  # stores the literal string "None" for "no masonry veneer"
)

# Numeric column whose NaN means "no veneer" -> 0 square feet.
NUMERIC_NA_TO_ZERO = ("MasVnrArea",)

# Columns that still contain genuine missingness after the rules above and are
# therefore imputed *inside* each CV fold by the scikit-learn pipeline.
# `LotFrontage` also gets an explicit missingness indicator (17.7% missing).
MISSING_INDICATORS = ("LotFrontage",)

# Sale-context features: known at the moment of sale, but not before it.
# Kept by default (the prediction task is "price at sale"); ablated in the
# leakage-sensitivity check in `src/train.py`.
SALE_CONTEXT = ("MoSold", "YrSold", "SaleType", "SaleCondition")

# `MSSubClass` is an integer *code* for the dwelling type, not a quantity.
CATEGORICAL_AS_CODE = ("MSSubClass",)

# Rare-category threshold for one-hot encoding (fewer than 5 training rows).
ONEHOT_MIN_FREQUENCY = 5

# --------------------------------------------------------------------------- #
# Out-of-time (temporal) evaluation
#
# The headline protocol randomises rows across 2006-2010.  A deployment that
# predicts *future* sales never sees the future, so we additionally train on
# sales through `TEMPORAL_TRAIN_MAX_YEAR` and test on every later sale.  This
# is reported as a stricter estimate, not as the headline number.
# --------------------------------------------------------------------------- #
TEMPORAL_YEAR_COL = "YrSold"
TEMPORAL_TRAIN_MAX_YEAR = 2007
