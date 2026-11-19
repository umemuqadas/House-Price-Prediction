"""Download the Ames Housing dataset used by this project.

Source: Kaggle "House Prices - Advanced Regression Techniques" training set,
retrieved through scikit-learn's OpenML mirror (dataset id 42193, name
`house_prices`).  The CSV is written to `data/raw/ames_housing.csv` with the
target column `SalePrice` appended as the last column.

Usage (from the project root):

    python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "raw" / "ames_housing.csv"


def main() -> int:
    if OUTPUT.exists():
        print(f"Already present, skipping download: {OUTPUT}")
        return 0

    print("Fetching `house_prices` from OpenML ...")
    bunch = fetch_openml("house_prices", version=1, as_frame=True, parser="auto")

    frame = bunch.data.copy()
    frame["SalePrice"] = bunch.target.astype("float64")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {frame.shape[0]} rows x {frame.shape[1]} columns -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
