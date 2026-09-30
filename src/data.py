"""Load and clean the Kaggle "Give Me Some Credit" training file.

Cleaning here is deliberately *stateless* (fixed rules, no statistics learned
from the data) so it can run before the train/test split without leakage.
Median imputation of missing values is learned on the training split only —
see `features.CreditPreprocessor`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATH, RENAME, TARGET

# Past-due counts of 96 / 98 are known sentinel codes in this dataset (not real
# counts). They co-occur with very high default rates, so rather than treating
# them as missing we map them to the cap — "a lot of late payments".
PAST_DUE_COLS = ["past_due_30_59", "past_due_60_89", "past_due_90_plus"]
SENTINEL_CODES = (96, 98)

# Domain caps (lower, upper). Values beyond these are data errors or so extreme
# that they only add noise (e.g. revolving utilisation of 50,000x the limit).
CAPS = {
    "revolving_utilization": (0.0, 2.0),
    "age": (18, 100),
    "debt_ratio": (0.0, 5.0),
    "monthly_income": (0.0, 50_000.0),
    "open_credit_lines": (0, 40),
    "real_estate_loans": (0, 15),
    "dependents": (0, 10),
    "past_due_30_59": (0, 20),
    "past_due_60_89": (0, 20),
    "past_due_90_plus": (0, 20),
}


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV, drop the unnamed index column and rename to snake_case."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download cs-training.csv from "
            "https://www.kaggle.com/c/GiveMeSomeCredit/data and place it in data/."
        )
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return df.rename(columns=RENAME)


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sentinel handling and fixed caps. NaNs are left untouched."""
    df = df.copy()
    for col in PAST_DUE_COLS:
        if col in df:
            df.loc[df[col].isin(SENTINEL_CODES), col] = CAPS[col][1]
    for col, (lo, hi) in CAPS.items():
        if col in df:
            df[col] = df[col].clip(lo, hi)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning for the training data."""
    # A handful of rows have age 0 (or < 18): not valid loan applicants.
    df = df[df["age"] >= 18].copy()
    df = cap_outliers(df)
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def load_clean(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


if __name__ == "__main__":
    d = load_clean()
    print(d.shape)
    print(d.isna().mean().round(3))
    print(d.describe().T[["min", "50%", "max"]])
    print("default rate:", round(d[TARGET].mean(), 4), "| any NaN left:", bool(np.any(d.isna())))
