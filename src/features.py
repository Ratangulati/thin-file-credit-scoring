"""Feature pipeline: imputation learned on training data + segment helpers."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import BEHAVIORAL_FEATURES, THIN_FILE_MAX_OPEN_LINES, TRADITIONAL_FEATURES
from src.data import cap_outliers

# Columns that are missing in the raw data. Missingness itself is informative
# (e.g. applicants who don't report income), so we add a 0/1 flag for each.
IMPUTE_COLS = ["monthly_income", "dependents"]


def feature_columns(use_behavioral: bool) -> list[str]:
    return TRADITIONAL_FEATURES + (BEHAVIORAL_FEATURES if use_behavioral else [])


class CreditPreprocessor(BaseEstimator, TransformerMixin):
    """Select model columns, re-apply caps, median-impute, add missing flags.

    Medians are learned in `fit` (training split only) and reused at scoring
    time, so the API handles e.g. a null `monthly_income` exactly like training.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        X = cap_outliers(X[self.columns])
        self.medians_ = {c: float(X[c].median()) for c in IMPUTE_COLS if c in self.columns}
        self.feature_names_out_ = list(self.columns) + [f"{c}_missing" for c in self.medians_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = cap_outliers(X[self.columns]).astype(float)
        for c, med in self.medians_.items():
            X[f"{c}_missing"] = X[c].isna().astype(float)
            X[c] = X[c].fillna(med)
        return X[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


def thin_file_mask(df: pd.DataFrame, max_open_lines: int = THIN_FILE_MAX_OPEN_LINES) -> pd.Series:
    """True for applicants with <= `max_open_lines` open credit lines/loans."""
    return df["open_credit_lines"] <= max_open_lines
