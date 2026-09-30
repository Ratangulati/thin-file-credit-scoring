"""The saved scoring object: preprocessing + classifier + probability calibration.

Kept separate from `train.py` so the API can unpickle it without importing any
training code.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import APPROVE_BELOW_PD, REJECT_ABOVE_PD


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class CreditScorer:
    name: str
    pipeline: Pipeline  # steps: "pre" (CreditPreprocessor) ... "clf"
    calibrator: LogisticRegression | None = None  # Platt scaling on the logit

    @property
    def feature_names(self) -> list[str]:
        return list(self.pipeline.named_steps["pre"].feature_names_out_)

    @property
    def input_columns(self) -> list[str]:
        return list(self.pipeline.named_steps["pre"].columns)

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated P(default); fine for ranking metrics (AUC etc.)."""
        return self.pipeline.predict_proba(df)[:, 1]

    def calibrate(self, df: pd.DataFrame, y) -> "CreditScorer":
        # Class weighting inflates raw probabilities; map them back to the
        # observed default rate using held-out data.
        self.calibrator = LogisticRegression().fit(_logit(self.raw_score(df)), y)
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.raw_score(df)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict_proba(_logit(raw))[:, 1]

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to the log-odds of default, one row per applicant.

        LightGBM: exact TreeSHAP values. Logistic regression: coefficient x
        standardised value (contribution relative to the average applicant).
        """
        steps = self.pipeline.named_steps
        X = self.pipeline[:-1].transform(df)  # everything before the classifier
        clf = steps["clf"]
        if isinstance(clf, LGBMClassifier):
            contrib = clf.predict(X, pred_contrib=True)[:, :-1]  # drop bias column
        else:
            contrib = np.asarray(X) * clf.coef_[0]
        return pd.DataFrame(contrib, columns=self.feature_names, index=df.index)


def credit_score(pd_: float | np.ndarray, base: float = 600, base_odds: float = 50, pdo: float = 40):
    """Map probability of default to a 300-850 style score.

    Standard "points to double the odds" scaling: a score of `base` means
    good:bad odds of `base_odds`:1 (PD ~2%), and every `pdo` points doubles
    the odds. Clipped to [300, 850].
    """
    p = np.clip(pd_, 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    score = base + factor * (np.log((1 - p) / p) - np.log(base_odds))
    return np.clip(np.round(score), 300, 850).astype(int)


def decision(pd_: float) -> str:
    if pd_ < APPROVE_BELOW_PD:
        return "approve"
    if pd_ > REJECT_ABOVE_PD:
        return "reject"
    return "review"
