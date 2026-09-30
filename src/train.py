"""Train baseline vs enhanced models (logistic regression + LightGBM).

    python -m src.train

Outputs (in models/):
    models.joblib     dict of the four CreditScorer objects
    test_set.joblib   held-out test rows (used by src.evaluate)
    scorer.joblib     the model served by the API (enhanced LightGBM)
    metadata.json     seeds, sizes, thresholds
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config
from src.config import SEED, TARGET
from src.data import load_clean
from src.features import CreditPreprocessor, feature_columns, thin_file_mask
from src.model import CreditScorer
from src.simulate import simulate_behavioral

FEATURE_SETS = {"baseline": False, "enhanced": True}  # name -> use_behavioral
SERVED_MODEL = "enhanced_lgbm"


def make_logreg(columns: list[str]) -> Pipeline:
    # All inputs are non-negative and most are right-skewed (income, counts),
    # so log1p + standardise keeps the linear model well-behaved and its
    # coefficients comparable.
    return Pipeline(
        [
            ("pre", CreditPreprocessor(columns)),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )


def make_lgbm(columns: list[str], pos_weight: float) -> Pipeline:
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,  # class imbalance (~1 default : 14 non-default)
        random_state=SEED,
        verbose=-1,
    )
    return Pipeline([("pre", CreditPreprocessor(columns)), ("clf", clf)])


def split(df: pd.DataFrame):
    """Stratified train / calibration / test split.

    Stratifying on target x thin-file keeps both the default rate and the
    (small) thin-file segment proportionally represented in every split.
    """
    strata = df[TARGET].astype(str) + "_" + thin_file_mask(df).astype(str)
    train, test = train_test_split(df, test_size=config.TEST_SIZE, stratify=strata, random_state=SEED)
    fit, cal = train_test_split(
        train, test_size=config.CALIBRATION_SIZE, stratify=strata.loc[train.index], random_state=SEED
    )
    return fit, cal, test


def train_all(df: pd.DataFrame) -> tuple[dict[str, CreditScorer], pd.DataFrame]:
    fit, cal, test = split(df)
    pos_weight = (fit[TARGET] == 0).sum() / (fit[TARGET] == 1).sum()

    scorers = {}
    for fs_name, use_behavioral in FEATURE_SETS.items():
        cols = feature_columns(use_behavioral)
        for algo, pipe in (("logreg", make_logreg(cols)), ("lgbm", make_lgbm(cols, pos_weight))):
            name = f"{fs_name}_{algo}"
            pipe.fit(fit, fit[TARGET])
            scorers[name] = CreditScorer(name, pipe).calibrate(cal, cal[TARGET])
            auc = roc_auc_score(test[TARGET], scorers[name].raw_score(test))
            print(f"  {name:<18} test ROC-AUC {auc:.4f}")
    return scorers, test


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading + cleaning data ...")
    df = simulate_behavioral(load_clean(), seed=SEED)
    print(f"  {len(df):,} rows | default rate {df[TARGET].mean():.3%} | "
          f"thin-file (<= {config.THIN_FILE_MAX_OPEN_LINES} lines) {thin_file_mask(df).mean():.2%}")

    print("Training ...")
    scorers, test = train_all(df)

    joblib.dump(scorers, config.MODELS_DIR / "models.joblib")
    joblib.dump(test, config.MODELS_DIR / "test_set.joblib")
    joblib.dump(scorers[SERVED_MODEL], config.SCORER_PATH)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "rows": len(df),
        "test_rows": len(test),
        "thin_file_max_open_lines": config.THIN_FILE_MAX_OPEN_LINES,
        "served_model": SERVED_MODEL,
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models to {config.MODELS_DIR}/")


if __name__ == "__main__":
    main()
