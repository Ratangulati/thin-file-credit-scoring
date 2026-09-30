import numpy as np
import pandas as pd

from src.config import TARGET
from src.data import CAPS, clean, load_raw
from src.features import CreditPreprocessor, feature_columns, thin_file_mask


def test_load_raw_drops_index_and_renames(fake_raw_csv):
    df = load_raw(fake_raw_csv)
    assert not any(c.startswith("Unnamed") for c in df.columns)
    assert TARGET in df.columns and "open_credit_lines" in df.columns


def test_clean_caps_sentinels_and_invalid_ages(fake_raw_csv):
    df = clean(load_raw(fake_raw_csv))
    assert df["age"].min() >= 18
    assert df["past_due_30_59"].max() <= CAPS["past_due_30_59"][1]
    assert df["revolving_utilization"].max() <= CAPS["revolving_utilization"][1]
    # sentinel 98 is mapped to the cap, not dropped
    assert (df["past_due_30_59"] == CAPS["past_due_30_59"][1]).sum() >= 4
    # clean() must NOT impute — that is learned on the train split
    assert df["monthly_income"].isna().any()


def test_preprocessor_imputes_with_train_medians_and_flags(fake_raw_csv):
    df = clean(load_raw(fake_raw_csv))
    cols = feature_columns(use_behavioral=False)
    pre = CreditPreprocessor(cols).fit(df)
    out = pre.transform(df)
    assert not out.isna().any().any()
    assert {"monthly_income_missing", "dependents_missing"} <= set(out.columns)
    assert out["monthly_income_missing"].sum() == df["monthly_income"].isna().sum()

    # New data with a missing income gets the *training* median
    new = df.head(1).copy()
    new["monthly_income"] = np.nan
    assert pre.transform(new)["monthly_income"].iloc[0] == pre.medians_["monthly_income"]


def test_preprocessor_caps_unseen_outliers(fake_raw_csv):
    df = clean(load_raw(fake_raw_csv))
    pre = CreditPreprocessor(feature_columns(False)).fit(df)
    row = df.head(1).copy()
    row["debt_ratio"] = 1e6
    assert pre.transform(row)["debt_ratio"].iloc[0] == CAPS["debt_ratio"][1]


def test_thin_file_mask_threshold_is_configurable():
    df = pd.DataFrame({"open_credit_lines": [0, 1, 2, 3, 10]})
    assert thin_file_mask(df, 2).tolist() == [True, True, True, False, False]
    assert thin_file_mask(df, 0).tolist() == [True, False, False, False, False]
