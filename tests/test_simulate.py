import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import BEHAVIORAL_FEATURES, TARGET
from src.data import clean, load_raw
from src.simulate import simulate_behavioral


@pytest.fixture
def clean_df(fake_raw_csv):
    return clean(load_raw(fake_raw_csv))


def test_same_seed_is_reproducible(clean_df):
    a = simulate_behavioral(clean_df, seed=7)
    b = simulate_behavioral(clean_df, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs(clean_df):
    a = simulate_behavioral(clean_df, seed=1)[BEHAVIORAL_FEATURES]
    b = simulate_behavioral(clean_df, seed=2)[BEHAVIORAL_FEATURES]
    assert not a.equals(b)


def test_input_not_mutated_and_columns_added(clean_df):
    before = clean_df.copy()
    out = simulate_behavioral(clean_df)
    pd.testing.assert_frame_equal(clean_df, before)
    assert set(BEHAVIORAL_FEATURES) <= set(out.columns)
    assert not out[BEHAVIORAL_FEATURES].isna().any().any()


def test_value_ranges(clean_df):
    out = simulate_behavioral(clean_df)
    assert out["on_time_payment_streak"].between(0, 36).all()
    assert out["active_driving_days_per_month"].between(0, 30).all()
    assert out["late_night_driving_ratio"].between(0, 1).all()
    assert (out["avg_daily_km_driven"] >= 0).all()
    assert (out["device_tamper_alerts"] >= 0).all()


def test_signal_is_mild_not_a_giveaway(clean_df):
    """Each feature should help a little, in the documented direction, but no
    single feature should come close to separating the classes."""
    out = simulate_behavioral(clean_df)
    y = out[TARGET]
    expected_sign = {  # +1: higher value => riskier
        "on_time_payment_streak": -1,
        "active_driving_days_per_month": -1,
        "late_night_driving_ratio": +1,
        "device_tamper_alerts": +1,
    }
    for col, sign in expected_sign.items():
        auc = roc_auc_score(y, sign * out[col])
        assert 0.52 < auc < 0.80, (col, auc)


def test_zero_signal_removes_label_dependence(clean_df):
    out = simulate_behavioral(clean_df, signal=0.0, credit_coupling=0.0)
    auc = roc_auc_score(out[TARGET], -out["on_time_payment_streak"])
    assert abs(auc - 0.5) < 0.07
