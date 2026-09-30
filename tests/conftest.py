"""Shared fixtures. Tests use a small fake dataset with the Kaggle schema so they
run without downloading the real file."""
import numpy as np
import pandas as pd
import pytest


def make_fake_raw(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Fake rows in the raw Kaggle schema (incl. NaNs, sentinels, outliers)."""
    rng = np.random.default_rng(seed)
    late = rng.poisson(0.3, n)
    y = (rng.random(n) < 0.04 + 0.15 * (late > 0)).astype(int)
    df = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(1, n + 1),
            "SeriousDlqin2yrs": y,
            "RevolvingUtilizationOfUnsecuredLines": rng.beta(1, 2, n),
            "age": rng.integers(21, 90, n),
            "NumberOfTime30-59DaysPastDueNotWorse": late,
            "DebtRatio": rng.gamma(1.0, 0.4, n),
            "MonthlyIncome": rng.lognormal(8.5, 0.6, n).round(),
            "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n),
            "NumberOfTimes90DaysLate": rng.poisson(0.1, n),
            "NumberRealEstateLoansOrLines": rng.poisson(1, n),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.1, n),
            "NumberOfDependents": rng.poisson(0.8, n).astype(float),
        }
    )
    df.loc[rng.random(n) < 0.2, "MonthlyIncome"] = np.nan
    df.loc[rng.random(n) < 0.03, "NumberOfDependents"] = np.nan
    df.loc[:4, "NumberOfTime30-59DaysPastDueNotWorse"] = 98  # sentinel code
    df.loc[5, "RevolvingUtilizationOfUnsecuredLines"] = 50_000.0  # outlier
    df.loc[6, "age"] = 0  # invalid
    df.loc[7:400, "NumberOfOpenCreditLinesAndLoans"] = rng.integers(0, 3, 394)  # thin-file rows
    return df


@pytest.fixture
def fake_raw_csv(tmp_path):
    path = tmp_path / "cs-training.csv"
    make_fake_raw().to_csv(path, index=False)
    return path
