"""Simulated behavioral features standing in for MCCS device data.

These features are synthetic. They are generated from the default label plus
noise, so they show how device data would be used, not how well it would work.

Assumptions:
- Each applicant has a hidden stress value
  z = SIGNAL * defaulted + CREDIT_COUPLING * credit_stress + N(0, 1).
- SIGNAL (0.8) sets how much defaulters differ. CREDIT_COUPLING (0.3) links
  behavior weakly to credit history, so the features overlap with bureau data.
- on_time_payment_streak: shorter under stress.
- active_driving_days_per_month: lower under stress (many drivers earn with the car).
- avg_daily_km_driven: slightly higher under stress; the weakest signal.
- late_night_driving_ratio: slightly higher under stress.
- device_tamper_alerts: rare, more likely under stress.
- The same seed always gives the same features.
"""
import numpy as np
import pandas as pd

from src.config import SEED, TARGET

SIGNAL = 0.8
CREDIT_COUPLING = 0.3


def _credit_stress(df: pd.DataFrame) -> np.ndarray:
    """Standardised proxy of bureau-visible stress (late payments + utilisation)."""
    late = df[["past_due_30_59", "past_due_60_89", "past_due_90_plus"]].sum(axis=1)
    raw = np.log1p(late.to_numpy(float)) + df["revolving_utilization"].to_numpy(float)
    return (raw - raw.mean()) / (raw.std() + 1e-9)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_behavioral(
    df: pd.DataFrame,
    seed: int = SEED,
    signal: float = SIGNAL,
    credit_coupling: float = CREDIT_COUPLING,
) -> pd.DataFrame:
    """Return a copy of `df` with the five simulated behavioral columns added."""
    rng = np.random.default_rng(seed)
    n = len(df)
    y = df[TARGET].to_numpy(float)

    z = signal * y + credit_coupling * _credit_stress(df) + rng.normal(0, 1, n)

    out = df.copy()

    streak_mean = 14.0 * np.exp(-0.35 * z + rng.normal(0, 0.3, n))
    out["on_time_payment_streak"] = np.clip(rng.poisson(streak_mean), 0, 36)

    p_active = _sigmoid(1.6 - 0.45 * z + rng.normal(0, 0.6, n))
    out["active_driving_days_per_month"] = rng.binomial(30, p_active)

    km = np.exp(np.log(38) + 0.10 * z + rng.normal(0, 0.45, n))
    out["avg_daily_km_driven"] = np.clip(km, 0, 400).round(1)

    mu = _sigmoid(-2.2 + 0.35 * z + rng.normal(0, 0.3, n))
    out["late_night_driving_ratio"] = rng.beta(mu * 12, (1 - mu) * 12).round(3)

    rate = 0.08 * np.exp(0.6 * z)
    out["device_tamper_alerts"] = np.clip(rng.poisson(rate), 0, 10)

    return out
