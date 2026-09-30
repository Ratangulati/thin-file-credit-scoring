"""SIMULATED behavioral / mobility features, standing in for MCCS device data.

!! These features are SYNTHETIC. They are generated from the label with noise,
!! so any lift they give demonstrates the *approach* (how such data would be
!! wired into scoring and evaluated on thin-file applicants), not real-world
!! predictive performance.

Generative story
----------------
Every applicant has an unobserved "financial stress" latent `z`:

    z = SIGNAL * defaulted  +  CREDIT_COUPLING * credit_stress  +  N(0, 1)

* `SIGNAL` (default 0.8) is the mean shift between defaulters and
  non-defaulters, in units of the noise SD. On its own, a latent with this
  shift ranks defaulters with AUC ~ Phi(0.8 / sqrt(2)) ~ 0.71. Each feature
  below adds its own noise, so individually they are weaker
  (single-feature AUC ~0.55-0.71 on the real data).
* `CREDIT_COUPLING` (default 0.3) ties behavior weakly to existing credit
  history (past-due counts, utilisation), so the new features are partly
  redundant with bureau data — as real telematics would be.
* The N(0,1) term is the part of behavior that has nothing to do with credit.

Each observed feature is a noisy, plausibly-shaped function of `z`:

  on_time_payment_streak        consecutive months paid on time (0-36).
                                Stressed borrowers break streaks sooner.
  active_driving_days_per_month days the vehicle was used (0-30). Many GMS
                                customers drive for income (ride-hail,
                                delivery), so consistent use ~ steady earnings.
  avg_daily_km_driven           km/day on active days. Deliberately the
                                weakest signal: heavy usage is only slightly
                                riskier (wear, fatigue), and noise dominates.
  late_night_driving_ratio      share of km driven 00:00-05:00 (0-1).
                                Mildly higher under stress (second jobs,
                                irregular hours) — a proxy, not a judgement.
  device_tamper_alerts          count of tamper/disconnect events. Rare
                                (mostly 0), much more likely when stressed.

Everything is drawn from a single `np.random.default_rng(seed)`, so the same
input and seed always reproduce the same features.
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

    # Poisson around a mean that shrinks as stress rises (~14 months at z=0).
    streak_mean = 14.0 * np.exp(-0.35 * z + rng.normal(0, 0.3, n))
    out["on_time_payment_streak"] = np.clip(rng.poisson(streak_mean), 0, 36)

    # Binomial over 30 days; p ~ 0.83 at z=0, falling with stress.
    p_active = _sigmoid(1.6 - 0.45 * z + rng.normal(0, 0.6, n))
    out["active_driving_days_per_month"] = rng.binomial(30, p_active)

    # Log-normal around ~38 km/day; very weak dependence on z.
    km = np.exp(np.log(38) + 0.10 * z + rng.normal(0, 0.45, n))
    out["avg_daily_km_driven"] = np.clip(km, 0, 400).round(1)

    # Beta with mean ~0.10 at z=0, concentration 12.
    mu = _sigmoid(-2.2 + 0.35 * z + rng.normal(0, 0.3, n))
    out["late_night_driving_ratio"] = rng.beta(mu * 12, (1 - mu) * 12).round(3)

    # Rare events: rate ~0.08/applicant at z=0, rising steeply with stress.
    rate = 0.08 * np.exp(0.6 * z)
    out["device_tamper_alerts"] = np.clip(rng.poisson(rate), 0, 10)

    return out
