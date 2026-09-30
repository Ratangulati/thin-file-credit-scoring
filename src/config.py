"""Paths, seed, feature lists and policy settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "cs-training.csv"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"
SCORER_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "scorer.joblib"))

SEED = 42
TEST_SIZE = 0.25
CALIBRATION_SIZE = 0.2

THIN_FILE_MAX_OPEN_LINES = int(os.getenv("THIN_FILE_MAX_OPEN_LINES", 2))

APPROVAL_RATE = float(os.getenv("APPROVAL_RATE", 0.80))

APPROVE_BELOW_PD = 0.08
REJECT_ABOVE_PD = 0.25

TARGET = "serious_dlq_2yrs"

RENAME = {
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "past_due_90_plus",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}

TRADITIONAL_FEATURES = [
    "revolving_utilization",
    "age",
    "past_due_30_59",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "past_due_90_plus",
    "real_estate_loans",
    "past_due_60_89",
    "dependents",
]

BEHAVIORAL_FEATURES = [
    "on_time_payment_streak",
    "avg_daily_km_driven",
    "active_driving_days_per_month",
    "late_night_driving_ratio",
    "device_tamper_alerts",
]
