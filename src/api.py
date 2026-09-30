"""Scoring API. Run: uvicorn src.api:app --port 8000"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.model import credit_score, decision

READABLE = {
    "revolving_utilization": "Revolving credit utilisation",
    "age": "Age",
    "past_due_30_59": "Times 30-59 days past due",
    "debt_ratio": "Debt-to-income ratio",
    "monthly_income": "Monthly income",
    "open_credit_lines": "Open credit lines / loans",
    "past_due_90_plus": "Times 90+ days late",
    "real_estate_loans": "Real-estate loans",
    "past_due_60_89": "Times 60-89 days past due",
    "dependents": "Number of dependents",
    "monthly_income_missing": "Monthly income not reported",
    "dependents_missing": "Dependents not reported",
    "on_time_payment_streak": "On-time payment streak (months)",
    "avg_daily_km_driven": "Average daily km driven",
    "active_driving_days_per_month": "Active driving days per month",
    "late_night_driving_ratio": "Share of late-night driving",
    "device_tamper_alerts": "Device tamper alerts",
}


EXAMPLE_APPLICANT = {
    "revolving_utilization": 0.35,
    "age": 29,
    "past_due_30_59": 0,
    "debt_ratio": 0.4,
    "monthly_income": 3200,
    "open_credit_lines": 1,
    "past_due_90_plus": 0,
    "real_estate_loans": 0,
    "past_due_60_89": 0,
    "dependents": 1,
    "on_time_payment_streak": 18,
    "avg_daily_km_driven": 42.0,
    "active_driving_days_per_month": 26,
    "late_night_driving_ratio": 0.05,
    "device_tamper_alerts": 0,
}


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_APPLICANT]})

    revolving_utilization: float = Field(ge=0, description="Balance / limit on unsecured revolving lines")
    age: int = Field(ge=18, le=120)
    past_due_30_59: int = Field(ge=0, description="Times 30-59 days past due in last 2 years")
    debt_ratio: float = Field(ge=0, description="Monthly debt payments / monthly income")
    monthly_income: float | None = Field(default=None, ge=0, description="Null if not reported")
    open_credit_lines: int = Field(ge=0)
    past_due_90_plus: int = Field(ge=0)
    real_estate_loans: int = Field(ge=0)
    past_due_60_89: int = Field(ge=0)
    dependents: int | None = Field(default=None, ge=0)

    on_time_payment_streak: int = Field(ge=0, le=600, description="Consecutive on-time months")
    avg_daily_km_driven: float = Field(ge=0, le=2000)
    active_driving_days_per_month: int = Field(ge=0, le=31)
    late_night_driving_ratio: float = Field(ge=0, le=1)
    device_tamper_alerts: int = Field(ge=0)


class Factor(BaseModel):
    feature: str
    description: str
    value: float | None
    effect: Literal["increases risk", "decreases risk"]
    impact: float = Field(description="Contribution to log-odds of default (SHAP)")


class ScoreResponse(BaseModel):
    probability_of_default: float
    credit_score: int = Field(ge=300, le=850)
    decision: Literal["approve", "review", "reject"]
    thin_file: bool
    top_factors: list[Factor]
    model: str


def load_scorer(path: Path):
    return joblib.load(path) if path.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = load_scorer(config.SCORER_PATH)
    yield


app = FastAPI(
    title="Thin-file credit scoring",
    description="Scores applicants from traditional credit + (simulated) vehicle-telematics features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    scorer = app.state.scorer
    return {"status": "ok", "model_loaded": scorer is not None,
            "model": scorer.name if scorer else None}


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant):
    scorer = app.state.scorer
    if scorer is None:
        raise HTTPException(503, "Model not loaded. Run `make train` first.")

    row = pd.DataFrame([applicant.model_dump()])
    row = row.astype({c: float for c in row.columns})
    pd_ = float(scorer.predict_pd(row)[0])

    contrib = scorer.contributions(row).iloc[0]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:3]
    features = scorer.pipeline.named_steps["pre"].transform(row).iloc[0]
    factors = [
        Factor(
            feature=f,
            description=READABLE.get(f, f),
            value=None if f in row and pd.isna(row.at[0, f]) else float(features[f]),
            effect="increases risk" if v > 0 else "decreases risk",
            impact=round(float(v), 4),
        )
        for f, v in top.items()
    ]
    return ScoreResponse(
        probability_of_default=round(pd_, 4),
        credit_score=int(credit_score(pd_)),
        decision=decision(pd_),
        thin_file=applicant.open_credit_lines <= config.THIN_FILE_MAX_OPEN_LINES,
        top_factors=factors,
        model=scorer.name,
    )
