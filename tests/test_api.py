import joblib
import pytest
from fastapi.testclient import TestClient

from src import api, config
from src.data import clean, load_raw
from src.simulate import simulate_behavioral
from src.train import SERVED_MODEL, train_all


@pytest.fixture(scope="module")
def scorer_path(tmp_path_factory):
    """Train the real pipeline on fake data once, save it like `make train` does."""
    tmp = tmp_path_factory.mktemp("models")
    from tests.conftest import make_fake_raw

    make_fake_raw(4000, seed=1).to_csv(tmp / "raw.csv", index=False)
    df = simulate_behavioral(clean(load_raw(tmp / "raw.csv")))
    scorers, _ = train_all(df)
    path = tmp / "scorer.joblib"
    joblib.dump(scorers[SERVED_MODEL], path)
    return path


@pytest.fixture
def client(scorer_path, monkeypatch):
    monkeypatch.setattr(config, "SCORER_PATH", scorer_path)
    with TestClient(api.app) as c:  # context manager runs the lifespan (model load)
        yield c


@pytest.fixture
def applicant():
    return dict(api.EXAMPLE_APPLICANT)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model": SERVED_MODEL}


def test_score_returns_valid_response(client, applicant):
    r = client.post("/score", json=applicant)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert 300 <= body["credit_score"] <= 850
    assert body["decision"] in {"approve", "review", "reject"}
    assert body["thin_file"] is True  # example has 1 open credit line
    assert len(body["top_factors"]) == 3
    impacts = [abs(f["impact"]) for f in body["top_factors"]]
    assert impacts == sorted(impacts, reverse=True)


def test_riskier_applicant_scores_lower(client, applicant):
    risky = applicant | {
        "past_due_30_59": 6, "past_due_90_plus": 3, "revolving_utilization": 1.5,
        "on_time_payment_streak": 0, "device_tamper_alerts": 4, "active_driving_days_per_month": 5,
    }
    good = client.post("/score", json=applicant).json()
    bad = client.post("/score", json=risky).json()
    assert bad["probability_of_default"] > good["probability_of_default"]
    assert bad["credit_score"] < good["credit_score"]


def test_missing_income_is_accepted(client, applicant):
    applicant["monthly_income"] = None
    r = client.post("/score", json=applicant)
    assert r.status_code == 200
    factors = {f["feature"]: f for f in r.json()["top_factors"]}
    if "monthly_income" in factors:
        assert factors["monthly_income"]["value"] is None


@pytest.mark.parametrize(
    "patch",
    [
        {"age": 12},
        {"late_night_driving_ratio": 1.5},
        {"past_due_30_59": -1},
        {"age": "thirty"},
        {"unexpected_field": 1},
    ],
)
def test_invalid_input_rejected(client, applicant, patch):
    assert client.post("/score", json=applicant | patch).status_code == 422


def test_missing_required_field_rejected(client, applicant):
    applicant.pop("on_time_payment_streak")
    assert client.post("/score", json=applicant).status_code == 422


def test_score_503_when_model_missing(monkeypatch, tmp_path, applicant):
    monkeypatch.setattr(config, "SCORER_PATH", tmp_path / "nope.joblib")
    with TestClient(api.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/score", json=applicant).status_code == 503
