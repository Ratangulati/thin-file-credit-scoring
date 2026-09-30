import numpy as np

from src.config import APPROVE_BELOW_PD, REJECT_ABOVE_PD
from src.model import credit_score, decision


def test_credit_score_is_monotonic_and_bounded():
    pds = np.linspace(0, 1, 101)
    scores = credit_score(pds)
    assert scores.min() == 300 and scores.max() == 850
    assert np.all(np.diff(scores) <= 0)  # higher PD -> lower score


def test_credit_score_anchor():
    assert credit_score(1 / 16) == 660  # odds 15:1


def test_decision_bands():
    assert decision(APPROVE_BELOW_PD / 2) == "approve"
    assert decision((APPROVE_BELOW_PD + REJECT_ABOVE_PD) / 2) == "review"
    assert decision(REJECT_ABOVE_PD + 0.1) == "reject"
