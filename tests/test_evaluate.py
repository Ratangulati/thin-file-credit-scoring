import numpy as np

from src.evaluate import bad_rate_at_approval_rate, recall_at_approval_rate


def test_recall_at_approval_rate_perfect_and_worst_ranking():
    y = np.array([0] * 8 + [1] * 2)
    perfect = np.arange(10)
    assert recall_at_approval_rate(y, perfect, approval_rate=0.8) == 1.0
    assert recall_at_approval_rate(y, -perfect, approval_rate=0.8) == 0.0


def test_bad_rate_among_approved():
    y = np.array([0] * 8 + [1] * 2)
    assert bad_rate_at_approval_rate(y, np.arange(10), approval_rate=0.8) == 0.0
    assert bad_rate_at_approval_rate(y, np.arange(10), approval_rate=1.0) == 0.2
