"""Tests for the epsilon-greedy exploration service."""
import math

import pytest

from data5580_hw.services.exploration_service import (
    ExplorationService,
    exploration_service,
    _BASE_EPSILON,
    _DECAY_RATE,
    _MIN_EPSILON,
)


# ---------------------------------------------------------------------------
# compute_epsilon
# ---------------------------------------------------------------------------

def test_epsilon_at_zero_feedback_is_base():
    service = ExplorationService()
    eps = service.compute_epsilon(0)
    assert eps == pytest.approx(_BASE_EPSILON)


def test_epsilon_decreases_with_feedback():
    service = ExplorationService()
    eps_0 = service.compute_epsilon(0)
    eps_10 = service.compute_epsilon(10)
    eps_50 = service.compute_epsilon(50)
    assert eps_0 > eps_10 > eps_50


def test_epsilon_never_below_min():
    service = ExplorationService()
    eps = service.compute_epsilon(10_000)
    assert eps == pytest.approx(_MIN_EPSILON)


def test_epsilon_formula():
    service = ExplorationService()
    n = 20
    expected = max(_MIN_EPSILON, _BASE_EPSILON * math.exp(-_DECAY_RATE * n))
    assert service.compute_epsilon(n) == pytest.approx(expected)


def test_epsilon_at_large_n_equals_min():
    service = ExplorationService()
    assert service.compute_epsilon(1000) == pytest.approx(_MIN_EPSILON)


# ---------------------------------------------------------------------------
# adjust_score: exploration branch (random seed control)
# ---------------------------------------------------------------------------

def test_adjust_score_explore_branch(monkeypatch):
    """Force the random roll to be below epsilon → should return a random value, not raw_score."""
    service = ExplorationService()

    # With zero feedback, epsilon ≈ 0.8; monkeypatch random.random to always return 0.0
    import random
    monkeypatch.setattr(random, "random", lambda: 0.0)  # 0.0 < 0.8 → explore

    raw_score = 0.99
    result = service.adjust_score(raw_score, n_feedback=0)
    # The adjusted score is a second call to random.random, also patched to 0.0
    assert result == pytest.approx(0.0)
    assert result != raw_score


def test_adjust_score_exploit_branch(monkeypatch):
    """Force the random roll to be above epsilon → should return raw_score unchanged."""
    service = ExplorationService()
    import random

    # epsilon at 100 feedback ≈ min (0.05); roll 0.99 > 0.05 → exploit
    monkeypatch.setattr(random, "random", lambda: 0.99)
    raw_score = 0.75
    result = service.adjust_score(raw_score, n_feedback=100)
    assert result == pytest.approx(raw_score)


def test_adjust_score_high_feedback_mostly_exploits():
    """Statistical check: with 200 feedback samples, ≥90% of adjustments should be exploitation."""
    service = ExplorationService()
    raw_score = 0.42
    exploitations = sum(
        1 for _ in range(500)
        if service.adjust_score(raw_score, n_feedback=200) == pytest.approx(raw_score, abs=1e-9)
    )
    # At n=200 epsilon≈0.05, so ~95% exploitation expected; use 85% as a loose bound
    assert exploitations / 500 > 0.85


def test_adjust_score_zero_feedback_rarely_exploits():
    """Statistical check: with 0 feedback, most adjustments should be exploration."""
    service = ExplorationService()
    raw_score = 0.42
    exploitations = sum(
        1 for _ in range(500)
        if service.adjust_score(raw_score, n_feedback=0) == pytest.approx(raw_score, abs=1e-9)
    )
    # At n=0 epsilon=0.8, so ~20% exploitation expected; use 35% as a loose upper bound
    assert exploitations / 500 < 0.35


def test_adjust_score_result_is_float():
    service = ExplorationService()
    result = service.adjust_score(0.5, 0)
    assert isinstance(result, float)


def test_adjust_score_explore_result_in_unit_interval():
    """Explored scores must be in [0, 1]."""
    service = ExplorationService()
    for _ in range(50):
        result = service.adjust_score(0.5, 0)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def test_singleton_exists():
    assert exploration_service is not None
    assert isinstance(exploration_service, ExplorationService)
