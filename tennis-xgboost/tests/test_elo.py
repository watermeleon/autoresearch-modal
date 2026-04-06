"""Tests for the ELO engine."""

from __future__ import annotations

import math

from tennis_predict.elo import (
    ELO_START,
    PlayerState,
    elo_expected,
    shrunk_surface_elo,
    updated_elo,
)


def test_elo_expected_equal_ratings() -> None:
    """Equal ratings should give 50% expected score."""
    assert elo_expected(1500.0, 1500.0) == 0.5


def test_elo_expected_higher_rating_favored() -> None:
    """Higher-rated player should have > 50% expected score."""
    assert elo_expected(1600.0, 1400.0) > 0.5


def test_elo_expected_symmetry() -> None:
    """Expected scores should sum to 1."""
    ea = elo_expected(1600.0, 1400.0)
    eb = elo_expected(1400.0, 1600.0)
    assert abs(ea + eb - 1.0) < 1e-10


def test_updated_elo_winner_gains() -> None:
    """Winner should gain rating points."""
    new_a, new_b = updated_elo(1500.0, 1500.0, 1.0, 32.0)
    assert new_a > 1500.0
    assert new_b < 1500.0
    assert abs(new_a + new_b - 3000.0) < 1e-10  # Zero-sum


def test_updated_elo_zero_sum() -> None:
    """ELO updates should be zero-sum."""
    new_a, new_b = updated_elo(1600.0, 1400.0, 1.0, 32.0)
    assert abs((new_a - 1600.0) + (new_b - 1400.0)) < 1e-10


def test_shrunk_surface_elo_no_surface_matches() -> None:
    """With zero surface matches, shrunk surface ELO should equal overall ELO."""
    state = PlayerState(elo=1600.0)
    result = shrunk_surface_elo(state, "Hard")
    assert result == 1600.0


def test_shrunk_surface_elo_many_surface_matches() -> None:
    """With many surface matches, shrunk ELO should approach raw surface ELO."""
    state = PlayerState(elo=1600.0)
    state.surface_elo["Hard"] = 1700.0
    state.surface_matches["Hard"] = 100
    result = shrunk_surface_elo(state, "Hard")
    assert abs(result - 1700.0) < 1.0  # Should be very close to surface ELO


def test_shrunk_surface_elo_partial_blend() -> None:
    """With some surface matches, should blend between overall and surface ELO."""
    state = PlayerState(elo=1500.0)
    state.surface_elo["Clay"] = 1600.0
    state.surface_matches["Clay"] = 10  # Half of SURFACE_PRIOR_MATCHES (20)
    result = shrunk_surface_elo(state, "Clay")
    # blend = 10/20 = 0.5, so result = 1500 + (1600-1500)*0.5 = 1550
    assert abs(result - 1550.0) < 1e-10
