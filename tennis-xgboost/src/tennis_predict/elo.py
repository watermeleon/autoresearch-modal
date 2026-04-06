"""ELO rating engine with overall and per-surface ratings.

Strict temporal ordering: all ELO values used for features are PRE-match.
Updates happen AFTER feature extraction for each match.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tennis_predict.config import (
    DEFAULT_ELO_K,
    ELO_START,
    ROLLING_WINDOWS,
    SURFACE_K_FACTORS,
    SURFACE_PRIOR_MATCHES,
)


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def updated_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k_factor: float,
) -> tuple[float, float]:
    """Compute new ELO ratings after a match.

    Returns (new_rating_a, new_rating_b).
    """
    expected_a = elo_expected(rating_a, rating_b)
    delta = k_factor * (score_a - expected_a)
    return rating_a + delta, rating_b - delta


def recency_factor(
    match_date: pd.Timestamp,
    reference_date: pd.Timestamp,
) -> float:
    """Linearly decay recency from 1.0 in the last year to 0.0 at 10+ years."""
    age_days = max((reference_date - match_date).days, 0)
    recent_window_days = 365.25
    stale_window_days = 365.25 * 10

    if age_days <= recent_window_days:
        return 1.0
    if age_days >= stale_window_days:
        return 0.0

    decay_span = stale_window_days - recent_window_days
    return 1.0 - ((age_days - recent_window_days) / decay_span)


def effective_k_factor(
    base_k_factor: float,
    match_date: pd.Timestamp,
    reference_date: pd.Timestamp,
) -> float:
    """Increase K for more recent matches while preserving the base K floor."""
    return base_k_factor * (1.0 + 0.5 * recency_factor(match_date, reference_date))


def surface_base_k_factor(surface: str) -> float:
    """Return the configured base K-factor for a surface."""
    return SURFACE_K_FACTORS.get(surface, DEFAULT_ELO_K)


def shrunk_rating(
    overall_rating: float,
    surface_rating: float,
    observation_count: int,
    prior_observations: float,
) -> float:
    """Blend a sparse specialist rating back toward the player's global level."""
    if prior_observations <= 0:
        return surface_rating
    blend = min(observation_count / prior_observations, 1.0)
    return overall_rating + (surface_rating - overall_rating) * blend


@dataclass(slots=True)
class MatchStats:
    """Per-match statistics stored in player history for rolling window features."""

    surface: str
    match_date: pd.Timestamp
    won: int
    opponent_hand: str | None
    ace_rate: float
    df_rate: float
    first_in_rate: float
    first_won_rate: float
    second_won_rate: float
    service_points_won_rate: float
    return_points_won_rate: float
    point_win_rate: float
    break_save_rate: float
    break_conversion_rate: float
    minutes: float
    elo_delta: float
    surface_elo_delta: float
    expected_win_prob: float
    performance_vs_expectation: float
    opponent_elo: float
    opponent_surface_elo: float
    serve_elo_score: float
    surface_serve_elo_score: float
    sets_played: float
    tiebreaks_played: float
    tiebreaks_won: float
    was_retirement: int
    straight_set_win: int
    game_margin: float


@dataclass
class PlayerState:
    """Mutable state tracking for a single player's ELO and match history."""

    elo: float = ELO_START
    surface_elo: dict[str, float] = field(default_factory=dict)
    serve_elo: float = ELO_START
    return_elo: float = ELO_START
    surface_serve_elo: dict[str, float] = field(default_factory=dict)
    surface_return_elo: dict[str, float] = field(default_factory=dict)
    total_matches: int = 0
    total_wins: int = 0
    best_of_3_matches: int = 0
    best_of_3_wins: int = 0
    best_of_5_matches: int = 0
    best_of_5_wins: int = 0
    serve_matches: int = 0
    return_matches: int = 0
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_serve_matches: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    surface_return_matches: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    last_match_date: pd.Timestamp | None = None
    last_surface_date: dict[str, pd.Timestamp] = field(default_factory=dict)
    history: list[MatchStats] = field(default_factory=list)

    def current_surface_elo(self, surface: str) -> float:
        """Get surface ELO, defaulting to ELO_START if no surface matches yet."""
        return self.surface_elo.get(surface, ELO_START)

    def current_surface_serve_elo(self, surface: str) -> float:
        """Get surface serve ELO, defaulting to ELO_START if absent."""
        return self.surface_serve_elo.get(surface, ELO_START)

    def current_surface_return_elo(self, surface: str) -> float:
        """Get surface return ELO, defaulting to ELO_START if absent."""
        return self.surface_return_elo.get(surface, ELO_START)

    def recent_history(self, n: int) -> list[MatchStats]:
        """Get last n matches from history."""
        return self.history[-n:] if self.history else []


def shrunk_surface_elo(state: PlayerState, surface: str) -> float:
    """Bayesian-shrunk surface ELO: blend toward overall ELO when few surface matches.

    Formula: overall + (surface - overall) * min(surface_count / PRIOR, 1.0)
    When surface_count >= SURFACE_PRIOR_MATCHES, returns raw surface ELO.
    When surface_count == 0, returns overall ELO.
    """
    surface_count = state.surface_matches.get(surface, 0)
    surface_rating = state.current_surface_elo(surface)
    return shrunk_rating(
        state.elo,
        surface_rating,
        surface_count,
        SURFACE_PRIOR_MATCHES,
    )


def shrunk_surface_serve_elo(state: PlayerState, surface: str) -> float:
    """Surface serve ELO blended back toward overall serve ELO."""
    surface_count = state.surface_serve_matches.get(surface, 0)
    surface_rating = state.current_surface_serve_elo(surface)
    return shrunk_rating(
        state.serve_elo,
        surface_rating,
        surface_count,
        SURFACE_PRIOR_MATCHES,
    )


def shrunk_surface_return_elo(state: PlayerState, surface: str) -> float:
    """Surface return ELO blended back toward overall return ELO."""
    surface_count = state.surface_return_matches.get(surface, 0)
    surface_rating = state.current_surface_return_elo(surface)
    return shrunk_rating(
        state.return_elo,
        surface_rating,
        surface_count,
        SURFACE_PRIOR_MATCHES,
    )


def updated_observed_elo(
    rating_a: float,
    rating_b: float,
    observed_score_a: float,
    k_factor: float,
) -> tuple[float, float]:
    """Update a paired rating track when the observed score is continuous."""
    if pd.isna(observed_score_a):
        return rating_a, rating_b
    bounded_score = float(np.clip(observed_score_a, 0.0, 1.0))
    return updated_elo(rating_a, rating_b, bounded_score, k_factor)


def apply_match_result(
    winner_state: PlayerState,
    loser_state: PlayerState,
    surface: str,
    match_date: pd.Timestamp,
    winner_stats: MatchStats,
    loser_stats: MatchStats,
    k_factor: float,
) -> None:
    """Update both players' ELO ratings and state after a match.

    This MUST be called AFTER features are extracted for this match.
    """
    # Compute new ratings
    winner_post, loser_post = updated_elo(
        winner_state.elo, loser_state.elo, 1.0, k_factor
    )
    winner_surface_post, loser_surface_post = updated_elo(
        winner_state.current_surface_elo(surface),
        loser_state.current_surface_elo(surface),
        1.0,
        k_factor,
    )
    winner_serve_post, loser_return_post = updated_observed_elo(
        winner_state.serve_elo,
        loser_state.return_elo,
        winner_stats.serve_elo_score,
        k_factor,
    )
    loser_serve_post, winner_return_post = updated_observed_elo(
        loser_state.serve_elo,
        winner_state.return_elo,
        loser_stats.serve_elo_score,
        k_factor,
    )
    winner_surface_serve_post, loser_surface_return_post = updated_observed_elo(
        winner_state.current_surface_serve_elo(surface),
        loser_state.current_surface_return_elo(surface),
        winner_stats.surface_serve_elo_score,
        k_factor,
    )
    loser_surface_serve_post, winner_surface_return_post = updated_observed_elo(
        loser_state.current_surface_serve_elo(surface),
        winner_state.current_surface_return_elo(surface),
        loser_stats.surface_serve_elo_score,
        k_factor,
    )

    # Update ELO
    winner_state.elo = winner_post
    loser_state.elo = loser_post
    winner_state.surface_elo[surface] = winner_surface_post
    loser_state.surface_elo[surface] = loser_surface_post
    winner_state.serve_elo = winner_serve_post
    loser_state.serve_elo = loser_serve_post
    winner_state.return_elo = winner_return_post
    loser_state.return_elo = loser_return_post
    winner_state.surface_serve_elo[surface] = winner_surface_serve_post
    loser_state.surface_serve_elo[surface] = loser_surface_serve_post
    winner_state.surface_return_elo[surface] = winner_surface_return_post
    loser_state.surface_return_elo[surface] = loser_surface_return_post

    # Update match counts
    winner_state.total_matches += 1
    winner_state.total_wins += 1
    loser_state.total_matches += 1

    winner_state.surface_matches[surface] += 1
    winner_state.surface_wins[surface] += 1
    loser_state.surface_matches[surface] += 1
    if not pd.isna(winner_stats.serve_elo_score):
        winner_state.serve_matches += 1
        loser_state.return_matches += 1
    if not pd.isna(loser_stats.serve_elo_score):
        loser_state.serve_matches += 1
        winner_state.return_matches += 1
    if not pd.isna(winner_stats.surface_serve_elo_score):
        winner_state.surface_serve_matches[surface] += 1
        loser_state.surface_return_matches[surface] += 1
    if not pd.isna(loser_stats.surface_serve_elo_score):
        loser_state.surface_serve_matches[surface] += 1
        winner_state.surface_return_matches[surface] += 1

    # Update dates
    winner_state.last_match_date = match_date
    loser_state.last_match_date = match_date
    winner_state.last_surface_date[surface] = match_date
    loser_state.last_surface_date[surface] = match_date

    # Append history
    winner_state.history.append(winner_stats)
    loser_state.history.append(loser_stats)

    # Cap history length to avoid unbounded growth
    max_history = max(ROLLING_WINDOWS) * 2
    if len(winner_state.history) > max_history:
        winner_state.history = winner_state.history[-max_history:]
    if len(loser_state.history) > max_history:
        loser_state.history = loser_state.history[-max_history:]
