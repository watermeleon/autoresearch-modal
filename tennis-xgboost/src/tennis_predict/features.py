"""Feature engineering for tennis match prediction.

Tour-agnostic: works identically for ATP and WTA data.
Uses elo.py for all rating computations.
Strict temporal ordering only — no leaky mode.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from tennis_predict.config import (
    DEFAULT_ELO_K,
    RANK_MOMENTUM_WINDOWS_DAYS,
    RECENT_ACTIVITY_WINDOWS_DAYS,
    ROLLING_WINDOWS,
    SURFACE_FORM_WINDOWS,
)
from tennis_predict.data import ROUND_ORDER
from tennis_predict.elo import (
    MatchStats,
    PlayerState,
    apply_match_result,
    effective_k_factor,
    elo_expected,
    surface_base_k_factor,
    shrunk_surface_elo,
    shrunk_surface_return_elo,
    shrunk_surface_serve_elo,
    updated_elo,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float:
    """Divide numerator by denominator, returning NaN on any invalid input."""
    if numerator is None or denominator in (None, 0):
        return np.nan
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def normalize_surface(surface: Any) -> str:
    """Normalize surface string to title case, defaulting to 'Unknown'."""
    if surface is None or pd.isna(surface):
        return "Unknown"
    return str(surface).strip().title() or "Unknown"


def numeric_or_nan(value: Any) -> float:
    """Convert to float, returning NaN for missing values."""
    if value is None or pd.isna(value):
        return np.nan
    return float(value)


def seed_to_numeric(seed: Any) -> float:
    """Convert seed to float, returning NaN for missing or non-numeric."""
    if seed is None or pd.isna(seed):
        return np.nan
    try:
        return float(seed)
    except (TypeError, ValueError):
        return np.nan


def normalize_entry(entry: Any) -> str | None:
    """Normalize entry status strings, returning None when absent."""
    if entry is None or pd.isna(entry):
        return None
    normalized = str(entry).strip().upper()
    return normalized or None


def normalize_hand(hand: Any) -> str | None:
    """Normalize handedness strings, returning None for unknown values."""
    if hand is None or pd.isna(hand):
        return None
    normalized = str(hand).strip().upper()
    if normalized in {"L", "R"}:
        return normalized
    return None


def normalize_ioc(ioc: Any) -> str | None:
    """Normalize IOC country codes, returning None when absent."""
    if ioc is None or pd.isna(ioc):
        return None
    normalized = str(ioc).strip().upper()
    return normalized or None


def days_since(previous_date: pd.Timestamp | None, current_date: pd.Timestamp) -> float:
    """Days between two dates, NaN if previous_date is None."""
    if previous_date is None:
        return np.nan
    return float((current_date - previous_date).days)


def mean_attribute(records: list[MatchStats], attr: str) -> float:
    """Mean of a MatchStats attribute across records, ignoring NaN."""
    values = [getattr(item, attr) for item in records if not pd.isna(getattr(item, attr))]
    return float(np.mean(values)) if values else np.nan


def weighted_mean_attribute(
    records: list[MatchStats],
    attr: str,
    weight_attr: str,
) -> float:
    """Weighted mean of a MatchStats attribute, ignoring missing values."""
    values: list[float] = []
    weights: list[float] = []
    for item in records:
        value = getattr(item, attr)
        raw_weight = getattr(item, weight_attr)
        if pd.isna(value) or pd.isna(raw_weight):
            continue
        weight = float(raw_weight) / 1500.0
        if weight <= 0:
            continue
        values.append(float(value))
        weights.append(weight)
    return float(np.average(values, weights=weights)) if weights else np.nan


def std_attribute(records: list[MatchStats], attr: str) -> float:
    """Standard deviation of a MatchStats attribute, ignoring NaN."""
    values = [getattr(item, attr) for item in records if not pd.isna(getattr(item, attr))]
    return float(np.std(values)) if len(values) >= 2 else np.nan


def ratio_of_sums(
    records: list[MatchStats],
    numerator_attr: str,
    denominator_attr: str,
) -> float:
    """Ratio of summed attributes across records, ignoring missing values."""
    numerator = 0.0
    denominator = 0.0
    has_value = False
    for item in records:
        numerator_value = getattr(item, numerator_attr)
        denominator_value = getattr(item, denominator_attr)
        if pd.isna(numerator_value) or pd.isna(denominator_value):
            continue
        numerator += float(numerator_value)
        denominator += float(denominator_value)
        has_value = True
    if not has_value or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def shrunk_binary_rate(
    successes: int,
    observations: int,
    fallback_rate: float,
    prior_observations: float,
) -> float:
    """Shrink sparse binary rates back toward a more stable fallback rate."""
    if pd.isna(fallback_rate):
        fallback_rate = 0.5
    if observations <= 0:
        return float(fallback_rate)
    numerator = float(fallback_rate) * prior_observations + float(successes)
    denominator = prior_observations + float(observations)
    return float(numerator / denominator)


def matchup_fallback_rate(*rates: float) -> float:
    """Return the first non-missing fallback rate, defaulting to 0.5."""
    for rate in rates:
        if pd.notna(rate):
            return float(rate)
    return 0.5


ROUND_STEPS_FROM_FINAL: dict[str, float] = {
    "F": 0.0,
    "BR": 0.0,
    "SF": 1.0,
    "QF": 2.0,
    "R4": 2.5,
    "R16": 3.0,
    "R32": 4.0,
    "R64": 5.0,
    "R128": 6.0,
}


def draw_size_to_rounds(draw_size: Any) -> float:
    """Approximate knockout rounds implied by the tournament draw size."""
    draw_size_value = numeric_or_nan(draw_size)
    if pd.isna(draw_size_value) or draw_size_value <= 0:
        return np.nan
    return float(np.ceil(np.log2(draw_size_value)))


def round_stage_index(round_name: Any, draw_size: Any) -> float:
    """Approximate stage index from tournament start, 0-based."""
    steps_from_final = ROUND_STEPS_FROM_FINAL.get(str(round_name).strip().upper())
    total_rounds = draw_size_to_rounds(draw_size)
    if steps_from_final is None or pd.isna(total_rounds):
        return np.nan
    return total_rounds - steps_from_final - 1.0


def round_stage_progress(round_name: Any, draw_size: Any) -> float:
    """Approximate match stage on a 0-1 scale from opening round to final."""
    stage_index = round_stage_index(round_name, draw_size)
    total_rounds = draw_size_to_rounds(draw_size)
    if pd.isna(stage_index) or pd.isna(total_rounds) or total_rounds <= 1:
        return np.nan
    return float(np.clip(stage_index / (total_rounds - 1.0), 0.0, 1.0))


def tournament_snapshot(
    player_id: int,
    tourney_name: str,
    match_date: pd.Timestamp,
    tournament_matches: dict[tuple[int, str], int],
    tournament_wins: dict[tuple[int, str], int],
    tournament_last_dates: dict[tuple[int, str], pd.Timestamp],
) -> dict[str, float]:
    """Pre-match player history at the current tournament/venue."""
    key = (player_id, tourney_name)
    matches = tournament_matches[key]
    return {
        "tourney_matches": float(matches),
        "tourney_win_rate": safe_ratio(tournament_wins[key], matches),
        "days_since_last_tourney_match": days_since(
            tournament_last_dates.get(key), match_date
        ),
    }


SERVICE_POINT_WIN_RATE_PRIOR = 0.60
SERVICE_POINT_WIN_RATE_PRIOR_OBSERVATIONS = 200.0
SURFACE_SERVICE_POINT_WIN_RATE_PRIOR_OBSERVATIONS = 100.0
SERVICE_POINT_ELO_SCORE_SPREAD = 0.30
FORMAT_WIN_RATE_PRIOR_MATCHES = 12.0
IOC_BUCKET_COUNT = 10
IOC_MATCHUP_PRIOR_MATCHES = 8.0
SCORE_TOKEN_RE = re.compile(
    r"^(?P<winner_games>\d+)-(?P<loser_games>\d+)(?:\(\d+\)|\[\d+(?:-\d+)?\])?$"
)
SCORE_RETIREMENT_MARKERS = (
    "RET",
    "W/O",
    "WALKOVER",
    "DEF",
    "DEFAULT",
    "ABD",
    "ABANDONED",
    "UNF",
    "UNFINISHED",
)


def tracked_ioc_buckets(matches: pd.DataFrame) -> tuple[str, ...]:
    """Return the most common IOC codes in the provided match frame."""
    if "winner_ioc" not in matches.columns or "loser_ioc" not in matches.columns:
        return ()
    all_iocs = pd.concat(
        [matches["winner_ioc"], matches["loser_ioc"]],
        ignore_index=True,
    )
    normalized = all_iocs.map(normalize_ioc)
    counts = normalized.dropna().value_counts()
    return tuple(counts.head(IOC_BUCKET_COUNT).index.tolist())


def ioc_bucket(ioc: str | None, tracked_buckets: tuple[str, ...]) -> str | None:
    """Collapse IOC codes into tracked buckets plus OTHER."""
    if ioc is None:
        return None
    return ioc if ioc in tracked_buckets else "OTHER"


@dataclass(frozen=True, slots=True)
class ParsedScore:
    """Winner-oriented match score summary parsed from the raw score string."""

    sets_played: float
    tiebreaks_played: float
    winner_tiebreaks_won: float
    loser_tiebreaks_won: float
    was_retirement: bool
    winner_straight_set_win: int
    loser_straight_set_win: int
    winner_game_margin: float
    loser_game_margin: float


def empty_parsed_score() -> ParsedScore:
    """Return an empty parsed score payload for missing score strings."""
    return ParsedScore(
        sets_played=np.nan,
        tiebreaks_played=0.0,
        winner_tiebreaks_won=0.0,
        loser_tiebreaks_won=0.0,
        was_retirement=False,
        winner_straight_set_win=0,
        loser_straight_set_win=0,
        winner_game_margin=np.nan,
        loser_game_margin=np.nan,
    )


def parse_score(score: Any) -> ParsedScore:
    """Parse a winner-oriented tennis score string into match-level features."""
    if score is None or pd.isna(score):
        return empty_parsed_score()

    normalized_score = str(score).strip().upper()
    if not normalized_score:
        return empty_parsed_score()

    was_retirement = any(marker in normalized_score for marker in SCORE_RETIREMENT_MARKERS)
    winner_sets = 0
    loser_sets = 0
    winner_games_total = 0
    loser_games_total = 0
    tiebreaks_played = 0
    winner_tiebreaks_won = 0
    loser_tiebreaks_won = 0

    for raw_token in normalized_score.replace(",", " ").split():
        token = raw_token.strip().rstrip(".")
        if not token:
            continue
        match = SCORE_TOKEN_RE.fullmatch(token)
        if match is None:
            continue
        winner_games = int(match.group("winner_games"))
        loser_games = int(match.group("loser_games"))
        winner_games_total += winner_games
        loser_games_total += loser_games
        if winner_games > loser_games:
            winner_sets += 1
        elif loser_games > winner_games:
            loser_sets += 1
        if "(" in token:
            tiebreaks_played += 1
            if winner_games > loser_games:
                winner_tiebreaks_won += 1
            elif loser_games > winner_games:
                loser_tiebreaks_won += 1

    sets_played = float(winner_sets + loser_sets) if (winner_sets + loser_sets) > 0 else np.nan
    winner_game_margin = (
        float(winner_games_total - loser_games_total)
        if (winner_sets + loser_sets) > 0
        else np.nan
    )
    loser_game_margin = -winner_game_margin if pd.notna(winner_game_margin) else np.nan
    winner_straight_set_win = int(
        not was_retirement and winner_sets > 0 and loser_sets == 0
    )

    return ParsedScore(
        sets_played=sets_played,
        tiebreaks_played=float(tiebreaks_played),
        winner_tiebreaks_won=float(winner_tiebreaks_won),
        loser_tiebreaks_won=float(loser_tiebreaks_won),
        was_retirement=was_retirement,
        winner_straight_set_win=winner_straight_set_win,
        loser_straight_set_win=0,
        winner_game_margin=winner_game_margin,
        loser_game_margin=loser_game_margin,
    )


def score_features_for_role(parsed_score: ParsedScore, role: str) -> dict[str, float | int]:
    """Convert a parsed winner-oriented score into player-oriented features."""
    if role == "winner":
        return {
            "sets_played": parsed_score.sets_played,
            "tiebreaks_played": parsed_score.tiebreaks_played,
            "tiebreaks_won": parsed_score.winner_tiebreaks_won,
            "was_retirement": int(parsed_score.was_retirement),
            "straight_set_win": parsed_score.winner_straight_set_win,
            "game_margin": parsed_score.winner_game_margin,
        }
    if role == "loser":
        return {
            "sets_played": parsed_score.sets_played,
            "tiebreaks_played": parsed_score.tiebreaks_played,
            "tiebreaks_won": parsed_score.loser_tiebreaks_won,
            "was_retirement": int(parsed_score.was_retirement),
            "straight_set_win": parsed_score.loser_straight_set_win,
            "game_margin": parsed_score.loser_game_margin,
        }
    raise ValueError(f"Unknown role: {role}")


def service_point_win_rate_baseline(
    observed_sum: float,
    observed_count: int,
) -> float:
    """Historical overall service-point win baseline with a weak prior."""
    numerator = (
        SERVICE_POINT_WIN_RATE_PRIOR * SERVICE_POINT_WIN_RATE_PRIOR_OBSERVATIONS
        + observed_sum
    )
    denominator = SERVICE_POINT_WIN_RATE_PRIOR_OBSERVATIONS + observed_count
    return float(numerator / denominator)


def surface_service_point_win_rate_baseline(
    surface: str,
    overall_baseline: float,
    surface_sums: dict[str, float],
    surface_counts: dict[str, int],
) -> float:
    """Surface-specific service-point baseline, shrunk toward the overall rate."""
    numerator = (
        overall_baseline * SURFACE_SERVICE_POINT_WIN_RATE_PRIOR_OBSERVATIONS
        + surface_sums[surface]
    )
    denominator = (
        SURFACE_SERVICE_POINT_WIN_RATE_PRIOR_OBSERVATIONS
        + surface_counts[surface]
    )
    return float(numerator / denominator)


def service_point_elo_score(rate: float, baseline: float) -> float:
    """Map service-point win rate onto a centered 0-1 ELO score."""
    if pd.isna(rate):
        return np.nan
    centered_score = 0.5 + ((float(rate) - baseline) / SERVICE_POINT_ELO_SCORE_SPREAD)
    return float(np.clip(centered_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Rankings index for rank momentum features
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RankingsIndex:
    """Pre-indexed rankings data for fast lookups by player_id and date."""

    dates_by_player: dict[int, np.ndarray]
    ranks_by_player: dict[int, np.ndarray]
    points_by_player: dict[int, np.ndarray]

    @classmethod
    def from_frame(cls, rankings: pd.DataFrame | None) -> RankingsIndex:
        """Build index from rankings DataFrame."""
        if rankings is None or rankings.empty:
            return cls({}, {}, {})

        ordered = rankings.sort_values(
            ["player_id", "ranking_date"], kind="mergesort"
        )
        dates_by_player: dict[int, np.ndarray] = {}
        ranks_by_player: dict[int, np.ndarray] = {}
        points_by_player: dict[int, np.ndarray] = {}

        for player_id, group in ordered.groupby("player_id", sort=False):
            pid = int(player_id)
            dates_by_player[pid] = group["ranking_date"].to_numpy(dtype="datetime64[ns]")
            ranks_by_player[pid] = group["rank"].to_numpy(dtype=float)
            points_by_player[pid] = group["points"].to_numpy(dtype=float)

        return cls(dates_by_player, ranks_by_player, points_by_player)

    @staticmethod
    def empty_snapshot() -> dict[str, float]:
        """Return empty rank momentum features."""
        snapshot: dict[str, float] = {}
        for days in RANK_MOMENTUM_WINDOWS_DAYS:
            snapshot[f"rank_change_{days}_days"] = np.nan
            snapshot[f"rank_points_change_{days}_days"] = np.nan
        return snapshot

    def snapshot(self, player_id: int, match_date: pd.Timestamp) -> dict[str, float]:
        """Compute rank momentum features for a player at a given date."""
        dates = self.dates_by_player.get(player_id)
        if dates is None or len(dates) == 0:
            return self.empty_snapshot()

        match_time = match_date.to_datetime64()
        current_index = int(np.searchsorted(dates, match_time, side="right") - 1)
        if current_index < 0:
            return self.empty_snapshot()

        ranks = self.ranks_by_player[player_id]
        points = self.points_by_player[player_id]
        current_rank = float(ranks[current_index])
        current_points = float(points[current_index])

        snapshot: dict[str, float] = {}
        for days in RANK_MOMENTUM_WINDOWS_DAYS:
            prior_time = match_time - np.timedelta64(days, "D")
            prior_index = int(np.searchsorted(dates, prior_time, side="right") - 1)
            if prior_index < 0:
                snapshot[f"rank_change_{days}_days"] = np.nan
                snapshot[f"rank_points_change_{days}_days"] = np.nan
                continue
            prior_rank = float(ranks[prior_index])
            prior_points = float(points[prior_index])
            snapshot[f"rank_change_{days}_days"] = prior_rank - current_rank
            snapshot[f"rank_points_change_{days}_days"] = current_points - prior_points
        return snapshot


# ---------------------------------------------------------------------------
# Stats extraction from match row
# ---------------------------------------------------------------------------

def stats_from_row(
    row: pd.Series,
    role: str,
    *,
    surface: str,
    match_date: pd.Timestamp,
    elo_delta: float,
    surface_elo_delta: float,
    expected_win_prob: float,
    opponent_elo: float,
    opponent_surface_elo: float,
    serve_elo_score: float,
    surface_serve_elo_score: float,
    score_features: dict[str, float | int],
) -> MatchStats:
    """Extract per-player match statistics from a raw match row."""
    if role not in {"winner", "loser"}:
        raise ValueError(f"Unknown role: {role}")
    prefix = "w" if role == "winner" else "l"
    opp_prefix = "l" if role == "winner" else "w"
    opponent_role = "loser" if role == "winner" else "winner"

    service_points = row[f"{prefix}_svpt"]
    first_in = row[f"{prefix}_1stIn"]
    first_won = row[f"{prefix}_1stWon"]
    second_won = row[f"{prefix}_2ndWon"]
    second_attempts = (
        service_points - first_in
        if pd.notna(service_points) and pd.notna(first_in)
        else np.nan
    )
    service_points_won = (
        first_won + second_won
        if pd.notna(first_won) and pd.notna(second_won)
        else np.nan
    )
    opponent_service_points = row[f"{opp_prefix}_svpt"]
    opponent_first_won = row[f"{opp_prefix}_1stWon"]
    opponent_second_won = row[f"{opp_prefix}_2ndWon"]
    opponent_service_points_won = (
        opponent_first_won + opponent_second_won
        if pd.notna(opponent_first_won) and pd.notna(opponent_second_won)
        else np.nan
    )
    return_points_won = (
        opponent_service_points - opponent_service_points_won
        if pd.notna(opponent_service_points) and pd.notna(opponent_service_points_won)
        else np.nan
    )
    total_points_won = (
        service_points_won + return_points_won
        if pd.notna(service_points_won) and pd.notna(return_points_won)
        else np.nan
    )
    total_points = (
        service_points + opponent_service_points
        if pd.notna(service_points) and pd.notna(opponent_service_points)
        else np.nan
    )
    opponent_bp_faced = row[f"{opp_prefix}_bpFaced"]
    opponent_bp_saved = row[f"{opp_prefix}_bpSaved"]
    break_points_won = (
        max(opponent_bp_faced - opponent_bp_saved, 0)
        if pd.notna(opponent_bp_faced) and pd.notna(opponent_bp_saved)
        else np.nan
    )
    won = 1 if role == "winner" else 0

    return MatchStats(
        surface=surface,
        match_date=match_date,
        won=won,
        opponent_hand=normalize_hand(row.get(f"{opponent_role}_hand")),
        ace_rate=safe_ratio(row[f"{prefix}_ace"], service_points),
        df_rate=safe_ratio(row[f"{prefix}_df"], service_points),
        first_in_rate=safe_ratio(first_in, service_points),
        first_won_rate=safe_ratio(first_won, first_in),
        second_won_rate=safe_ratio(second_won, second_attempts),
        service_points_won_rate=safe_ratio(service_points_won, service_points),
        return_points_won_rate=safe_ratio(return_points_won, opponent_service_points),
        point_win_rate=safe_ratio(total_points_won, total_points),
        break_save_rate=safe_ratio(row[f"{prefix}_bpSaved"], row[f"{prefix}_bpFaced"]),
        break_conversion_rate=safe_ratio(break_points_won, opponent_bp_faced),
        minutes=float(row["minutes"]) if pd.notna(row["minutes"]) else np.nan,
        elo_delta=elo_delta,
        surface_elo_delta=surface_elo_delta,
        expected_win_prob=expected_win_prob,
        performance_vs_expectation=won - expected_win_prob,
        opponent_elo=opponent_elo,
        opponent_surface_elo=opponent_surface_elo,
        serve_elo_score=serve_elo_score,
        surface_serve_elo_score=surface_serve_elo_score,
        sets_played=float(score_features["sets_played"]),
        tiebreaks_played=float(score_features["tiebreaks_played"]),
        tiebreaks_won=float(score_features["tiebreaks_won"]),
        was_retirement=int(score_features["was_retirement"]),
        straight_set_win=int(score_features["straight_set_win"]),
        game_margin=float(score_features["game_margin"]),
    )


def service_points_won_rate_from_row(row: pd.Series, role: str) -> float:
    """Compute service-point win rate for a winner/loser row role."""
    if role not in {"winner", "loser"}:
        raise ValueError(f"Unknown role: {role}")
    prefix = "w" if role == "winner" else "l"
    service_points = row[f"{prefix}_svpt"]
    first_won = row[f"{prefix}_1stWon"]
    second_won = row[f"{prefix}_2ndWon"]
    if pd.isna(service_points) or pd.isna(first_won) or pd.isna(second_won):
        return np.nan
    return safe_ratio(first_won + second_won, service_points)


# ---------------------------------------------------------------------------
# Snapshot builders (rolling window features)
# ---------------------------------------------------------------------------

def history_snapshot(state: PlayerState) -> dict[str, float]:
    """Compute rolling window features from player match history."""
    snapshot: dict[str, float] = {}
    for window in ROLLING_WINDOWS:
        sample = state.recent_history(window)
        score_sample = [record for record in sample if not record.was_retirement]
        if not sample:
            for metric in (
                "win_rate", "ace_rate", "df_rate", "first_in_rate",
                "first_won_rate", "second_won_rate", "service_points_won_rate",
                "return_points_won_rate", "point_win_rate", "break_save_rate",
                "break_conversion_rate", "minutes_avg", "elo_delta_avg",
                "surface_elo_delta_avg", "performance_vs_expectation",
                "opponent_elo_avg", "opponent_surface_elo_avg",
                "quality_weighted_win_rate", "quality_weighted_point_win_rate",
                "quality_weighted_performance_vs_expectation",
                "point_win_rate_std", "performance_vs_expectation_std",
                "avg_sets_played", "tiebreak_win_rate",
                "game_margin_avg", "straight_set_win_pct",
            ):
                snapshot[f"{metric}_last_{window}"] = np.nan
            continue

        snapshot[f"win_rate_last_{window}"] = mean_attribute(sample, "won")
        snapshot[f"ace_rate_last_{window}"] = mean_attribute(sample, "ace_rate")
        snapshot[f"df_rate_last_{window}"] = mean_attribute(sample, "df_rate")
        snapshot[f"first_in_rate_last_{window}"] = mean_attribute(sample, "first_in_rate")
        snapshot[f"first_won_rate_last_{window}"] = mean_attribute(sample, "first_won_rate")
        snapshot[f"second_won_rate_last_{window}"] = mean_attribute(sample, "second_won_rate")
        snapshot[f"service_points_won_rate_last_{window}"] = mean_attribute(
            sample, "service_points_won_rate"
        )
        snapshot[f"return_points_won_rate_last_{window}"] = mean_attribute(
            sample, "return_points_won_rate"
        )
        snapshot[f"point_win_rate_last_{window}"] = mean_attribute(sample, "point_win_rate")
        snapshot[f"break_save_rate_last_{window}"] = mean_attribute(sample, "break_save_rate")
        snapshot[f"break_conversion_rate_last_{window}"] = mean_attribute(
            sample, "break_conversion_rate"
        )
        snapshot[f"minutes_avg_last_{window}"] = mean_attribute(sample, "minutes")
        snapshot[f"elo_delta_avg_last_{window}"] = mean_attribute(sample, "elo_delta")
        snapshot[f"surface_elo_delta_avg_last_{window}"] = mean_attribute(
            sample, "surface_elo_delta"
        )
        snapshot[f"performance_vs_expectation_last_{window}"] = mean_attribute(
            sample, "performance_vs_expectation"
        )
        snapshot[f"opponent_elo_avg_last_{window}"] = mean_attribute(sample, "opponent_elo")
        snapshot[f"opponent_surface_elo_avg_last_{window}"] = mean_attribute(
            sample, "opponent_surface_elo"
        )
        snapshot[f"quality_weighted_win_rate_last_{window}"] = weighted_mean_attribute(
            sample, "won", "opponent_elo"
        )
        snapshot[f"quality_weighted_point_win_rate_last_{window}"] = (
            weighted_mean_attribute(sample, "point_win_rate", "opponent_elo")
        )
        snapshot[f"quality_weighted_performance_vs_expectation_last_{window}"] = (
            weighted_mean_attribute(
                sample, "performance_vs_expectation", "opponent_elo"
            )
        )
        snapshot[f"point_win_rate_std_last_{window}"] = std_attribute(
            sample, "point_win_rate"
        )
        snapshot[f"performance_vs_expectation_std_last_{window}"] = std_attribute(
            sample, "performance_vs_expectation"
        )
        snapshot[f"avg_sets_played_last_{window}"] = mean_attribute(
            score_sample, "sets_played"
        )
        snapshot[f"tiebreak_win_rate_last_{window}"] = ratio_of_sums(
            score_sample, "tiebreaks_won", "tiebreaks_played"
        )
        snapshot[f"game_margin_avg_last_{window}"] = mean_attribute(
            score_sample, "game_margin"
        )
        snapshot[f"straight_set_win_pct_last_{window}"] = mean_attribute(
            score_sample, "straight_set_win"
        )
    return snapshot


def surface_form_snapshot(state: PlayerState, surface: str) -> dict[str, float]:
    """Compute surface-specific rolling window features."""
    snapshot: dict[str, float] = {}
    surface_history = [r for r in state.history if r.surface == surface]

    for window in SURFACE_FORM_WINDOWS:
        sample = surface_history[-window:]
        score_sample = [record for record in sample if not record.was_retirement]
        if not sample:
            for metric in (
                "surface_win_rate", "surface_service_points_won_rate",
                "surface_return_points_won_rate", "surface_point_win_rate",
                "surface_performance_vs_expectation", "surface_minutes_avg",
                "surface_avg_sets_played", "surface_tiebreak_win_rate",
                "surface_game_margin_avg", "surface_straight_set_win_pct",
            ):
                snapshot[f"{metric}_last_{window}"] = np.nan
            continue

        snapshot[f"surface_win_rate_last_{window}"] = mean_attribute(sample, "won")
        snapshot[f"surface_service_points_won_rate_last_{window}"] = mean_attribute(
            sample, "service_points_won_rate"
        )
        snapshot[f"surface_return_points_won_rate_last_{window}"] = mean_attribute(
            sample, "return_points_won_rate"
        )
        snapshot[f"surface_point_win_rate_last_{window}"] = mean_attribute(
            sample, "point_win_rate"
        )
        snapshot[f"surface_performance_vs_expectation_last_{window}"] = mean_attribute(
            sample, "performance_vs_expectation"
        )
        snapshot[f"surface_minutes_avg_last_{window}"] = mean_attribute(sample, "minutes")
        snapshot[f"surface_avg_sets_played_last_{window}"] = mean_attribute(
            score_sample, "sets_played"
        )
        snapshot[f"surface_tiebreak_win_rate_last_{window}"] = ratio_of_sums(
            score_sample, "tiebreaks_won", "tiebreaks_played"
        )
        snapshot[f"surface_game_margin_avg_last_{window}"] = mean_attribute(
            score_sample, "game_margin"
        )
        snapshot[f"surface_straight_set_win_pct_last_{window}"] = mean_attribute(
            score_sample, "straight_set_win"
        )
    return snapshot


def recent_activity_snapshot(
    state: PlayerState, match_date: pd.Timestamp
) -> dict[str, float]:
    """Compute recent activity features (matches/minutes in last N days)."""
    snapshot: dict[str, float] = {}
    for days in RECENT_ACTIVITY_WINDOWS_DAYS:
        active_sample = [
            record for record in state.history
            if 0 < (match_date - record.match_date).days <= days
        ]
        snapshot[f"matches_last_{days}_days"] = float(len(active_sample))
        snapshot[f"minutes_last_{days}_days"] = float(
            np.nansum([record.minutes for record in active_sample])
        )
        snapshot[f"win_rate_last_{days}_days"] = mean_attribute(active_sample, "won")
    return snapshot


def season_snapshot(
    state: PlayerState,
    surface: str,
    match_date: pd.Timestamp,
) -> dict[str, float]:
    """Compute current-season form features up to the pre-match snapshot."""
    season_history = [
        record for record in state.history
        if record.match_date.year == match_date.year
    ]
    surface_season_history = [
        record for record in season_history if record.surface == surface
    ]
    season_matches = float(len(season_history))
    season_surface_matches = float(len(surface_season_history))
    return {
        "season_matches": season_matches,
        "season_win_rate": mean_attribute(season_history, "won"),
        "season_service_points_won_rate": mean_attribute(
            season_history, "service_points_won_rate"
        ),
        "season_return_points_won_rate": mean_attribute(
            season_history, "return_points_won_rate"
        ),
        "season_point_win_rate": mean_attribute(season_history, "point_win_rate"),
        "season_performance_vs_expectation": mean_attribute(
            season_history, "performance_vs_expectation"
        ),
        "season_opponent_elo_avg": mean_attribute(season_history, "opponent_elo"),
        "season_surface_matches": season_surface_matches,
        "season_surface_share": safe_ratio(season_surface_matches, season_matches),
        "season_surface_win_rate": mean_attribute(surface_season_history, "won"),
        "season_surface_point_win_rate": mean_attribute(
            surface_season_history, "point_win_rate"
        ),
        "season_surface_performance_vs_expectation": mean_attribute(
            surface_season_history, "performance_vs_expectation"
        ),
    }


def signed_streak(records: list[MatchStats]) -> float:
    """Signed consecutive-result streak: wins are positive, losses negative."""
    if not records:
        return 0.0
    last_result = int(records[-1].won)
    streak = 0
    for record in reversed(records):
        if int(record.won) != last_result:
            break
        streak += 1
    return float(streak if last_result == 1 else -streak)


def streak_snapshot(state: PlayerState, surface: str) -> dict[str, float]:
    """Compute overall and surface-specific signed streak features."""
    surface_history = [record for record in state.history if record.surface == surface]
    return {
        "current_streak": signed_streak(state.history),
        "surface_streak": signed_streak(surface_history),
    }


def hand_matchup_snapshot(
    state: PlayerState,
    surface: str,
    opponent_hand: str | None,
) -> dict[str, float]:
    """Summarize prior performance against the upcoming opponent's hand."""
    keys = (
        "vs_opp_hand_matches",
        "vs_opp_hand_win_rate",
        "vs_opp_hand_point_win_rate",
        "vs_opp_hand_performance_vs_expectation",
        "surface_vs_opp_hand_matches",
        "surface_vs_opp_hand_win_rate",
        "surface_vs_opp_hand_point_win_rate",
        "surface_vs_opp_hand_performance_vs_expectation",
    )
    if opponent_hand is None:
        return {key: np.nan for key in keys}

    opponent_hand_history = [
        record for record in state.history if record.opponent_hand == opponent_hand
    ]
    surface_opponent_hand_history = [
        record for record in opponent_hand_history if record.surface == surface
    ]
    return {
        "vs_opp_hand_matches": float(len(opponent_hand_history)),
        "vs_opp_hand_win_rate": mean_attribute(opponent_hand_history, "won"),
        "vs_opp_hand_point_win_rate": mean_attribute(
            opponent_hand_history, "point_win_rate"
        ),
        "vs_opp_hand_performance_vs_expectation": mean_attribute(
            opponent_hand_history, "performance_vs_expectation"
        ),
        "surface_vs_opp_hand_matches": float(len(surface_opponent_hand_history)),
        "surface_vs_opp_hand_win_rate": mean_attribute(
            surface_opponent_hand_history, "won"
        ),
        "surface_vs_opp_hand_point_win_rate": mean_attribute(
            surface_opponent_hand_history, "point_win_rate"
        ),
        "surface_vs_opp_hand_performance_vs_expectation": mean_attribute(
            surface_opponent_hand_history, "performance_vs_expectation"
        ),
    }


def format_snapshot(state: PlayerState, best_of: float) -> dict[str, float]:
    """Summarize prior player performance by match format."""
    career_win_rate = safe_ratio(state.total_wins, state.total_matches)
    bo5_win_rate = shrunk_binary_rate(
        state.best_of_5_wins,
        state.best_of_5_matches,
        career_win_rate,
        FORMAT_WIN_RATE_PRIOR_MATCHES,
    )
    bo3_win_rate = shrunk_binary_rate(
        state.best_of_3_wins,
        state.best_of_3_matches,
        career_win_rate,
        FORMAT_WIN_RATE_PRIOR_MATCHES,
    )
    if pd.isna(best_of):
        format_matches = np.nan
        format_win_rate = np.nan
        format_win_rate_gap = np.nan
    elif best_of >= 5.0:
        format_matches = float(state.best_of_5_matches)
        format_win_rate = bo5_win_rate
        format_win_rate_gap = bo5_win_rate - bo3_win_rate
    else:
        format_matches = float(state.best_of_3_matches)
        format_win_rate = bo3_win_rate
        format_win_rate_gap = bo3_win_rate - bo5_win_rate
    return {
        "best_of_5_matches": float(state.best_of_5_matches),
        "best_of_5_win_rate": bo5_win_rate,
        "best_of_3_matches": float(state.best_of_3_matches),
        "best_of_3_win_rate": bo3_win_rate,
        "best_of_5_win_rate_gap": bo5_win_rate - bo3_win_rate,
        "format_matches": format_matches,
        "format_win_rate": format_win_rate,
        "format_win_rate_gap": format_win_rate_gap,
    }


def ioc_matchup_snapshot(
    player_id: int,
    state: PlayerState,
    surface: str,
    opponent_ioc_bucket: str | None,
    same_nationality_matches: dict[int, int],
    same_nationality_wins: dict[int, int],
    ioc_bucket_matches: dict[tuple[int, str], int],
    ioc_bucket_wins: dict[tuple[int, str], int],
    surface_ioc_bucket_matches: dict[tuple[int, str, str], int],
    surface_ioc_bucket_wins: dict[tuple[int, str, str], int],
) -> dict[str, float]:
    """Summarize prior performance against the opponent's nationality bucket."""
    career_win_rate = safe_ratio(state.total_wins, state.total_matches)
    surface_matches = state.surface_matches.get(surface, 0)
    surface_win_rate = safe_ratio(state.surface_wins.get(surface, 0), surface_matches)
    same_matches = same_nationality_matches[player_id]
    same_rate = shrunk_binary_rate(
        same_nationality_wins[player_id],
        same_matches,
        career_win_rate,
        IOC_MATCHUP_PRIOR_MATCHES,
    )
    snapshot = {
        "same_nationality_matches": float(same_matches),
        "same_nationality_win_rate": same_rate,
        "same_nationality_win_rate_gap": same_rate - career_win_rate,
    }
    if opponent_ioc_bucket is None:
        snapshot.update(
            {
                "vs_opp_ioc_matches": np.nan,
                "vs_opp_ioc_win_rate": np.nan,
                "vs_opp_ioc_win_rate_gap": np.nan,
                "surface_vs_opp_ioc_matches": np.nan,
                "surface_vs_opp_ioc_win_rate": np.nan,
                "surface_vs_opp_ioc_win_rate_gap": np.nan,
            }
        )
        return snapshot

    matchup_matches = ioc_bucket_matches[(player_id, opponent_ioc_bucket)]
    matchup_rate = shrunk_binary_rate(
        ioc_bucket_wins[(player_id, opponent_ioc_bucket)],
        matchup_matches,
        career_win_rate,
        IOC_MATCHUP_PRIOR_MATCHES,
    )
    surface_matchup_matches = surface_ioc_bucket_matches[
        (player_id, surface, opponent_ioc_bucket)
    ]
    surface_matchup_rate = shrunk_binary_rate(
        surface_ioc_bucket_wins[(player_id, surface, opponent_ioc_bucket)],
        surface_matchup_matches,
        matchup_fallback_rate(matchup_rate, surface_win_rate, career_win_rate),
        IOC_MATCHUP_PRIOR_MATCHES,
    )
    snapshot.update(
        {
            "vs_opp_ioc_matches": float(matchup_matches),
            "vs_opp_ioc_win_rate": matchup_rate,
            "vs_opp_ioc_win_rate_gap": matchup_rate - career_win_rate,
            "surface_vs_opp_ioc_matches": float(surface_matchup_matches),
            "surface_vs_opp_ioc_win_rate": surface_matchup_rate,
            "surface_vs_opp_ioc_win_rate_gap": surface_matchup_rate - surface_win_rate,
        }
    )
    return snapshot


def player_snapshot(
    state: PlayerState,
    player_id: int,
    surface: str,
    match_date: pd.Timestamp,
    rankings: RankingsIndex,
    opponent_hand: str | None,
    best_of: float,
    opponent_ioc_bucket: str | None,
    same_nationality_matches: dict[int, int],
    same_nationality_wins: dict[int, int],
    ioc_bucket_matches: dict[tuple[int, str], int],
    ioc_bucket_wins: dict[tuple[int, str], int],
    surface_ioc_bucket_matches: dict[tuple[int, str, str], int],
    surface_ioc_bucket_wins: dict[tuple[int, str, str], int],
) -> dict[str, float]:
    """Build complete feature snapshot for a player before a match."""
    surface_matches = state.surface_matches.get(surface, 0)
    surface_wins = state.surface_wins.get(surface, 0)
    surface_elo_value = state.current_surface_elo(surface)
    surface_elo_shrunk = shrunk_surface_elo(state, surface)
    surface_serve_elo_value = state.current_surface_serve_elo(surface)
    surface_return_elo_value = state.current_surface_return_elo(surface)
    surface_serve_elo_shrunk = shrunk_surface_serve_elo(state, surface)
    surface_return_elo_shrunk = shrunk_surface_return_elo(state, surface)
    snapshot = {
        "elo": state.elo,
        "surface_elo": surface_elo_value,
        "surface_elo_shrunk": surface_elo_shrunk,
        "surface_elo_gap": surface_elo_value - state.elo,
        "serve_elo": state.serve_elo,
        "return_elo": state.return_elo,
        "serve_return_gap": state.serve_elo - state.return_elo,
        "surface_serve_elo": surface_serve_elo_value,
        "surface_return_elo": surface_return_elo_value,
        "surface_serve_elo_shrunk": surface_serve_elo_shrunk,
        "surface_return_elo_shrunk": surface_return_elo_shrunk,
        "surface_serve_return_gap": (
            surface_serve_elo_shrunk - surface_return_elo_shrunk
        ),
        "career_matches": float(state.total_matches),
        "career_win_rate": safe_ratio(state.total_wins, state.total_matches),
        "surface_matches": float(surface_matches),
        "surface_win_rate": safe_ratio(surface_wins, surface_matches),
        "days_since_last_match": days_since(state.last_match_date, match_date),
        "days_since_last_surface_match": days_since(
            state.last_surface_date.get(surface), match_date
        ),
    }
    snapshot.update(history_snapshot(state))
    snapshot.update(surface_form_snapshot(state, surface))
    snapshot.update(recent_activity_snapshot(state, match_date))
    snapshot.update(season_snapshot(state, surface, match_date))
    snapshot.update(streak_snapshot(state, surface))
    snapshot.update(hand_matchup_snapshot(state, surface, opponent_hand))
    snapshot.update(format_snapshot(state, best_of))
    snapshot.update(
        ioc_matchup_snapshot(
            player_id,
            state,
            surface,
            opponent_ioc_bucket,
            same_nationality_matches,
            same_nationality_wins,
            ioc_bucket_matches,
            ioc_bucket_wins,
            surface_ioc_bucket_matches,
            surface_ioc_bucket_wins,
        )
    )
    snapshot.update(rankings.snapshot(player_id, match_date))
    return snapshot


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def oriented_players(row: pd.Series) -> tuple[str, str]:
    """Deterministically assign player A/B roles to remove winner/loser bias.

    Uses (id, name) tuple comparison for stable ordering.
    """
    winner_key = (int(row["winner_id"]), str(row["winner_name"]))
    loser_key = (int(row["loser_id"]), str(row["loser_name"]))
    return ("winner", "loser") if winner_key < loser_key else ("loser", "winner")


def role_field(row: pd.Series, role: str, field_name: str) -> Any:
    """Get a field for a player by their role (winner/loser)."""
    prefix = "winner" if role == "winner" else "loser"
    return row[f"{prefix}_{field_name}"]


def optional_role_field(row: pd.Series, role: str, field_name: str) -> Any:
    """Get an optional role field, returning None when the column is absent."""
    prefix = "winner" if role == "winner" else "loser"
    return row.get(f"{prefix}_{field_name}")


def match_id(row: pd.Series) -> str:
    """Generate unique match identifier."""
    return f"{row['tourney_id']}-{int(row['match_num'])}"


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature building."""

    k_factor: float = DEFAULT_ELO_K


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_feature_frame(
    matches: pd.DataFrame,
    config: FeatureConfig | None = None,
    rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the complete feature DataFrame from raw match data.

    Processes matches in chronological order. For each match:
    1. Snapshot both players' pre-match state as features
    2. Update ELO and state AFTER feature extraction (strict temporal ordering)

    Args:
        matches: raw match DataFrame from data.load_matches()
        config: feature configuration (K-factor, etc.)
        rankings: optional rankings DataFrame for rank momentum features

    Returns:
        DataFrame with one row per match, features as diff/sum columns
    """
    if config is None:
        config = FeatureConfig()

    states: dict[int, PlayerState] = defaultdict(PlayerState)
    rankings_index = RankingsIndex.from_frame(rankings)
    head_to_head: dict[tuple[int, int], int] = defaultdict(int)
    surface_head_to_head: dict[tuple[int, int, str], int] = defaultdict(int)
    ioc_buckets = tracked_ioc_buckets(matches)
    same_nationality_matches: dict[int, int] = defaultdict(int)
    same_nationality_wins: dict[int, int] = defaultdict(int)
    ioc_bucket_matches: dict[tuple[int, str], int] = defaultdict(int)
    ioc_bucket_wins: dict[tuple[int, str], int] = defaultdict(int)
    surface_ioc_bucket_matches: dict[tuple[int, str, str], int] = defaultdict(int)
    surface_ioc_bucket_wins: dict[tuple[int, str, str], int] = defaultdict(int)
    tournament_matches: dict[tuple[int, str], int] = defaultdict(int)
    tournament_wins: dict[tuple[int, str], int] = defaultdict(int)
    tournament_last_dates: dict[tuple[int, str], pd.Timestamp] = {}
    service_point_win_rate_sum = 0.0
    service_point_win_rate_count = 0
    surface_service_point_win_rate_sums: dict[str, float] = defaultdict(float)
    surface_service_point_win_rate_counts: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    latest_match_date = pd.Timestamp(matches["match_date"].max())

    for row in matches.itertuples(index=False):
        series = pd.Series(row._asdict())
        surface = normalize_surface(series["surface"])
        tourney_name = str(series["tourney_name"])
        match_date = series["match_date"]
        winner_id = int(series["winner_id"])
        loser_id = int(series["loser_id"])

        winner_state = states[winner_id]
        loser_state = states[loser_id]
        best_of = numeric_or_nan(series.get("best_of"))

        # Deterministic orientation: player A/B assignment
        player_a_role, player_b_role = oriented_players(series)
        player_a_id = int(role_field(series, player_a_role, "id"))
        player_b_id = int(role_field(series, player_b_role, "id"))
        player_a_name = str(role_field(series, player_a_role, "name"))
        player_b_name = str(role_field(series, player_b_role, "name"))
        player_a_hand = normalize_hand(
            optional_role_field(series, player_a_role, "hand")
        )
        player_b_hand = normalize_hand(
            optional_role_field(series, player_b_role, "hand")
        )
        player_a_entry = normalize_entry(
            optional_role_field(series, player_a_role, "entry")
        )
        player_b_entry = normalize_entry(
            optional_role_field(series, player_b_role, "entry")
        )
        player_a_ioc = normalize_ioc(optional_role_field(series, player_a_role, "ioc"))
        player_b_ioc = normalize_ioc(optional_role_field(series, player_b_role, "ioc"))
        player_a_opp_ioc_bucket = ioc_bucket(player_b_ioc, ioc_buckets)
        player_b_opp_ioc_bucket = ioc_bucket(player_a_ioc, ioc_buckets)
        same_nationality = (
            player_a_ioc is not None
            and player_b_ioc is not None
            and player_a_ioc == player_b_ioc
        )
        label = 1 if player_a_role == "winner" else 0

        # Snapshot PRE-MATCH state (strict: no leakage)
        player_a_state = states[player_a_id]
        player_b_state = states[player_b_id]
        player_a_snapshot = player_snapshot(
            player_a_state,
            player_a_id,
            surface,
            match_date,
            rankings_index,
            player_b_hand,
            best_of,
            player_a_opp_ioc_bucket,
            same_nationality_matches,
            same_nationality_wins,
            ioc_bucket_matches,
            ioc_bucket_wins,
            surface_ioc_bucket_matches,
            surface_ioc_bucket_wins,
        )
        player_b_snapshot = player_snapshot(
            player_b_state,
            player_b_id,
            surface,
            match_date,
            rankings_index,
            player_a_hand,
            best_of,
            player_b_opp_ioc_bucket,
            same_nationality_matches,
            same_nationality_wins,
            ioc_bucket_matches,
            ioc_bucket_wins,
            surface_ioc_bucket_matches,
            surface_ioc_bucket_wins,
        )
        player_a_tourney_snapshot = tournament_snapshot(
            player_a_id,
            tourney_name,
            match_date,
            tournament_matches,
            tournament_wins,
            tournament_last_dates,
        )
        player_b_tourney_snapshot = tournament_snapshot(
            player_b_id,
            tourney_name,
            match_date,
            tournament_matches,
            tournament_wins,
            tournament_last_dates,
        )

        # Head-to-head record (also pre-match)
        h2h_a = head_to_head[(player_a_id, player_b_id)]
        h2h_b = head_to_head[(player_b_id, player_a_id)]
        surface_h2h_a = surface_head_to_head[(player_a_id, player_b_id, surface)]
        surface_h2h_b = surface_head_to_head[(player_b_id, player_a_id, surface)]
        draw_size = numeric_or_nan(series.get("draw_size"))
        round_name = str(series["round"])

        # Build row
        row_data: dict[str, Any] = {
            "match_id": match_id(series),
            "match_date": match_date,
            "tourney_name": tourney_name,
            "tourney_level": series["tourney_level"],
            "surface": surface,
            "round": round_name,
            "score": series["score"],
            "winner_name": series["winner_name"],
            "loser_name": series["loser_name"],
            "player_a_id": player_a_id,
            "player_a_name": player_a_name,
            "player_b_id": player_b_id,
            "player_b_name": player_b_name,
            "label": label,
            "age_diff": (
                numeric_or_nan(role_field(series, player_a_role, "age"))
                - numeric_or_nan(role_field(series, player_b_role, "age"))
            ),
            "height_diff": (
                numeric_or_nan(role_field(series, player_a_role, "ht"))
                - numeric_or_nan(role_field(series, player_b_role, "ht"))
            ),
            "seed_diff": (
                seed_to_numeric(role_field(series, player_a_role, "seed"))
                - seed_to_numeric(role_field(series, player_b_role, "seed"))
            ),
            "rank_edge": (
                numeric_or_nan(role_field(series, player_b_role, "rank"))
                - numeric_or_nan(role_field(series, player_a_role, "rank"))
            ),
            "rank_points_diff": (
                numeric_or_nan(role_field(series, player_a_role, "rank_points"))
                - numeric_or_nan(role_field(series, player_b_role, "rank_points"))
            ),
            "entry_q_diff": float(player_a_entry == "Q") - float(player_b_entry == "Q"),
            "entry_q_sum": float(player_a_entry == "Q") + float(player_b_entry == "Q"),
            "entry_wc_diff": float(player_a_entry == "WC") - float(player_b_entry == "WC"),
            "entry_wc_sum": float(player_a_entry == "WC") + float(player_b_entry == "WC"),
            "hand_left_diff": float(player_a_hand == "L") - float(player_b_hand == "L"),
            "hand_left_sum": float(player_a_hand == "L") + float(player_b_hand == "L"),
            "hand_unknown_sum": float(player_a_hand is None) + float(player_b_hand is None),
            "same_nationality": float(same_nationality),
            "hand_same_known": float(
                player_a_hand is not None
                and player_b_hand is not None
                and player_a_hand == player_b_hand
            ),
            "hand_lefty_matchup": float(
                {player_a_hand, player_b_hand} == {"L", "R"}
            ),
            "h2h_diff": float(h2h_a - h2h_b),
            "h2h_total": float(h2h_a + h2h_b),
            "surface_h2h_diff": float(surface_h2h_a - surface_h2h_b),
            "surface_h2h_total": float(surface_h2h_a + surface_h2h_b),
            "draw_size": draw_size,
            "draw_size_rounds": draw_size_to_rounds(draw_size),
            "best_of": best_of,
            "best_of_5": float(best_of >= 5.0) if pd.notna(best_of) else np.nan,
            "round_order": float(ROUND_ORDER.get(round_name, 0)),
            "round_stage_index": round_stage_index(round_name, draw_size),
            "round_stage_progress": round_stage_progress(round_name, draw_size),
        }

        # Diff and sum features from player snapshots
        for key, value in player_a_snapshot.items():
            row_data[f"{key}_diff"] = value - player_b_snapshot[key]
            row_data[f"{key}_sum"] = value + player_b_snapshot[key]
        for key, value in player_a_tourney_snapshot.items():
            row_data[f"{key}_diff"] = value - player_b_tourney_snapshot[key]
            row_data[f"{key}_sum"] = value + player_b_tourney_snapshot[key]

        rows.append(row_data)

        # --- POST-MATCH updates (after features are extracted) ---
        head_to_head[(winner_id, loser_id)] += 1
        surface_head_to_head[(winner_id, loser_id, surface)] += 1
        base_k_factor = surface_base_k_factor(surface)
        match_k_factor = effective_k_factor(
            base_k_factor, match_date, latest_match_date
        )
        overall_service_baseline = service_point_win_rate_baseline(
            service_point_win_rate_sum,
            service_point_win_rate_count,
        )
        surface_service_baseline = surface_service_point_win_rate_baseline(
            surface,
            overall_service_baseline,
            surface_service_point_win_rate_sums,
            surface_service_point_win_rate_counts,
        )
        winner_service_point_win_rate = service_points_won_rate_from_row(
            series, "winner"
        )
        loser_service_point_win_rate = service_points_won_rate_from_row(
            series, "loser"
        )
        parsed_score = parse_score(series.get("score"))

        # Compute ELO deltas for history stats
        winner_pre = winner_state.elo
        loser_pre = loser_state.elo
        winner_surface_pre = winner_state.current_surface_elo(surface)
        loser_surface_pre = loser_state.current_surface_elo(surface)
        winner_expected = elo_expected(winner_pre, loser_pre)
        loser_expected = 1.0 - winner_expected
        winner_post, loser_post = updated_elo(
            winner_pre, loser_pre, 1.0, match_k_factor
        )
        winner_surface_post, loser_surface_post = updated_elo(
            winner_surface_pre, loser_surface_pre, 1.0, match_k_factor
        )

        winner_stats = stats_from_row(
            series, "winner",
            surface=surface, match_date=match_date,
            elo_delta=winner_post - winner_pre,
            surface_elo_delta=winner_surface_post - winner_surface_pre,
            expected_win_prob=winner_expected,
            opponent_elo=loser_pre,
            opponent_surface_elo=loser_surface_pre,
            serve_elo_score=service_point_elo_score(
                winner_service_point_win_rate, overall_service_baseline
            ),
            surface_serve_elo_score=service_point_elo_score(
                winner_service_point_win_rate, surface_service_baseline
            ),
            score_features=score_features_for_role(parsed_score, "winner"),
        )
        loser_stats = stats_from_row(
            series, "loser",
            surface=surface, match_date=match_date,
            elo_delta=loser_post - loser_pre,
            surface_elo_delta=loser_surface_post - loser_surface_pre,
            expected_win_prob=loser_expected,
            opponent_elo=winner_pre,
            opponent_surface_elo=winner_surface_pre,
            serve_elo_score=service_point_elo_score(
                loser_service_point_win_rate, overall_service_baseline
            ),
            surface_serve_elo_score=service_point_elo_score(
                loser_service_point_win_rate, surface_service_baseline
            ),
            score_features=score_features_for_role(parsed_score, "loser"),
        )

        apply_match_result(
            winner_state, loser_state, surface, match_date,
            winner_stats, loser_stats, match_k_factor,
        )
        winner_ioc = normalize_ioc(series.get("winner_ioc"))
        loser_ioc = normalize_ioc(series.get("loser_ioc"))
        winner_opp_ioc_bucket = ioc_bucket(loser_ioc, ioc_buckets)
        loser_opp_ioc_bucket = ioc_bucket(winner_ioc, ioc_buckets)
        if same_nationality:
            same_nationality_matches[winner_id] += 1
            same_nationality_matches[loser_id] += 1
            same_nationality_wins[winner_id] += 1
        if winner_opp_ioc_bucket is not None:
            ioc_bucket_matches[(winner_id, winner_opp_ioc_bucket)] += 1
            ioc_bucket_wins[(winner_id, winner_opp_ioc_bucket)] += 1
            surface_ioc_bucket_matches[(winner_id, surface, winner_opp_ioc_bucket)] += 1
            surface_ioc_bucket_wins[(winner_id, surface, winner_opp_ioc_bucket)] += 1
        if loser_opp_ioc_bucket is not None:
            ioc_bucket_matches[(loser_id, loser_opp_ioc_bucket)] += 1
            surface_ioc_bucket_matches[(loser_id, surface, loser_opp_ioc_bucket)] += 1
        if pd.notna(best_of):
            if best_of >= 5.0:
                winner_state.best_of_5_matches += 1
                winner_state.best_of_5_wins += 1
                loser_state.best_of_5_matches += 1
            else:
                winner_state.best_of_3_matches += 1
                winner_state.best_of_3_wins += 1
                loser_state.best_of_3_matches += 1
        for service_point_win_rate in (
            winner_service_point_win_rate,
            loser_service_point_win_rate,
        ):
            if pd.isna(service_point_win_rate):
                continue
            service_point_win_rate_sum += float(service_point_win_rate)
            service_point_win_rate_count += 1
            surface_service_point_win_rate_sums[surface] += float(
                service_point_win_rate
            )
            surface_service_point_win_rate_counts[surface] += 1
        winner_tourney_key = (winner_id, tourney_name)
        loser_tourney_key = (loser_id, tourney_name)
        tournament_matches[winner_tourney_key] += 1
        tournament_matches[loser_tourney_key] += 1
        tournament_wins[winner_tourney_key] += 1
        tournament_last_dates[winner_tourney_key] = match_date
        tournament_last_dates[loser_tourney_key] = match_date

    features = pd.DataFrame(rows)
    features["match_date"] = pd.to_datetime(features["match_date"])
    return features
