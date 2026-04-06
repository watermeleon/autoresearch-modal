"""Tests for feature engineering."""

from __future__ import annotations

import pandas as pd

from tennis_predict.features import FeatureConfig, build_feature_frame


def sample_matches() -> pd.DataFrame:
    """Create minimal match data for testing."""
    rows = [
        {
            "tourney_id": "2024-TEST-1",
            "tourney_name": "Sample Slam",
            "surface": "Hard",
            "tourney_level": "G",
            "tourney_date": 20240101,
            "match_num": 1,
            "winner_id": 100,
            "winner_name": "Alice Ace",
            "winner_age": 24.0,
            "winner_ht": 180.0,
            "winner_seed": 1,
            "winner_rank": 5,
            "winner_rank_points": 4000,
            "loser_id": 200,
            "loser_name": "Bella Baseline",
            "loser_age": 25.0,
            "loser_ht": 175.0,
            "loser_seed": 4,
            "loser_rank": 18,
            "loser_rank_points": 2300,
            "score": "6-4 6-4",
            "round": "R32",
            "minutes": 90,
            "w_ace": 5, "w_df": 2, "w_svpt": 70, "w_1stIn": 40,
            "w_1stWon": 30, "w_2ndWon": 15, "w_SvGms": 10,
            "w_bpSaved": 3, "w_bpFaced": 4,
            "l_ace": 3, "l_df": 4, "l_svpt": 68, "l_1stIn": 38,
            "l_1stWon": 24, "l_2ndWon": 14, "l_SvGms": 10,
            "l_bpSaved": 2, "l_bpFaced": 5,
        },
        {
            "tourney_id": "2024-TEST-1",
            "tourney_name": "Sample Slam",
            "surface": "Hard",
            "tourney_level": "G",
            "tourney_date": 20240103,
            "match_num": 2,
            "winner_id": 200,
            "winner_name": "Bella Baseline",
            "winner_age": 25.0,
            "winner_ht": 175.0,
            "winner_seed": 4,
            "winner_rank": 18,
            "winner_rank_points": 2300,
            "loser_id": 100,
            "loser_name": "Alice Ace",
            "loser_age": 24.0,
            "loser_ht": 180.0,
            "loser_seed": 1,
            "loser_rank": 5,
            "loser_rank_points": 4000,
            "score": "4-6 6-3 6-2",
            "round": "R16",
            "minutes": 125,
            "w_ace": 6, "w_df": 3, "w_svpt": 80, "w_1stIn": 45,
            "w_1stWon": 33, "w_2ndWon": 18, "w_SvGms": 12,
            "w_bpSaved": 6, "w_bpFaced": 8,
            "l_ace": 8, "l_df": 5, "l_svpt": 82, "l_1stIn": 48,
            "l_1stWon": 32, "l_2ndWon": 16, "l_SvGms": 12,
            "l_bpSaved": 5, "l_bpFaced": 9,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["match_date"] = pd.to_datetime(frame["tourney_date"].astype(str), format="%Y%m%d")
    return frame


def sample_rankings() -> pd.DataFrame:
    """Create minimal rankings data for testing."""
    frame = pd.DataFrame([
        {"ranking_date": 20231002, "player_id": 100, "rank": 10, "points": 3200},
        {"ranking_date": 20231204, "player_id": 100, "rank": 8, "points": 3600},
        {"ranking_date": 20240101, "player_id": 100, "rank": 5, "points": 4000},
        {"ranking_date": 20231002, "player_id": 200, "rank": 28, "points": 1700},
        {"ranking_date": 20231204, "player_id": 200, "rank": 22, "points": 2000},
        {"ranking_date": 20240101, "player_id": 200, "rank": 18, "points": 2300},
    ])
    frame["ranking_date"] = pd.to_datetime(
        frame["ranking_date"].astype(str), format="%Y%m%d"
    )
    return frame


def test_strict_mode_uses_pre_match_elo() -> None:
    """First match should have zero ELO diff (both players start at 1500)."""
    features = build_feature_frame(
        sample_matches(),
        config=FeatureConfig(k_factor=32.0),
    )
    first_row = features.iloc[0]
    assert first_row["elo_diff"] == 0.0
    assert first_row["surface_elo_diff"] == 0.0


def test_head_to_head_available_before_second_match() -> None:
    """After first match, h2h should be available for second."""
    features = build_feature_frame(
        sample_matches(),
        config=FeatureConfig(k_factor=32.0),
    )
    second_row = features.iloc[1]
    assert second_row["h2h_total"] == 1.0


def test_features_include_shrunk_elo() -> None:
    """Features should contain surface_elo_shrunk_diff."""
    features = build_feature_frame(
        sample_matches(),
        config=FeatureConfig(k_factor=32.0),
    )
    first_row = features.iloc[0]
    assert "surface_elo_shrunk_diff" in features.columns
    assert first_row["surface_elo_shrunk_diff"] == 0.0


def test_features_include_surface_and_rank_momentum() -> None:
    """Features should contain surface form and rank momentum columns."""
    features = build_feature_frame(
        sample_matches(),
        config=FeatureConfig(k_factor=32.0),
        rankings=sample_rankings(),
    )
    first_row = features.iloc[0]
    second_row = features.iloc[1]

    assert "surface_win_rate_last_10_diff" in features.columns
    assert "matches_last_14_days_diff" in features.columns
    assert "rank_points_change_28_days_diff" in features.columns
    assert first_row["matches_last_14_days_diff"] == 0.0
    assert first_row["rank_points_change_28_days_diff"] == 100.0
    assert second_row["surface_win_rate_last_10_diff"] == 1.0


def test_label_is_binary() -> None:
    """Label should be 0 or 1."""
    features = build_feature_frame(
        sample_matches(),
        config=FeatureConfig(k_factor=32.0),
    )
    assert set(features["label"].unique()).issubset({0, 1})
