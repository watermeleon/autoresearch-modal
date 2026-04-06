"""Data loading and synchronization for ATP and WTA Sackmann repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from tennis_predict.config import (
    DEFAULT_START_YEAR,
    MATCH_FILE_PREFIXES,
    RANKING_FILE_PREFIXES,
    SACKMANN_REPOS,
)

# Round-to-ordinal mapping for chronological sorting within a tournament.
#
# WHY THIS EXISTS: Sackmann's match_num encoding changed in 2025. Before 2025,
# finals had the highest match_num (~300), so ascending sort by match_num
# produced chronological order (R128 → F). From 2025 onward, finals have
# match_num=1 and early rounds have high numbers, so ascending match_num sort
# produces REVERSE chronological order (F → R128). This causes temporal
# leakage: ELO updates from later rounds contaminate features for earlier
# rounds within the same tournament.
#
# Fix: sort by round ordinal (early rounds first, final last) instead of
# match_num. match_num is used only as a tiebreaker within the same round.
# This is convention-agnostic and handles any future match_num changes.
ROUND_ORDER: dict[str, int] = {
    "RR": 0,   # Round robin (group stage), before knockouts
    "BR": 0,   # Bronze medal match — concurrent with final
    "ER": 0,   # Early rounds / qualifying overflow
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "R4": 5,   # 4th round in WTA 1000 64-draws (between R16 and QF)
    "QF": 6,
    "SF": 7,
    "F": 8,
}


def sync_repo(tour: str, destination: Path, refresh: bool = False) -> Path:
    """Clone or pull a Sackmann tennis data repository.

    Args:
        tour: 'atp' or 'wta'
        destination: target directory for the git clone
        refresh: if True, pull latest changes
    """
    destination = destination.resolve()
    if destination.exists() and (destination / ".git").exists():
        if refresh:
            subprocess.run(
                ["git", "-C", str(destination), "pull", "--ff-only"], check=True
            )
        return destination

    url = SACKMANN_REPOS[tour]
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(destination)], check=True
    )
    return destination


def main_match_files(
    raw_repo_dir: Path, tour: str, start_year: int = DEFAULT_START_YEAR
) -> list[Path]:
    """Find main-tour match CSV files, excluding qualifiers and ITF."""
    prefix = MATCH_FILE_PREFIXES[tour]
    candidates = sorted(raw_repo_dir.glob(f"{prefix}_*.csv"))
    files: list[Path] = []
    for path in candidates:
        stem = path.stem
        if "qual" in stem or "itf" in stem or "futures" in stem or "chall" in stem:
            continue
        year_token = stem.split("_")[-1]
        if not year_token.isdigit():
            continue
        if int(year_token) < start_year:
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No main-tour {tour.upper()} match files found in {raw_repo_dir}"
        )
    return files


def load_matches(
    raw_repo_dir: Path, tour: str, start_year: int = DEFAULT_START_YEAR
) -> pd.DataFrame:
    """Load and clean match data from Sackmann CSV files.

    Enforces temporal ordering by match_date, tourney_name, and round ordinal.
    Uses ROUND_ORDER to sort by round stage (R128 before R64 before ... before F)
    with match_num as a secondary tiebreaker within the same round. This avoids
    temporal leakage from Sackmann's 2025+ match_num convention change where
    finals have match_num=1 (lowest) instead of the highest value.
    """
    frames: list[pd.DataFrame] = []
    for path in main_match_files(raw_repo_dir, tour, start_year=start_year):
        frame = pd.read_csv(path, low_memory=False)
        frame["source_year"] = int(path.stem.split("_")[-1])
        frames.append(frame)

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.dropna(
        subset=["winner_id", "loser_id", "tourney_date", "match_num"]
    ).copy()

    integerish = [
        "winner_id", "loser_id", "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points", "winner_seed", "loser_seed",
        "match_num", "minutes",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
    ]
    floatish = ["winner_age", "loser_age", "winner_ht", "loser_ht"]

    for column in integerish + floatish:
        if column in matches.columns:
            matches[column] = pd.to_numeric(matches[column], errors="coerce")

    matches["winner_id"] = matches["winner_id"].astype("int64")
    matches["loser_id"] = matches["loser_id"].astype("int64")
    matches["match_num"] = matches["match_num"].astype("int64")
    # tourney_date can be float (e.g. 20240101.0) in some data files
    matches["tourney_date"] = pd.to_numeric(matches["tourney_date"], errors="coerce")
    matches = matches.dropna(subset=["tourney_date"]).copy()
    matches["match_date"] = pd.to_datetime(
        matches["tourney_date"].astype(int).astype(str), format="%Y%m%d"
    )
    matches["surface"] = matches["surface"].fillna("Unknown")
    matches["tourney_name"] = matches["tourney_name"].fillna("Unknown")
    matches["round"] = matches["round"].fillna("Unknown")
    matches["tourney_level"] = matches["tourney_level"].fillna("Unknown")

    matches["_round_ord"] = matches["round"].map(ROUND_ORDER).fillna(0).astype(int)
    matches = matches.sort_values(
        by=["match_date", "tourney_name", "_round_ord", "match_num", "winner_name", "loser_name"],
        kind="mergesort",
    ).reset_index(drop=True)
    matches = matches.drop(columns=["_round_ord"])
    return matches


def ranking_files(raw_repo_dir: Path, tour: str) -> list[Path]:
    """Find ranking CSV files for a given tour."""
    prefix = RANKING_FILE_PREFIXES[tour]
    files = sorted(raw_repo_dir.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No {tour.upper()} rankings files found in {raw_repo_dir}"
        )
    return files


def load_rankings(
    raw_repo_dir: Path, tour: str, start_year: int = DEFAULT_START_YEAR
) -> pd.DataFrame:
    """Load and clean rankings data from Sackmann CSV files."""
    frames: list[pd.DataFrame] = []
    for path in ranking_files(raw_repo_dir, tour):
        frame = pd.read_csv(
            path,
            usecols=["ranking_date", "rank", "player", "points"],
            low_memory=False,
        )
        frames.append(frame)

    rankings = pd.concat(frames, ignore_index=True)
    rankings = rankings.rename(columns={"player": "player_id"})
    for column in ("player_id", "rank", "points"):
        rankings[column] = pd.to_numeric(rankings[column], errors="coerce")

    rankings["ranking_date"] = pd.to_datetime(
        rankings["ranking_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    rankings = rankings.dropna(subset=["player_id", "ranking_date", "rank"]).copy()

    minimum_date = pd.Timestamp(year=max(start_year - 1, 1968), month=1, day=1)
    rankings = rankings.loc[rankings["ranking_date"] >= minimum_date].copy()
    rankings["player_id"] = rankings["player_id"].astype("int64")
    rankings = rankings.sort_values(
        ["player_id", "ranking_date", "rank"], kind="mergesort"
    )
    rankings = rankings.drop_duplicates(
        subset=["player_id", "ranking_date"], keep="first"
    )
    return rankings.reset_index(drop=True)


def event_slice(
    features: pd.DataFrame, event_name: str, year: int
) -> pd.DataFrame:
    """Extract rows matching a tournament name and year."""
    names = features["tourney_name"].fillna("")
    years = pd.to_datetime(features["match_date"]).dt.year
    mask = names.str.contains(event_name, case=False, na=False) & (years == year)
    return features.loc[mask].copy()


def date_slice(features: pd.DataFrame, after_date: str) -> pd.DataFrame:
    """Extract all rows with match_date strictly after the given date.

    Args:
        features: DataFrame with 'match_date' column.
        after_date: cutoff date string (e.g. '2025-12-31').

    Returns:
        DataFrame of matches after the cutoff date.
    """
    cutoff = pd.Timestamp(after_date)
    mask = pd.to_datetime(features["match_date"]) > cutoff
    return features.loc[mask].copy()


def load_validation_csvs(validation_dir: Path, tour: str) -> pd.DataFrame | None:
    """Load supplementary validation CSV files for a tour.

    Looks for any CSV in the validation directory matching *_{tour}.csv.
    Returns concatenated DataFrame, or None if no files found.
    """
    pattern = f"*_{tour}.csv"
    files = sorted(validation_dir.glob(pattern))
    if not files:
        return None
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(frames, ignore_index=True)


def load_matches_with_validation(
    raw_repo_dir: Path,
    tour: str,
    validation_dir: Path | None = None,
    start_year: int = DEFAULT_START_YEAR,
) -> pd.DataFrame:
    """Load main match data and merge in validation CSVs with deduplication.

    This extends load_matches() to also include matches from validation files
    (e.g. scraped tournament data not yet in the Sackmann repo).

    Deduplication is by (tourney_id, match_num) to avoid double-counting.
    """
    matches = load_matches(raw_repo_dir, tour, start_year=start_year)

    if validation_dir is None:
        return matches

    extra = load_validation_csvs(validation_dir, tour)
    if extra is None or extra.empty:
        return matches

    # Apply the same cleaning as load_matches
    extra = extra.dropna(
        subset=["winner_id", "loser_id", "tourney_date", "match_num"]
    ).copy()

    integerish = [
        "winner_id", "loser_id", "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points", "winner_seed", "loser_seed",
        "match_num", "minutes",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
    ]
    floatish = ["winner_age", "loser_age", "winner_ht", "loser_ht"]
    for column in integerish + floatish:
        if column in extra.columns:
            extra[column] = pd.to_numeric(extra[column], errors="coerce")

    extra["winner_id"] = extra["winner_id"].astype("int64")
    extra["loser_id"] = extra["loser_id"].astype("int64")
    extra["match_num"] = extra["match_num"].astype("int64")
    extra["tourney_date"] = pd.to_numeric(extra["tourney_date"], errors="coerce")
    extra = extra.dropna(subset=["tourney_date"]).copy()
    extra["match_date"] = pd.to_datetime(
        extra["tourney_date"].astype(int).astype(str), format="%Y%m%d"
    )
    extra["surface"] = extra["surface"].fillna("Unknown")
    extra["tourney_name"] = extra["tourney_name"].fillna("Unknown")
    extra["round"] = extra["round"].fillna("Unknown")
    extra["tourney_level"] = extra["tourney_level"].fillna("Unknown")

    pre_dedup = len(matches) + len(extra)
    combined = pd.concat([matches, extra], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["tourney_id", "match_num"], keep="first"
    )
    combined["_round_ord"] = combined["round"].map(ROUND_ORDER).fillna(0).astype(int)
    combined = combined.sort_values(
        by=["match_date", "tourney_name", "_round_ord", "match_num", "winner_name", "loser_name"],
        kind="mergesort",
    ).reset_index(drop=True)
    combined = combined.drop(columns=["_round_ord"])

    added = len(combined) - len(matches)
    print(f"Merged validation data: {added} new matches added ({pre_dedup - len(combined)} duplicates removed)")
    return combined
