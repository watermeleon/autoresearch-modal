"""Configuration constants and path management for tennis prediction pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# --- ELO parameters ---
DEFAULT_ELO_K = 48.0
ELO_START = 1500.0
SURFACE_PRIOR_MATCHES = 20.0
SURFACE_K_FACTORS = {
    "Hard": 32.0,
    "Clay": 28.0,
    "Grass": 36.0,
    "Indoor": 32.0,
}

# --- Data parameters ---
DEFAULT_START_YEAR = 1985

# --- Rolling window sizes for historical features ---
ROLLING_WINDOWS = (10, 25, 50, 100)
SURFACE_FORM_WINDOWS = (10, 25)
RECENT_ACTIVITY_WINDOWS_DAYS = (14, 30)
RANK_MOMENTUM_WINDOWS_DAYS = (28, 91)

# --- Train/test split defaults ---
DEFAULT_CUTOFF_DATE = "2025-12-31"
DEFAULT_TEST_EVENT = "Australian Open"
DEFAULT_TEST_YEAR = 2024
DEFAULT_TEST_MODE = "date"  # "date" = all matches after cutoff; "event" = single tournament

# --- Per-tour cutoff dates for balanced test sets ---
# ATP has dense 2026 data (607 matches after 2025-12-31).
# WTA has sparse early-2026 data (335 matches after 2025-12-31).
# Using an earlier WTA cutoff (2025-09-30 -> ~614 matches) balances the test
# sets so COMBINED_ROC_AUC weights both tours equally in sample count.
CUTOFF_DATES_BY_TOUR: dict[str, str] = {
    "atp": "2025-12-31",
    "wta": "2025-09-30",
}

# --- Model parameters ---
DEFAULT_RANDOM_STATE = 42

# --- Column classifications ---
CATEGORICAL_FEATURES = ("surface", "tourney_level", "round")
META_COLUMNS = (
    "match_id",
    "match_date",
    "tourney_name",
    "score",
    "winner_name",
    "loser_name",
    "player_a_id",
    "player_a_name",
    "player_b_id",
    "player_b_name",
    "label",
)
TARGET_COLUMN = "label"

# --- Tournament → host-country IOC code mapping ---
# Regular ATP/WTA tournaments, alphabetically sorted.
# Davis Cup ties are handled by tourney_country() below — the first IOC code
# in the tie name (e.g. "Davis Cup QLS R1: FRA vs BRA" → "FRA") is the home team.
TOURNEY_COUNTRY: dict[str, str] = {
    "'s-Hertogenbosch": "NED",
    "ATP Finals": "ITA",               # Turin since 2021
    "Abu Dhabi": "UAE",
    "Acapulco": "MEX",
    "Adelaide": "AUS",
    "Almaty": "KAZ",
    "Athens": "GRE",
    "Auckland": "NZL",
    "Austin": "USA",
    "Australian Open": "AUS",
    "Bad Homburg": "GER",
    "Barcelona": "ESP",
    "Basel": "SUI",
    "Bastad": "SWE",
    "Beijing": "CHN",
    "Berlin": "GER",
    "Bogota": "COL",
    "Brisbane": "AUS",
    "Brussels": "BEL",
    "Bucharest": "ROU",
    "Buenos Aires": "ARG",
    "Canada Masters": "CAN",            # alternates Montreal/Toronto
    "Charleston": "USA",
    "Chengdu": "CHN",
    "Chennai": "IND",
    "Cincinnati": "USA",
    "Cincinnati Masters": "USA",
    "Cleveland": "USA",
    "Dallas": "USA",
    "Delray Beach": "USA",
    "Doha": "QAT",
    "Dubai": "UAE",
    "Eastbourne": "GBR",
    "Geneva": "SUI",
    "Gstaad": "SUI",
    "Guadalajara": "MEX",
    "Guangzhou": "CHN",
    "Halle": "GER",
    "Hamburg": "GER",
    "Hangzhou": "CHN",
    "Hobart": "AUS",
    "Hong Kong": "HKG",
    "Houston": "USA",
    "Iasi": "ROU",
    "Indian Wells": "USA",
    "Indian Wells Masters": "USA",
    "Jiujiang": "CHN",
    "Kitzbuhel": "AUT",
    "Laver Cup": "USA",                 # rotates; 2025 San Francisco
    "Linz": "AUT",
    "Los Cabos": "MEX",
    "Madrid": "ESP",
    "Madrid Masters": "ESP",
    "Mallorca": "ESP",
    "Marrakech": "MAR",
    "Marseille": "FRA",
    "Merida": "MEX",
    "Metz": "FRA",
    "Miami": "USA",
    "Miami Masters": "USA",
    "Monte Carlo Masters": "MON",
    "Monterrey": "MEX",
    "Montpellier": "FRA",
    "Munich": "GER",
    "Next Gen ATP Finals": "KSA",       # Jeddah since 2023
    "Ningbo": "CHN",
    "Nottingham": "GBR",
    "Pan Pacific Open": "JPN",
    "Paris Masters": "FRA",
    "Prague": "CZE",
    "Queen's Club": "GBR",
    "Rabat": "MAR",
    "Rio de Janeiro": "BRA",
    "Riyadh Finals": "KSA",
    "Roland Garros": "FRA",
    "Rome": "ITA",
    "Rome Masters": "ITA",
    "Rotterdam": "NED",
    "Rouen": "FRA",
    "Santiago": "CHI",
    "Sao Paulo": "BRA",
    "Seoul": "KOR",
    "Shanghai": "CHN",
    "Singapore": "SGP",
    "Stockholm": "SWE",
    "Strasbourg": "FRA",
    "Stuttgart": "GER",
    "Tokyo": "JPN",
    "Toronto": "CAN",
    "US Open": "USA",
    "Umag": "CRO",
    "United Cup": "AUS",               # Sydney since 2023
    "Us Open": "USA",                   # alternate casing in some data files
    "Vienna": "AUT",
    "Washington": "USA",
    "Wimbledon": "GBR",
    "Winston-Salem": "USA",
    "Wuhan": "CHN",
}

# Regex for Davis Cup tie names: "Davis Cup ... : XXX vs YYY"
_DC_HOME_RE = re.compile(r"Davis Cup .+: (\w{2,3}) vs ")


def tourney_country(tourney_name: str) -> str | None:
    """Return 3-letter IOC code for a tournament, or None if unknown.

    Handles both static TOURNEY_COUNTRY entries and dynamic Davis Cup ties
    where the home team IOC code is embedded in the name.
    """
    if tourney_name in TOURNEY_COUNTRY:
        return TOURNEY_COUNTRY[tourney_name]
    m = _DC_HOME_RE.match(tourney_name)
    if m:
        return m.group(1)
    return None


# --- Tour definitions ---
TOURS = ("atp", "wta")
SACKMANN_REPOS = {
    "atp": "https://github.com/JeffSackmann/tennis_atp",
    "wta": "https://github.com/JeffSackmann/tennis_wta",
}
MATCH_FILE_PREFIXES = {
    "atp": "atp_matches",
    "wta": "wta_matches",
}
RANKING_FILE_PREFIXES = {
    "atp": "atp_rankings",
    "wta": "wta_rankings",
}


@dataclass(frozen=True)
class RepoPaths:
    """Resolved filesystem paths for a specific tour."""

    root: Path
    tour: str
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path

    @classmethod
    def from_root(cls, root: Path | None = None, tour: str = "wta") -> RepoPaths:
        """Build paths from project root and tour name."""
        resolved_root = (root or Path(__file__).resolve().parents[2]).resolve()
        data_dir = resolved_root / "data"
        return cls(
            root=resolved_root,
            tour=tour,
            data_dir=data_dir,
            raw_dir=data_dir / "raw",
            processed_dir=data_dir / "processed",
            models_dir=resolved_root / "models" / tour / "xgboost",
        )

    @property
    def raw_repo_dir(self) -> Path:
        """Path to the Sackmann git clone for this tour."""
        return self.raw_dir / f"tennis_{self.tour}"

    @property
    def features_parquet(self) -> Path:
        """Path to the processed features parquet file."""
        return self.processed_dir / f"{self.tour}_features_strict.parquet"

    def ensure(self) -> None:
        """Create all directories if they don't exist."""
        for path in (self.data_dir, self.raw_dir, self.processed_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)
