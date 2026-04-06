# Validation Data

Validation sets for forward-testing model predictions on unseen tournaments.

## Current Setup

Validation is built into the train/test split:
- **Holdout:** Australian Open 2024
- **Cutoff:** 2023-12-31 (all training data before this date)

The holdout event is extracted from the Sackmann data by matching tournament name and year.

## Indian Wells 2026 (Scraped)

Two files scraped and normalized into Sackmann-compatible CSV format:

- `indian_wells_2026_atp.csv` — ATP BNP Paribas Open 2026 (93 matches, R128 through SF)
- `indian_wells_2026_wta.csv` — WTA BNP Paribas Open 2026 (96 matches, R128 through F)

### Source & Date

- **Source:** tennisexplorer.com (results pages)
- **Date scraped:** 2026-03-14
- **Script:** `scripts/scrape_indian_wells_2026.py`

### Tournament Metadata

| Field | ATP | WTA |
|-------|-----|-----|
| tourney_id | 2026-0404 | 2026-609 |
| tourney_name | Indian Wells Masters | Indian Wells |
| surface | Hard | Hard |
| draw_size | 128 | 128 |
| tourney_level | M (Masters) | PM (Premier Mandatory) |
| tourney_date | 20260304 | 20260306 |
| best_of | 3 | 3 |

### What's Populated

- tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date
- match_num (reverse-chronological: Final = 1, earliest R128 = highest)
- winner/loser: name, id (from Sackmann historical DB), seed, entry (WC/Q/LL)
- winner/loser: hand, height, IOC country code (from historical player DB)
- score, best_of, round

### What's Empty

- All serve/return stats (w_ace, w_df, w_svpt, etc.) — not available from results-only scraping
- minutes (match duration)
- winner_age, loser_age (could be computed from birth dates but not done)
- winner_rank, winner_rank_points, loser_rank, loser_rank_points (current rankings not scraped)

### Caveats

1. **ATP Final not played yet** — tournament through SF only (March 14, 2026; Final is March 15)
2. **WTA Final included** — Sabalenka d. Rybakina 6-3 6-4
3. **Bracket inconsistencies** — The WTA data has minor bracket inconsistencies due to scraping artifacts (5 QFs instead of 4, 9 R16 instead of 8). These don't affect the ML model which uses individual match results, not bracket structure.
4. **Player ID matching** — 4 ATP and 3 WTA players have no Sackmann ID (new/minor tour players not in historical data). These have empty winner_id/loser_id fields.
5. **Name normalization** — Player names were mapped to Sackmann canonical forms via aliases. Some edge cases may have incorrect mappings.

### Adding New Tournaments

To add a new validation set:

1. Create a CSV matching the Sackmann schema (49 columns, same header as `atp_matches_YYYY.csv`)
2. Place it here as `{event}_{year}_{tour}.csv`
3. Pass `--test-event "Indian Wells" --test-year 2026` to the CLI
