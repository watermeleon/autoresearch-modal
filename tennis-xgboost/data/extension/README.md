# Extension Match Data (2025-2026)

Jeff Sackmann's `tennis_atp` and `tennis_wta` repos are updated throughout each season but can lag behind by weeks or months. These extension files fill the gap with 2025 late-season and 2026 early-season matches that the pipeline needs for its validation split.

## Files

| File | Rows | Period | Tour |
|------|------|--------|------|
| `atp_matches_2025.csv` | ~2,944 | Full 2025 season | ATP |
| `atp_matches_2026.csv` | ~518 | Jan-Mar 2026 | ATP |
| `wta_matches_2025.csv` | ~2,416 | Full 2025 season | WTA |
| `wta_matches_2026.csv` | ~246 | Jan-Mar 2026 | WTA |

## Schema

All files follow the Sackmann CSV schema (49 columns for ATP, 49 for WTA). They can be dropped directly into `data/raw/tennis_atp/` and `data/raw/tennis_wta/` respectively and the pipeline will pick them up.

## Provenance

The bulk of each file comes from Sackmann's repos as of March 2026. Gaps were filled using:

- **TML-Database** (github.com/Tennismylife/TML-Database): ATP 2025-2026 matches converted to Sackmann schema via `scripts/convert_tml_to_sackmann.py`
- **tennisexplorer.com**: ATP Feb-Mar 2026 results (see `scripts/fill_atp_feb_mar_2026.py`)
- **tennisabstract.com**: WTA Finals 2025, late-season WTA matches (see `scripts/fill_wta_nov_dec_2025.py`)

All manually-sourced matches were cross-verified against multiple sources.

## License

The underlying match data is derived from Jeff Sackmann's tennis_atp and tennis_wta repositories, licensed under CC BY-NC-SA 4.0. The same license applies to these extension files.
