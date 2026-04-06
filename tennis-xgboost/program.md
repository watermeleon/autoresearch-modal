# program.md — Tennis Match Prediction Auto-Research

## Objective

Maximize `COMBINED_ROC_AUC = (ATP_ROC_AUC + WTA_ROC_AUC) / 2` on a 2026 date-split validation set. Higher is better. Single scalar. That's the only number that matters.

**Current best:** 0.7454 (ATP: 0.7473, WTA: 0.7435) — after 8 iterations of ELO tuning.
**Baseline:** 0.7377 (ATP: 0.7472, WTA: 0.7282)
**Target:** 0.76+. Stretch: 0.78+. Theoretical ceiling ~0.80-0.82 (intrinsic match variance).

## Evaluation

```bash
bash gate.sh
# Outputs: COMBINED_ROC_AUC=0.XXXX
# Runs both tours, averages ROC-AUC, enforces all guard rails
# Deterministic: XGBoost random_state=42, tree_method="hist"
# Must complete in < 5 minutes total
```

## What You Can Change

| File | Arena | Hard Constraints |
|------|-------|-----------------|
| `src/tennis_predict/config.py` | K-factors, windows, priors, constants | Keep `META_COLUMNS`, `TARGET_COLUMN`, `TOURS`, `RepoPaths` |
| `src/tennis_predict/elo.py` | Rating system — algorithms, K-logic, state tracking, new rating approaches (Glicko, TrueSkill, etc.) | Keep function signatures: `elo_expected()`, `updated_elo()`, `apply_match_result()`. `PlayerState` stays a dataclass |
| `src/tennis_predict/features.py` | Features — add, remove, transform, combine. This is where most wins live | Keep `build_feature_frame()` signature. Keep player A/B orientation. Keep strict temporal ordering (pre-match snapshot, post-match update) |
| `src/tennis_predict/models.py` | Model — hyperparameters, architecture, ensembles, calibration, feature selection, training logic | Keep `train_and_report()` signature. Evaluation logic lives in `evaluate.py` (immutable). Output must include `roc_auc`, `accuracy`, `brier_score`, `log_loss` |
| `pyproject.toml` | Dependencies only | Don't change package name, Python version, or build system |

## What You Cannot Change

`data.py`, `cli.py`, `evaluate.py`, `gate.sh`, `run-research.sh`, `RESEARCH_LOG.md`, `data/raw/**`, `data/validation/**`, `tests/**`. These are structural. Touching them invalidates everything.

**Why `evaluate.py` is immutable:** It contains all scoring/metric computation (`evaluate_model`, `extract_feature_importances`, metric calculations). Previously this logic lived in `models.py` (mutable), which allowed agents to inject post-hoc probability adjustments into the evaluation path to game validation scores. Extracting evaluation into an immutable file closes that attack vector. `gate.sh` enforces immutability via git diff check and also runs prediction sanity checks (probability range, mean, std) to catch gaming attempts in the model's predict path.

## The System You're Improving

Tennis match prediction via XGBoost with ELO-based features. Two separate models (ATP/WTA). ~132K ATP training matches, ~112K WTA, spanning 1985-2025.

**Validation split:** Per-tour temporal cutoffs for balanced test sets:
- ATP: matches after 2025-12-31 (~607 test matches)
- WTA: matches after 2025-09-30 (~614 test matches)

The WTA cutoff is earlier because WTA has sparser early-2026 data (only 335 matches after 2025-12-31). Using per-tour cutoffs ensures COMBINED_ROC_AUC weights both tours equally by sample count. Cutoff dates are configured in `config.py` via `CUTOFF_DATES_BY_TOUR`.

**Current architecture:**
- ELO engine: K=48, surface-specific K (Hard=32, Clay=28, Grass=36), tournament-level K, recency-weighted K, Bayesian surface shrinkage (prior=20 matches)
- Features (~230 after one-hot): ELO diffs, surface ELO, career stats, rolling windows (10/25/50/100), surface form, recent activity, rank momentum, H2H, age/height/seed diffs, categoricals (surface, tourney_level, round)
- Model: XGBoost 500 trees, depth=4, lr=0.05, subsample=0.85

**Read the source files.** Understand what's there before changing anything. Profile what's weak. Form your own hypotheses. Implement what you believe will improve ROC-AUC.

## Research Directions (not prescriptions)

These are ideas, not assignments. Use your judgment. Try what seems most promising based on the code you read and the research log.

**Rating system:** Glicko-2 (rating deviation as uncertainty), margin-of-victory adjustments, ELO volatility tracking, temporal decay toward mean for inactive players, momentum-weighted K.

**Features:** H2H on specific surfaces, tournament venue affinity (Nadal at Roland Garros), fatigue (days since last match, matches in 7 days), win/loss streaks, age-surface interactions, opponent quality weighting, set-level performance, draw difficulty, serve/return composites, handedness matchup effects.

**Model:** Depth/lr/n_estimators sweeps, LightGBM comparison, feature selection (drop importance < 0.001), calibration (Platt or isotonic), per-surface or per-round specialized models, ensemble/stacking.

**Data engineering:** Retirement/walkover filtering from rolling stats, indoor/outdoor proxy, best-of-3 vs best-of-5 flag.

## Synthetic Metrics — High-Value Research Direction

The biggest gains in this domain come from better signal extraction, not narrower segmentation. Invest in novel features and synthetic metrics that capture tennis dynamics the current feature set misses. Think like a domain expert who watches tennis, not like an optimizer who watches the scoreboard.

**Synthetic metric ideas to explore:**

- **Momentum ELO:** ELO that weighs recent matches more heavily via exponential decay. A player's rating 3 months ago matters less than their rating from the last 2 weeks. Implement as a separate ELO track with a decay half-life parameter.
- **Fatigue-adjusted ELO:** Factor in tournament schedule density and travel distance between events. A player coming off 3 tournaments in 3 weeks on 3 continents is not the same as a rested player.
- **Clutch ELO:** Separate rating for performance in decisive sets and tiebreaks. Some players systematically over- or under-perform in high-pressure moments. This is a real, measurable signal.
- **Surface transition penalty:** How much a player's form drops when switching surfaces. A clay-to-grass transition is brutal for most baseliners. Measure the delta between a player's surface-specific ELO and their performance in the first N matches after a surface switch.
- **Form volatility:** Standard deviation of recent results (win/loss, margin) as a feature. A player with volatile recent form is harder to predict — the model should know this.
- **Serve dominance composite:** First-serve percentage, ace rate, and service game hold rate combined into a single metric. If the raw stats are available, this compresses serve-side signal.

These are not exhaustive. If you see a tennis dynamic that the current features do not capture, build a metric for it.

### Anti-Gaming Clause

Any iteration that adds segment specialists (tournament-level, tournament-name, or narrower carve-outs) MUST also demonstrate that the global model's ROC-AUC improved, not just the blended score. If global model score drops while blended score rises, that iteration is gaming the blend weights and should be rejected. The gate enforces COMBINED_ROC_AUC as a single scalar — do not try to decompose it into segments that individually look better while the aggregate stagnates.

## Dead Ends — Do Not Retry

| Approach | Why |
|----------|-----|
| Post-match ELO as feature | Temporal leakage. CRITICAL — inflates by 5-10% |
| match_num for ordering | Convention changed in 2025. Fixed in data.py |
| Height features on WTA | ~40% missing. Noise > signal |
| Merged ATP/WTA model | Structural tour differences add noise |
| ELO_START tuning (1400-1600) | Cancels out in diffs. No impact |
| MLP / RandomForest / DecisionTree | XGBoost dominates. Tested |
| Removing one-hot categoricals | Surface/tourney_level/round matter |

## Guard Rails

- Training time < 5 min per tour
- Model size < 100MB per tour
- Feature count < 500 after one-hot
- `pytest` must pass
- No test rows in training, no training rows in test
- Memory < 8GB during pipeline

## Research Log

Read `RESEARCH_LOG.md` before every iteration. It contains what was tried, what worked, what didn't. Learn from it. Don't repeat failures. Build on successes.

Your changes will be evaluated by `gate.sh`. If ROC-AUC improves, your change is committed. If not, it's rolled back. Make every iteration count.
