# Research Log

Append-only log of auto-research iterations. Each entry records what changed, why, and whether it improved the metric.

---

## Iteration 0 -- BASELINE
- **Phase:** Setup
- **Change:** Initial baseline established with date-split evaluation (train <=2025, test 2026), round-ordinal ordering fix, validation CSVs, 2025-2026 data pipeline
- **Hypothesis:** Establish honest baseline with strict temporal integrity for the auto-research ratchet
- **Files modified:** all (initial build)
- **ATP ROC-AUC:** 0.7472
- **WTA ROC-AUC:** 0.7282
- **Combined ROC-AUC:** 0.7377
- **ATP Accuracy:** 0.6870
- **WTA Accuracy:** 0.6657
- **Brier (avg):** 0.2089
- **Training time:** ~180s
- **Committed:** yes (8b00a9f)
- **Notes:** K=32, n_estimators=500, max_depth=4, lr=0.05. ELO diff is top feature (14.6% ATP, 12.1% WTA). WTA 90.5% accuracy bug caught and fixed (match_num ordering reversal in 2025+ data). See docs/baseline-methodology.html for full write-up.

## Iteration 1 -- IMPROVED
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7497 (delta from baseline)
- **WTA ROC-AUC:** 0.7327 (delta from baseline)
- **Combined ROC-AUC:** 0.7412 (delta: +0.0035 from previous best 0.7377)
- **ATP Accuracy:** 0.6903
- **WTA Accuracy:** 0.6597
- **Committed:** yes (0db45a0)

## Iteration 2 -- NO_CHANGE
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7486
- **WTA ROC-AUC:** 0.7321
- **Combined ROC-AUC:** 0.7404 (delta: -0.0008 from best 0.7412)
- **Committed:** no (rolled back)

## Iteration 3 -- NO_CHANGE
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7474
- **WTA ROC-AUC:** 0.7331
- **Combined ROC-AUC:** 0.7402 (delta: -0.0010 from best 0.7412)
- **Committed:** no (rolled back)

## Iteration 4 -- IMPROVED
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7496 (delta from baseline)
- **WTA ROC-AUC:** 0.7392 (delta from baseline)
- **Combined ROC-AUC:** 0.7444 (delta: +0.0032 from previous best 0.7412)
- **ATP Accuracy:** 0.6804
- **WTA Accuracy:** 0.6687
- **Committed:** yes (c199041)

## Iteration 5 -- NO_CHANGE
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7490
- **WTA ROC-AUC:** 0.7358
- **Combined ROC-AUC:** 0.7424 (delta: -0.0020 from best 0.7444)
- **Committed:** no (rolled back)

## Iteration 6 -- NO_CHANGE
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7494
- **WTA ROC-AUC:** 0.7358
- **Combined ROC-AUC:** 0.7426 (delta: -0.0018 from best 0.7444)
- **Committed:** no (rolled back)

## Iteration 7 -- IMPROVED
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7511 (delta from baseline)
- **WTA ROC-AUC:** 0.7394 (delta from baseline)
- **Combined ROC-AUC:** 0.7452 (delta: +0.0008 from previous best 0.7444)
- **ATP Accuracy:** 0.6771
- **WTA Accuracy:** 0.6866
- **Committed:** yes (2168acd)

## Iteration 8 -- IMPROVED
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7473 (delta from baseline)
- **WTA ROC-AUC:** 0.7435 (delta from baseline)
- **Combined ROC-AUC:** 0.7454 (delta: +0.0002 from previous best 0.7452)
- **ATP Accuracy:** 0.6837
- **WTA Accuracy:** 0.6896
- **Committed:** yes (80826a1)

## Iteration 1 -- NO_CHANGE
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7473
- **WTA ROC-AUC:** 0.7435
- **Combined ROC-AUC:** 0.7454 (delta: 0.0000 from best 0.7454)
- **Committed:** no (rolled back)

## Iteration 2 -- NO_CHANGE
- **Phase:** A (ELO System Tuning)
- **ATP ROC-AUC:** 0.7502
- **WTA ROC-AUC:** 0.7370
- **Combined ROC-AUC:** 0.7436 (delta: -0.0018 from best 0.7454)
- **Committed:** no (rolled back)
