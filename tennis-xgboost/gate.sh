#!/bin/bash
# =============================================================================
# Verification Gate: Tennis XGBoost Auto-Research
# =============================================================================
#
# Runs the full pipeline for BOTH tours (ATP + WTA), averages ROC-AUC,
# and outputs a single scalar for the ratchet.
#
# Usage:   bash gate.sh
# Output:  COMBINED_ROC_AUC=0.XXXX  (stdout, single line)
# Exit:    0 on success, 1 on any failure
# Time:    < 5 minutes total (both tours)
#
# Individual tour scores are printed to stderr for diagnostics.
# The ONLY stdout output is the final COMBINED_ROC_AUC line.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Pre-flight checks ---

if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv not found. Run 'make install' first." >&2
    exit 1
fi

# Verify data.py has not been modified (immutable guard rail)
DATA_PY_STATUS=$(git diff --name-only -- src/tennis_predict/data.py 2>/dev/null || echo "")
if [[ -n "$DATA_PY_STATUS" ]]; then
    echo "ERROR: data.py has been modified. This file is IMMUTABLE." >&2
    echo "Revert changes to data.py before running the gate." >&2
    exit 1
fi

# Verify cli.py has not been modified (immutable guard rail)
CLI_PY_STATUS=$(git diff --name-only -- src/tennis_predict/cli.py 2>/dev/null || echo "")
if [[ -n "$CLI_PY_STATUS" ]]; then
    echo "ERROR: cli.py has been modified. This file is IMMUTABLE." >&2
    echo "Revert changes to cli.py before running the gate." >&2
    exit 1
fi

# Verify tests/ has not been modified (immutable guard rail)
TESTS_STATUS=$(git diff --name-only -- tests/ 2>/dev/null || echo "")
if [[ -n "$TESTS_STATUS" ]]; then
    echo "ERROR: tests/ has been modified. Test files are IMMUTABLE." >&2
    exit 1
fi

# Verify evaluate.py has not been modified (immutable guard rail)
# evaluate.py contains all scoring/metric computation. Keeping it immutable
# prevents agents from injecting post-hoc probability adjustments that game
# validation scores without genuinely improving the model.
EVAL_PY_STATUS=$(git diff --name-only -- src/tennis_predict/evaluate.py 2>/dev/null || echo "")
if [[ -n "$EVAL_PY_STATUS" ]]; then
    echo "ERROR: evaluate.py has been modified. This file is IMMUTABLE." >&2
    echo "Revert changes to evaluate.py before running the gate." >&2
    exit 1
fi

# --- Run tests first (fast gate) ---

echo "Running pytest..." >&2
if ! (. .venv/bin/activate && pytest -q 2>&1) >&2; then
    echo "ERROR: pytest failed. Fix tests before running the gate." >&2
    exit 1
fi
echo "Tests passed." >&2

# --- Run ATP pipeline ---

echo "Running ATP pipeline..." >&2
START_ATP=$(date +%s)

ATP_OUTPUT=$(. .venv/bin/activate && tennis-predict --tour atp run-pipeline 2>&1) || {
    echo "ERROR: ATP pipeline failed:" >&2
    echo "$ATP_OUTPUT" >&2
    exit 1
}

END_ATP=$(date +%s)
ATP_TIME=$((END_ATP - START_ATP))

ATP_ROC_LINE=$(echo "$ATP_OUTPUT" | grep "^ROC_AUC=" | tail -1)
if [[ -z "$ATP_ROC_LINE" ]]; then
    echo "ERROR: No ROC_AUC output from ATP pipeline" >&2
    echo "$ATP_OUTPUT" >&2
    exit 1
fi

ATP_ROC=$(echo "$ATP_ROC_LINE" | sed 's/ROC_AUC=//')
echo "ATP: ROC_AUC=$ATP_ROC (${ATP_TIME}s)" >&2

# Guard rail: training time < 5 min per tour
if [[ $ATP_TIME -gt 300 ]]; then
    echo "ERROR: ATP training exceeded 5-minute guard rail (${ATP_TIME}s)" >&2
    exit 1
fi

# --- Run WTA pipeline ---

echo "Running WTA pipeline..." >&2
START_WTA=$(date +%s)

WTA_OUTPUT=$(. .venv/bin/activate && tennis-predict --tour wta run-pipeline 2>&1) || {
    echo "ERROR: WTA pipeline failed:" >&2
    echo "$WTA_OUTPUT" >&2
    exit 1
}

END_WTA=$(date +%s)
WTA_TIME=$((END_WTA - START_WTA))

WTA_ROC_LINE=$(echo "$WTA_OUTPUT" | grep "^ROC_AUC=" | tail -1)
if [[ -z "$WTA_ROC_LINE" ]]; then
    echo "ERROR: No ROC_AUC output from WTA pipeline" >&2
    echo "$WTA_OUTPUT" >&2
    exit 1
fi

WTA_ROC=$(echo "$WTA_ROC_LINE" | sed 's/ROC_AUC=//')
echo "WTA: ROC_AUC=$WTA_ROC (${WTA_TIME}s)" >&2

# Guard rail: training time < 5 min per tour
if [[ $WTA_TIME -gt 300 ]]; then
    echo "ERROR: WTA training exceeded 5-minute guard rail (${WTA_TIME}s)" >&2
    exit 1
fi

# --- Guard rail: prediction sanity checks ---
# Catches post-hoc probability manipulation (hardcoded overrides, systematic
# bias, degenerate models). Predictions are written by train_and_report() to
# models/{tour}/xgboost/predictions.csv with a prob_player_a_wins column.

for TOUR in atp wta; do
    PRED_FILE="models/${TOUR}/xgboost/predictions.csv"
    if [[ ! -f "$PRED_FILE" ]]; then
        echo "ERROR: ${TOUR} predictions file not found at ${PRED_FILE}" >&2
        exit 1
    fi

    SANITY_RESULT=$(. .venv/bin/activate && python -c "
import sys
import pandas as pd
import numpy as np

df = pd.read_csv('${PRED_FILE}')
probs = df['prob_player_a_wins'].values

# Check 1: No extreme probabilities (catches hardcoded overrides)
max_prob = float(np.max(probs))
min_prob = float(np.min(probs))
if max_prob > 0.99:
    print(f'FAIL: max prediction {max_prob:.4f} > 0.99 (hardcoded override?)')
    sys.exit(1)
if min_prob < 0.01:
    print(f'FAIL: min prediction {min_prob:.4f} < 0.01 (hardcoded override?)')
    sys.exit(1)

# Check 2: Mean prediction in reasonable range (catches systematic bias)
mean_prob = float(np.mean(probs))
if mean_prob < 0.35 or mean_prob > 0.65:
    print(f'FAIL: mean prediction {mean_prob:.4f} outside [0.35, 0.65] (systematic bias?)')
    sys.exit(1)

# Check 3: Non-degenerate distribution (catches constant models)
std_prob = float(np.std(probs))
if std_prob < 0.05:
    print(f'FAIL: prediction std {std_prob:.4f} < 0.05 (degenerate model?)')
    sys.exit(1)

print(f'OK: range=[{min_prob:.4f}, {max_prob:.4f}], mean={mean_prob:.4f}, std={std_prob:.4f}')
" 2>&1) || {
        echo "ERROR: ${TOUR} prediction sanity check failed:" >&2
        echo "$SANITY_RESULT" >&2
        exit 1
    }

    if [[ "$SANITY_RESULT" == FAIL* ]]; then
        echo "ERROR: ${TOUR} prediction sanity check failed: ${SANITY_RESULT}" >&2
        exit 1
    fi
    echo "${TOUR} predictions: ${SANITY_RESULT}" >&2
done

# --- Guard rail: model size < 100MB per tour ---

for TOUR in atp wta; do
    MODEL_FILE="models/${TOUR}/xgboost/model.joblib"
    if [[ -f "$MODEL_FILE" ]]; then
        MODEL_SIZE=$(wc -c < "$MODEL_FILE" | tr -d ' ')
        MAX_SIZE=$((100 * 1024 * 1024))  # 100MB
        if [[ $MODEL_SIZE -gt $MAX_SIZE ]]; then
            echo "ERROR: ${TOUR} model exceeds 100MB guard rail ($(( MODEL_SIZE / 1024 / 1024 ))MB)" >&2
            exit 1
        fi
    fi
done

# --- Guard rail: feature count < 500 ---

for TOUR in atp wta; do
    PARQUET_FILE="data/processed/${TOUR}_features_strict.parquet"
    if [[ -f "$PARQUET_FILE" ]]; then
        NCOLS=$(. .venv/bin/activate && python -c "
import pandas as pd
from tennis_predict.config import META_COLUMNS
df = pd.read_parquet('${PARQUET_FILE}')
feature_cols = [c for c in df.columns if c not in set(META_COLUMNS)]
print(len(feature_cols))
" 2>/dev/null || echo "0")
        if [[ $NCOLS -gt 500 ]]; then
            echo "ERROR: ${TOUR} feature count ($NCOLS) exceeds 500 guard rail" >&2
            exit 1
        fi
        echo "${TOUR}: ${NCOLS} features" >&2
    fi
done

# --- Compute combined score ---

COMBINED=$(python3 -c "
atp = float('${ATP_ROC}')
wta = float('${WTA_ROC}')
combined = (atp + wta) / 2.0
print(f'{combined:.4f}')
")

TOTAL_TIME=$((ATP_TIME + WTA_TIME))
echo "Total time: ${TOTAL_TIME}s" >&2
echo "Combined: (${ATP_ROC} + ${WTA_ROC}) / 2 = ${COMBINED}" >&2

# --- Extract additional diagnostics ---

ATP_ACC=$(echo "$ATP_OUTPUT" | grep "accuracy:" | head -1 | awk '{print $2}' || echo "N/A")
WTA_ACC=$(echo "$WTA_OUTPUT" | grep "accuracy:" | head -1 | awk '{print $2}' || echo "N/A")
echo "Accuracy: ATP=${ATP_ACC}, WTA=${WTA_ACC}" >&2

# --- Output the single scalar (ONLY stdout output) ---

echo "COMBINED_ROC_AUC=${COMBINED}"
