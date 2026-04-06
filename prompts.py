"""Prompt generation and timer script for the tennis-xgboost research loop.

The per-iteration prompt is adapted from run-research.sh (tennis-xgboost repo).
"""

import time


# ---------------------------------------------------------------------------
# Prompt template — verbatim from run-research.sh lines 55-70
# ---------------------------------------------------------------------------

ITERATION_PROMPT_TEMPLATE = """\
You are an ML researcher optimizing a tennis match prediction pipeline.

## Objective
Maximize COMBINED_ROC_AUC (average of ATP and WTA ROC-AUC on 2026 validation).
Current best: {best}. Baseline: {baseline}. This is iteration {iteration} of {max_iters}.

## How to work
1. Read program.md -- it defines what you can change, what is off limits, and dead ends to avoid.
2. Read RESEARCH_LOG.md -- it shows what was tried before and what worked. Learn from it. Do not repeat failures. Build on successes.
3. Read the source files you plan to modify. Understand what is there before changing anything.
4. Form your own hypothesis about what will improve ROC-AUC. You decide what to try.
5. Implement your change. Be bold -- structural changes often beat parameter tweaks.
6. Run pytest to verify tests pass.
7. Do NOT modify immutable files: data.py, cli.py, evaluate.py, gate.sh, RESEARCH_LOG.md, data/raw/**, data/validation/**, tests/**"""

CLAUDE_PROMPT_SUFFIX = (
    "\nYou are running in a non-interactive mode. So make sure every process "
    "you are running finishes before you write your last message."
)


def build_iteration_prompt(
    best: float,
    baseline: float,
    iteration: int,
    max_iters: int,
    agent: str = "claude",
    extra_prompt: str = "",
) -> str:
    """Build the per-iteration prompt for the agent."""
    prompt = ITERATION_PROMPT_TEMPLATE.format(
        best=f"{best:.4f}",
        baseline=f"{baseline:.4f}",
        iteration=iteration,
        max_iters=max_iters,
    )
    if extra_prompt:
        prompt += "\n\n## Additional Instructions\n" + extra_prompt
    if agent == "claude":
        prompt += CLAUDE_PROMPT_SUFFIX
    return prompt


def generate_timer_script(num_hours: float) -> str:
    """Generate a timer.sh script the agent can call to check remaining time."""
    creation_time = int(time.time())
    total_seconds = int(num_hours * 3600)
    return f"""#!/bin/bash
TOTAL_SECONDS={total_seconds}
CREATION_DATE={creation_time}
DEADLINE=$((CREATION_DATE + TOTAL_SECONDS))
NOW=$(date +%s)
REMAINING=$((DEADLINE - NOW))
if [ $REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes):"
    HOURS=$((REMAINING / 3600))
    MINUTES=$(((REMAINING % 3600) / 60))
    printf "%d:%02d\\n" $HOURS $MINUTES
fi
"""
