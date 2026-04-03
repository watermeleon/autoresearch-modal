"""
Validation script: runs training via Modal dispatch and directly on Modal,
then compares the output structure to verify they're equivalent.

Usage:
    uv run validate.py
"""

import os
import re
import sys
import textwrap

EXPECTED_KEYS = [
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
]


def parse_summary(log: str) -> dict:
    """Extract the key-value summary block after '---' in training output."""
    results = {}
    in_summary = False
    for line in log.splitlines():
        if line.strip() == "---":
            in_summary = True
            continue
        if in_summary:
            match = re.match(r"^(\w+):\s+(.+)$", line.strip())
            if match:
                results[match.group(1)] = match.group(2).strip()
    return results


def check_log_structure(log: str, label: str) -> dict:
    """Validate that a training log has the expected structure. Returns parsed summary."""
    print(f"\n{'='*60}")
    print(f"Checking: {label}")
    print(f"{'='*60}")

    # Check log is non-empty
    if not log.strip():
        print(f"  FAIL: log is empty")
        return {}

    # Check for the --- separator
    if "---" not in log:
        print(f"  FAIL: no '---' separator found")
        print(f"  Last 10 lines of log:")
        for line in log.strip().splitlines()[-10:]:
            print(f"    {line}")
        return {}

    # Parse summary
    summary = parse_summary(log)

    # Check all expected keys are present
    missing = [k for k in EXPECTED_KEYS if k not in summary]
    if missing:
        print(f"  FAIL: missing keys: {missing}")
    else:
        print(f"  OK: all {len(EXPECTED_KEYS)} expected keys present")

    # Print values
    for key in EXPECTED_KEYS:
        val = summary.get(key, "MISSING")
        print(f"  {key:20s}: {val}")

    # Check for training progress lines (step NNNNN ...)
    step_lines = [l for l in log.splitlines() if re.match(r".*step \d{5}", l)]
    print(f"  Training step lines: {len(step_lines)}")

    return summary


def run_modal_dispatch() -> tuple[str, int]:
    """Run train.py via the Modal dispatch preamble (as the agent would)."""
    import subprocess
    print("\nRunning: Modal dispatch (uv run train.py) ...")
    result = subprocess.run(
        ["uv", "run", "train.py"],
        capture_output=True,
        text=True,
        timeout=700,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return result.stdout + result.stderr, result.returncode


def run_direct_on_modal() -> tuple[str, int]:
    """Run the original train.py directly on Modal (no preamble) as a baseline."""
    from modal_app import run_training

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Read train.py and strip the preamble so it runs directly
    with open(os.path.join(base_dir, "train.py")) as f:
        train_py = f.read()

    # Remove the preamble block
    filtered = []
    skip = False
    for line in train_py.splitlines():
        if "MODAL DISPATCH" in line and "DO NOT MODIFY" in line:
            skip = True
            continue
        if "END MODAL DISPATCH" in line:
            skip = False
            continue
        if skip:
            continue
        filtered.append(line)
    train_py_clean = "\n".join(filtered)

    with open(os.path.join(base_dir, "prepare.py")) as f:
        prepare_py = f.read()

    print("\nRunning: Direct on Modal (baseline, no preamble) ...")
    result = run_training.remote(train_py_clean, prepare_py)
    return result["stdout"] + result["stderr"], result["exit_code"]


def compare_summaries(a: dict, b: dict, label_a: str, label_b: str):
    """Compare two parsed summaries."""
    print(f"\n{'='*60}")
    print(f"Comparison: {label_a} vs {label_b}")
    print(f"{'='*60}")

    for key in EXPECTED_KEYS:
        va = a.get(key, "MISSING")
        vb = b.get(key, "MISSING")
        try:
            fa, fb = float(va), float(vb)
            pct_diff = abs(fa - fb) / max(abs(fa), 1e-10) * 100
            status = "OK" if pct_diff < 50 else "WARN"
            print(f"  {key:20s}: {va:>12s} vs {vb:>12s}  ({pct_diff:5.1f}% diff) [{status}]")
        except ValueError:
            match = "OK" if va == vb else "DIFF"
            print(f"  {key:20s}: {va:>12s} vs {vb:>12s}  [{match}]")


def main():
    print("Autoresearch Modal Validation")
    print("This will run training TWICE on Modal H100s (~10 min total).")
    print()

    # Run both
    dispatch_log, dispatch_exit = run_modal_dispatch()
    direct_log, direct_exit = run_direct_on_modal()

    # Save logs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "validate_dispatch.log"), "w") as f:
        f.write(dispatch_log)
    with open(os.path.join(base_dir, "validate_direct.log"), "w") as f:
        f.write(direct_log)

    print(f"\nExit codes: dispatch={dispatch_exit}, direct={direct_exit}")

    # Check structure
    summary_dispatch = check_log_structure(dispatch_log, "Modal Dispatch (agent path)")
    summary_direct = check_log_structure(direct_log, "Direct on Modal (baseline)")

    # Compare
    if summary_dispatch and summary_direct:
        compare_summaries(summary_dispatch, summary_direct, "dispatch", "direct")

    # Final verdict
    print(f"\n{'='*60}")
    if summary_dispatch and summary_direct and dispatch_exit == 0 and direct_exit == 0:
        print("VERDICT: Both runs produced valid output. Dispatch is working.")
    else:
        print("VERDICT: Something went wrong. Check logs above.")
    print(f"{'='*60}")

    print(f"\nLogs saved to: validate_dispatch.log, validate_direct.log")


if __name__ == "__main__":
    main()
