"""
Modal launcher for tennis-xgboost auto-research.
Runs an LLM agent (Claude Code / Codex) on Modal CPU to iteratively optimize
a tennis match prediction pipeline via XGBoost.

The research loop follows the same pattern as run-research.sh:
  for each iteration:
    1. dispatch agent with per-iteration prompt
    2. agent reads program.md + RESEARCH_LOG.md, modifies src/
    3. run gate.sh to evaluate
    4. commit if improved, rollback if not

Usage:
    # Basic run (Claude, 2 hours)
    python3 -m modal run launch.py --agent claude --agent-config claude-opus-4-6 --num-hours 2

    # Detached (survives terminal close)
    python3 -m modal run --detach launch.py --agent claude --agent-config claude-opus-4-6 --num-hours 4

    # Codex agent
    python3 -m modal run --detach launch.py --agent codex --agent-config gpt-5.4 --num-hours 4

    # Short test
    python3 -m modal run launch.py --agent claude --agent-config claude-opus-4-6 --num-hours 0.5 --max-iters 3
"""

import modal
import os
import re
import sys
import time

from agents import VALID_AGENTS, AGENT_RUNNERS
from prompts import build_iteration_prompt, generate_timer_script
from results import _make_run_id, save_results


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TENNIS_REPO_DIR = os.environ.get(
    "TENNIS_REPO_PATH",
    os.path.join(_THIS_DIR, "tennis-xgboost"),
)

# ---------------------------------------------------------------------------
# Logging helper (from spar-maded)
# ---------------------------------------------------------------------------


class _TeeWriter:
    """Write to both a file and the original stream."""
    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file

    def write(self, data):
        self._original.write(data)
        self._log.write(data)

    def flush(self):
        self._original.flush()
        self._log.flush()


# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "make")
    # Node.js for agent CLIs
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
    )
    # Agent CLIs
    .run_commands(
        "npm install -g @openai/codex@0.79.0",
        "npm install -g @anthropic-ai/claude-code@2.0.55",
    )
    # uv for fast pip installs
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    # Copy bundled tennis-xgboost code into image
    .add_local_dir(
        TENNIS_REPO_DIR,
        remote_path="/repo/tennis-xgboost",
        copy=True,
    )
    # Create venv + install tennis deps (gate.sh requires .venv)
    .run_commands(
        "cd /repo/tennis-xgboost && /root/.local/bin/uv venv --clear",
        "cd /repo/tennis-xgboost && /root/.local/bin/uv pip install --python /repo/tennis-xgboost/.venv/bin/python -e '.[dev]'",
    )
    # Sync Sackmann data (cached in image layer)
    .run_commands(
        "cd /repo/tennis-xgboost && . .venv/bin/activate && tennis-predict --tour atp sync-data",
        "cd /repo/tennis-xgboost && . .venv/bin/activate && tennis-predict --tour wta sync-data",
    )
    # Copy extension data into raw repos (2025-2026 gap fill)
    .run_commands(
        "cp /repo/tennis-xgboost/data/extension/atp_matches_*.csv /repo/tennis-xgboost/data/raw/tennis_atp/ 2>/dev/null || true",
        "cp /repo/tennis-xgboost/data/extension/wta_matches_*.csv /repo/tennis-xgboost/data/raw/tennis_wta/ 2>/dev/null || true",
    )
    # Copy launcher modules for remote imports
    .add_local_file(os.path.join(_THIS_DIR, "agents.py"), remote_path="/root/agents.py", copy=True)
    .add_local_file(os.path.join(_THIS_DIR, "prompts.py"), remote_path="/root/prompts.py", copy=True)
    .add_local_file(os.path.join(_THIS_DIR, "results.py"), remote_path="/root/results.py", copy=True)
)

app = modal.App("tennis-xgboost-autoresearch", image=image)

# Persistent volume for results
results_volume = modal.Volume.from_name("tennis-xgboost-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# The main Modal function — runs on CPU
# ---------------------------------------------------------------------------

@app.function(
    cpu=8,
    memory=16384,
    timeout=12 * 3600,  # 12h max
    volumes={"/results": results_volume},
    secrets=[
        # modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("openai-api-key"),
        # modal.Secret.from_name("codex-auth"),
    ],
)
def run_research(
    num_hours: float = 2,
    max_iters: int = 50,
    agent: str = "claude",
    agent_config: str = "claude-opus-4-6",
    extra_prompt: str = "",
) -> dict:
    """Run the tennis-xgboost auto-research loop on Modal."""
    import subprocess
    from pathlib import Path

    if agent not in VALID_AGENTS:
        raise ValueError(f"agent must be one of {VALID_AGENTS}, got '{agent}'")

    start_time = time.time()
    deadline = start_time + num_hours * 3600

    # --- Working directory (use repo in-place, Modal containers are ephemeral) ---
    task_dir = Path("/repo/tennis-xgboost")

    # --- Capture orchestration logs ---
    _out_log = open(task_dir / "output.log", "w")
    _err_log = open(task_dir / "error.log", "w")
    sys.stdout = _TeeWriter(sys.__stdout__, _out_log)
    sys.stderr = _TeeWriter(sys.__stderr__, _err_log)

    # --- Initialize git repo (gate.sh needs it for immutability checks) ---
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "modal",
        "GIT_AUTHOR_EMAIL": "modal@local",
        "GIT_COMMITTER_NAME": "modal",
        "GIT_COMMITTER_EMAIL": "modal@local",
    }
    subprocess.run(["git", "init"], cwd=task_dir, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=task_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial baseline"],
        cwd=task_dir, capture_output=True, env=git_env,
    )
    print("Git repo initialized for gate.sh immutability checks", flush=True)

    # --- Generate timer script ---
    timer_script = generate_timer_script(num_hours)
    (task_dir / "timer.sh").write_text(timer_script)
    (task_dir / "timer.sh").chmod(0o755)

    # --- Common env vars ---
    env = os.environ.copy()
    env["CI"] = "1"
    env["PYTHONNOUSERSITE"] = "1"

    # --- Establish baseline ---
    print("=" * 60, flush=True)
    print("ESTABLISHING BASELINE", flush=True)
    print("=" * 60, flush=True)

    baseline_result = subprocess.run(
        ["bash", "gate.sh"],
        cwd=task_dir,
        capture_output=True,
        text=True,
        timeout=1200,
    )

    if baseline_result.returncode != 0:
        print(f"GATE FAILED during baseline:\n{baseline_result.stderr}", flush=True)
        _out_log.close(); _err_log.close()
        sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__
        return {"status": "baseline_failed", "error": baseline_result.stderr[:500]}

    baseline_match = re.search(r"COMBINED_ROC_AUC=([\d.]+)", baseline_result.stdout)
    if not baseline_match:
        print(f"Could not parse baseline from: {baseline_result.stdout}", flush=True)
        _out_log.close(); _err_log.close()
        sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__
        return {"status": "baseline_parse_failed", "stdout": baseline_result.stdout[:500]}

    baseline = float(baseline_match.group(1))
    best = baseline
    print(f"Baseline: COMBINED_ROC_AUC={baseline:.4f}", flush=True)

    # --- Research loop ---
    print("=" * 60, flush=True)
    print(f"STARTING RESEARCH LOOP", flush=True)
    print(f"Agent: {agent} ({agent_config})", flush=True)
    print(f"Time budget: {num_hours}h | Max iterations: {max_iters}", flush=True)
    print("=" * 60, flush=True)

    consecutive_failures = 0
    circuit_breaker = 10
    iterations_improved = 0

    for i in range(1, max_iters + 1):
        # Check time remaining
        remaining = deadline - time.time()
        if remaining < 60:
            print(f"\nTime budget exhausted ({remaining:.0f}s remaining). Stopping.", flush=True)
            break

        print(f"\n{'=' * 40}", flush=True)
        print(f"--- Iteration {i} / {max_iters} ---", flush=True)
        print(f"--- Best: {best:.4f} | Remaining: {remaining / 60:.0f}min ---", flush=True)
        print(f"{'=' * 40}", flush=True)

        # --- Build prompt ---
        prompt = build_iteration_prompt(
            best=best,
            baseline=baseline,
            iteration=i,
            max_iters=max_iters,
            agent=agent,
            extra_prompt=extra_prompt,
        )

        # --- Dispatch agent ---
        per_iter_timeout = min(remaining - 60, 1800)  # 30 min cap, keep 60s for gate
        if per_iter_timeout <= 0:
            print("Not enough time for another iteration. Stopping.", flush=True)
            break

        transcript_name = f"transcript_iter_{i}.json"
        runner = AGENT_RUNNERS[agent]

        try:
            proc, transcript_path = runner(
                prompt, agent_config, task_dir, env.copy(),
                int(per_iter_timeout), stderr_fh=_err_log,
                transcript_name=transcript_name,
            )
        except Exception as e:
            print(f"AGENT DISPATCH FAILED: {e}", flush=True)
            consecutive_failures += 1
            if consecutive_failures >= circuit_breaker:
                print(f"CIRCUIT BREAKER: {circuit_breaker} consecutive failures. Stopping.", flush=True)
                break
            continue

        print(f"Agent completed (exit code: {proc.returncode}). Running gate...", flush=True)

        # --- Run gate ---
        try:
            gate_result = subprocess.run(
                ["bash", "gate.sh"],
                cwd=task_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            print("GATE TIMED OUT", flush=True)
            _rollback(task_dir, git_env)
            _append_research_log(task_dir, i, "GATE_TIMEOUT", {})
            consecutive_failures += 1
            if consecutive_failures >= circuit_breaker:
                print(f"CIRCUIT BREAKER: {circuit_breaker} consecutive failures. Stopping.", flush=True)
                break
            continue

        if gate_result.returncode != 0:
            print(f"GATE FAILED:\n{gate_result.stderr[-200:]}", flush=True)
            _rollback(task_dir, git_env)
            _append_research_log(task_dir, i, "GATE_FAILED",
                                 {"gate_output": gate_result.stderr[:200]})
            consecutive_failures += 1
            if consecutive_failures >= circuit_breaker:
                print(f"CIRCUIT BREAKER: {circuit_breaker} consecutive failures. Stopping.", flush=True)
                break
            continue

        # --- Parse score ---
        score_match = re.search(r"COMBINED_ROC_AUC=([\d.]+)", gate_result.stdout)
        if not score_match:
            print(f"Could not parse COMBINED_ROC_AUC from gate output", flush=True)
            _rollback(task_dir, git_env)
            consecutive_failures += 1
            continue

        current = float(score_match.group(1))

        # Parse diagnostics from stderr
        diagnostics = _parse_gate_diagnostics(gate_result.stderr)

        # --- Compare ---
        if current > best:
            delta = current - best
            prev_best = best
            best = current
            consecutive_failures = 0
            iterations_improved += 1

            print(f"IMPROVEMENT: {prev_best:.4f} -> {best:.4f} (+{delta:.4f})", flush=True)

            # Commit
            subprocess.run(["git", "add", "src/", "pyproject.toml"],
                           cwd=task_dir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m",
                 f"auto-research iter {i}: COMBINED_ROC_AUC={best:.4f} (+{delta:.4f})"],
                cwd=task_dir, capture_output=True, env=git_env,
            )
            commit_sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=task_dir, capture_output=True, text=True,
            ).stdout.strip()

            _append_research_log(task_dir, i, "IMPROVED", {
                **diagnostics,
                "combined": f"{best:.4f}",
                "delta": f"+{delta:.4f}",
                "prev_best": f"{prev_best:.4f}",
                "commit": commit_sha,
            })
        else:
            delta = current - best
            print(f"No improvement: {current:.4f} <= {best:.4f} ({delta:.4f})", flush=True)

            _rollback(task_dir, git_env)
            consecutive_failures += 1

            _append_research_log(task_dir, i, "NO_CHANGE", {
                **diagnostics,
                "combined": f"{current:.4f}",
                "delta": f"{delta:.4f}",
                "best": f"{best:.4f}",
            })

            if consecutive_failures >= circuit_breaker:
                print(f"CIRCUIT BREAKER: {circuit_breaker} consecutive failures. Stopping.", flush=True)
                break

    # --- Finalize ---
    iterations_run = i if 'i' in dir() else 0

    print("\n" + "=" * 60, flush=True)
    print("AUTO-RESEARCH COMPLETE", flush=True)
    print(f"Best COMBINED_ROC_AUC: {best:.4f} (baseline was {baseline:.4f})", flush=True)
    print(f"Improvement: +{best - baseline:.4f}", flush=True)
    print(f"Iterations: {iterations_run} ({iterations_improved} improved)", flush=True)
    print("=" * 60, flush=True)

    # --- Save results to volume ---
    _out_log.flush(); _err_log.flush()

    run_id = _make_run_id(agent, agent_config, start_time)
    end_time = time.time()

    save_results(
        run_id=run_id,
        task_dir=task_dir,
        agent=agent,
        agent_config=agent_config,
        start_time=start_time,
        end_time=end_time,
        baseline=baseline,
        best_score=best,
        iterations_run=iterations_run,
        iterations_improved=iterations_improved,
    )
    results_volume.commit()

    _out_log.close(); _err_log.close()
    sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__

    return {
        "status": "ok",
        "run_id": run_id,
        "baseline": baseline,
        "best_score": best,
        "improvement": best - baseline,
        "iterations_run": iterations_run,
        "iterations_improved": iterations_improved,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rollback(task_dir, git_env):
    """Roll back agent changes (matches run-research.sh)."""
    import subprocess
    subprocess.run(["git", "checkout", "--", "src/", "pyproject.toml"],
                   cwd=task_dir, capture_output=True, env=git_env)
    subprocess.run(["git", "clean", "-fd", "src/"],
                   cwd=task_dir, capture_output=True, env=git_env)


def _parse_gate_diagnostics(stderr: str) -> dict:
    """Extract per-tour scores from gate.sh stderr output."""
    info = {}
    atp_match = re.search(r"ATP: ROC_AUC=([\d.]+)", stderr)
    if atp_match:
        info["atp_roc"] = atp_match.group(1)
    wta_match = re.search(r"WTA: ROC_AUC=([\d.]+)", stderr)
    if wta_match:
        info["wta_roc"] = wta_match.group(1)
    acc_match = re.search(r"Accuracy: ATP=([\d.]+), WTA=([\d.]+)", stderr)
    if acc_match:
        info["atp_acc"] = acc_match.group(1)
        info["wta_acc"] = acc_match.group(1)
    return info


def _append_research_log(task_dir, iteration: int, status: str, info: dict):
    """Append an entry to RESEARCH_LOG.md (matches run-research.sh format)."""
    from pathlib import Path

    log_path = Path(task_dir) / "RESEARCH_LOG.md"
    entry = f"\n## Iteration {iteration} -- {status}\n"

    if "atp_roc" in info:
        entry += f"- **ATP ROC-AUC:** {info['atp_roc']}\n"
    if "wta_roc" in info:
        entry += f"- **WTA ROC-AUC:** {info['wta_roc']}\n"

    if status == "IMPROVED":
        entry += f"- **Combined ROC-AUC:** {info.get('combined', 'N/A')} (delta: {info.get('delta', 'N/A')} from previous best {info.get('prev_best', 'N/A')})\n"
        if "atp_acc" in info:
            entry += f"- **ATP Accuracy:** {info['atp_acc']}\n"
        if "wta_acc" in info:
            entry += f"- **WTA Accuracy:** {info['wta_acc']}\n"
        entry += f"- **Committed:** yes ({info.get('commit', 'N/A')})\n"
    elif status == "NO_CHANGE":
        entry += f"- **Combined ROC-AUC:** {info.get('combined', 'N/A')} (delta: {info.get('delta', 'N/A')} from best {info.get('best', 'N/A')})\n"
        entry += f"- **Committed:** no (rolled back)\n"
    elif status == "GATE_FAILED":
        entry += f"- **Gate output:** {info.get('gate_output', 'N/A')}\n"
        entry += f"- **Committed:** no (rolled back)\n"
    elif status == "GATE_TIMEOUT":
        entry += f"- **Committed:** no (rolled back, gate timed out)\n"

    with open(log_path, "a") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    num_hours: float = 2,
    max_iters: int = 50,
    agent: str = "claude",
    agent_config: str = "claude-opus-4-6",
    extra_prompt: str = "",
):
    print(f"Launching tennis-xgboost auto-research on Modal...")
    print(f"  Agent: {agent} ({agent_config})")
    print(f"  Time:  {num_hours}h")
    print(f"  Iters: {max_iters}")
    if extra_prompt:
        print(f"  Extra: {extra_prompt[:80]}{'...' if len(extra_prompt) > 80 else ''}")
    print()

    result = run_research.remote(
        num_hours=num_hours,
        max_iters=max_iters,
        agent=agent,
        agent_config=agent_config,
        extra_prompt=extra_prompt,
    )

    print()
    print("=" * 60)
    print("RESULT:", result)
    print("=" * 60)
