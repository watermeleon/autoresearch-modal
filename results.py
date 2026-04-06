"""Result saving — persist run artifacts to the Modal volume."""

import json
import shutil
from pathlib import Path


def _make_run_id(agent: str, agent_config: str, start_time: float) -> str:
    """Generate a unique, filesystem-safe run ID."""
    config_safe = agent_config.replace("/", "_").replace(":", "_")
    timestamp = int(start_time)
    return f"tennis_{agent}_{config_safe}_{timestamp}"


def save_results(
    run_id: str,
    task_dir: Path,
    agent: str,
    agent_config: str,
    start_time: float,
    end_time: float,
    baseline: float,
    best_score: float,
    iterations_run: int,
    iterations_improved: int,
) -> None:
    """Save run artifacts to /results/{run_id}/ on the Modal volume."""
    results_dir = Path("/results") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    task_dir = Path(task_dir)

    # --- metadata.json ---
    metadata = {
        "run_id": run_id,
        "agent": agent,
        "agent_config": agent_config,
        "start_time": start_time,
        "end_time": end_time,
        "wall_time_seconds": end_time - start_time,
        "baseline": baseline,
        "best_score": best_score,
        "improvement": best_score - baseline,
        "iterations_run": iterations_run,
        "iterations_improved": iterations_improved,
        "status": "complete",
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # --- RESEARCH_LOG.md ---
    research_log = task_dir / "RESEARCH_LOG.md"
    if research_log.exists():
        shutil.copy(research_log, results_dir / "RESEARCH_LOG.md")

    # --- per-iteration transcripts ---
    for transcript in sorted(task_dir.glob("transcript_iter_*.json")):
        shutil.copy(transcript, results_dir / transcript.name)

    # --- orchestrator logs ---
    for log_name in ("output.log", "error.log"):
        log_path = task_dir / log_name
        if log_path.exists():
            shutil.copy(log_path, results_dir / log_name)

    # --- final source code snapshot (best version) ---
    src_dir = task_dir / "src"
    if src_dir.is_dir():
        shutil.copytree(str(src_dir), str(results_dir / "src"))

    # --- program.md for reference ---
    program_md = task_dir / "program.md"
    if program_md.exists():
        shutil.copy(program_md, results_dir / "program.md")

    print(f"Results saved to volume: /results/{run_id}", flush=True)
    print(f"  - metadata.json, RESEARCH_LOG.md", flush=True)
    transcripts = list(results_dir.glob("transcript_iter_*.json"))
    if transcripts:
        total_mb = sum(t.stat().st_size for t in transcripts) / (1024 * 1024)
        print(f"  - {len(transcripts)} transcripts ({total_mb:.1f} MB total)", flush=True)
    if (results_dir / "src").is_dir():
        print(f"  - src/ (final source snapshot)", flush=True)
