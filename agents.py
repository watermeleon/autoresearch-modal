"""Agent runners — launch Codex CLI or Claude Code as subprocesses.

Adapted from spar-maded/src/agents.py for the tennis-xgboost research loop.
Each iteration dispatches the agent once; transcript is saved per iteration.
"""

import subprocess
import sys
from pathlib import Path

VALID_AGENTS = {"codex", "claude"}

GRACEFUL_KILL_TIMEOUT_SEC = 30


def _run_agent_subprocess(cmd: list, task_dir, env: dict, timeout_seconds: int,
                          stderr_fh=None, transcript_name: str = "transcript.json") -> tuple:
    """Run an agent subprocess, capture transcript, handle timeout.

    Returns (subprocess.CompletedProcess-like, Path to transcript).
    """
    transcript_path = Path(task_dir) / transcript_name
    with open(transcript_path, "w") as transcript_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(task_dir),
            env=env,
            stdout=transcript_fh,
            stderr=stderr_fh if stderr_fh is not None else sys.__stderr__,
        )
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"Agent timed out after {timeout_seconds}s, terminating...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=GRACEFUL_KILL_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    if transcript_path.exists():
        size_mb = transcript_path.stat().st_size / (1024 * 1024)
        print(f"Transcript saved: {transcript_path.name} ({size_mb:.1f} MB)", flush=True)

    return proc, transcript_path


def run_codex(prompt: str, agent_config: str, task_dir, env: dict,
              timeout_seconds: int, stderr_fh=None,
              transcript_name: str = "transcript.json"):
    """Run the Codex CLI agent. Returns (proc, path_to_transcript)."""
    CODEX_COMMAND_TIMEOUT_MS = 7200000  # 2 hours

    # Codex config
    codex_dir = Path.home() / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    (codex_dir / "config.toml").write_text(
        '[projects."/home/"]\ntrust_level = "trusted"\n\n'
        '[projects."/repo/"]\ntrust_level = "trusted"\n\n'
        "[shell_environment_policy]\n"
        'inherit = "all"\n\n'
        "[shell]\n"
        f"command_timeout = {CODEX_COMMAND_TIMEOUT_MS}\n"
    )

    # Auth: prefer ChatGPT subscription (full auth.json) over API key
    codex_auth_json = env.get("CODEX_AUTH_JSON", "")
    if codex_auth_json:
        (codex_dir / "auth.json").write_text(codex_auth_json)
        (codex_dir / "auth.json").chmod(0o600)
        print("Codex auth: using ChatGPT subscription (auth.json)", flush=True)
        env.pop("CODEX_API_KEY", None)
    else:
        env["CODEX_API_KEY"] = env.get("OPENAI_API_KEY", "")
        print("Codex auth: using API key", flush=True)

    # Strip other agent keys to prevent cross-contamination
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    env.pop("CODEX_AUTH_JSON", None)

    return _run_agent_subprocess(
        [
            "codex",
            "--search", "exec",
            "--json",
            "-c", "model_reasoning_summary=detailed",
            "--skip-git-repo-check",
            "--yolo",
            "--model", agent_config,
            prompt,
        ],
        task_dir, env, timeout_seconds, stderr_fh, transcript_name,
    )


def run_claude(prompt: str, agent_config: str, task_dir, env: dict,
               timeout_seconds: int, stderr_fh=None,
               transcript_name: str = "transcript.json"):
    """Run the Claude Code agent. Returns (proc, path_to_transcript).

    Authentication: uses CLAUDE_CODE_OAUTH_TOKEN (Max subscription, free) if available,
    otherwise falls back to ANTHROPIC_API_KEY (API billing).
    """
    BASH_MAX_TIMEOUT_MS = 36000000  # 10 hours

    env["BASH_MAX_TIMEOUT_MS"] = str(BASH_MAX_TIMEOUT_MS)
    env["IS_SANDBOX"] = "1"

    # Strip other agent keys
    env.pop("CODEX_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("CODEX_AUTH_JSON", None)

    # Prefer OAuth token (Max subscription) over API key
    oauth_token = env.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if oauth_token:
        env.pop("ANTHROPIC_API_KEY", None)
        print("Claude auth: using OAuth token (Max subscription)", flush=True)
    elif env.get("ANTHROPIC_API_KEY"):
        print("Claude auth: using API key", flush=True)
    else:
        raise RuntimeError(
            "No Claude credentials found. Set either:\n"
            "  - CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`, uses Max subscription)\n"
            "  - ANTHROPIC_API_KEY (API billing)"
        )

    return _run_agent_subprocess(
        [
            "claude",
            "--print",
            "--verbose",
            "--model", agent_config,
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            prompt,
        ],
        task_dir, env, timeout_seconds, stderr_fh, transcript_name,
    )


AGENT_RUNNERS = {
    "codex": run_codex,
    "claude": run_claude,
}
