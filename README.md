# autoresearch-modal

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that runs training on [Modal](https://modal.com) GPUs. In the original repo, GPU time burns while the agent thinks or codes. Here, Modal allows us to only pay for GPU time when training is happening.

The agent runs on any machine (laptop, CPU server, etc.) and edits `train.py` as usual. When it runs `uv run train.py`, the training is transparently dispatched to a Modal H100 and the output is streamed back. The agent doesn't need to know Modal exists.

## How it works

A small preamble at the top of `train.py` checks whether it's running inside Modal; if not, it dispatches the run to a Modal GPU via `modal_dispatch.py`, which sends the current `train.py` + `prepare.py` to a Modal GPU function. The Modal container runs training, and stdout/stderr are returned to the local machine exactly as if training ran locally.

```
Agent's machine (no GPU)          Modal (H100, ~5 min)
─────────────────────────         ────────────────────
1. Agent edits train.py
2. uv run train.py
3. Preamble → modal_dispatch  ──→ 4. Writes files to container
                                  5. Runs python train.py
                              ←── 6. Returns stdout + exit code
7. Output written to run.log
8. Agent reads results
```

## Validation (can you trust this?)
- uv run train.py on a Mac (no GPU) → dispatches to Modal H100
- Run the original karpathy train.py directly on a GPU
- These two setups produce the same output format, the same exit codes, and the same results (within GPU variance)

## Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Authenticate with Modal
modal token set

# 4. Prepare data on Modal (one-time, downloads shards + trains tokenizer)
modal run modal_app.py --command prepare --num-shards 10

# 5. Run a single training experiment (dispatches to Modal H100)
uv run train.py
```

## Running the agent

Same as the original — point your agent at `program.md`:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Project structure

```
train.py          — model, optimizer, training loop (agent modifies this)
prepare.py        — constants, data prep + runtime utilities (do not modify)
program.md        — agent instructions
modal_app.py      — Modal function definitions (GPU training + data prep)
modal_dispatch.py — local dispatch logic (sends train.py to Modal)
pyproject.toml    — dependencies
```

## Credits

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). See the original repo for full context on the project.

## License

MIT
