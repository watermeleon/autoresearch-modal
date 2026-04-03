# autoresearch-modal

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that runs training on [Modal](https://modal.com) GPUs instead of requiring a local GPU.

The agent runs on any machine (laptop, CPU server, etc.) and edits `train.py` as usual. When it runs `uv run train.py`, the training is transparently dispatched to a Modal H100 and the output is streamed back. The agent doesn't know Modal exists.

## How it works

A small preamble at the top of `train.py` detects that there's no local GPU and calls `modal_dispatch.py`, which sends the current `train.py` + `prepare.py` to a Modal GPU function. The Modal container runs training, and stdout/stderr are returned to the local machine exactly as if training ran locally.

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
