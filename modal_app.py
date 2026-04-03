"""
Modal app for autoresearch. Defines GPU functions for training and data prep.

Usage:
    uv run modal_app.py prepare              # one-time: populate Volume with data + tokenizer
    uv run modal_app.py prepare --num-shards 20  # download more shards
"""

import modal

app = modal.App("autoresearch")

# Volume for data shards + tokenizer (persists across runs)
data_volume = modal.Volume.from_name("autoresearch-data", create_if_missing=True)
DATA_MOUNT = "/root/.cache/autoresearch"

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "kernels>=0.11.7",
        "numpy>=2.2.6",
        "pyarrow>=21.0.0",
        "requests>=2.32.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
    )
)


@app.function(
    image=image,
    gpu="H100",
    volumes={DATA_MOUNT: data_volume},
    timeout=600,
)
def run_training(train_py: str, prepare_py: str) -> dict:
    """Run train.py on an H100. Returns stdout, stderr, and exit code."""
    import os
    import subprocess
    import tempfile

    workspace = tempfile.mkdtemp()
    train_path = os.path.join(workspace, "train.py")
    prepare_path = os.path.join(workspace, "prepare.py")

    with open(train_path, "w") as f:
        f.write(train_py)
    with open(prepare_path, "w") as f:
        f.write(prepare_py)

    env = os.environ.copy()
    env["MODAL_IS_REMOTE"] = "1"

    result = subprocess.run(
        ["python", train_path],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=540,  # 9 min hard kill (training budget is 5 min + startup)
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
    }


@app.function(
    image=image,
    volumes={DATA_MOUNT: data_volume},
    timeout=3600,
)
def prepare_data_with_content(prepare_py: str, num_shards: int = 10):
    """Run prepare.py on Modal to populate the data Volume."""
    import os
    import subprocess
    import tempfile

    workspace = tempfile.mkdtemp()
    prepare_path = os.path.join(workspace, "prepare.py")

    with open(prepare_path, "w") as f:
        f.write(prepare_py)

    result = subprocess.run(
        ["python", prepare_path, "--num-shards", str(num_shards)],
        cwd=workspace,
        text=True,
        timeout=3500,
    )

    if result.returncode != 0:
        raise RuntimeError(f"prepare.py failed with exit code {result.returncode}")

    # Commit Volume so data persists
    data_volume.commit()
    print(f"Data prepared with {num_shards} shards. Volume committed.")


@app.local_entrypoint()
def main(command: str = "prepare", num_shards: int = 10):
    """CLI entrypoint: uv run modal_app.py [prepare|train]"""
    import os

    if command == "prepare":
        prepare_path = os.path.join(os.path.dirname(__file__), "prepare.py")
        with open(prepare_path) as f:
            prepare_py = f.read()
        prepare_data_with_content.remote(prepare_py, num_shards)
        print("Done. Data volume populated.")
    else:
        print(f"Unknown command: {command}")
        print("Usage: modal run modal_app.py [--command prepare] [--num-shards 10]")
