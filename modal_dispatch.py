"""
Local dispatch: sends train.py to Modal for GPU execution.
Called by the preamble in train.py when no local GPU is available.
"""

import os
import sys


def run_on_modal(train_py_path: str) -> int:
    """Read train.py + prepare.py, dispatch to Modal, print output, return exit code."""
    from modal_app import app, run_training

    # Read current file contents
    base_dir = os.path.dirname(os.path.abspath(train_py_path))

    with open(train_py_path) as f:
        train_py = f.read()

    prepare_path = os.path.join(base_dir, "prepare.py")
    with open(prepare_path) as f:
        prepare_py = f.read()

    # Start the Modal app and dispatch
    with app.run():
        result = run_training.remote(train_py, prepare_py)

    # Print output exactly as if it ran locally
    if result["stdout"]:
        sys.stdout.write(result["stdout"])
        sys.stdout.flush()
    if result["stderr"]:
        sys.stderr.write(result["stderr"])
        sys.stderr.flush()

    return result["exit_code"]
