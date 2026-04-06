"""Immutable evaluation module for tennis prediction pipeline.

This file is IMMUTABLE — it must not be modified by auto-research agents.
gate.sh enforces this via git diff checks.

Why: evaluation logic was previously inside models.py (mutable), which allowed
agents to inject post-hoc probability adjustments that gamed validation scores
without genuinely improving the model. By extracting scoring/metric computation
into an immutable file, the evaluation path is tamper-proof.

The functions here receive a trained model and data, run predictions via the
model's predict_proba method, and compute metrics. The model's training,
architecture, and feature engineering remain in models.py (mutable).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline


def evaluate_model(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Fit model on train data and evaluate on test data."""
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1].astype(float)
    predictions = (probabilities >= 0.5).astype(int)
    scores = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities))
        if len(np.unique(y_test)) > 1
        else float("nan"),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
    }
    return scores


def extract_feature_importances(
    model: Any,
) -> pd.DataFrame | None:
    """Extract feature importance from fitted XGBoost model.

    Handles Pipeline, WeightedXGBoostEnsemble, and SegmentBlendModel types
    by duck-typing their attributes rather than importing model classes.
    """
    # SegmentBlendModel: delegate to global model
    if hasattr(model, "primary_model_"):
        return extract_feature_importances(model.primary_model_)
    # WeightedXGBoostEnsemble: use primary estimator + preprocessor
    if hasattr(model, "primary_estimator_") and hasattr(model, "preprocessor_"):
        estimator = model.primary_estimator_
        preprocessor = model.preprocessor_
    # Pipeline: use named steps
    elif hasattr(model, "named_steps"):
        estimator = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]
    else:
        return None
    if not hasattr(estimator, "feature_importances_"):
        return None
    feature_names = preprocessor.get_feature_names_out()
    importances = estimator.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table."""
    headers = list(frame.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header_row, separator_row, *data_rows])


def markdown_report(
    summary: dict[str, Any],
    scores: dict[str, float],
    feature_importances: pd.DataFrame | None,
    per_tournament: pd.DataFrame | None = None,
) -> str:
    """Generate a markdown report for a training run."""
    tour = summary["tour"].upper()
    test_mode = summary.get("test_mode", "event")
    lines = [
        f"# {tour} XGBoost Report",
        "",
        f"- tour: `{summary['tour']}`",
        f"- train rows: `{summary['train_rows']}`",
        f"- test rows: `{summary['test_rows']}`",
        f"- cutoff date: `{summary['cutoff_date']}`",
        f"- test mode: `{test_mode}`",
    ]
    if test_mode == "event":
        lines.append(f"- test event: `{summary['test_event']}` ({summary['test_year']})")
    else:
        lines.append(f"- test set: all matches after cutoff ({summary.get('test_tournaments', 'N/A')} tournaments)")
    lines.extend([
        "",
        "## Metrics",
        "",
        f"| metric | value |",
        f"| --- | --- |",
    ])
    for metric, value in scores.items():
        lines.append(f"| {metric} | {value:.4f} |")

    if per_tournament is not None and not per_tournament.empty:
        lines.extend([
            "",
            "## Per-Tournament Breakdown",
            "",
            dataframe_to_markdown(per_tournament),
        ])

    if feature_importances is not None and not feature_importances.empty:
        lines.extend([
            "",
            "## Top 20 Features",
            "",
            dataframe_to_markdown(feature_importances.head(20)),
        ])
    return "\n".join(lines)


def per_tournament_accuracy(
    predictions_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compute accuracy breakdown by tournament."""
    if "tourney_name" not in predictions_frame.columns:
        return pd.DataFrame()
    grouped = predictions_frame.groupby("tourney_name").apply(
        lambda g: pd.Series({
            "matches": len(g),
            "correct": int((g["label"] == g["predicted_label"]).sum()),
            "accuracy": float((g["label"] == g["predicted_label"]).mean()),
        }),
        include_groups=False,
    ).reset_index()
    return grouped.sort_values("matches", ascending=False).reset_index(drop=True)
