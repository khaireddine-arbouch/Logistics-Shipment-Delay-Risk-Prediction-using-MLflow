"""Shared helpers: MLflow setup, plotting, and small file utilities."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend so scripts work outside an IDE
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from . import config


def configure_mlflow(experiment_name: str) -> None:
    """Point MLflow at the configured tracking store and experiment."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def log_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    title: str,
    artifact_dir: str = "plots",
) -> Path:
    """Save and log a confusion matrix plot, returning its local path."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["on_time", "delayed"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()

    out_path = Path(tempfile.mkdtemp()) / f"confusion_matrix_{_slug(title)}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    mlflow.log_artifact(str(out_path), artifact_path=artifact_dir)
    return out_path


def log_classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    title: str,
    artifact_dir: str = "reports",
) -> Path:
    """Persist a JSON classification report and log it as an artifact."""
    report: dict[str, Any] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    out_path = Path(tempfile.mkdtemp()) / f"classification_report_{_slug(title)}.json"
    out_path.write_text(json.dumps(report, indent=2))
    mlflow.log_artifact(str(out_path), artifact_path=artifact_dir)
    return out_path


def log_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    title: str,
    artifact_dir: str = "plots",
    top_n: int = 20,
) -> Path | None:
    """Log a bar plot of feature importances if the model exposes them."""
    if importances is None or len(importances) == 0:
        return None

    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(order))))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#3b82f6",
    )
    ax.set_title(f"Top {len(order)} feature importances — {title}")
    ax.set_xlabel("importance")
    fig.tight_layout()

    out_path = Path(tempfile.mkdtemp()) / f"feature_importance_{_slug(title)}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    mlflow.log_artifact(str(out_path), artifact_path=artifact_dir)
    return out_path


def log_dict_artifact(payload: dict[str, Any], filename: str, artifact_dir: str) -> Path:
    """Write a dict as JSON and log it as an MLflow artifact."""
    out_path = Path(tempfile.mkdtemp()) / filename
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    mlflow.log_artifact(str(out_path), artifact_path=artifact_dir)
    return out_path


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
