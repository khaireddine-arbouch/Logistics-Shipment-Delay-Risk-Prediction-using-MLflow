"""Monitoring runner: log batch performance and drift to MLflow.

For each pre-generated monitoring batch under `data/monitoring/`, this script:

1. Loads the production model from the registry.
2. Predicts the batch.
3. Compares feature distributions to the reference training set.
4. Logs an MLflow run under the `logistics-delay-monitoring` experiment with
   batch size, prediction stats, performance metrics (where labels exist),
   and per-feature drift scores.
5. Saves a `monitoring_report.json`, a `drift_summary.csv`, and a
   `prediction_distribution.png` artifact per batch.

Run as a CLI:

    python -m src.monitor
    python -m src.monitor --batch weather_drift
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from . import config, utils

DRIFT_NUMERIC_THRESHOLD: float = 0.50  # normalized mean/std diff for numeric features
DRIFT_CATEGORICAL_THRESHOLD: float = 0.20  # total-variation distance for categorical features


def _load_batch(batch_name: str) -> tuple[pd.DataFrame, Path]:
    if batch_name not in config.MONITORING_BATCHES:
        raise ValueError(
            f"Unknown batch '{batch_name}'. Choose from "
            f"{list(config.MONITORING_BATCHES.keys())}."
        )
    path = config.MONITORING_DIR / config.MONITORING_BATCHES[batch_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Batch file not found at {path}. "
            "Run `python -m src.data_generation` first."
        )
    return pd.read_csv(path), path


def _load_reference() -> pd.DataFrame:
    if not config.RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Training set not found at {config.RAW_DATASET_PATH}. "
            "Run `python -m src.data_generation` first."
        )
    return pd.read_csv(config.RAW_DATASET_PATH)


def _load_production_model(alias: str) -> tuple[Pipeline, str]:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, alias)
    uri = f"models:/{config.REGISTERED_MODEL_NAME}@{alias}"
    # Native sklearn load so we can call .predict_proba — pyfunc would return
    # class labels only and we need probabilities for ROC-AUC and the
    # prediction-distribution plot.
    model = mlflow.sklearn.load_model(uri)
    return model, str(mv.version)


def _predict(model: Pipeline, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[config.FEATURE_COLUMNS]
    probs = model.predict_proba(X)[:, 1].astype(float)
    labels = (probs >= 0.5).astype(int)
    return labels, probs


def _numeric_drift(reference: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    """Per-numeric-feature normalized mean / std drift."""
    rows = []
    for col in config.NUMERIC_FEATURES:
        ref_mean, ref_std = float(reference[col].mean()), float(reference[col].std())
        bat_mean, bat_std = float(batch[col].mean()), float(batch[col].std())
        denom = max(ref_std, 1e-9)
        norm_mean_diff = abs(bat_mean - ref_mean) / denom
        norm_std_ratio = abs(bat_std - ref_std) / denom
        rows.append(
            {
                "feature": col,
                "type": "numeric",
                "reference_mean": ref_mean,
                "batch_mean": bat_mean,
                "reference_std": ref_std,
                "batch_std": bat_std,
                "normalized_mean_diff": norm_mean_diff,
                "normalized_std_diff": norm_std_ratio,
                "drift_score": float(max(norm_mean_diff, norm_std_ratio)),
            }
        )
    return pd.DataFrame(rows)


def _categorical_drift(reference: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    """Total-variation distance between category frequency distributions."""
    rows = []
    for col in config.CATEGORICAL_FEATURES:
        ref_freq = reference[col].value_counts(normalize=True)
        bat_freq = batch[col].value_counts(normalize=True)
        all_categories = ref_freq.index.union(bat_freq.index)
        ref = ref_freq.reindex(all_categories, fill_value=0.0)
        bat = bat_freq.reindex(all_categories, fill_value=0.0)
        tvd = float(0.5 * np.abs(ref.to_numpy() - bat.to_numpy()).sum())
        rows.append(
            {
                "feature": col,
                "type": "categorical",
                "tvd": tvd,
                "drift_score": tvd,
            }
        )
    return pd.DataFrame(rows)


def _save_prediction_plot(probs: np.ndarray, batch_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(probs, bins=25, color="#3b82f6", edgecolor="white")
    ax.set_title(f"Predicted delay probability — {batch_name}")
    ax.set_xlabel("P(delayed)")
    ax.set_ylabel("count")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out_path = Path(tempfile.mkdtemp()) / f"prediction_distribution_{batch_name}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _log_drift_summary(numeric_df: pd.DataFrame, categorical_df: pd.DataFrame) -> Path:
    summary = pd.concat([numeric_df, categorical_df], ignore_index=True)
    out_path = Path(tempfile.mkdtemp()) / "drift_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def run_monitoring_batch(
    batch_name: str,
    model: Pipeline,
    model_version: str,
    reference: pd.DataFrame,
    numeric_threshold: float,
    categorical_threshold: float,
) -> dict:
    batch, path = _load_batch(batch_name)
    labels_pred, probs = _predict(model, batch)

    has_labels = config.TARGET_COLUMN in batch.columns
    perf_metrics: dict[str, float] = {}
    if has_labels:
        y_true = batch[config.TARGET_COLUMN].to_numpy()
        perf_metrics = {
            "accuracy": float(accuracy_score(y_true, labels_pred)),
            "precision": float(precision_score(y_true, labels_pred, zero_division=0)),
            "recall": float(recall_score(y_true, labels_pred, zero_division=0)),
            "f1": float(f1_score(y_true, labels_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
        }

    numeric_df = _numeric_drift(reference, batch)
    categorical_df = _categorical_drift(reference, batch)
    numeric_drift_score = float(numeric_df["drift_score"].max())
    categorical_drift_score = float(categorical_df["drift_score"].max())
    drift_detected = int(
        numeric_drift_score >= numeric_threshold
        or categorical_drift_score >= categorical_threshold
    )

    metrics = {
        "batch_size": float(len(batch)),
        "positive_prediction_rate": float(labels_pred.mean()),
        "average_delay_probability": float(probs.mean()),
        "numeric_drift_score": numeric_drift_score,
        "categorical_drift_score": categorical_drift_score,
        "drift_detected": float(drift_detected),
        **perf_metrics,
    }

    drift_csv = _log_drift_summary(numeric_df, categorical_df)
    plot_path = _save_prediction_plot(probs, batch_name)
    report = {
        "batch": batch_name,
        "batch_path": str(path),
        "n_rows": int(len(batch)),
        "model_name": config.REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "metrics": metrics,
        "numeric_threshold": numeric_threshold,
        "categorical_threshold": categorical_threshold,
        "drift_detected": bool(drift_detected),
        "top_numeric_drift": numeric_df.sort_values("drift_score", ascending=False)
        .head(3)
        .to_dict(orient="records"),
        "top_categorical_drift": categorical_df.sort_values("drift_score", ascending=False)
        .head(3)
        .to_dict(orient="records"),
    }

    with mlflow.start_run(run_name=f"monitoring_{batch_name}"):
        mlflow.log_params(
            {
                "batch": batch_name,
                "model_name": config.REGISTERED_MODEL_NAME,
                "model_version": model_version,
                "numeric_threshold": numeric_threshold,
                "categorical_threshold": categorical_threshold,
                "n_rows": len(batch),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(drift_csv), artifact_path="drift")
        mlflow.log_artifact(str(plot_path), artifact_path="plots")
        utils.log_dict_artifact(report, "monitoring_report.json", "monitoring")

    summary_line = (
        f"[{batch_name:<17}] n={len(batch):>4}  "
        f"pos_rate={metrics['positive_prediction_rate']:.3f}  "
        f"avg_prob={metrics['average_delay_probability']:.3f}  "
        f"num_drift={numeric_drift_score:.3f}  "
        f"cat_drift={categorical_drift_score:.3f}  "
        f"drift={'YES' if drift_detected else 'no'}"
    )
    if perf_metrics:
        summary_line += (
            f"  acc={perf_metrics['accuracy']:.3f}  f1={perf_metrics['f1']:.3f}"
        )
    print(summary_line)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        choices=list(config.MONITORING_BATCHES.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--alias",
        default="production",
        help="Registry alias for the model to evaluate",
    )
    parser.add_argument(
        "--experiment",
        default=config.MLFLOW_MONITORING_EXPERIMENT,
    )
    parser.add_argument(
        "--numeric-threshold",
        type=float,
        default=DRIFT_NUMERIC_THRESHOLD,
        help="Numeric drift score above which `drift_detected` flips to 1",
    )
    parser.add_argument(
        "--categorical-threshold",
        type=float,
        default=DRIFT_CATEGORICAL_THRESHOLD,
        help="Categorical drift score (TVD) above which `drift_detected` flips to 1",
    )
    args = parser.parse_args()

    utils.configure_mlflow(args.experiment)
    print(f"MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")
    print(f"Monitoring experiment: {args.experiment}")

    model, model_version = _load_production_model(args.alias)
    print(
        f"Loaded model '{config.REGISTERED_MODEL_NAME}@{args.alias}' "
        f"(version {model_version})"
    )
    reference = _load_reference()

    batches: Iterable[str] = (
        config.MONITORING_BATCHES.keys() if args.batch == "all" else [args.batch]
    )
    for batch in batches:
        run_monitoring_batch(
            batch_name=batch,
            model=model,
            model_version=model_version,
            reference=reference,
            numeric_threshold=args.numeric_threshold,
            categorical_threshold=args.categorical_threshold,
        )


if __name__ == "__main__":
    main()
