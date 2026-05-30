"""Copy the latest run artifacts into reports/figures for the report."""

from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

REPORTS = Path("reports/figures")
REPORTS.mkdir(parents=True, exist_ok=True)


def latest_run_for(name: str, experiment: str) -> str | None:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"attributes.run_name = '{name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def copy_artifact(run_id: str, artifact_path: str, dest_name: str) -> None:
    local_dir = Path(client.download_artifacts(run_id, artifact_path))
    dest = REPORTS / dest_name
    shutil.copy(local_dir, dest)
    print(f"  {artifact_path} -> {dest}")


targets = [
    ("logistic_regression", "plots/confusion_matrix_logistic_regression.png", "confusion_matrix_logistic_regression.png"),
    ("logistic_regression", "plots/feature_importance_logistic_regression.png", "feature_importance_logistic_regression.png"),
    ("random_forest", "plots/confusion_matrix_random_forest.png", "confusion_matrix_random_forest.png"),
    ("random_forest", "plots/feature_importance_random_forest.png", "feature_importance_random_forest.png"),
    ("gradient_boosting", "plots/confusion_matrix_gradient_boosting.png", "confusion_matrix_gradient_boosting.png"),
    ("gradient_boosting", "plots/feature_importance_gradient_boosting.png", "feature_importance_gradient_boosting.png"),
    ("tuned_gradient_boosting", "plots/confusion_matrix_tuned_gradient_boosting.png", "confusion_matrix_tuned_gradient_boosting.png"),
    ("tuned_gradient_boosting", "plots/feature_importance_tuned_gradient_boosting.png", "feature_importance_tuned_gradient_boosting.png"),
]

for run_name, art, dest in targets:
    run_id = latest_run_for(run_name, "logistics-delay-risk")
    if run_id is None:
        print(f"  [skip] no run named {run_name}")
        continue
    print(f"{run_name} -> {run_id}")
    try:
        copy_artifact(run_id, art, dest)
    except Exception as exc:
        print(f"  [skip] {art}: {exc}")

monitoring_targets = [
    "stable",
    "weather_drift",
    "congestion_drift",
    "carrier_drift",
    "route_drift",
]
for batch in monitoring_targets:
    run_id = latest_run_for(f"monitoring_{batch}", "logistics-delay-monitoring")
    if run_id is None:
        print(f"  [skip] no run for monitoring_{batch}")
        continue
    print(f"monitoring_{batch} -> {run_id}")
    try:
        copy_artifact(
            run_id,
            f"plots/prediction_distribution_{batch}.png",
            f"prediction_distribution_{batch}.png",
        )
    except Exception as exc:
        print(f"  [skip] {batch}: {exc}")

print("\nDone.")
