"""Register the best training runs in the MLflow Model Registry.

Searches the training experiment, picks the top runs by the chosen metric,
registers them as versions of `LogisticsDelayRiskModel`, and assigns aliases
(`production`, `staging`) instead of the deprecated stage transitions.

The script is idempotent: re-running it after another tuning pass simply
adds a new version and reassigns the aliases to the new best version.

Run as a CLI:

    python -m src.register_model
    python -m src.register_model --metric roc_auc --top 3
"""

from __future__ import annotations

import argparse

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException, RestException
from mlflow.tracking import MlflowClient

from . import config, utils


def _find_logged_model_uri(client: MlflowClient, run_id: str, experiment_id: str) -> str:
    """Return the canonical model URI for a logged sklearn model in a run."""
    # MLflow 3 stores the model under the run as a "logged model" with a
    # stable model_id. Use search_logged_models to look it up by run_id.
    try:
        logged_models = client.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"source_run_id='{run_id}'",
            max_results=1,
        )
    except TypeError:
        # Older MLflow signatures may not accept experiment_ids; fall back.
        logged_models = []
    if not logged_models:
        # Fallback: artifact-style URI used in older MLflow versions.
        return f"runs:/{run_id}/model"
    return f"models:/{logged_models[0].model_id}"


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except (RestException, MlflowException):
        client.create_registered_model(
            name=name,
            description=(
                "Logistics shipment delay risk classifier. Versions are "
                "registered from runs in the `logistics-delay-risk` experiment "
                "and managed via aliases (production, staging)."
            ),
        )


def register_top_runs(
    experiment_name: str,
    registered_model_name: str,
    metric: str,
    top_n: int,
) -> list[dict]:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{experiment_name}' not found. "
            "Run `python -m src.train` first."
        )

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=max(top_n, 1) * 4,
    )
    runs_df = runs_df.dropna(subset=[f"metrics.{metric}"])
    if runs_df.empty:
        raise RuntimeError(
            f"No runs in experiment '{experiment_name}' have metric "
            f"'{metric}'. Train models first."
        )

    selected = runs_df.head(top_n)

    _ensure_registered_model(client, registered_model_name)

    print(
        f"Registering top {len(selected)} runs from '{experiment_name}' "
        f"by metric '{metric}' into model '{registered_model_name}':\n"
    )

    results: list[dict] = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        run_id = row["run_id"]
        run_name = row.get("tags.mlflow.runName", "") or "(unnamed)"
        metric_value = float(row[f"metrics.{metric}"])
        model_uri = _find_logged_model_uri(client, run_id, experiment.experiment_id)

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
            tags={
                "source_run_id": run_id,
                "source_run_name": run_name,
                "selection_metric": metric,
                "rank": str(rank),
            },
        )
        version = int(registered.version)

        client.update_model_version(
            name=registered_model_name,
            version=str(version),
            description=(
                f"Registered from run '{run_name}' (run_id={run_id}). "
                f"Selected by {metric}={metric_value:.4f}, rank #{rank} in "
                f"experiment '{experiment_name}'."
            ),
        )

        print(
            f"  v{version:<3} <- {run_name:<24} {metric}={metric_value:.4f} "
            f"run={run_id}"
        )
        results.append(
            {
                "version": version,
                "run_id": run_id,
                "run_name": run_name,
                "metric_value": metric_value,
            }
        )

    return results


def assign_aliases(
    registered_model_name: str,
    registrations: list[dict],
) -> dict[str, int]:
    """Map best/second-best registrations to production/staging aliases."""
    client = MlflowClient()
    if not registrations:
        return {}

    aliases: dict[str, int] = {"production": registrations[0]["version"]}
    if len(registrations) >= 2:
        aliases["staging"] = registrations[1]["version"]

    for alias, version in aliases.items():
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias,
            version=str(version),
        )
        print(f"  alias '{alias}' -> v{version}")

    return aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default=config.MLFLOW_TRAINING_EXPERIMENT,
        help="Experiment to scan for candidate runs",
    )
    parser.add_argument(
        "--name",
        default=config.REGISTERED_MODEL_NAME,
        help="Registered model name",
    )
    parser.add_argument(
        "--metric",
        default="f1",
        choices=["f1", "roc_auc", "accuracy", "precision", "recall"],
    )
    parser.add_argument("--top", type=int, default=2)
    args = parser.parse_args()

    utils.configure_mlflow(args.experiment)
    print(f"MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")

    registrations = register_top_runs(
        experiment_name=args.experiment,
        registered_model_name=args.name,
        metric=args.metric,
        top_n=args.top,
    )

    print("\nAssigning aliases:")
    aliases = assign_aliases(args.name, registrations)

    print("\nDone. Summary:")
    summary = pd.DataFrame(registrations)
    print(summary.to_string(index=False))
    print("\nAliases:", aliases)


if __name__ == "__main__":
    main()
