"""Train the baseline lineup and log everything to MLflow.

Trains Logistic Regression, Random Forest, and Gradient Boosting using a
preprocessing pipeline, logs params, metrics, plots, and the fitted model
to MLflow as one run per model under the training experiment.

Run as a CLI:

    python -m src.train
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from . import config, preprocessing, utils


@dataclass
class ModelSpec:
    name: str
    estimator: ClassifierMixin
    params: dict[str, Any]


def build_model_specs(seed: int = config.RANDOM_SEED) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", random_state=seed
            ),
            params={"max_iter": 1000, "C": 1.0, "solver": "lbfgs"},
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1,
                random_state=seed,
            ),
            params={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
            },
        ),
        ModelSpec(
            name="gradient_boosting",
            estimator=GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=seed,
            ),
            params={
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 3,
            },
        ),
    ]


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _extract_feature_importances(
    pipeline: Pipeline,
) -> tuple[list[str], np.ndarray] | None:
    """Pull feature importances out of a fitted pipeline if available."""
    preproc = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = list(preproc.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        return feature_names, np.asarray(model.feature_importances_)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        return feature_names, np.abs(coef)
    return None


def train_one_model(
    spec: ModelSpec,
    splits: preprocessing.DataSplits,
    schema_info: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """Train a single model and log everything under one MLflow run."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", splits.preprocessor),
            ("model", spec.estimator),
        ]
    )

    with mlflow.start_run(run_name=spec.name):
        mlflow.log_params(
            {
                "model_type": spec.name,
                "random_seed": seed,
                "test_size": config.TEST_SIZE,
                "feature_set_version": "v1",
                "n_numeric_features": len(config.NUMERIC_FEATURES),
                "n_categorical_features": len(config.CATEGORICAL_FEATURES),
                **{f"hp_{k}": v for k, v in spec.params.items()},
            }
        )

        pipeline.fit(splits.X_train, splits.y_train)

        y_pred = pipeline.predict(splits.X_test)
        y_proba = pipeline.predict_proba(splits.X_test)[:, 1]
        metrics = compute_metrics(
            splits.y_test.to_numpy(), np.asarray(y_pred), np.asarray(y_proba)
        )
        mlflow.log_metrics(metrics)

        utils.log_confusion_matrix(splits.y_test, y_pred, title=spec.name)
        utils.log_classification_report(splits.y_test, y_pred, title=spec.name)
        utils.log_dict_artifact(schema_info, "dataset_schema.json", "schema")

        importance = _extract_feature_importances(pipeline)
        if importance is not None:
            utils.log_feature_importance(importance[0], importance[1], spec.name)

        signature = infer_signature(splits.X_test.head(5), y_pred[:5])
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=splits.X_test.head(2),
        )

        run = mlflow.active_run()
        print(
            f"[{spec.name}] f1={metrics['f1']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} "
            f"run_id={run.info.run_id}"
        )
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default=config.MLFLOW_TRAINING_EXPERIMENT,
        help="MLflow experiment name",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    utils.configure_mlflow(args.experiment)
    print(f"MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")
    print(f"Experiment: {args.experiment}")

    splits = preprocessing.load_and_split(seed=args.seed)
    schema_info = preprocessing.schema_summary(
        pd.concat([splits.X_train, splits.y_train], axis=1)
    )

    results: list[tuple[str, dict[str, float]]] = []
    for spec in build_model_specs(seed=args.seed):
        metrics = train_one_model(spec, splits, schema_info, args.seed)
        results.append((spec.name, metrics))

    print("\n=== Training summary ===")
    summary = pd.DataFrame(
        [{"model": name, **metrics} for name, metrics in results]
    ).set_index("model")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
