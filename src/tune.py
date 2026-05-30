"""Hyperparameter tuning with Optuna and nested MLflow runs.

A parent run, `optuna_tuning_<model>`, owns the search. Each Optuna trial
becomes a nested MLflow run that logs its sampled hyperparameters and the
mean cross-validated metric. After the search, the best parameters are
written to the parent run, the final model is retrained on the full training
split, and a separate run under the regular training experiment logs the
tuned model so the registry script can pick it up alongside the baselines.

Run as a CLI:

    python -m src.tune
    python -m src.tune --model random_forest --trials 30
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow.models.signature import infer_signature
from optuna.samplers import TPESampler
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import config, preprocessing, utils


@dataclass
class TunableModel:
    name: str
    factory: Callable[[dict[str, Any], int], ClassifierMixin]
    space: Callable[[optuna.Trial], dict[str, Any]]


def _gb_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


def _gb_factory(params: dict[str, Any], seed: int) -> ClassifierMixin:
    return GradientBoostingClassifier(random_state=seed, **params)


def _rf_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _rf_factory(params: dict[str, Any], seed: int) -> ClassifierMixin:
    return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)


TUNABLE: dict[str, TunableModel] = {
    "gradient_boosting": TunableModel(
        name="gradient_boosting", factory=_gb_factory, space=_gb_space
    ),
    "random_forest": TunableModel(
        name="random_forest", factory=_rf_factory, space=_rf_space
    ),
}


def _build_pipeline(spec: TunableModel, params: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            ("model", spec.factory(params, seed)),
        ]
    )


def _objective(
    trial: optuna.Trial,
    spec: TunableModel,
    splits: preprocessing.DataSplits,
    metric: str,
    seed: int,
    cv_splits: int,
) -> float:
    params = spec.space(trial)
    pipeline = _build_pipeline(spec, params, seed)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(
        pipeline,
        splits.X_train,
        splits.y_train,
        scoring=metric,
        cv=cv,
        n_jobs=-1,
    )
    score = float(np.mean(scores))

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric(f"cv_{metric}_mean", score)
        mlflow.log_metric(f"cv_{metric}_std", float(np.std(scores)))
    return score


def _compute_test_metrics(
    pipeline: Pipeline, splits: preprocessing.DataSplits
) -> dict[str, float]:
    y_pred = pipeline.predict(splits.X_test)
    y_proba = pipeline.predict_proba(splits.X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(splits.y_test, y_pred)),
        "precision": float(precision_score(splits.y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(splits.y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(splits.y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(splits.y_test, y_proba)),
    }, y_pred


def tune_model(
    model_name: str,
    n_trials: int,
    metric: str,
    seed: int,
    cv_splits: int,
    tuning_experiment: str,
    training_experiment: str,
) -> dict[str, Any]:
    if model_name not in TUNABLE:
        raise ValueError(
            f"Unknown model {model_name!r}. Choose from {sorted(TUNABLE)}."
        )
    spec = TUNABLE[model_name]

    splits = preprocessing.load_and_split(seed=seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name=f"optuna_{spec.name}",
    )

    utils.configure_mlflow(tuning_experiment)
    parent_run_name = f"optuna_tuning_{spec.name}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.log_params(
            {
                "tuner": "optuna",
                "sampler": "TPE",
                "model_type": spec.name,
                "n_trials": n_trials,
                "cv_splits": cv_splits,
                "metric": metric,
                "random_seed": seed,
            }
        )

        study.optimize(
            lambda trial: _objective(trial, spec, splits, metric, seed, cv_splits),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_value = float(study.best_value)
        mlflow.log_metric(f"best_cv_{metric}", best_value)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        utils.log_dict_artifact(
            {
                "best_params": best_params,
                "best_cv_metric": best_value,
                "metric": metric,
                "n_trials": n_trials,
                "cv_splits": cv_splits,
            },
            "best_params.json",
            artifact_dir="tuning",
        )
        parent_run_id = parent_run.info.run_id

    print(
        f"\n[{spec.name}] best cv_{metric}={best_value:.4f} with params={best_params}"
    )

    final_pipeline = _build_pipeline(spec, best_params, seed)
    final_pipeline.fit(splits.X_train, splits.y_train)
    test_metrics, y_pred = _compute_test_metrics(final_pipeline, splits)

    schema_info = preprocessing.schema_summary(
        pd.concat([splits.X_train, splits.y_train], axis=1)
    )

    # Final tuned-model run lives in the training experiment so it sits
    # alongside the baseline runs and `register_model.py` can rank them all
    # together.
    utils.configure_mlflow(training_experiment)
    final_run_name = f"tuned_{spec.name}"
    with mlflow.start_run(run_name=final_run_name) as final_run:
        mlflow.log_params(
            {
                "model_type": spec.name,
                "tuned": True,
                "tuning_parent_run_id": parent_run_id,
                "random_seed": seed,
                **{f"hp_{k}": v for k, v in best_params.items()},
            }
        )
        mlflow.log_metrics(test_metrics)

        utils.log_confusion_matrix(splits.y_test, y_pred, title=final_run_name)
        utils.log_classification_report(splits.y_test, y_pred, title=final_run_name)
        utils.log_dict_artifact(schema_info, "dataset_schema.json", "schema")

        importances = getattr(final_pipeline.named_steps["model"], "feature_importances_", None)
        if importances is not None:
            feature_names = list(
                final_pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
            utils.log_feature_importance(
                feature_names, np.asarray(importances), final_run_name
            )

        signature = infer_signature(splits.X_test.head(5), y_pred[:5])
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="model",
            signature=signature,
            input_example=splits.X_test.head(2),
        )
        final_run_id = final_run.info.run_id

    print(
        f"[{spec.name}] tuned test f1={test_metrics['f1']:.4f} "
        f"roc_auc={test_metrics['roc_auc']:.4f} run_id={final_run_id}"
    )

    return {
        "model": spec.name,
        "best_params": best_params,
        "best_cv_metric": best_value,
        "test_metrics": test_metrics,
        "parent_run_id": parent_run_id,
        "final_run_id": final_run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(TUNABLE.keys()),
        default="gradient_boosting",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--metric",
        default="f1",
        choices=["f1", "roc_auc", "accuracy"],
    )
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--training-experiment",
        default=config.MLFLOW_TRAINING_EXPERIMENT,
        help="Experiment that holds the final tuned-model run",
    )
    parser.add_argument(
        "--tuning-experiment",
        default=config.MLFLOW_TUNING_EXPERIMENT,
        help="Experiment that holds the parent + nested trial runs",
    )
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")
    print(f"Tuning experiment:   {args.tuning_experiment}")
    print(f"Training experiment: {args.training_experiment}")

    result = tune_model(
        model_name=args.model,
        n_trials=args.trials,
        metric=args.metric,
        seed=args.seed,
        cv_splits=args.cv,
        tuning_experiment=args.tuning_experiment,
        training_experiment=args.training_experiment,
    )

    print(
        f"\nTuned model logged into experiment '{args.training_experiment}' "
        f"as run {result['final_run_id']}"
    )


if __name__ == "__main__":
    main()
