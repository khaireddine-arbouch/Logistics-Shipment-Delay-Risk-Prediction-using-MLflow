"""Preprocessing pipeline and dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass
class DataSplits:
    """Container for the train/test split and the unfit preprocessing pipeline."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


def load_raw_dataset(path: Path = config.RAW_DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python -m src.data_generation` first."
        )
    df = pd.read_csv(path)
    missing = [c for c in config.FEATURE_COLUMNS + [config.TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median impute + standard scale. Categorical: mode impute + one-hot."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return expanded feature names after one-hot encoding (must be fitted)."""
    return list(preprocessor.get_feature_names_out())


def split_dataset(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
) -> DataSplits:
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    return DataSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=build_preprocessor(),
    )


def load_and_split(
    path: Path = config.RAW_DATASET_PATH,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
) -> DataSplits:
    df = load_raw_dataset(path)
    return split_dataset(df, test_size=test_size, seed=seed)


def schema_summary(df: pd.DataFrame) -> dict[str, object]:
    """Compact schema description for logging as an artifact."""
    return {
        "n_rows": int(len(df)),
        "columns": {
            col: {
                "dtype": str(df[col].dtype),
                "n_missing": int(df[col].isna().sum()),
                "n_unique": int(df[col].nunique(dropna=True)),
            }
            for col in df.columns
        },
        "target_distribution": {
            str(k): int(v)
            for k, v in df[config.TARGET_COLUMN].value_counts().to_dict().items()
        }
        if config.TARGET_COLUMN in df.columns
        else None,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
    }


__all__ = [
    "DataSplits",
    "build_preprocessor",
    "get_feature_names",
    "load_and_split",
    "load_raw_dataset",
    "schema_summary",
    "split_dataset",
]
