"""Central configuration: paths, MLflow settings, schema, and seeds.

Anything that's used in more than one module should live here. This keeps
training, monitoring, and serving aligned on the same column names and
MLflow tracking destination.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MONITORING_DIR: Path = DATA_DIR / "monitoring"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

RAW_DATASET_PATH: Path = RAW_DATA_DIR / "train_shipments.csv"

MONITORING_BATCHES: dict[str, str] = {
    "stable": "batch_stable.csv",
    "weather_drift": "batch_weather_drift.csv",
    "congestion_drift": "batch_congestion_drift.csv",
    "carrier_drift": "batch_carrier_drift.csv",
    "route_drift": "batch_route_drift.csv",
}

for _path in (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MONITORING_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
):
    _path.mkdir(parents=True, exist_ok=True)

# Default to a local sqlite store so `python -m src.train` works without
# starting a separate MLflow server. Override via MLFLOW_TRACKING_URI to
# point at e.g. http://127.0.0.1:5000 once the tracking server is running.
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)

MLFLOW_TRAINING_EXPERIMENT: str = os.getenv(
    "MLFLOW_TRAINING_EXPERIMENT", "logistics-delay-risk"
)
MLFLOW_TUNING_EXPERIMENT: str = os.getenv(
    "MLFLOW_TUNING_EXPERIMENT", "logistics-delay-risk-tuning"
)
MLFLOW_MONITORING_EXPERIMENT: str = os.getenv(
    "MLFLOW_MONITORING_EXPERIMENT", "logistics-delay-monitoring"
)

REGISTERED_MODEL_NAME: str = os.getenv(
    "MLFLOW_REGISTERED_MODEL", "LogisticsDelayRiskModel"
)

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE: float = 0.2

TARGET_COLUMN: str = "is_delayed"

NUMERIC_FEATURES: list[str] = [
    "route_distance_km",
    "planned_transit_days",
    "customs_complexity_score",
    "weather_risk_score",
    "historical_route_delay_rate",
    "cargo_value_usd",
    "port_congestion_score",
]

CATEGORICAL_FEATURES: list[str] = [
    "shipment_mode",
    "carrier_type",
    "origin_region",
    "destination_region",
    "shipment_priority",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORICAL_VALUES: dict[str, list[str]] = {
    "shipment_mode": ["air", "sea", "road", "rail"],
    "carrier_type": ["premium", "standard", "low_cost"],
    "origin_region": ["Europe", "MENA", "Asia", "Africa"],
    "destination_region": ["Europe", "MENA", "Asia", "Africa"],
    "shipment_priority": ["low", "normal", "high", "critical"],
}

DATASET_SIZE: int = 6000
MONITORING_BATCH_SIZE: int = 1000

# Fraction of labels randomly flipped after the deterministic threshold.
# Small but non-zero so the task isn't trivially separable.
LABEL_NOISE_RATE: float = 0.05
