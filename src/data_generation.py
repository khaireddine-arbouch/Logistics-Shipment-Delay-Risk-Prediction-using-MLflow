"""Generate training and monitoring datasets for logistics delay prediction.

Produces six deterministic CSVs:

- `data/raw/train_shipments.csv`              — main training set
- `data/monitoring/batch_stable.csv`          — same distribution as training
- `data/monitoring/batch_weather_drift.csv`   — higher weather_risk_score
- `data/monitoring/batch_congestion_drift.csv` — higher port_congestion_score
- `data/monitoring/batch_carrier_drift.csv`   — more low_cost carriers
- `data/monitoring/batch_route_drift.csv`     — longer route_distance_km

Drift batches keep the same target-generation logic so the model's predicted
delay rate moves only because the input distribution moves, which is the
behavior the monitoring step needs.

Run as a CLI:

    python -m src.data_generation                 # generate every file
    python -m src.data_generation --only train    # just the training set
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class ScenarioConfig:
    """Knobs for shifting the input distribution per monitoring scenario."""

    name: str
    weather_shift: float = 0.0  # added to weather_risk_score (clipped to [0, 10])
    congestion_shift: float = 0.0
    carrier_probs: tuple[float, float, float] | None = None  # (premium, standard, low_cost)
    route_distance_multiplier: float = 1.0


SCENARIOS: dict[str, ScenarioConfig] = {
    "train": ScenarioConfig(name="train"),
    "stable": ScenarioConfig(name="stable"),
    "weather_drift": ScenarioConfig(name="weather_drift", weather_shift=3.0),
    "congestion_drift": ScenarioConfig(name="congestion_drift", congestion_shift=3.0),
    "carrier_drift": ScenarioConfig(
        name="carrier_drift", carrier_probs=(0.10, 0.30, 0.60)
    ),
    "route_drift": ScenarioConfig(
        name="route_drift", route_distance_multiplier=1.8
    ),
}


@dataclass
class _SeedPlan:
    """Per-scenario seed offsets keep batches independent but reproducible."""

    seeds: dict[str, int] = field(
        default_factory=lambda: {
            "train": 0,
            "stable": 101,
            "weather_drift": 202,
            "congestion_drift": 303,
            "carrier_drift": 404,
            "route_drift": 505,
        }
    )


def generate_dataset(
    n_rows: int,
    seed: int,
    scenario: ScenarioConfig = SCENARIOS["train"],
) -> pd.DataFrame:
    """Build a deterministic synthetic logistics dataset under a given scenario.

    Drift scenarios shift the *input* distribution. The target-generation
    logic stays constant so monitoring sees a real distribution change rather
    than a label-rule change.
    """
    rng = np.random.default_rng(seed)

    shipment_mode = rng.choice(
        config.CATEGORICAL_VALUES["shipment_mode"],
        size=n_rows,
        p=[0.30, 0.35, 0.25, 0.10],
    )
    carrier_probs = scenario.carrier_probs or (0.25, 0.55, 0.20)
    carrier_type = rng.choice(
        config.CATEGORICAL_VALUES["carrier_type"],
        size=n_rows,
        p=list(carrier_probs),
    )
    origin_region = rng.choice(
        config.CATEGORICAL_VALUES["origin_region"], size=n_rows
    )
    destination_region = rng.choice(
        config.CATEGORICAL_VALUES["destination_region"], size=n_rows
    )
    shipment_priority = rng.choice(
        config.CATEGORICAL_VALUES["shipment_priority"],
        size=n_rows,
        p=[0.20, 0.50, 0.20, 0.10],
    )

    route_distance_km = (
        rng.gamma(shape=2.0, scale=1500.0, size=n_rows)
        * scenario.route_distance_multiplier
    ).clip(50, 30_000)
    planned_transit_days = (
        0.5 + route_distance_km / 1500.0 + rng.normal(0, 1.2, size=n_rows)
    ).clip(1, 60)

    customs_complexity_score = rng.beta(2, 5, size=n_rows) * 10
    weather_risk_score = (rng.beta(2, 4, size=n_rows) * 10 + scenario.weather_shift).clip(
        0, 10
    )
    port_congestion_score = (
        rng.beta(2, 5, size=n_rows) * 10 + scenario.congestion_shift
    ).clip(0, 10)
    historical_route_delay_rate = rng.beta(2, 6, size=n_rows)

    cargo_value_usd = np.exp(rng.normal(8.5, 1.4, size=n_rows)).clip(50, 5_000_000)

    mode_risk = np.select(
        [shipment_mode == "sea", shipment_mode == "road", shipment_mode == "rail"],
        [0.50, 0.20, 0.10],
        default=-0.20,  # air baseline reduction
    )
    carrier_risk = np.select(
        [carrier_type == "low_cost", carrier_type == "standard"],
        [0.40, 0.10],
        default=-0.20,  # premium
    )
    priority_risk = np.select(
        [shipment_priority == "critical", shipment_priority == "high"],
        [-0.30, -0.10],
        default=0.0,
    )

    cross_region_penalty = (origin_region != destination_region).astype(float) * 0.20

    logits = (
        -3.6
        + 0.30 * customs_complexity_score
        + 0.28 * weather_risk_score
        + 0.25 * port_congestion_score
        + 4.5 * historical_route_delay_rate
        + 0.07 * planned_transit_days
        + mode_risk
        + carrier_risk
        + priority_risk
        + cross_region_penalty
        + rng.normal(0, 0.05, size=n_rows)
    )
    delay_prob = 1.0 / (1.0 + np.exp(-logits))
    # Deterministic threshold + small label-flip noise. This keeps the task
    # learnable enough for baseline metrics in the 0.80-0.95 range while still
    # leaving room for tuning and monitoring drift to matter.
    base_label = (delay_prob > 0.5).astype(int)
    flip_mask = rng.uniform(0, 1, size=n_rows) < config.LABEL_NOISE_RATE
    is_delayed = np.where(flip_mask, 1 - base_label, base_label)

    df = pd.DataFrame(
        {
            "shipment_mode": shipment_mode,
            "carrier_type": carrier_type,
            "origin_region": origin_region,
            "destination_region": destination_region,
            "shipment_priority": shipment_priority,
            "route_distance_km": route_distance_km.round(1),
            "planned_transit_days": planned_transit_days.round(2),
            "customs_complexity_score": customs_complexity_score.round(3),
            "weather_risk_score": weather_risk_score.round(3),
            "historical_route_delay_rate": historical_route_delay_rate.round(4),
            "cargo_value_usd": cargo_value_usd.round(2),
            "port_congestion_score": port_congestion_score.round(3),
            config.TARGET_COLUMN: is_delayed,
        }
    )
    return df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]]


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _output_path_for(scenario_name: str) -> Path:
    if scenario_name == "train":
        return config.RAW_DATASET_PATH
    return config.MONITORING_DIR / config.MONITORING_BATCHES[scenario_name]


def _print_summary(df: pd.DataFrame, label: str, output_path: Path) -> None:
    positives = int(df[config.TARGET_COLUMN].sum())
    total = len(df)
    print(
        f"[{label:<17}] {total:>5} rows  "
        f"delayed={positives / total:.1%}  "
        f"weather_mean={df['weather_risk_score'].mean():.2f}  "
        f"congestion_mean={df['port_congestion_score'].mean():.2f}  "
        f"low_cost_share={(df['carrier_type'] == 'low_cost').mean():.1%}  "
        f"route_mean={df['route_distance_km'].mean():.0f}km  "
        f"-> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=config.DATASET_SIZE,
        help="Rows for the training set",
    )
    parser.add_argument(
        "--monitoring-rows",
        type=int,
        default=config.MONITORING_BATCH_SIZE,
        help="Rows per monitoring batch",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--only",
        choices=list(SCENARIOS.keys()) + ["monitoring"],
        help="Generate just one scenario (or only the monitoring batches)",
    )
    args = parser.parse_args()

    seed_plan = _SeedPlan().seeds
    selected: list[str]
    if args.only is None:
        selected = list(SCENARIOS.keys())
    elif args.only == "monitoring":
        selected = [s for s in SCENARIOS if s != "train"]
    else:
        selected = [args.only]

    for scenario_name in selected:
        scenario = SCENARIOS[scenario_name]
        n_rows = args.rows if scenario_name == "train" else args.monitoring_rows
        df = generate_dataset(
            n_rows=n_rows,
            seed=args.seed + seed_plan[scenario_name],
            scenario=scenario,
        )
        out_path = _output_path_for(scenario_name)
        save_dataset(df, out_path)
        _print_summary(df, scenario_name, out_path)


if __name__ == "__main__":
    main()
