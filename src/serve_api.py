"""FastAPI service for the registered LogisticsDelayRiskModel.

Loads the model from the MLflow Model Registry once at startup using the
`production` alias (or whichever alias is configured) and exposes:

- `GET  /health`         — liveness + loaded-model metadata
- `POST /predict`        — single-shipment inference
- `POST /predict-batch`  — batched inference for multiple shipments

The service never trains or refits. Bring it up with:

    uvicorn src.serve_api:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from . import config, utils

ShipmentMode = Literal["air", "sea", "road", "rail"]
CarrierType = Literal["premium", "standard", "low_cost"]
Region = Literal["Europe", "MENA", "Asia", "Africa"]
Priority = Literal["low", "normal", "high", "critical"]


class Shipment(BaseModel):
    """One shipment row. Field names match the training schema exactly."""

    model_config = ConfigDict(extra="forbid")

    shipment_mode: ShipmentMode
    carrier_type: CarrierType
    origin_region: Region
    destination_region: Region
    shipment_priority: Priority
    route_distance_km: float = Field(..., ge=0, le=30_000)
    planned_transit_days: float = Field(..., ge=0, le=90)
    customs_complexity_score: float = Field(..., ge=0, le=10)
    weather_risk_score: float = Field(..., ge=0, le=10)
    historical_route_delay_rate: float = Field(..., ge=0, le=1)
    cargo_value_usd: float = Field(..., ge=0)
    port_congestion_score: float = Field(..., ge=0, le=10)


class PredictionResponse(BaseModel):
    prediction: int
    label: Literal["delayed", "on_time"]
    delay_probability: float
    model_name: str
    model_alias: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    shipments: list[Shipment]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    n: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_name: str
    model_alias: str
    model_version: str
    tracking_uri: str


_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "production")


class _ModelHandle:
    """Singleton-style holder so we load the model exactly once."""

    name: str = config.REGISTERED_MODEL_NAME
    alias: str = _MODEL_ALIAS
    version: str = ""
    pipeline: Pipeline | None = None

    def load(self) -> None:
        utils.configure_mlflow(config.MLFLOW_TRAINING_EXPERIMENT)
        client = MlflowClient()
        try:
            mv = client.get_model_version_by_alias(self.name, self.alias)
        except Exception as exc:  # mlflow raises RestException / MlflowException
            raise RuntimeError(
                f"Could not load model '{self.name}' alias '{self.alias}'. "
                "Run `python -m src.register_model` first. "
                f"Underlying error: {exc}"
            ) from exc
        uri = f"models:/{self.name}@{self.alias}"
        # Load as native sklearn so we can call .predict_proba; the pyfunc
        # wrapper returns class labels only, which would lose the probability.
        self.pipeline = mlflow.sklearn.load_model(uri)
        self.version = str(mv.version)
        print(f"Loaded {uri}  (version {self.version})")


_handle = _ModelHandle()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    _handle.load()
    yield


app = FastAPI(
    title="Logistics Delay Risk API",
    version="1.0.0",
    description="Inference service for the LogisticsDelayRiskModel.",
    lifespan=lifespan,
)


def _ensure_loaded() -> Pipeline:
    if _handle.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Try again in a moment.",
        )
    return _handle.pipeline


def _shipments_to_frame(shipments: list[Shipment]) -> pd.DataFrame:
    return pd.DataFrame([s.model_dump() for s in shipments])[config.FEATURE_COLUMNS]


def _predict_df(model: Pipeline, df: pd.DataFrame) -> tuple[list[int], list[float]]:
    """Run the sklearn pipeline and return labels + delay probabilities."""
    probs = model.predict_proba(df)[:, 1].astype(float).tolist()
    labels = [1 if p >= 0.5 else 0 for p in probs]
    return labels, probs


def _build_response(label: int, prob: float) -> PredictionResponse:
    return PredictionResponse(
        prediction=label,
        label="delayed" if label == 1 else "on_time",
        delay_probability=round(prob, 4),
        model_name=_handle.name,
        model_alias=_handle.alias,
        model_version=_handle.version,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=_handle.name,
        model_alias=_handle.alias,
        model_version=_handle.version,
        tracking_uri=config.MLFLOW_TRACKING_URI,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(shipment: Shipment) -> PredictionResponse:
    model = _ensure_loaded()
    df = _shipments_to_frame([shipment])
    labels, probs = _predict_df(model, df)
    return _build_response(labels[0], probs[0])


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if not request.shipments:
        raise HTTPException(status_code=400, detail="`shipments` cannot be empty.")
    model = _ensure_loaded()
    df = _shipments_to_frame(request.shipments)
    labels, probs = _predict_df(model, df)
    responses = [_build_response(label, prob) for label, prob in zip(labels, probs)]
    return BatchPredictionResponse(predictions=responses, n=len(responses))
