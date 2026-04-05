"""Model loading utility: MLflow registry first, local PKL fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.neighbors import NearestNeighbors


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_env_value(key: str, default_value: str) -> str:
    """Read runtime config from environment with sane defaults."""
    return os.getenv(key, default_value).strip()


def _resolve_path(value: str) -> Path:
    """Resolve paths relative to project root to work locally and in containers."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


DEFAULT_TRACKING_PATH = "mlruns"
DEFAULT_MODEL_NAME = "music-recommender-knn"
DEFAULT_MODEL_STAGE = "Staging"
DEFAULT_FALLBACK_MODEL_PATH = "data/processed/knn_model.pkl"


@dataclass
class ModelLoadResult:
    """Container for loaded model metadata."""

    model: NearestNeighbors
    source: str
    version: str


def _configure_tracking() -> None:
    """Configure local MLflow tracking URI."""
    tracking_uri = _get_env_value("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_PATH)
    parsed = urlparse(tracking_uri)

    if parsed.scheme:
        mlflow.set_tracking_uri(tracking_uri)
        return

    tracking_path = _resolve_path(tracking_uri)
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(tracking_path))


def load_recommender_model(
    model_name: str | None = None,
    stage: str | None = None,
    fallback_model_path: str | None = None,
) -> ModelLoadResult:
    """Load model from MLflow registry, fallback to local PKL when unavailable."""
    _configure_tracking()

    resolved_model_name = model_name or _get_env_value("MODEL_NAME", DEFAULT_MODEL_NAME)
    resolved_stage = stage or _get_env_value("MODEL_STAGE", DEFAULT_MODEL_STAGE)
    resolved_fallback_model_path = fallback_model_path or _get_env_value(
        "FALLBACK_MODEL_PATH",
        DEFAULT_FALLBACK_MODEL_PATH,
    )

    try:
        model_uri = f"models:/{resolved_model_name}/{resolved_stage}"
        model = mlflow.sklearn.load_model(model_uri)
        if not isinstance(model, NearestNeighbors):
            raise TypeError("Loaded MLflow model is not NearestNeighbors")

        client = MlflowClient()
        latest_versions = client.get_latest_versions(
            resolved_model_name,
            stages=[resolved_stage],
        )
        version = str(latest_versions[0].version) if latest_versions else "unknown"
        return ModelLoadResult(model=model, source="mlflow", version=version)
    except Exception as exc:  # noqa: BLE001
        fallback_path = _resolve_path(resolved_fallback_model_path)
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"Fallback model file not found: {fallback_path}"
            ) from exc

        LOGGER.warning(
            "MLflow model unavailable, using local PKL fallback at %s. Reason: %s",
            fallback_path,
            exc,
        )
        fallback_model = joblib.load(fallback_path)
        if not isinstance(fallback_model, NearestNeighbors):
            raise TypeError("Fallback PKL model is not NearestNeighbors")
        return ModelLoadResult(model=fallback_model, source="pkl", version="local")
