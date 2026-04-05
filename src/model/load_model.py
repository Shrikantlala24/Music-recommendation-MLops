"""Model loading utility: MLflow registry first, local PKL fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.neighbors import NearestNeighbors


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = MLFLOW_TRACKING_DIR.resolve().as_uri()
MODEL_NAME = "music-recommender-knn"
MODEL_STAGE = "Staging"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "knn_model.pkl"


@dataclass
class ModelLoadResult:
    """Container for loaded model metadata."""

    model: NearestNeighbors
    source: str
    version: str


def _configure_tracking() -> None:
    """Configure local MLflow tracking URI."""
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_recommender_model(
    model_name: str = MODEL_NAME,
    stage: str = MODEL_STAGE,
    fallback_model_path: str = str(FALLBACK_MODEL_PATH),
) -> ModelLoadResult:
    """Load model from MLflow registry, fallback to local PKL when unavailable."""
    _configure_tracking()

    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        if not isinstance(model, NearestNeighbors):
            raise TypeError("Loaded MLflow model is not NearestNeighbors")

        client = MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        version = str(latest_versions[0].version) if latest_versions else "unknown"
        return ModelLoadResult(model=model, source="mlflow", version=version)
    except Exception as exc:  # noqa: BLE001
        fallback_path = Path(fallback_model_path)
        if not fallback_path.is_absolute():
            fallback_path = PROJECT_ROOT / fallback_path
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
