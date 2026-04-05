"""Model training utilities for fitting and persisting a KNN recommender."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# ----------------------------
# MLflow Configuration
# ----------------------------
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"
MLFLOW_TRACKING_URI = MLFLOW_TRACKING_DIR.resolve().as_uri()
MLFLOW_ARTIFACT_URI = MLFLOW_ARTIFACTS_DIR.resolve().as_uri()
MLFLOW_EXPERIMENT_NAME = "music-recommender"
MODEL_REGISTRY_NAME = "music-recommender-knn"

# Fallback defaults (overridden by params.yaml)
DEFAULT_FEATURE_COLS = [
	"danceability",
	"energy",
	"loudness",
	"speechiness",
	"acousticness",
	"instrumentalness",
	"valence",
	"tempo",
]
DEFAULT_ALGORITHM = "brute"
DEFAULT_DECAY = 0.85
DEFAULT_DATASET_VERSION = "v1"
DEFAULT_EVAL_SAMPLE_SIZE = 100
EVAL_RANDOM_SEED = 42


def _load_params(params_path: str) -> dict:
	"""Load training parameters from params.yaml."""
	path = Path(params_path)
	if not path.exists():
		raise FileNotFoundError(f"params.yaml not found: {path}")

	with path.open("r", encoding="utf-8") as f:
		params = yaml.safe_load(f) or {}

	feature_cols = params.get("feature_cols", DEFAULT_FEATURE_COLS)
	if not isinstance(feature_cols, list) or not feature_cols:
		raise ValueError("params.yaml must define feature_cols as a non-empty list")

	n_features = int(params.get("n_features", len(feature_cols)))
	if n_features != len(feature_cols):
		raise ValueError(
			"n_features in params.yaml must match length of feature_cols "
			f"({n_features} != {len(feature_cols)})"
		)

	params["feature_cols"] = feature_cols
	params["n_features"] = n_features
	params["algorithm"] = str(params.get("algorithm", DEFAULT_ALGORITHM))
	params["decay"] = float(params.get("decay", DEFAULT_DECAY))
	params["dataset_version"] = str(
		params.get("dataset_version", DEFAULT_DATASET_VERSION)
	)
	params["eval_sample_size"] = int(
		params.get("eval_sample_size", DEFAULT_EVAL_SAMPLE_SIZE)
	)
	return params


def _configure_mlflow() -> None:
	"""Configure local MLflow tracking and experiment."""
	MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
	MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

	mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

	client = MlflowClient()
	existing = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
	if existing is None:
		client.create_experiment(
			name=MLFLOW_EXPERIMENT_NAME,
			artifact_location=MLFLOW_ARTIFACT_URI,
		)

	mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def _average_nearest_neighbor_distance(
	model: NearestNeighbors,
	X: np.ndarray,
	eval_sample_size: int,
) -> float:
	"""Compute a quality proxy on a bounded deterministic sample."""
	if len(X) == 0:
		return float("nan")

	query_size = min(len(X), eval_sample_size)
	rng = np.random.default_rng(EVAL_RANDOM_SEED)
	query_indices = rng.choice(len(X), size=query_size, replace=False)
	query_X = X[query_indices]

	neighbors_to_query = 2 if len(X) > 1 else 1
	distances, _ = model.kneighbors(query_X, n_neighbors=neighbors_to_query)

	if neighbors_to_query == 2:
		nearest_non_self = distances[:, 1]
	else:
		nearest_non_self = distances[:, 0]

	return float(np.mean(nearest_non_self))


def _register_and_promote_model(run_id: str) -> Any:
	"""Register MLflow model and transition new version to Staging."""
	model_uri = f"runs:/{run_id}/knn_model"
	registered = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

	client = MlflowClient()
	client.transition_model_version_stage(
		name=MODEL_REGISTRY_NAME,
		version=registered.version,
		stage="Staging",
		archive_existing_versions=False,
	)

	promoted = client.get_model_version(
		name=MODEL_REGISTRY_NAME,
		version=registered.version,
	)
	print(
		"Registered model "
		f"{MODEL_REGISTRY_NAME} version={promoted.version} "
		f"stage={promoted.current_stage}"
	)
	return promoted


def train(
	features_path: str,
	model_output_path: str,
	n_neighbors: int,
	metric: str,
	params_path: str = str(PARAMS_PATH),
) -> NearestNeighbors:
	"""Train a NearestNeighbors model from processed features and save it."""
	params = _load_params(params_path)
	feature_cols = params["feature_cols"]
	algorithm = params["algorithm"]
	decay = params["decay"]
	dataset_version = params["dataset_version"]
	eval_sample_size = params["eval_sample_size"]

	_configure_mlflow()

	features_file = Path(features_path)
	if not features_file.exists():
		raise FileNotFoundError(f"Features file not found: {features_file}")

	df = pd.read_csv(features_file)

	missing_features = [c for c in feature_cols if c not in df.columns]
	if missing_features:
		raise ValueError(f"Missing feature columns in features file: {missing_features}")

	X = df[feature_cols].to_numpy(dtype=np.float64)

	if int(params["n_features"]) != X.shape[1]:
		raise ValueError(
			"n_features in params.yaml does not match matrix shape "
			f"({params['n_features']} != {X.shape[1]})"
		)

	model = NearestNeighbors(
		n_neighbors=n_neighbors,
		metric=metric,
		algorithm=algorithm,
	)

	with mlflow.start_run() as run:
		model.fit(X)

		# Dual-save behavior: keep local PKL while also logging/registering in MLflow.
		output_path = Path(model_output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(model, output_path)

		# Track parameters and metadata.
		mlflow.log_param("n_neighbors", n_neighbors)
		mlflow.log_param("metric", metric)
		mlflow.log_param("algorithm", algorithm)
		mlflow.log_param("decay", decay)
		mlflow.log_param("n_songs", int(X.shape[0]))
		mlflow.log_param("n_features", int(X.shape[1]))
		mlflow.log_param("dataset_version", dataset_version)
		mlflow.log_param("eval_sample_size", int(min(len(X), eval_sample_size)))

		mlflow.set_tag("model_type", "KNN")
		mlflow.set_tag("filtering_type", "content-based")

		# Metrics.
		mlflow.log_metric("train_size", int(X.shape[0]))
		mlflow.log_metric(
			"avg_nearest_neighbor_distance",
			_average_nearest_neighbor_distance(
				model=model,
				X=X,
				eval_sample_size=eval_sample_size,
			),
		)

		# Artifact logging.
		scaler_file = features_file.parent / "scaler.pkl"
		if scaler_file.exists():
			mlflow.log_artifact(str(scaler_file), artifact_path="artifacts")
		mlflow.log_artifact(str(features_file), artifact_path="artifacts")
		mlflow.log_artifact(str(output_path), artifact_path="artifacts")

		# MLflow model logging.
		mlflow.sklearn.log_model(model, artifact_path="knn_model")

		# Model registry and auto-promotion to Staging.
		_register_and_promote_model(run_id=run.info.run_id)

	return model


def _parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for local/DVC training runs."""
	params = _load_params(str(PARAMS_PATH))
	parser = argparse.ArgumentParser(description="Train KNN with MLflow tracking")
	parser.add_argument(
		"--features-path",
		default=str(PROJECT_ROOT / "data" / "processed" / "features.csv"),
		help="Path to processed features CSV",
	)
	parser.add_argument(
		"--model-output-path",
		default=str(PROJECT_ROOT / "data" / "processed" / "knn_model.pkl"),
		help="Output path for serialized KNN model",
	)
	parser.add_argument(
		"--params-path",
		default=str(PARAMS_PATH),
		help="Path to params.yaml",
	)
	parser.add_argument(
		"--n-neighbors",
		type=int,
		default=int(params.get("n_neighbors", 10)),
		help="Number of neighbors",
	)
	parser.add_argument(
		"--metric",
		default=str(params.get("metric", "cosine")),
		help="Distance metric for NearestNeighbors",
	)
	return parser.parse_args()


def main() -> None:
	"""CLI entry point for training and MLflow logging."""
	args = _parse_args()
	model = train(
		features_path=args.features_path,
		model_output_path=args.model_output_path,
		n_neighbors=args.n_neighbors,
		metric=args.metric,
		params_path=args.params_path,
	)
	print(
		"Training complete. "
		f"Saved local model to {args.model_output_path}. "
		f"Model class: {type(model).__name__}"
	)


if __name__ == "__main__":
	main()

