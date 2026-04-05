"""Model training utilities for fitting and persisting a KNN recommender."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


FEATURE_COLS = [
	"danceability",
	"energy",
	"loudness",
	"speechiness",
	"acousticness",
	"instrumentalness",
	"valence",
	"tempo",
]


def train(
	features_path: str,
	model_output_path: str,
	n_neighbors: int,
	metric: str,
) -> NearestNeighbors:
	"""Train a NearestNeighbors model from processed features and save it.

	Args:
		features_path: Path to processed features CSV.
		model_output_path: Output path for serialized KNN model.
		n_neighbors: Number of neighbors used by KNN.
		metric: Distance metric to use (for example, 'cosine').

	Returns:
		Fitted NearestNeighbors instance.

	Raises:
		FileNotFoundError: If features file is missing.
		ValueError: If required feature columns are missing.
	"""
	features_file = Path(features_path)
	if not features_file.exists():
		raise FileNotFoundError(f"Features file not found: {features_file}")

	df = pd.read_csv(features_file)

	missing_features = [c for c in FEATURE_COLS if c not in df.columns]
	if missing_features:
		raise ValueError(f"Missing feature columns in features file: {missing_features}")

	X = df[FEATURE_COLS].to_numpy(dtype=np.float64)

	model = NearestNeighbors(
		n_neighbors=n_neighbors,
		metric=metric,
		algorithm="brute",
	)
	model.fit(X)

	output_path = Path(model_output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, output_path)

	return model

