"""Recommendation utilities for profile-vector nearest-neighbor retrieval."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def get_recommendations(
	profile_vector: np.ndarray,
	X: np.ndarray,
	df_meta: pd.DataFrame,
	model: NearestNeighbors,
	n: int,
	exclude_indices: list[int],
) -> pd.DataFrame:
	"""Return top-N recommendations for a profile vector.

	Args:
		profile_vector: User profile vector in model feature space.
		X: Feature matrix used to fit the KNN model.
		df_meta: Metadata dataframe aligned by row index with X.
		model: Fitted NearestNeighbors model.
		n: Number of recommendations to return.
		exclude_indices: Row indices to exclude (already played songs).

	Returns:
		Dataframe containing recommended rows from df_meta and cosine_distance.

	Raises:
		ValueError: If input shapes are inconsistent.
	"""
	if X.shape[0] != len(df_meta):
		raise ValueError("X and df_meta must have the same number of rows")

	query = np.asarray(profile_vector, dtype=np.float64).reshape(1, -1)
	if query.shape[1] != X.shape[1]:
		raise ValueError(
			f"profile_vector has {query.shape[1]} features, expected {X.shape[1]}"
		)

	excluded = set(exclude_indices)
	requested = n + len(excluded)
	n_neighbors = min(max(requested, n), len(df_meta))

	distances, indices = model.kneighbors(query, n_neighbors=n_neighbors)

	selected_indices: list[int] = []
	selected_distances: list[float] = []
	for idx, dist in zip(indices[0], distances[0]):
		idx_int = int(idx)
		if idx_int in excluded:
			continue
		selected_indices.append(idx_int)
		selected_distances.append(float(dist))
		if len(selected_indices) >= n:
			break

	result = df_meta.loc[selected_indices].copy()
	result.insert(0, "index", selected_indices)
	result["cosine_distance"] = selected_distances
	return result.reset_index(drop=True)

