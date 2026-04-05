"""Profile utilities for session-based preference aggregation."""

import numpy as np


def compute_profile_vector(history: list[dict], decay: float) -> np.ndarray:
	"""Compute a weighted user profile from song vectors using exponential decay.

	Weights are applied from oldest to newest as:
	- oldest: decay^(n-1)
	- newest: decay^0 = 1.0

	Args:
		history: List of {'vector': np.ndarray} entries in chronological order.
		decay: Exponential decay factor in (0, 1].

	Returns:
		Weighted average profile vector.

	Raises:
		ValueError: If history is empty or malformed.
	"""
	

    
	if not history:
		raise ValueError("history must not be empty")
	if not (0 < decay <= 1):
		raise ValueError("decay must be in the range (0, 1]")

	try:
		vectors = [np.asarray(item["vector"], dtype=np.float64) for item in history]
	except (KeyError, TypeError) as exc:
		raise ValueError("each history item must be a dict containing key 'vector'") from exc

	n = len(vectors)
	weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)
	stacked = np.vstack(vectors)

	if stacked.ndim != 2:
		raise ValueError("history vectors must be one-dimensional arrays")

	profile = (stacked * weights[:, None]).sum(axis=0) / weights.sum()
	return profile


def update_history(history: list, new_vector: np.ndarray) -> list:
	"""Append a new song vector to the user history and return the updated list."""
	history.append({"vector": np.asarray(new_vector, dtype=np.float64)})
	return history

