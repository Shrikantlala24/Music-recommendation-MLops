"""Deterministic tests for profile utilities."""

from __future__ import annotations

import numpy as np

from src.utils.profile import compute_profile_vector, update_history


def test_update_history_appends_vector() -> None:
    history: list[dict] = []
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    updated = update_history(history, vec)

    assert len(updated) == 1
    assert np.allclose(updated[0]["vector"], vec)


def test_compute_profile_vector_known_values() -> None:
    history = [
        {"vector": np.array([1.0, 0.0], dtype=np.float64)},
        {"vector": np.array([0.0, 1.0], dtype=np.float64)},
    ]
    decay = 0.5

    profile = compute_profile_vector(history, decay)

    expected = np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)
    assert np.allclose(profile, expected)
