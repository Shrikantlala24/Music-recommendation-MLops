"""Smoke test recommendation call path with lightweight dummy data."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from src.model.recommend import get_recommendations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "knn_model.pkl"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="knn_model.pkl is not available in CI",
)
def test_get_recommendations_with_dummy_data_and_saved_model() -> None:
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, NearestNeighbors)

    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dummy_model = NearestNeighbors(
        n_neighbors=3,
        metric="cosine",
        algorithm="brute",
    )
    dummy_model.fit(X)

    df_meta = pd.DataFrame(
        {
            "track_id": ["a", "b", "c", "d"],
            "track_name": ["A", "B", "C", "D"],
            "artists": ["x", "y", "z", "w"],
            "track_genre": ["g1", "g2", "g3", "g4"],
            "popularity": [10, 20, 30, 40],
        }
    )

    profile_vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    recs = get_recommendations(
        profile_vector=profile_vector,
        X=X,
        df_meta=df_meta,
        model=dummy_model,
        n=2,
        exclude_indices=[0],
    )

    assert len(recs) == 2
    assert "track_id" in recs.columns
    assert "cosine_distance" in recs.columns
