"""Data ingestion utilities for loading raw Spotify data."""

from pathlib import Path

import pandas as pd


EXPECTED_COLUMN_COUNT = 21
EXPECTED_FEATURE_COLUMNS = [
	"danceability",
	"energy",
	"loudness",
	"speechiness",
	"acousticness",
	"instrumentalness",
	"valence",
	"tempo",
]


def load_raw_data(raw_path: str) -> pd.DataFrame:
	"""Load raw dataset and validate schema requirements.

	Args:
		raw_path: Path to raw CSV file.

	Returns:
		Loaded raw dataframe.

	Raises:
		FileNotFoundError: If the input file does not exist.
		ValueError: If expected column count or feature columns are missing.
	"""
	path = Path(raw_path)
	if not path.exists():
		raise FileNotFoundError(f"Raw data file not found: {path}")

	df = pd.read_csv(path)

	if len(df.columns) != EXPECTED_COLUMN_COUNT:
		raise ValueError(
			f"Expected {EXPECTED_COLUMN_COUNT} columns, found {len(df.columns)}"
		)

	missing_features = [c for c in EXPECTED_FEATURE_COLUMNS if c not in df.columns]
	if missing_features:
		raise ValueError(f"Missing expected feature columns: {missing_features}")

	return df

