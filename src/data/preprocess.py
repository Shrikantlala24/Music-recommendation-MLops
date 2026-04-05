"""Data preprocessing utilities for feature preparation and artifact saving."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(
	df: pd.DataFrame,
	feature_cols: list[str],
	drop_cols: list[str],
	processed_dir: str,
) -> pd.DataFrame:
	"""Clean data, scale model features, and save preprocessing artifacts.

	Args:
		df: Raw input dataframe.
		feature_cols: Audio feature columns to scale and use for modeling.
		drop_cols: Columns to drop before modeling.
		processed_dir: Output directory for processed artifacts.

	Returns:
		Processed dataframe with scaled feature columns.

	Raises:
		ValueError: If required feature columns are missing.
	"""
	missing_features = [c for c in feature_cols if c not in df.columns]
	if missing_features:
		raise ValueError(f"Missing feature columns in dataframe: {missing_features}")

	df_processed = df.copy()

	cols_to_drop = [c for c in drop_cols if c in df_processed.columns]
	if cols_to_drop:
		df_processed = df_processed.drop(columns=cols_to_drop)

	df_processed = df_processed.dropna(subset=feature_cols)

	if "track_id" in df_processed.columns:
		df_processed = df_processed.drop_duplicates(subset=["track_id"], keep="first")

	scaler = StandardScaler()
	df_processed.loc[:, feature_cols] = scaler.fit_transform(df_processed[feature_cols])

	out_dir = Path(processed_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	features_path = out_dir / "features.csv"
	scaler_path = out_dir / "scaler.pkl"

	df_processed.to_csv(features_path, index=False)
	joblib.dump(scaler, scaler_path)

	return df_processed

