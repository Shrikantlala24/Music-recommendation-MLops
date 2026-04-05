"""Data preprocessing utilities for feature preparation and artifact saving."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


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


def _load_params(params_path: str) -> dict:
	"""Load preprocessing parameters from params.yaml."""
	path = Path(params_path)
	if not path.exists():
		raise FileNotFoundError(f"params.yaml not found: {path}")

	with path.open("r", encoding="utf-8") as f:
		params = yaml.safe_load(f) or {}

	if "feature_cols" not in params or not isinstance(params["feature_cols"], list):
		raise ValueError("params.yaml must define feature_cols as a list")
	if "drop_cols" not in params or not isinstance(params["drop_cols"], list):
		raise ValueError("params.yaml must define drop_cols as a list")

	n_features = int(params.get("n_features", len(params["feature_cols"])))
	if n_features != len(params["feature_cols"]):
		raise ValueError(
			"n_features in params.yaml must match length of feature_cols "
			f"({n_features} != {len(params['feature_cols'])})"
		)

	return params


def _parse_args() -> argparse.Namespace:
	"""Parse CLI args for DVC preprocess stage."""
	parser = argparse.ArgumentParser(description="Preprocess raw dataset for KNN training")
	parser.add_argument(
		"--raw-path",
		default=str(RAW_DATA_PATH),
		help="Path to raw dataset CSV",
	)
	parser.add_argument(
		"--processed-dir",
		default=str(PROCESSED_DIR),
		help="Directory to write processed artifacts",
	)
	parser.add_argument(
		"--params-path",
		default=str(PARAMS_PATH),
		help="Path to params.yaml",
	)
	return parser.parse_args()


def main() -> None:
	"""CLI entrypoint for DVC preprocess stage."""
	args = _parse_args()
	params = _load_params(args.params_path)

	raw_file = Path(args.raw_path)
	if not raw_file.exists():
		raise FileNotFoundError(f"Raw dataset not found: {raw_file}")

	df_raw = pd.read_csv(raw_file)
	df_processed = preprocess(
		df=df_raw,
		feature_cols=params["feature_cols"],
		drop_cols=params["drop_cols"],
		processed_dir=args.processed_dir,
	)

	print(
		"Preprocess complete. "
		f"Rows={len(df_processed)}, Cols={len(df_processed.columns)}, "
		f"Outputs={Path(args.processed_dir) / 'features.csv'}, "
		f"{Path(args.processed_dir) / 'scaler.pkl'}"
	)


if __name__ == "__main__":
	main()

