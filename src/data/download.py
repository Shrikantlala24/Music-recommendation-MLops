"""Dataset download/prepare stage for DVC pipeline.

If the target raw dataset already exists, this stage is a no-op.
If it does not exist, this stage tries (in order):
1) RAW_DATA_SOURCE_PATH copy
2) Kaggle download via kagglehub for the public Spotify dataset
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"
KAGGLE_DATASET_REF = "maharshipandya/-spotify-tracks-dataset"


def _copy_from_source_env() -> Path | None:
    """Copy dataset from RAW_DATA_SOURCE_PATH when provided."""
    source = os.getenv("RAW_DATA_SOURCE_PATH")
    if not source:
        return None

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"RAW_DATA_SOURCE_PATH does not exist: {source_path}")

    shutil.copy2(source_path, RAW_DATA_PATH)
    print(f"Copied raw dataset from {source_path} to {RAW_DATA_PATH}")
    return RAW_DATA_PATH


def _download_from_kagglehub() -> Path | None:
    """Download dataset using kagglehub and copy best CSV candidate."""
    try:
        import kagglehub  # type: ignore
    except Exception:
        return None

    try:
        dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_REF))
    except Exception as exc:
        print(f"Kaggle download unavailable: {exc}")
        return None

    csv_candidates = sorted(dataset_dir.rglob("*.csv"))
    if not csv_candidates:
        return None

    # Prefer file explicitly named dataset.csv, else largest CSV file.
    named = [p for p in csv_candidates if p.name.lower() == "dataset.csv"]
    source_csv = named[0] if named else max(csv_candidates, key=lambda p: p.stat().st_size)

    shutil.copy2(source_csv, RAW_DATA_PATH)
    print(f"Downloaded and copied dataset from {source_csv} to {RAW_DATA_PATH}")
    return RAW_DATA_PATH


def download_or_prepare_dataset() -> Path:
    """Ensure raw dataset exists for downstream stages."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists():
        print(f"Raw dataset already present, skipping download: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    copied = _copy_from_source_env()
    if copied is not None:
        return copied

    downloaded = _download_from_kagglehub()
    if downloaded is not None:
        return downloaded

    raise FileNotFoundError(
        "Raw dataset missing and could not be resolved automatically. "
        "Place dataset at data/raw/dataset.csv, or set RAW_DATA_SOURCE_PATH, "
        "or configure Kaggle access for kagglehub."
    )


if __name__ == "__main__":
    download_or_prepare_dataset()
