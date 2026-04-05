"""Dataset download/prepare stage for DVC pipeline.

If the target raw dataset already exists, this stage is a no-op.
If it does not exist, set RAW_DATA_SOURCE_PATH to copy a local file into place.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"


def download_or_prepare_dataset() -> Path:
    """Ensure raw dataset exists for downstream stages."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists():
        print(f"Raw dataset already present, skipping download: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    source = os.getenv("RAW_DATA_SOURCE_PATH")
    if source:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(
                f"RAW_DATA_SOURCE_PATH does not exist: {source_path}"
            )
        shutil.copy2(source_path, RAW_DATA_PATH)
        print(f"Copied raw dataset from {source_path} to {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    raise FileNotFoundError(
        "Raw dataset missing and no RAW_DATA_SOURCE_PATH provided. "
        "Place dataset at data/raw/dataset.csv or set RAW_DATA_SOURCE_PATH."
    )


if __name__ == "__main__":
    download_or_prepare_dataset()
