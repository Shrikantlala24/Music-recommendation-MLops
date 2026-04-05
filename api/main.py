"""FastAPI entry point for music recommendation backend."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.recommend import router as recommend_router
from api.routes.search import router as search_router
from api.schemas.models import ErrorResponse, HealthData, HealthResponse
from api.session.store import SessionStore
from src.model.load_model import load_recommender_model


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path_from_env(env_key: str, default_relative: str) -> Path:
	"""Resolve a filesystem path from env, falling back to a project-relative default."""
	raw_value = os.getenv(env_key)
	if not raw_value:
		return (PROJECT_ROOT / default_relative).resolve()

	path_value = Path(raw_value)
	if not path_value.is_absolute():
		path_value = (PROJECT_ROOT / path_value).resolve()
	return path_value


FEATURES_PATH = _resolve_path_from_env(
	"FEATURES_PATH",
	"data/processed/features.csv",
)
SCALER_PATH = _resolve_path_from_env(
	"SCALER_PATH",
	"data/processed/scaler.pkl",
)

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

META_COLS = [
	"track_id",
	"track_name",
	"artists",
	"track_genre",
	"popularity",
]


def _load_feature_data() -> tuple[np.ndarray, pd.DataFrame]:
	"""Load processed features and aligned metadata."""
	if not FEATURES_PATH.exists():
		raise FileNotFoundError(f"Missing processed features file: {FEATURES_PATH}")

	df = pd.read_csv(FEATURES_PATH)

	missing_features = [c for c in FEATURE_COLS if c not in df.columns]
	if missing_features:
		raise ValueError(f"Missing feature columns in features.csv: {missing_features}")

	missing_meta = [c for c in META_COLS if c not in df.columns]
	if missing_meta:
		raise ValueError(f"Missing metadata columns in features.csv: {missing_meta}")

	X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
	df_meta = df[META_COLS].reset_index(drop=True)
	return X, df_meta


app = FastAPI(title="Music Recommender API", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
	"""Format all HTTP errors with a consistent wrapper."""
	return JSONResponse(
		status_code=exc.status_code,
		content=ErrorResponse(message=str(exc.detail)).model_dump(),
	)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
	_: Request,
	exc: RequestValidationError,
) -> JSONResponse:
	"""Format request validation errors with the standard error wrapper."""
	first_error = exc.errors()[0] if exc.errors() else None
	message = "Invalid request"
	if first_error and "msg" in first_error:
		message = str(first_error["msg"])

	return JSONResponse(
		status_code=422,
		content=ErrorResponse(message=message).model_dump(),
	)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
	"""Format unexpected errors with a consistent wrapper."""
	LOGGER.exception("Unhandled API error: %s", exc)
	return JSONResponse(
		status_code=500,
		content=ErrorResponse(message="Internal server error").model_dump(),
	)


@app.on_event("startup")
def startup() -> None:
	"""Load model, data, and session store at application startup."""
	app.state.model = None
	app.state.model_source = None
	app.state.model_version = None
	app.state.X = None
	app.state.df_meta = None
	app.state.scaler = None
	app.state.session_store = SessionStore(ttl_minutes=30)

	try:
		loaded = load_recommender_model()
		app.state.model = loaded.model
		app.state.model_source = loaded.source
		app.state.model_version = loaded.version
		LOGGER.info(
			"Model loaded from %s (version=%s)",
			app.state.model_source,
			app.state.model_version,
		)
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("Model failed to load at startup: %s", exc)

	try:
		app.state.X, app.state.df_meta = _load_feature_data()
		LOGGER.info("Feature data loaded with %d songs", len(app.state.df_meta))
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("Feature data failed to load at startup: %s", exc)

	try:
		if SCALER_PATH.exists():
			app.state.scaler = joblib.load(SCALER_PATH)
			LOGGER.info("Scaler loaded from %s", SCALER_PATH)
		else:
			LOGGER.warning("Scaler file missing: %s", SCALER_PATH)
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("Scaler failed to load at startup: %s", exc)


@app.get(
	"/health",
	response_model=HealthResponse,
	responses={503: {"model": ErrorResponse}},
)
def health() -> HealthResponse:
	"""Health endpoint with model/data runtime status."""
	n_songs = 0
	if app.state.df_meta is not None:
		n_songs = int(len(app.state.df_meta))

	return HealthResponse(
		status="ok",
		data=HealthData(
			api="music-recommender-api",
			model_loaded=app.state.model is not None,
			model_source=app.state.model_source,
			model_version=app.state.model_version,
			n_songs=n_songs,
		),
	)


app.include_router(search_router)
app.include_router(recommend_router)
