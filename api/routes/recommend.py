"""Recommendation and session lifecycle endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from api.schemas.models import (
    ErrorResponse,
    RecommendationItem,
    SessionDeleteData,
    SessionDeleteResponse,
    SessionHistoryData,
    SessionHistoryEvent,
    SessionHistoryResponse,
    SessionNextData,
    SessionNextRequest,
    SessionNextResponse,
    SessionStartData,
    SessionStartRequest,
    SessionStartResponse,
)
from src.model.recommend import get_recommendations
from src.utils.profile import compute_profile_vector, update_history


router = APIRouter()

DECAY = 0.85
TOP_N = 5


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_runtime_ready(request: Request) -> None:
    """Ensure required model and data resources are loaded."""
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if request.app.state.X is None or request.app.state.df_meta is None:
        raise HTTPException(status_code=503, detail="Feature data is not loaded")


def _validate_song_index(song_index: int, n_rows: int) -> None:
    """Validate index bounds for song lookup."""
    if song_index < 0 or song_index >= n_rows:
        raise HTTPException(
            status_code=400,
            detail=f"song_index must be between 0 and {n_rows - 1}",
        )


def _get_song_name(df_meta, song_index: int) -> str:
    value = df_meta.iloc[song_index]["track_name"]
    return str(value)


def _build_recommendation_items(rec_df) -> list[RecommendationItem]:
    """Convert recommendation dataframe into response schema list."""
    items: list[RecommendationItem] = []
    for _, row in rec_df.iterrows():
        items.append(
            RecommendationItem(
                track_id=str(row.track_id),
                track_name=str(row.track_name),
                artists=None if row.artists is None else str(row.artists),
                track_genre=None if row.track_genre is None else str(row.track_genre),
                popularity=None if row.popularity is None else row.popularity,
                song_index=int(row["index"]),
                cosine_distance=float(row.cosine_distance),
            )
        )
    return items


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    similarity = float(np.dot(a, b) / denom)
    similarity = max(min(similarity, 1.0), -1.0)
    return float(1.0 - similarity)


@router.post(
    "/session/start",
    response_model=SessionStartResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def start_session(request: Request, payload: SessionStartRequest) -> SessionStartResponse:
    """Start a session and return initial recommendations."""
    _ensure_runtime_ready(request)

    X = request.app.state.X
    df_meta = request.app.state.df_meta
    model = request.app.state.model
    store = request.app.state.session_store

    _validate_song_index(payload.song_index, len(df_meta))

    session = store.create()
    selected_vector = np.asarray(X[payload.song_index], dtype=np.float64)

    update_history(session.history_vectors, selected_vector)
    session.played_indices.append(int(payload.song_index))
    session.events.append(
        {
            "song_index": int(payload.song_index),
            "song_name": _get_song_name(df_meta, payload.song_index),
            "action": "start",
            "timestamp": _now_utc(),
        }
    )

    session.current_profile_vector = compute_profile_vector(session.history_vectors, decay=DECAY)
    session.previous_profile_vector = None
    store.touch(session)

    rec_df = get_recommendations(
        profile_vector=session.current_profile_vector,
        X=X,
        df_meta=df_meta,
        model=model,
        n=TOP_N,
        exclude_indices=session.played_indices,
    )

    played_names = [_get_song_name(df_meta, idx) for idx in session.played_indices]
    recommendations = _build_recommendation_items(rec_df)

    return SessionStartResponse(
        status="ok",
        data=SessionStartData(
            session_id=session.session_id,
            played=played_names,
            recommendations=recommendations,
        ),
    )


@router.post(
    "/session/{session_id}/next",
    response_model=SessionNextResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def next_session_pick(
    session_id: str,
    payload: SessionNextRequest,
    request: Request,
) -> SessionNextResponse:
    """Advance a session with one user action and return refreshed recommendations."""
    _ensure_runtime_ready(request)

    X = request.app.state.X
    df_meta = request.app.state.df_meta
    model = request.app.state.model
    store = request.app.state.session_store

    _validate_song_index(payload.song_index, len(df_meta))

    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    song_vector = np.asarray(X[payload.song_index], dtype=np.float64)
    song_name = _get_song_name(df_meta, payload.song_index)

    previous_profile = (
        None
        if session.current_profile_vector is None
        else np.asarray(session.current_profile_vector, dtype=np.float64).copy()
    )

    if payload.action == "play":
        update_history(session.history_vectors, song_vector)
        if payload.song_index not in session.played_indices:
            session.played_indices.append(int(payload.song_index))
    elif payload.action == "replay":
        update_history(session.history_vectors, song_vector * 1.5)
        if payload.song_index not in session.played_indices:
            session.played_indices.append(int(payload.song_index))
    elif payload.action == "skip":
        pass

    session.events.append(
        {
            "song_index": int(payload.song_index),
            "song_name": song_name,
            "action": payload.action,
            "timestamp": _now_utc(),
        }
    )

    session.previous_profile_vector = previous_profile

    if payload.action in {"play", "replay"}:
        session.current_profile_vector = compute_profile_vector(
            session.history_vectors,
            decay=DECAY,
        )

    if session.current_profile_vector is None:
        raise HTTPException(status_code=400, detail="Session profile is not initialized")

    profile_shift = 0.0
    if session.previous_profile_vector is not None:
        profile_shift = _cosine_distance(
            np.asarray(session.previous_profile_vector, dtype=np.float64),
            np.asarray(session.current_profile_vector, dtype=np.float64),
        )

    rec_df = get_recommendations(
        profile_vector=np.asarray(session.current_profile_vector, dtype=np.float64),
        X=X,
        df_meta=df_meta,
        model=model,
        n=TOP_N,
        exclude_indices=session.played_indices,
    )

    store.touch(session)

    played_names = [_get_song_name(df_meta, idx) for idx in session.played_indices]
    recommendations = _build_recommendation_items(rec_df)

    return SessionNextResponse(
        status="ok",
        data=SessionNextData(
            session_id=session.session_id,
            played=played_names,
            recommendations=recommendations,
            profile_shift=float(profile_shift),
        ),
    )


@router.get(
    "/session/{session_id}/history",
    response_model=SessionHistoryResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_session_history(session_id: str, request: Request) -> SessionHistoryResponse:
    """Return full session history with actions and current profile vector."""
    _ensure_runtime_ready(request)

    df_meta = request.app.state.df_meta
    store = request.app.state.session_store

    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    profile_vector = (
        []
        if session.current_profile_vector is None
        else [float(v) for v in np.asarray(session.current_profile_vector, dtype=np.float64)]
    )

    played_names = [_get_song_name(df_meta, idx) for idx in session.played_indices]
    events = [
        SessionHistoryEvent(
            song_index=int(event["song_index"]),
            song_name=str(event["song_name"]),
            action=str(event["action"]),
            timestamp=event["timestamp"],
        )
        for event in session.events
    ]

    return SessionHistoryResponse(
        status="ok",
        data=SessionHistoryData(
            session_id=session.session_id,
            played=played_names,
            actions=events,
            current_profile_vector=profile_vector,
        ),
    )


@router.delete(
    "/session/{session_id}",
    response_model=SessionDeleteResponse,
    responses={404: {"model": ErrorResponse}},
)
def delete_session(session_id: str, request: Request) -> SessionDeleteResponse:
    """Delete one session from in-memory store."""
    store = request.app.state.session_store
    deleted = store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionDeleteResponse(
        status="ok",
        data=SessionDeleteData(
            session_id=session_id,
            message="Session deleted",
        ),
    )
