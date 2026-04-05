"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model with strict payload handling."""

    model_config = ConfigDict(extra="forbid")


class ErrorResponse(StrictModel):
    """Standard error response payload."""

    status: Literal["error"] = "error"
    message: str


class HealthData(StrictModel):
    """Health check payload."""

    api: str
    model_loaded: bool
    model_source: str | None
    model_version: str | None
    n_songs: int


class HealthResponse(StrictModel):
    """Standard success wrapper for health endpoint."""

    status: Literal["ok"] = "ok"
    data: HealthData


class SongMatch(StrictModel):
    """Search result item."""

    track_id: str
    track_name: str
    artists: str | None
    track_genre: str | None
    popularity: int | float | None
    song_index: int


class SearchData(StrictModel):
    """Search response payload."""

    query: str
    matches: list[SongMatch]


class SearchResponse(StrictModel):
    """Standard success wrapper for search endpoint."""

    status: Literal["ok"] = "ok"
    data: SearchData


class RecommendationItem(StrictModel):
    """Recommendation item payload."""

    track_id: str
    track_name: str
    artists: str | None
    track_genre: str | None
    popularity: int | float | None
    song_index: int
    cosine_distance: float


class SessionStartRequest(StrictModel):
    """Request schema for creating a session."""

    song_index: int = Field(ge=0)


class SessionStartData(StrictModel):
    """Payload returned when a session starts."""

    session_id: str
    played: list[str]
    recommendations: list[RecommendationItem]


class SessionStartResponse(StrictModel):
    """Standard success wrapper for session start endpoint."""

    status: Literal["ok"] = "ok"
    data: SessionStartData


class SessionNextRequest(StrictModel):
    """Request schema for progressing a session."""

    song_index: int = Field(ge=0)
    action: Literal["play", "skip", "replay"]


class SessionNextData(StrictModel):
    """Payload returned when a session advances."""

    session_id: str
    played: list[str]
    recommendations: list[RecommendationItem]
    profile_shift: float


class SessionNextResponse(StrictModel):
    """Standard success wrapper for session next endpoint."""

    status: Literal["ok"] = "ok"
    data: SessionNextData


class SessionHistoryEvent(StrictModel):
    """One action in session history."""

    song_index: int
    song_name: str
    action: Literal["start", "play", "skip", "replay"]
    timestamp: datetime


class SessionHistoryData(StrictModel):
    """Payload returned for session history endpoint."""

    session_id: str
    played: list[str]
    actions: list[SessionHistoryEvent]
    current_profile_vector: list[float]


class SessionHistoryResponse(StrictModel):
    """Standard success wrapper for session history endpoint."""

    status: Literal["ok"] = "ok"
    data: SessionHistoryData


class SessionDeleteData(StrictModel):
    """Payload returned for delete session endpoint."""

    session_id: str
    message: str


class SessionDeleteResponse(StrictModel):
    """Standard success wrapper for delete endpoint."""

    status: Literal["ok"] = "ok"
    data: SessionDeleteData
