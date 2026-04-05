"""Song search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from api.schemas.models import ErrorResponse, SearchData, SearchResponse, SongMatch


router = APIRouter()


@router.get(
    "/search",
    response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def search_songs(
    request: Request,
    q: str = Query(..., min_length=1, description="Partial song name"),
) -> SearchResponse:
    """Search songs by partial track name match (case-insensitive)."""
    df_meta = request.app.state.df_meta
    if df_meta is None or df_meta.empty:
        raise HTTPException(status_code=503, detail="Metadata is not loaded")

    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter q must not be empty")

    mask = df_meta["track_name"].fillna("").str.contains(query, case=False, na=False)
    matches_df = df_meta.loc[mask].head(25)

    matches = [
        SongMatch(
            track_id=str(row.track_id),
            track_name=str(row.track_name),
            artists=None if row.artists is None else str(row.artists),
            track_genre=None if row.track_genre is None else str(row.track_genre),
            popularity=None if row.popularity is None else row.popularity,
            song_index=int(idx),
        )
        for idx, row in matches_df.iterrows()
    ]

    return SearchResponse(status="ok", data=SearchData(query=query, matches=matches))
