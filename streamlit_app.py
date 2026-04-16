"""Streamlit UI for local Music Recommender API."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st


def _api_get(base_url: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(f"{base_url}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def _api_post(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{base_url}{path}", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Music Recommender", page_icon="music", layout="wide")

st.title("Music Recommender")

with st.sidebar:
    st.header("API Settings")
    base_url = st.text_input("Base URL", value=st.session_state.get("base_url", "http://127.0.0.1:8001"))
    st.session_state["base_url"] = base_url

    if st.button("Health Check"):
        try:
            health = _api_get(base_url, "/health")
            st.success("API is healthy")
            st.json(health)
        except requests.RequestException as exc:
            st.error(f"Health check failed: {exc}")


st.subheader("Search for a seed song")
search_query = st.text_input("Song name", value=st.session_state.get("search_query", "sicko mode"))

if st.button("Search"):
    try:
        payload = _api_get(base_url, "/search", params={"q": search_query})
        st.session_state["search_query"] = search_query
        st.session_state["matches"] = payload["data"]["matches"]
    except requests.RequestException as exc:
        st.error(f"Search failed: {exc}")

matches = st.session_state.get("matches", [])

if matches:
    st.write("Matches")
    st.dataframe(matches, use_container_width=True)

    options = [
        f"{item['song_index']} | {item['track_name']} - {item.get('artists') or 'Unknown'}"
        for item in matches
    ]
    selected = st.selectbox("Pick a seed song", options=options, index=0)
    selected_index = int(selected.split("|")[0].strip())

    if st.button("Start session"):
        try:
            start = _api_post(base_url, "/session/start", {"song_index": selected_index})
            st.session_state["session_id"] = start["data"]["session_id"]
            st.session_state["recommendations"] = start["data"]["recommendations"]
            st.success("Session started")
        except requests.RequestException as exc:
            st.error(f"Session start failed: {exc}")


st.subheader("Recommendations")
recs = st.session_state.get("recommendations", [])

if recs:
    sort_order = st.radio("Sort by cosine distance", ["Most similar", "Least similar"], horizontal=True)
    reverse = sort_order == "Least similar"
    recs_sorted = sorted(recs, key=lambda r: r["cosine_distance"], reverse=reverse)

    st.dataframe(recs_sorted, use_container_width=True)

    rec_options = [
        f"{item['song_index']} | {item['track_name']} - {item.get('artists') or 'Unknown'}"
        for item in recs_sorted
    ]
    pick = st.selectbox("Pick a recommended song", options=rec_options, index=0)
    pick_index = int(pick.split("|")[0].strip())
    action = st.selectbox("Action", options=["play", "replay", "skip"], index=0)

    if st.button("Send action"):
        session_id = st.session_state.get("session_id")
        if not session_id:
            st.error("Session is missing. Start a session first.")
        else:
            try:
                next_payload = _api_post(
                    base_url,
                    f"/session/{session_id}/next",
                    {"song_index": pick_index, "action": action},
                )
                st.session_state["recommendations"] = next_payload["data"]["recommendations"]
                st.success("Session updated")
            except requests.RequestException as exc:
                st.error(f"Next action failed: {exc}")
else:
    st.info("Start a session to see recommendations.")
