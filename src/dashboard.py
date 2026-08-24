"""Streamlit replay dashboard: step through a precomputed game's plays with
live predictions, a running accuracy counter, and team tendency charts.

Reads data/replay.db (written by precompute_predictions.py). Run with:
    streamlit run src/dashboard.py
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path("data/replay.db")


@st.cache_data
def load_plays() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM play_predictions ORDER BY play_index", conn)
    finally:
        conn.close()
    if df.empty:
        raise ValueError(f"No plays found in {DB_PATH} — run precompute_predictions.py first")
    return df


def init_state(max_idx: int) -> None:
    st.session_state.setdefault("play_idx", 0)
    st.session_state.setdefault("playing", False)
    st.session_state.play_idx = min(st.session_state.play_idx, max_idx)


def render_sidebar(plays: pd.DataFrame) -> float:
    st.sidebar.header("Playback")
    speed = st.sidebar.slider("Speed (plays/sec)", 0.5, 5.0, 1.0, 0.5)

    label = "Pause" if st.session_state.playing else "Play"
    if st.sidebar.button(label, width='stretch'):
        st.session_state.playing = not st.session_state.playing

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Prev", width='stretch'):
        st.session_state.playing = False
        st.session_state.play_idx = max(0, st.session_state.play_idx - 1)
    if col2.button("Next", width='stretch'):
        st.session_state.playing = False
        st.session_state.play_idx = min(len(plays) - 1, st.session_state.play_idx + 1)

    idx = st.sidebar.slider("Play", 0, len(plays) - 1, st.session_state.play_idx)
    if idx != st.session_state.play_idx:
        st.session_state.playing = False
        st.session_state.play_idx = idx

    return speed


def render_play_card(row: pd.Series) -> None:
    st.subheader(f"Q{row.quarter} — {row.game_clock or ''}")
    c1, c2, c3 = st.columns(3)
    down_txt = f"{int(row.down)} & {int(row.distance)}" if pd.notna(row.down) else "—"
    c1.metric("Down & distance", down_txt)
    c2.metric("Yardline (to opp EZ)", int(row.yardline_100) if pd.notna(row.yardline_100) else "—")
    c3.metric("Offense / Defense", f"{row.offense_team} vs {row.defense_team}")

    if row.is_scored:
        prob_pass = row.predicted_prob_pass
        st.progress(prob_pass, text=f"Pass {prob_pass:.0%} / Run {1 - prob_pass:.0%}")
        correct = row.predicted_label == row.actual_play_type
        mark = "correct" if correct else "wrong"
        st.write(f"Predicted **{row.predicted_label}** — actual **{row.actual_play_type}** ({mark})")
    else:
        st.info(f"Not scored — {row.skip_reason} (actual: {row.actual_play_type})")


def render_accuracy(plays: pd.DataFrame, upto_idx: int) -> None:
    seen = plays[(plays.play_index <= upto_idx) & (plays.is_scored == 1)]
    if seen.empty:
        st.metric("Accuracy so far", "—")
        return
    correct = (seen.predicted_label == seen.actual_play_type).sum()
    st.metric("Accuracy so far", f"{correct}/{len(seen)} ({correct / len(seen):.0%})")


def render_team_tendency(plays: pd.DataFrame, upto_idx: int) -> None:
    seen = plays[(plays.play_index <= upto_idx) & (plays.is_scored == 1)]
    if seen.empty:
        st.write("No scored plays yet.")
        return
    counts = seen.groupby(["offense_team", "actual_play_type"]).size().reset_index(name="count")
    fig = px.bar(counts, x="offense_team", y="count", color="actual_play_type", barmode="group",
                 title="Run/pass tendency so far")
    st.plotly_chart(fig, width='stretch')


def autoplay_tick(plays: pd.DataFrame, speed: float) -> None:
    if not st.session_state.playing:
        return
    if st.session_state.play_idx >= len(plays) - 1:
        st.session_state.playing = False
        return
    time.sleep(1 / speed)
    st.session_state.play_idx += 1
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="NFL Replay", layout="wide")
    plays = load_plays()
    init_state(len(plays) - 1)

    st.title(f"Replay: {plays.game_id.iloc[0]}")
    speed = render_sidebar(plays)

    row = plays.iloc[st.session_state.play_idx]
    left, right = st.columns([2, 1])
    with left:
        render_play_card(row)
        render_team_tendency(plays, st.session_state.play_idx)
    with right:
        render_accuracy(plays, st.session_state.play_idx)
        st.caption(f"Play {st.session_state.play_idx + 1} of {len(plays)}")

    autoplay_tick(plays, speed)


if __name__ == "__main__":
    main()
