"""Streamlit replay dashboard: step through a precomputed game's plays with
live predictions, a football-field visualization, a running accuracy donut,
and a team-tendency chart.

Reads data/replay.db (written by precompute_predictions.py). Run with:
    streamlit run src/dashboard.py
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path("data/replay.db")
GAME_TITLE = "LA Rams at Seattle Seahawks, Week 16, 2025"

# Scheme A — team tendency chart.
RAMS_BLUE = "#003594"
SEA_GREEN = "#69BE28"
SCHEME_A = {"LA": RAMS_BLUE, "SEA": SEA_GREEN}

# Scheme B — field end zones.
SEA_NAVY = "#002244"
RAMS_YELLOW = "#FFA300"
SCHEME_B = {"LA": RAMS_YELLOW, "SEA": SEA_NAVY}

# Functional field colors (broadcast-standard, not team colors).
LOS_BLUE = "#1E90FF"
FIRST_DOWN_YELLOW = "#FFD700"
GAIN_RED = "#D62728"
FIELD_GREENS = ("#3C8C40", "#2F7A33")


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
    speed = st.sidebar.slider("Speed (plays/sec)", 0.5, 5.0, 0.5, 0.5)

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


def render_prediction(row: pd.Series) -> None:
    if row.is_scored:
        prob_pass = row.predicted_prob_pass
        st.progress(prob_pass, text=f"Pass {prob_pass:.0%} / Run {1 - prob_pass:.0%}")
        correct = row.predicted_label == row.actual_play_type
        mark = "correct" if correct else "wrong"
        st.write(f"Predicted **{row.predicted_label}** — actual **{row.actual_play_type}** ({mark})")
        if pd.notna(row.yards_gained):
            st.caption(f"Gain: {row.yards_gained:+.0f} yds")
    else:
        st.info(f"Not scored — {row.skip_reason} (actual: {row.actual_play_type})")


def _ball_x(row: pd.Series) -> float | None:
    if pd.isna(row.yardline_100) or row.offense_team not in ("LA", "SEA"):
        return None
    # Field spans x=0 (SEA/navy end zone) to x=100 (LA/yellow end zone).
    return row.yardline_100 if row.offense_team == "LA" else 100 - row.yardline_100


def render_field(row: pd.Series) -> None:
    ball_x = _ball_x(row)
    if ball_x is None:
        st.caption("Field data unavailable for this play.")
        return

    direction = -1 if row.offense_team == "LA" else 1
    fig = go.Figure()

    for i, x0 in enumerate(range(0, 100, 10)):
        fig.add_shape(type="rect", x0=x0, x1=x0 + 10, y0=0, y1=53.3,
                       fillcolor=FIELD_GREENS[i % 2], line_width=0, layer="below")

    fig.add_shape(type="rect", x0=-10, x1=0, y0=0, y1=53.3, fillcolor=SCHEME_B["SEA"], line_width=0)
    fig.add_shape(type="rect", x0=100, x1=110, y0=0, y1=53.3, fillcolor=SCHEME_B["LA"], line_width=0)
    fig.add_annotation(x=-5, y=26.65, text="SEA", textangle=-90, showarrow=False,
                        font=dict(color="white", size=16, family="Arial Black"))
    fig.add_annotation(x=105, y=26.65, text="LA", textangle=-90, showarrow=False,
                        font=dict(color=SEA_NAVY, size=16, family="Arial Black"))

    for x in range(0, 101, 10):
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=53.3, line=dict(color="white", width=1))
        if 0 < x < 100:
            label = str(x if x <= 50 else 100 - x)
            fig.add_annotation(x=x, y=6, text=label, showarrow=False, font=dict(color="white", size=10))
            fig.add_annotation(x=x, y=47, text=label, showarrow=False, font=dict(color="white", size=10))

    if row.is_scored and pd.notna(row.yards_gained):
        gain_end = max(0, min(100, ball_x + direction * row.yards_gained))
        x0, x1 = sorted([ball_x, gain_end])
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=53.3, fillcolor=GAIN_RED,
                       opacity=0.35, line_width=0)

    fig.add_shape(type="line", x0=ball_x, x1=ball_x, y0=0, y1=53.3, line=dict(color=LOS_BLUE, width=3))

    if pd.notna(row.distance):
        marker_x = max(0, min(100, ball_x + direction * row.distance))
        fig.add_shape(type="line", x0=marker_x, x1=marker_x, y0=0, y1=53.3,
                      line=dict(color=FIRST_DOWN_YELLOW, width=3, dash="dash"))

    fig.add_trace(go.Scatter(x=[ball_x], y=[26.65], mode="markers",
                              marker=dict(size=14, color="#5C3A21"), showlegend=False, hoverinfo="skip"))

    fig.update_xaxes(range=[-10, 110], visible=False)
    fig.update_yaxes(range=[0, 53.3], visible=False)
    fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10),
                       plot_bgcolor="#FAF9F6", paper_bgcolor="#FAF9F6", showlegend=False)
    st.plotly_chart(fig, width='stretch', config={"staticPlot": True})


def render_accuracy(plays: pd.DataFrame, upto_idx: int) -> None:
    seen = plays[(plays.play_index <= upto_idx) & (plays.is_scored == 1)]
    if seen.empty:
        st.write("No scored plays yet.")
        return
    correct = int((seen.predicted_label == seen.actual_play_type).sum())
    total = len(seen)
    fig = go.Figure(go.Pie(
        values=[correct, total - correct], labels=["Correct", "Incorrect"], hole=0.65,
        marker=dict(colors=["#2CA02C", "#D62728"]), textinfo="none", sort=False,
    ))
    fig.update_layout(
        annotations=[dict(text=f"{correct / total:.0%}<br>{correct}/{total}", x=0.5, y=0.5,
                           font_size=18, showarrow=False)],
        height=220, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="#FAF9F6", legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig, width='stretch')


def render_team_tendency(plays: pd.DataFrame, upto_idx: int) -> None:
    seen = plays[(plays.play_index <= upto_idx) & (plays.is_scored == 1)]
    if seen.empty:
        st.write("No scored plays yet.")
        return
    counts = seen.groupby(["actual_play_type", "offense_team"]).size().reset_index(name="count")
    fig = px.bar(counts, x="actual_play_type", y="count", color="offense_team", barmode="group",
                 color_discrete_map=SCHEME_A, title="Run/pass tendency so far")
    fig.update_layout(paper_bgcolor="#FAF9F6", plot_bgcolor="#FAF9F6")
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

    st.title(GAME_TITLE)
    st.caption(plays.game_id.iloc[0])

    speed = render_sidebar(plays)

    row = plays.iloc[st.session_state.play_idx]
    render_play_card(row)
    render_field(row)

    left, right = st.columns(2)
    with left:
        render_prediction(row)
    with right:
        render_accuracy(plays, st.session_state.play_idx)

    render_team_tendency(plays, st.session_state.play_idx)
    st.caption(f"Play {st.session_state.play_idx + 1} of {len(plays)}")

    autoplay_tick(plays, speed)


if __name__ == "__main__":
    main()
