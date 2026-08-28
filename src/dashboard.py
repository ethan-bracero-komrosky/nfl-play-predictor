"""Streamlit replay dashboard: step through a precomputed game's plays with
live predictions, a football-field visualization, a running accuracy donut,
and a team-tendency chart.

Reads data/replay.db (written by precompute_predictions.py). Run with:
    streamlit run src/dashboard.py
"""

import sqlite3
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

# Prediction bar: pass/run split per team, reusing the same four colors.
TEAM_SPLIT_COLORS = {
    "LA": {"pass": RAMS_BLUE, "run": RAMS_YELLOW},
    "SEA": {"pass": SEA_GREEN, "run": SEA_NAVY},
}
TEXT_COLOR_ON = {RAMS_BLUE: "white", RAMS_YELLOW: "#1A1A1A", SEA_GREEN: "#1A1A1A", SEA_NAVY: "white"}

# Functional field colors (broadcast-standard, not team colors).
LOS_BLUE = "#1E90FF"
FIRST_DOWN_YELLOW = "#FFD700"
GAIN_RED = "#D62728"
FIELD_GREENS = ("#3C8C40", "#2F7A33")


def _build_static_field() -> tuple[list[dict], list[dict]]:
    """Grass, end zones, gridlines, yard numbers — identical on every play."""
    shapes = []
    for i, x0 in enumerate(range(0, 100, 10)):
        shapes.append(dict(type="rect", x0=x0, x1=x0 + 10, y0=0, y1=53.3,
                            fillcolor=FIELD_GREENS[i % 2], line_width=0, layer="below"))
    shapes.append(dict(type="rect", x0=-10, x1=0, y0=0, y1=53.3, fillcolor=SCHEME_B["SEA"], line_width=0))
    shapes.append(dict(type="rect", x0=100, x1=110, y0=0, y1=53.3, fillcolor=SCHEME_B["LA"], line_width=0))
    for x in range(0, 101, 10):
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=53.3, line=dict(color="white", width=1)))

    annotations = [
        dict(x=-5, y=26.65, text="SEA", textangle=-90, showarrow=False,
             font=dict(color="white", size=16, family="Arial Black")),
        dict(x=105, y=26.65, text="LA", textangle=-90, showarrow=False,
             font=dict(color=SEA_NAVY, size=16, family="Arial Black")),
    ]
    for x in range(0, 101, 10):
        if 0 < x < 100:
            label = str(x if x <= 50 else 100 - x)
            annotations.append(dict(x=x, y=6, text=label, showarrow=False, font=dict(color="white", size=10)))
            annotations.append(dict(x=x, y=47, text=label, showarrow=False, font=dict(color="white", size=10)))
    return shapes, annotations


STATIC_FIELD_SHAPES, STATIC_FIELD_ANNOTATIONS = _build_static_field()


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


def render_playback_toggle() -> None:
    # Speed + Play/Pause live outside the fragment on purpose: changing either
    # must trigger a full rerun so render_play_viewer can recompute run_every
    # (a fragment's run_every is fixed at the moment it's defined, so the thing
    # that decides "should this auto-tick, and how fast" can't live inside it).
    st.session_state.setdefault("speed", 0.5)
    c1, c2 = st.columns([3, 1])
    st.session_state.speed = c1.slider("Speed (plays/sec)", 0.5, 5.0, st.session_state.speed, 0.5)
    label = "Pause" if st.session_state.playing else "Play"
    if c2.button(label, width='stretch'):
        st.session_state.playing = not st.session_state.playing


def render_play_card(row: pd.Series) -> None:
    st.subheader(f"Q{row.quarter} — {row.game_clock or ''}")
    c1, c2, c3 = st.columns(3)
    down_txt = f"{int(row.down)} & {int(row.distance)}" if pd.notna(row.down) else "—"
    c1.metric("Down & distance", down_txt)
    c2.metric("Yardline (to opp EZ)", int(row.yardline_100) if pd.notna(row.yardline_100) else "—")
    c3.metric("Offense / Defense", f"{row.offense_team} vs {row.defense_team}")


def _prediction_bar_figure(row: pd.Series) -> go.Figure:
    colors = TEAM_SPLIT_COLORS.get(row.offense_team, {"pass": "#888888", "run": "#CCCCCC"})
    prob_pass = row.predicted_prob_pass
    prob_run = 1 - prob_pass
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[prob_pass], y=[""], orientation="h", marker_color=colors["pass"],
        text=f"Pass {prob_pass:.0%}", textposition="inside", insidetextanchor="middle",
        textfont=dict(color=TEXT_COLOR_ON.get(colors["pass"], "white")), hoverinfo="skip",
    ))
    fig.add_trace(go.Bar(
        x=[prob_run], y=[""], orientation="h", marker_color=colors["run"],
        text=f"Run {prob_run:.0%}", textposition="inside", insidetextanchor="middle",
        textfont=dict(color=TEXT_COLOR_ON.get(colors["run"], "white")), hoverinfo="skip",
    ))
    fig.update_layout(barmode="stack", height=70, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                       xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False),
                       paper_bgcolor="#FAF9F6", plot_bgcolor="#FAF9F6")
    return fig


def render_prediction(row: pd.Series) -> None:
    if row.is_scored:
        st.plotly_chart(_prediction_bar_figure(row), width='stretch',
                         config={"staticPlot": True}, key="prediction_bar")
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

    if row.is_scored and pd.notna(row.yards_gained):
        gain_end = max(0, min(100, ball_x + direction * row.yards_gained))
        gain_x0, gain_x1 = sorted([ball_x, gain_end])
        gain_opacity = 0.35
    else:
        gain_x0 = gain_x1 = ball_x
        gain_opacity = 0.0

    if pd.notna(row.distance):
        marker_x = max(0, min(100, ball_x + direction * row.distance))
        marker_opacity = 1.0
    else:
        marker_x = ball_x
        marker_opacity = 0.0

    # Always exactly 3 dynamic shapes in this order, never appended/omitted —
    # keeps shape count/order constant across reruns so Plotly can diff-update
    # instead of a full remount (which was causing a white flash on play change).
    dynamic_shapes = [
        dict(type="rect", x0=gain_x0, x1=gain_x1, y0=0, y1=53.3, fillcolor=GAIN_RED,
             opacity=gain_opacity, line_width=0),
        dict(type="line", x0=ball_x, x1=ball_x, y0=0, y1=53.3, line=dict(color=LOS_BLUE, width=3)),
        dict(type="line", x0=marker_x, x1=marker_x, y0=0, y1=53.3, opacity=marker_opacity,
             line=dict(color=FIRST_DOWN_YELLOW, width=3, dash="dash")),
    ]

    fig = go.Figure()
    fig.update_layout(shapes=STATIC_FIELD_SHAPES + dynamic_shapes, annotations=STATIC_FIELD_ANNOTATIONS)
    fig.add_trace(go.Scatter(x=[ball_x], y=[26.65], mode="markers",
                              marker=dict(size=14, color="#5C3A21"), showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(range=[-10, 110], visible=False)
    fig.update_yaxes(range=[0, 53.3], visible=False)
    fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10),
                       plot_bgcolor="#FAF9F6", paper_bgcolor="#FAF9F6", showlegend=False)
    st.plotly_chart(fig, width='stretch', config={"staticPlot": True}, key="field_chart")


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
    st.plotly_chart(fig, width='stretch', key="accuracy_donut")


def render_team_tendency(plays: pd.DataFrame, upto_idx: int) -> None:
    seen = plays[(plays.play_index <= upto_idx) & (plays.is_scored == 1)]
    if seen.empty:
        st.write("No scored plays yet.")
        return
    counts = seen.groupby(["actual_play_type", "offense_team"]).size().reset_index(name="count")
    fig = px.bar(counts, x="actual_play_type", y="count", color="offense_team", barmode="group",
                 color_discrete_map=SCHEME_A, title="Run/pass tendency so far")
    fig.update_layout(paper_bgcolor="#FAF9F6", plot_bgcolor="#FAF9F6")
    st.plotly_chart(fig, width='stretch', key="tendency_chart")


def render_play_viewer(plays: pd.DataFrame) -> None:
    # run_every is decided here (outside the fragment) from session_state set by
    # render_playback_toggle, and baked into the fragment when it's (re)defined
    # on this full rerun — see the Streamlit fragment auto-rerun tutorial pattern.
    interval = 1 / st.session_state.speed if st.session_state.playing else None

    @st.fragment(run_every=interval)
    def _viewer() -> None:
        col1, col2 = st.columns(2)
        if col1.button("Prev", width='stretch'):
            st.session_state.playing = False
            st.session_state.play_idx = max(0, st.session_state.play_idx - 1)
        if col2.button("Next", width='stretch'):
            st.session_state.playing = False
            st.session_state.play_idx = min(len(plays) - 1, st.session_state.play_idx + 1)

        idx = st.slider("Play", 0, len(plays) - 1, st.session_state.play_idx, label_visibility="collapsed")
        if idx != st.session_state.play_idx:
            st.session_state.playing = False
            st.session_state.play_idx = idx

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

        if st.session_state.playing:
            if st.session_state.play_idx >= len(plays) - 1:
                st.session_state.playing = False
                st.rerun()  # full rerun so run_every gets recomputed to None
            else:
                st.session_state.play_idx += 1

    _viewer()


def main() -> None:
    st.set_page_config(page_title="NFL Replay", layout="wide")
    plays = load_plays()
    init_state(len(plays) - 1)

    st.title(GAME_TITLE)
    st.caption(plays.game_id.iloc[0])

    render_playback_toggle()
    render_play_viewer(plays)


if __name__ == "__main__":
    main()
