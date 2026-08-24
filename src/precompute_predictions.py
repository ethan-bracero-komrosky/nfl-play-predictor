"""Precompute run/pass predictions for one game and store them for replay.

Loads the trained XGBoost model (notebooks/nfl-play-predictor.ipynb, exported
to models/xgb_run_pass_model.json), pulls play-by-play for GAME_ID via
nflreadpy, reproduces the notebook's exact pre-snap feature engineering, and
writes one row per play to data/replay.db (SQLite).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

GAME_ID = "2025_16_LA_SEA"
MODEL_PATH = Path("models/xgb_run_pass_model.json")
DB_PATH = Path("data/replay.db")

# Exact order used in training (notebooks/nfl-play-predictor.ipynb, cell 745d1ba3).
TEAM_COLS = [
    "posteam_ARI", "posteam_ATL", "posteam_BAL", "posteam_BUF", "posteam_CAR",
    "posteam_CHI", "posteam_CIN", "posteam_CLE", "posteam_DAL", "posteam_DEN",
    "posteam_DET", "posteam_GB", "posteam_HOU", "posteam_IND", "posteam_JAX",
    "posteam_KC", "posteam_LA", "posteam_LAC", "posteam_LV", "posteam_MIA",
    "posteam_MIN", "posteam_NE", "posteam_NO", "posteam_NYG", "posteam_NYJ",
    "posteam_PHI", "posteam_PIT", "posteam_SEA", "posteam_SF", "posteam_TB",
    "posteam_TEN", "posteam_WAS",
]

FEATURE_COLS = [
    "down", "ydstogo", "yardline_100", "goal_to_go", "score_differential", "qtr",
    "game_seconds_remaining", "half_seconds_remaining", "game_half",
    "quarter_seconds_remaining", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "shotgun", "no_huddle", "wp", "ep",
] + TEAM_COLS

RAW_COLS = [
    "game_id", "play_id", "qtr", "time", "down", "ydstogo", "yardline_100",
    "goal_to_go", "score_differential", "game_seconds_remaining",
    "half_seconds_remaining", "game_half", "quarter_seconds_remaining",
    "home_timeouts_remaining", "away_timeouts_remaining", "posteam_type",
    "shotgun", "no_huddle", "posteam", "defteam", "wp", "ep", "play_type", "desc",
]

GAME_MARKER_DESCS = {"GAME", "END GAME"}


def load_model(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model


def fetch_game_pbp(game_id: str) -> pd.DataFrame:
    import nflreadpy as nfl

    season = int(game_id.split("_")[0])
    pbp = nfl.load_pbp([season]).to_pandas()
    game = pbp[pbp["game_id"] == game_id][RAW_COLS].copy()
    if game.empty:
        raise ValueError(f"No play-by-play rows found for game_id={game_id!r}")
    game = game.sort_values("play_id").reset_index(drop=True)

    # Drop non-play sentinel rows (kickoff/quarter/game markers with no play_type).
    is_marker = game["play_type"].isna() & (
        game["desc"].isin(GAME_MARKER_DESCS) | game["desc"].str.startswith("END QUARTER")
    )
    return game[~is_marker].reset_index(drop=True)


def classify_play(row: pd.Series) -> str | None:
    """Return None if scoreable, else a reason the play can't be scored."""
    if row["play_type"] in ("pass", "run") and pd.notna(row["down"]):
        return None
    if pd.isna(row["down"]) and row["play_type"] in ("pass", "run"):
        return "two_point_conversion"
    if row["play_type"] in ("punt", "field_goal", "extra_point", "kickoff"):
        return "special_teams"
    if row["play_type"] == "no_play":
        return "no_play"
    if row["play_type"] in ("qb_spike", "qb_kneel"):
        return "spike_or_kneel"
    return "missing_down" if pd.isna(row["down"]) else "other_non_play"


def build_features(scoreable: pd.DataFrame) -> pd.DataFrame:
    df = scoreable.copy()

    df["posteam_timeouts_remaining"] = np.where(
        df["posteam_type"] == "home", df["home_timeouts_remaining"], df["away_timeouts_remaining"]
    )
    df["defteam_timeouts_remaining"] = np.where(
        df["posteam_type"] == "home", df["away_timeouts_remaining"], df["home_timeouts_remaining"]
    )

    df["game_half"] = df["game_half"].map({"Half1": 0, "Half2": 1}).fillna(2)

    int_cols = [
        "down", "ydstogo", "yardline_100", "goal_to_go", "score_differential", "qtr",
        "game_seconds_remaining", "half_seconds_remaining", "quarter_seconds_remaining",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining", "shotgun", "no_huddle",
        "game_half",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)

    for col in TEAM_COLS:
        df[col] = (df["posteam"] == col.removeprefix("posteam_")).astype(int)

    return df[FEATURE_COLS]


def predict(model: XGBClassifier, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    prob_pass = model.predict_proba(X)[:, 1]
    label = np.where(prob_pass >= 0.5, "pass", "run")
    return prob_pass, label


def write_to_db(rows: list[dict], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS play_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                play_index INTEGER NOT NULL,
                play_id INTEGER,
                quarter INTEGER,
                game_clock TEXT,
                down INTEGER,
                distance INTEGER,
                yardline_100 INTEGER,
                offense_team TEXT,
                defense_team TEXT,
                predicted_prob_pass REAL,
                predicted_label TEXT,
                actual_play_type TEXT,
                is_scored INTEGER NOT NULL,
                skip_reason TEXT,
                UNIQUE(game_id, play_index)
            )
        """)
        conn.execute("DELETE FROM play_predictions WHERE game_id = ?", (rows[0]["game_id"],))
        conn.executemany(
            """
            INSERT INTO play_predictions (
                game_id, play_index, play_id, quarter, game_clock, down, distance,
                yardline_100, offense_team, defense_team, predicted_prob_pass,
                predicted_label, actual_play_type, is_scored, skip_reason
            ) VALUES (
                :game_id, :play_index, :play_id, :quarter, :game_clock, :down, :distance,
                :yardline_100, :offense_team, :defense_team, :predicted_prob_pass,
                :predicted_label, :actual_play_type, :is_scored, :skip_reason
            )
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    model = load_model(MODEL_PATH)
    game = fetch_game_pbp(GAME_ID)

    game["skip_reason"] = game.apply(classify_play, axis=1)
    scoreable_mask = game["skip_reason"].isna()

    prob_pass = pd.Series(np.nan, index=game.index)
    label = pd.Series(None, index=game.index, dtype=object)
    if scoreable_mask.any():
        X = build_features(game[scoreable_mask])
        p, l = predict(model, X)
        prob_pass[scoreable_mask] = p
        label[scoreable_mask] = l

    rows = [
        {
            "game_id": r["game_id"],
            "play_index": i,
            "play_id": int(r["play_id"]) if pd.notna(r["play_id"]) else None,
            "quarter": int(r["qtr"]) if pd.notna(r["qtr"]) else None,
            "game_clock": r["time"],
            "down": int(r["down"]) if pd.notna(r["down"]) else None,
            "distance": int(r["ydstogo"]) if pd.notna(r["ydstogo"]) else None,
            "yardline_100": int(r["yardline_100"]) if pd.notna(r["yardline_100"]) else None,
            "offense_team": r["posteam"],
            "defense_team": r["defteam"],
            "predicted_prob_pass": float(prob_pass[i]) if pd.notna(prob_pass[i]) else None,
            "predicted_label": label[i],
            "actual_play_type": r["play_type"],
            "is_scored": int(scoreable_mask[i]),
            "skip_reason": r["skip_reason"],
        }
        for i, r in game.iterrows()
    ]

    write_to_db(rows, DB_PATH)

    scored = scoreable_mask.sum()
    print(f"Wrote {len(rows)} rows for {GAME_ID} to {DB_PATH} "
          f"({scored} scored, {len(rows) - scored} flagged)")
    print(game["skip_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
