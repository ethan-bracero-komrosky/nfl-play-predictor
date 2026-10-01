"""Pre-game PROE from historical down/distance rates and the previous five games."""

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def build_proe_features(
    plays: pd.DataFrame, *, history_start_year: int, training_end_year: int,
) -> pd.DataFrame:
    """Return aligned diagnostics and a situation-adjusted, five-game rolling PROE.

    Only seasons in the training range update history. Within that range, a
    game's expected rates use earlier dates, excluding every game on its date.
    Test-season outcomes never update history. Windows carry across seasons.
    Games have equal weight in the five-game window, as in the notebook draft.
    """
    if history_start_year > training_end_year:
        raise ValueError("PROE history starts after the training period ends")
    columns = ["posteam", "game_id", "season", "game_date", "down", "ydstogo",
               "two_point_attempt", "play_type"]
    df = plays[columns].copy().reset_index(drop=True)
    if df[["posteam", "game_id", "season", "game_date"]].isna().any().any():
        raise ValueError("PROE needs team, game, season, and date for every play")
    if not df["two_point_attempt"].isin([0, 1]).all():
        raise ValueError("Unexpected two_point_attempt values or nulls")
    if not df["ydstogo"].between(0, 100).all():
        raise ValueError("Unexpected ydstogo values outside the bucket range")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="raise").dt.normalize()
    df["down"] = df["down"].mask(df["two_point_attempt"].eq(1), 0)
    if not df["down"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Unexpected down for PROE")
    bucket = pd.cut(df["ydstogo"], bins=[0, 3, 6, 10, 15, 100],
                    labels=["short", "medium", "long", "very_long", "forever"])
    # Conversions keep their own expectation instead of disappearing in groupby.
    df["situation"] = bucket.astype(object).mask(df["two_point_attempt"].eq(1), "two_point")
    if df["situation"].isna().any():
        raise ValueError("Missing distance bucket on a regular play")
    df["is_pass"] = df["play_type"].map({"run": 0, "pass": 1, 0: 0, 1: 1})
    df["use_history"] = df["season"].between(history_start_year, training_end_year)
    if df.loc[df["use_history"], "is_pass"].isna().any():
        raise ValueError("Historical PROE plays must have run/pass labels")

    league_rates = np.full(len(df), 0.5)
    rolling_rates = np.full(len(df), 0.5)
    rolling_expected = np.full(len(df), 0.5)
    league_counts = defaultdict(lambda: [0.0, 0])
    team_games = defaultdict(lambda: deque(maxlen=5))
    total_passes = 0.0
    total_plays = 0

    for _, day in df.groupby("game_date", sort=True):
        fallback_rate = total_passes / total_plays if total_plays else 0.5
        for (down, situation), rows in day.groupby(["down", "situation"], sort=False):
            passes, count = league_counts[(down, situation)]
            league_rates[rows.index] = passes / count if count else fallback_rate

        pending_games = []
        for (team, _), rows in day.groupby(["posteam", "game_id"], sort=False):
            history = team_games[team]
            if history:
                rolling_rates[rows.index] = np.mean([game[0] for game in history])
                rolling_expected[rows.index] = np.mean([game[1] for game in history])
            if rows["use_history"].all():
                pending_games.append((team, rows["is_pass"].mean(),
                                      league_rates[rows.index].mean()))

        # Commit only after assigning features to every game on this date.
        for team, actual, expected in pending_games:
            team_games[team].append((actual, expected))
        historical = day[day["use_history"]]
        for key, rows in historical.groupby(["down", "situation"], sort=False):
            league_counts[key][0] += rows["is_pass"].sum()
            league_counts[key][1] += len(rows)
        total_passes += historical["is_pass"].sum()
        total_plays += len(historical)

    return pd.DataFrame({
        "league_pass_rate": league_rates,
        "rolling_pass_rate": rolling_rates,
        "rolling_expected_pass_rate": rolling_expected,
        # No prior games means neutral PROE (0), without using future outcomes.
        "proe": rolling_rates - rolling_expected,
    }, index=plays.index)
