# Pre-Snap Intelligence

A machine learning tool that predicts offensive play-calling (run vs. pass) before the snap, built for defensive coordinators to quantify the expected value of different defensive alignments.

Given pre-snap game state — down, distance, field position, score differential, time remaining, personnel tendencies — the model estimates the probability of a run or pass call, so a defense can weigh its alignment options against what's actually likely to happen next.

## Project status

**Phase 1 (data science) — nearly complete. Phase 2 (ETL) — in progress.**

- Trained and compared Logistic Regression, Random Forest, and XGBoost on play-by-play data (2016–2024), testing on the 2025 season
- XGBoost is the current best performer (~0.72–0.73 accuracy)
- Final feature set is pre-snap only: down, distance, field position, score differential, game/quarter/half clock, timeouts remaining, shotgun/no-huddle, win probability (wp), expected points (ep), and one-hot encoded teams
- Identified and removed a data leakage bug (`pass_oe`, an advanced stat computed post-snap, had inflated accuracy to 0.96)
- Trained model exported to `models/xgb_run_pass_model.json`
- `src/precompute_predictions.py` scores one game's plays and writes them to `data/replay.db` (SQLite) for the replay dashboard; plays outside the training labels (two-point conversions, special teams, no-plays) are flagged rather than scored, so it doesn't crash on the Phase 4 demo game's OT/two-point-conversion edge cases

See `notebooks/` for the full modeling notebook.

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Data science notebook: data prep, feature engineering, model comparison | Nearly done |
| 2 | ETL pipeline into SQLite/PostgreSQL | In progress — model export + single-game precompute script done |
| 3 | FastAPI model-serving endpoint | Not started |
| 4 | Streamlit dashboard: play-by-play game replay with live predictions | Not started |
| 5 | Deploy (API → Railway/Render, dashboard → Streamlit Community Cloud) | Not started |

### Phase 4 demo target

The dashboard's flagship replay will be **Rams @ Seahawks, Week 16, 2025** — a Thursday Night Football game between two 11+ win division rivals that went to overtime after Seattle erased a 16-point fourth-quarter deficit. Chosen for the dramatic win-probability swing, the OT/two-point-conversion edge cases, and the visual contrast between team colors.

Dashboard can include:
- Watch it play out snap by snap at adjustable speed
- Model predicts pass/run probability before each play
- After play resolves, accuracy counter updates
- Team tendency charts update as the game progresses

> Note: `nflreadpy` updates weekly/nightly, not in real time. The dashboard replays completed games rather than tracking live ones, with a documented upgrade path to a commercial live-data API noted as a future enhancement.

## Tech stack

- **Data**: [`nflreadpy`](https://nflreadr.nflverse.com/) (play-by-play), converted from Polars to pandas
- **Modeling**: pandas, NumPy, scikit-learn, XGBoost
- **Planned**: SQLite/PostgreSQL, FastAPI, Streamlit, Plotly, deployed on Railway/Render + Streamlit Community Cloud

## Key technical decisions

- **Strict pre-snap feature discipline**: any feature that could leak post-snap information (like `pass_oe`) is excluded, even if it improves accuracy
- **Two-point conversions** and plays with a null `down` are dropped from training (not standard down-and-distance plays)
- **Overtime** plays are retained, with `game_half` encoded as a distinct value for OT
- **Timeout features** are engineered from home/away columns into offense/defense (`posteam`/`defteam`) using `posteam_type`

## Planned enhancement

A rolling Pass Rate Over Expected (PROE) feature is deferred — it requires restructuring the pipeline so `posteam` is one-hot encoded *after* the rolling PROE calculation runs, rather than before.

## Repo structure

```
notebooks/   modeling notebook (data prep, feature engineering, model comparison, export)
src/         precompute_predictions.py — score one game, write to data/replay.db
data/        replay.db (SQLite, gitignored)
models/      xgb_run_pass_model.json (gitignored)
config/      (empty, reserved for Phase 3+)
```
