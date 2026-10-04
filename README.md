# Pre-Snap Intelligence

A machine learning tool that predicts offensive play-calling (run vs. pass) before the snap, built for defensive coordinators to quantify the expected value of different defensive alignments.

Given pre-snap game state — down, distance, field position, score differential, time remaining, team tendencies — the model estimates the probability of a run or pass call, so a defense can weigh its alignment options against what's actually likely to happen next.

## Project status

**Phase 1 (data science) — done. Phase 2 (ETL) — partial. Phase 4 (replay dashboard) — built, polishing.**

- Trained and compared Logistic Regression, Random Forest, and XGBoost on 2016–2025 play-by-play data, testing on the **2026 season through Week 4 (5,955 plays)** — Week 4 is only the Thursday night game so far
- Shipped model: **XGBoost trained on 2021–2025 — 73.8% accuracy** on 5,955 2026 plays (always guessing "pass" scores 57.3%)
- Added a rolling **Pass Rate Over Expected (PROE)** team-tendency feature (`src/proe.py`, unit-tested in `tests/`)
- Two-point conversions are now kept and scored, flagged by a `two_point_attempt` input
- Trained model exported to `models/xgb_run_pass_model.json`, with its PROE history window stored in the model file so replay reproduces the same features
- `src/precompute_predictions.py` scores one game's plays and writes them to `data/replay.db` (SQLite); special teams, no-plays, and spikes/kneels are flagged rather than scored
- `src/dashboard.py` is a Streamlit replay of the demo game with a field view, live prediction bar, running accuracy, and team-tendency chart

See `notebooks/` for the full modeling notebook.

## Results (2026 through Week 4)

Test set: 5,955 run/pass plays from the 2026 season through Week 4 (17 of them two-point attempts). Week 4 includes only the Thursday night game so far. Each training range gets its own PROE history.

| Training range | Training plays | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|---|
| 2016–2025 | 344,820 | 72.8% | 72.9% | 73.7% |
| 2019–2025 | 244,232 | 72.9% | 72.6% | 73.7% |
| **2021–2025** | 176,478 | 72.9% | 72.8% | **73.8%** |

- XGBoost wins in every training range, by about 1 percentage point
- More history doesn't help: for XGBoost, 2021–2025 beats 2016–2025 with about half the plays
- Shipped model (XGBoost, 2021–2025): balanced accuracy 73.7%, macro F1 73.5%; pass recall 0.75, run recall 0.73
- A yards-to-go bucket feature (short/medium/long/…) was tested and dropped — it added nothing for XGBoost and hurt the other two models

Full per-model metrics are written to `data/*.json` by the notebook's comparison cell.

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Data science notebook: data prep, feature engineering, model comparison | Done |
| 2 | ETL pipeline into SQLite/PostgreSQL | Partial — model export + single-game precompute to SQLite done |
| 3 | FastAPI model-serving endpoint | Not started |
| 4 | Streamlit dashboard: play-by-play game replay with live predictions | Built — fixing a chart flash / scroll-jump bug during autoplay |
| 5 | Deploy (API → Railway/Render, dashboard → Streamlit Community Cloud) | Not started |

### Phase 4 demo game

The dashboard replays **Rams @ Seahawks, Week 16, 2025** — a Thursday Night Football game between two 11+ win division rivals that went to overtime after Seattle erased a 16-point fourth-quarter deficit. Chosen for the dramatic win-probability swing, the OT/two-point-conversion edge cases, and the visual contrast between team colors.

The dashboard:
- Plays the game snap by snap at adjustable speed (Play/Pause + speed slider)
- Shows the model's pass/run probability before each play, split in team colors
- Draws each play on a football field (line of scrimmage, first-down marker, yards gained)
- Updates a running accuracy donut and a team-tendency chart as the game progresses

> Note: `nflreadpy` updates weekly/nightly, not in real time. The dashboard replays completed games rather than tracking live ones, with a documented upgrade path to a commercial live-data API noted as a future enhancement.

## Running it

```bash
pip install -r requirements.txt

# 1. Train + export the model: run notebooks/nfl-play-predictor.ipynb top to bottom
# 2. Score the demo game into data/replay.db
python src/precompute_predictions.py
# 3. Launch the replay
streamlit run src/dashboard.py

# Tests
python -m unittest tests.test_proe
```

## Tech stack

- **Data**: [`nflreadpy`](https://nflreadr.nflverse.com/) (play-by-play), converted from Polars to pandas
- **Modeling**: pandas, NumPy, scikit-learn, XGBoost
- **Storage / app**: SQLite, Streamlit, Plotly
- **Planned**: PostgreSQL, FastAPI, deployed on Railway/Render + Streamlit Community Cloud

## Key technical decisions

- **Strict pre-snap feature discipline**: any feature that could leak post-snap information is excluded, even if it improves accuracy. `pass_oe` (computed after the play) once inflated accuracy to 0.96 and was removed
- **PROE without leakage**: a team's PROE is the average of (actual − expected pass rate) over its previous five games. Expected rates come from league down-and-distance rates on *earlier dates only*; the current game and all test-season outcomes never feed history
- **Two-point conversions are kept**: they have no regular down, so `down = 0` is a placeholder and `two_point_attempt = 1` marks them
- **Overtime** plays are retained, with `game_half` encoded as a distinct value for OT
- **Timeout features** are engineered from home/away columns into offense/defense (`posteam`/`defteam`) using `posteam_type`
- **Temporal split**: train on earlier seasons, test on the newest, matching the order information becomes available
- **Replay must match training**: `precompute_predictions.py` reproduces the notebook's feature engineering exactly and refuses to run if the model's feature list or PROE window doesn't match

## Repo structure

```
notebooks/   modeling notebook (data prep, feature engineering, model comparison, export)
src/         proe.py — rolling PROE feature
             precompute_predictions.py — score one game, write to data/replay.db
             dashboard.py — Streamlit replay dashboard
tests/       unit tests for PROE
data/        replay.db + model comparison JSON (gitignored)
models/      xgb_run_pass_model.json (gitignored)
config/      (empty, reserved for Phase 3)
```
