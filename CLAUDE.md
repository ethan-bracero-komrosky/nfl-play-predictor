# nfl-play-predictor

Pre-snap run/pass prediction tool for defensive coordinators. Full project
description, roadmap, and modeling decisions live in `README.md` — this file
is Claude-focused: where things are, what patterns to follow, and a running
log of what happened each session.

## Repo structure

```
notebooks/nfl-play-predictor.ipynb   Modeling notebook: data prep, feature engineering,
                                      model comparison (LogReg/RF/XGBoost), export.
                                      Source of truth for feature engineering — src/ scripts
                                      must reproduce it exactly, not reinvent it.
src/precompute_predictions.py        Loads models/xgb_run_pass_model.json, pulls one game's
                                      play-by-play via nflreadpy, reproduces the notebook's
                                      pre-snap feature engineering (see TEAM_COLS/FEATURE_COLS/
                                      RAW_COLS constants), scores every play, writes to
                                      data/replay.db. GAME_ID is hardcoded to the Phase 4 demo
                                      game (2025_16_LA_SEA, Rams @ Seahawks). Plays outside the
                                      training labels (two-point conversions, special teams,
                                      no-plays) are flagged via skip_reason rather than scored.
src/dashboard.py                     Streamlit replay dashboard. Run with
                                      `streamlit run src/dashboard.py`. Reads data/replay.db.
                                      Steps through the game's plays with a football-field
                                      Plotly visualization, prediction bar, running accuracy
                                      donut, and team-tendency chart.
data/replay.db                       SQLite, gitignored. Written by precompute_predictions.py,
                                      read by dashboard.py. Regenerate by rerunning the script.
models/xgb_run_pass_model.json       Trained XGBoost model, gitignored. Exported from the
                                      notebook.
config/                              Empty, reserved for Phase 3 (FastAPI serving).
.streamlit/config.toml               Light theme (bg #FAF9F6) for the dashboard.
requirements.txt                     pandas, numpy, scikit-learn, xgboost, streamlit, plotly,
                                      nflreadpy. Unpinned.
```

## Key patterns

- **Pre-snap feature discipline**: any feature that could leak post-snap
  information is excluded from training, even if it improves accuracy (a
  `pass_oe` leak previously inflated accuracy to 0.96 — see README "Key
  technical decisions"). Don't add features to `FEATURE_COLS` in
  `precompute_predictions.py` without checking whether they're known before
  the snap.
- **Feature engineering must match the notebook exactly** — `precompute_predictions.py`
  reproduces cell-by-cell logic from `notebooks/nfl-play-predictor.ipynb`
  (comment at the top of `TEAM_COLS` cites the specific cell). If the
  notebook's feature engineering changes, the script must be updated to match,
  not diverge.
- **Streamlit dashboard structure** (`src/dashboard.py`):
  - `render_play_viewer()` wraps the per-play UI in `@st.fragment(run_every=interval)`
    so autoplay ticks only rerun that fragment, not the whole page. Play/Pause
    and the speed slider live *outside* the fragment on purpose (in
    `render_playback_toggle()`) so changing them forces a full rerun to
    recompute `interval` — a fragment's `run_every` is fixed at definition
    time and can't be changed from inside it.
  - `render_field()` builds `STATIC_FIELD_SHAPES`/`STATIC_FIELD_ANNOTATIONS`
    once at module load (grass, end zones, gridlines never change) and always
    emits exactly 3 dynamic shapes in the same order (gain rect, LOS line,
    first-down marker), toggling opacity rather than adding/removing shapes,
    so Plotly can diff-update instead of remounting.
  - All charts are Plotly via `st.plotly_chart(..., key=...)` — never
    `st.pyplot`/Altair. Team colors are split into two intentionally different
    schemes: `SCHEME_A` (tendency chart) vs `SCHEME_B` (field end zones) vs
    `TEAM_SPLIT_COLORS` (prediction bar pass/run split) — don't consolidate
    these, they're colored for different visual purposes.
- **Only commit when explicitly asked** — standard Claude Code convention,
  applies here too.

## Session log

Append one dated entry per session, most recent first. Keep entries short —
what changed, what's still open, what to do next. When resuming, read the
top entry (and the plan file it points to, if any) before doing anything
else.

### 2026-10-04

Reran the notebook on the newest 2026 data and updated its markdown and the
README (commit "update for Week 4 TNF"). Test set is now 5,955 plays: Weeks
1–3 plus only Week 4's Thursday game (`2026_04_PIT_CLE`) — Sunday games
weren't published yet. Shipped XGBoost (2021–2025): 73.82% (was 73.80%);
always-pass 57.26%. Rankings unchanged.

Cell 128 used to `assert` that the saved no-PROE baseline
(`data/training_range_comparison_2026.json`) had the same play counts; that
file is from the Week 1 run (1,911 plays), so the assert crashed. It now
prints a skip message instead. The "+0.52 pp from PROE" figure in cell 129
is labeled as a Week 1-only result.

Not done: `data/replay.db` not regenerated (training data unchanged, so
predictions should match); no current PROE-gain measurement (would need
code to rebuild the no-PROE baseline on the new test set).

**Next**: once full Week 4 is published, rerun the notebook and hand-update
README + notebook markdown numbers (cells 98, 104, 109, 114, 124, 127, 129
cite the test set; 28/48/63/93 cite total play counts). README should then
drop the "Thursday night game only" note.

### 2026-08-28

Investigated (not yet implemented) the replay dashboard's flash/scroll-jump
bug — the prior fix in commit `09b7ffe` didn't actually address it: it still
flashes white on every play and jumps to the top of the page, and now the
accuracy/tendency charts flash too (they got no treatment last time).

Root-caused both symptoms against Streamlit's own GitHub issues (not
guesses):
- **Chart flash**: `st.plotly_chart` defaults to `theme="streamlit"`, which
  reprocesses the figure spec every rerun and breaks Plotly's diff-update,
  forcing a remount (paints white first). Fix: `theme=None` on all 4 chart
  calls. ([streamlit/streamlit#8782](https://github.com/streamlit/streamlit/issues/8782),
  [PR #12078](https://github.com/streamlit/streamlit/pull/12078))
- **Scroll jump**: confirmed open Streamlit bug — `st.fragment(run_every=...)`
  redraws its whole block on every autoplay tick, and Streamlit resets scroll
  on that redraw. No clean Python fix exists yet; workaround is a small JS
  snippet (via `st.components.v1.html`) that saves and restores
  `window.parent.scrollY` around each redraw.
  ([streamlit/streamlit#11971](https://github.com/streamlit/streamlit/issues/11971))

Full plan (approved, not yet executed) is saved at
`/Users/neptune/.claude/plans/the-glitch-wasn-t-fixed-adaptive-pnueli.md`.

**Next**: implement the plan in `src/dashboard.py`, then run
`streamlit run src/dashboard.py` and verify in-browser — Play button, check
no white flash on any of the 4 charts and no scroll movement during autoplay
ticks at both slow and fast speed.
