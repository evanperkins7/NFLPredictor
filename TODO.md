# TODO

## Repository setup

- [ ] Migrate `NFL Project` into its own local Git clone connected directly to
  `https://github.com/evanperkins7/NFLPredictor.git`.
- [ ] Retire the temporary subtree-push workflow after verifying the standalone clone
  has the complete project history.

## Step 3: Leakage audit

- [x] Prove that each prediction row uses only information available before kickoff.
- [x] Add temporal-ordering, duplicate-row, missing-feature, and same-game-contamination
  checks.
- [x] Add adversarial tests that mutate current/future game statistics and confirm
  earlier predictions do not change.
- [x] Generate a leakage-audit report for the 2016–2025 feature table.

Audit result: passed on 2026-08-05. The report covers 2,547 rows, with no structural
violations. Mutating `2021_06_HOU_IND` left all 1,281 current-or-earlier protected rows
unchanged. See `artifacts/model/leakage_audit.json`.

## Step 4: Ablations

- [x] Compare feature groups with chronological evaluation.
- [x] Compare expanding, 3-game, 5-game, and 8-game rolling windows.
- [x] Use walk-forward evaluation as the primary selection protocol and retain the
  2024–2025 fixed holdout as a reference.
- [x] Preserve reproducible per-season and pooled result tables.

Result: the 8-game EPA-only configuration is recommended for the weekly workflow based
on walk-forward log loss and Brier score. See `artifacts/model/ablations/report.md`.

## Step 5: Weekly workflow

- [x] Generate features for unplayed regular-season schedule rows from completed
  historical team statistics.
- [x] Add the weekly prediction CLI using the selected 8-game EPA-only model.
- [x] Export probabilities, predicted winner, and confidence tier to CSV.
- [x] Export a static probability chart and Markdown report.
- [x] Verify the real workflow against the 2026 Week 1 schedule.

Result: the workflow generated 16 predictions successfully. See
`artifacts/predictions/predictions_2026_week_1.csv` and the adjacent chart/report.

## Step 7: Neutral-situation pace experiment

- [x] Add a schema-checked play-by-play data boundary.
- [x] Define and test an explicit neutral-play filter.
- [x] Add leakage-aware rolling neutral pace features.
- [x] Compare neutral pace against EPA-only and existing feature groups.
- [x] Preserve the full 2016–2025 experiment outputs.

Result: neutral pace did not improve walk-forward log loss or Brier score, so the
8-game EPA-only model remains selected. See `artifacts/model/ablations/report.md`.

## Step 8: Probability calibration

- [x] Add time-aware reliability metrics and calibration candidates.
- [x] Keep calibration data strictly before each outer test season.
- [x] Compare raw, sigmoid, and isotonic probabilities over 2016–2025.
- [x] Use sigmoid calibration for weekly predictions based on walk-forward evidence.
- [x] Retain the existing confidence-tier thresholds pending stronger tier-level evidence.

Result: sigmoid calibration slightly improves the primary walk-forward probability
metrics; see `artifacts/model/calibration/report.md`.

## Step 9: Local Streamlit demo

- [x] Add a polished, artifact-first local dashboard.
- [x] Show weekly predictions, calibration evidence, ablations, and leakage status.
- [x] Add an explicit refresh action that delegates to the verified weekly CLI.
- [x] Add dashboard artifact-loading tests and a Streamlit smoke test.
- [x] Document local setup and usage.

Result: `app.py` is the recruiter-facing local demo. Public deployment remains a
follow-up after standalone repository migration.

## Step 6: Recruiter packaging

- [x] Add a Mermaid architecture and data-flow diagram.
- [x] Add a feature data dictionary.
- [x] Add a model card with metrics, leakage evidence, and limitations.
- [x] Add a recruiter-oriented case study with linked evidence.
- [x] Link the package from the README and verify the documentation claims.
