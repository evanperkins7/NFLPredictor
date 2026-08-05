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

## Step 6: Recruiter packaging

- [x] Add a Mermaid architecture and data-flow diagram.
- [x] Add a feature data dictionary.
- [x] Add a model card with metrics, leakage evidence, and limitations.
- [x] Add a recruiter-oriented case study with linked evidence.
- [x] Link the package from the README and verify the documentation claims.
