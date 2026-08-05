# TODO

## Repository setup

- [ ] Migrate `NFL Project` into its own local Git clone connected directly to
  `https://github.com/evanperkins7/NFLPredictor.git`.
- [ ] Retire the temporary subtree-push workflow after verifying the standalone clone
  has the complete project history.

## Step 3: Leakage audit

- [ ] Prove that each prediction row uses only information available before kickoff.
- [ ] Add temporal-ordering, duplicate-row, missing-feature, and same-game-contamination
  checks.
- [ ] Add adversarial tests that mutate current/future game statistics and confirm
  earlier predictions do not change.
- [ ] Generate a leakage-audit report for the 2016–2025 feature table.

