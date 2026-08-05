# NFL Weekly Game Predictor

A reproducible, leakage-aware NFL game prediction project built with `nflreadpy`,
`pandas`, and scikit-learn.

## MVP scope

- Use the last ten completed regular seasons as a configurable training window.
- Build pre-game team features from prior completed games only.
- Predict the home-team win probability with logistic regression.
- Export predicted winner and confidence tiers; add expected margin/spread as a separate
  regression model after the classification baseline is understood.
- Evaluate chronologically with log loss, Brier score, accuracy, and calibration—not
  just a random train/test split.

The initial feature set is rolling offensive EPA, defensive EPA, play volume (a
transparent pace proxy), and rest-day differential. The feature builder deliberately
fails with a useful error if the nflverse schema changes instead of silently selecting
the wrong column.

## Setup

```powershell
py -m pip install -e ".[dev]"
pytest
```

The next runnable milestone will add a CLI that downloads seasons, materializes a
training table, fits the model, and writes evaluation artifacts under `artifacts/`.

## Data and modeling notes

`nflreadpy` returns Polars DataFrames; this project converts them to pandas at the
boundary so feature engineering and scikit-learn remain straightforward. Only
completed regular-season games are used for the baseline. Week-one rows are omitted
until a prior-season rating policy is chosen, avoiding an arbitrary cold-start value.
