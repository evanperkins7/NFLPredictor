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
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

The project includes a CLI that downloads seasons, materializes a training table, fits
the selected model, and writes prediction artifacts under `artifacts/`.

## Weekly predictions

Generate predictions for an upcoming season or week using the selected sigmoid-calibrated
8-game EPA-only model:

```powershell
.\.venv\Scripts\python.exe scripts/predict_week.py --season 2026 --week 1
```

The command downloads the previous ten completed seasons plus the target schedule and
writes a CSV, probability chart, and Markdown report under `artifacts/predictions/`.
The output probabilities are estimates, and confidence tiers are communication aids,
not guarantees.

## Local dashboard

Run the recruiter-facing Streamlit demo with saved artifacts by default:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard includes weekly predictions, walk-forward calibration evidence, feature
ablations, leakage-audit status, and links to the supporting documentation. Its refresh
button runs the existing weekly CLI explicitly, so the interface does not duplicate
model or feature-engineering logic. The app opens in read-only artifact mode by default;
enable local refreshes explicitly with `$env:NFL_ENABLE_REFRESH = "true"` before
starting Streamlit.

## Streamlit Community Cloud deployment

The project is prepared for Streamlit Community Cloud with `app.py` as the entrypoint,
`requirements.txt` in the repository root, and a deployment-safe read-only default.
After pushing the standalone repository, create an app in Community Cloud and choose
the `main` branch plus `app.py`. Keep `NFL_ENABLE_REFRESH` unset so public visitors use
the committed artifacts rather than triggering data downloads.

## Project documentation

- [Architecture](docs/architecture.md)
- [Data dictionary](docs/data-dictionary.md)
- [Model card](docs/model-card.md)
- [Recruiter case study](docs/case-study.md)

## Data and modeling notes

`nflreadpy` returns Polars DataFrames; this project converts them to pandas at the
boundary so feature engineering and scikit-learn remain straightforward. Only
completed regular-season games are used for the baseline. Week-one rows are omitted
until a prior-season rating policy is chosen, avoiding an arbitrary cold-start value.
