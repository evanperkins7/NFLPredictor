# Case study: building an honest weekly NFL predictor

## The problem

The project started with a practical question: can a small weekly NFL prediction
system be made reproducible and inspectable from raw data through evaluation? The goal
was not to claim that a simple model beats the market. The goal was to demonstrate
sound engineering around temporal data, leakage prevention, and probability quality.

## Engineering decisions

I built a batch pipeline around nflverse data and scikit-learn. The feature builder
normalizes the changing team-stat schema, shifts each team’s statistics before rolling
aggregation, and refuses to silently select an unexpected column. The evaluation uses
chronological splits instead of a random train/test split.

The leakage audit checks required columns, duplicates, missing features, invalid
matchups, and adversarial mutations. The full 2016–2025 audit passed, including a
mutation test that preserved all earlier and same-game protected rows.

## What the experiments showed

The baseline was a standardized logistic regression. I compared feature groups and
expanding, 3-, 5-, and 8-game histories with walk-forward evaluation. The selected
configuration uses EPA-only features over the previous eight games because it produced
the best walk-forward log loss and Brier score: 0.645 and 0.227 across 1,615 test rows.

The 5-game EPA-only model had the highest accuracy at 62.9%, but the 8-game model had
better probability quality. That distinction shaped the product decision: choose the
model that communicates uncertainty better, not the one that wins a single accuracy
comparison.

## Deliverable

The weekly workflow now refreshes the previous ten completed seasons plus a target
schedule and writes a CSV, probability chart, and Markdown report. The real 2026 Week 1
run produced 16 predictions and is preserved in
[`artifacts/predictions/`](../artifacts/predictions/).

Run it with:

```powershell
.\.venv\Scripts\python.exe scripts/predict_week.py --season 2026 --week 1
```

## Lessons and next improvements

The strongest lesson was that evaluation design mattered more than adding features:
recent EPA history improved the walk-forward metrics, while the current pace proxy and
rest differential added little. The play-by-play neutral-situation pace experiment did
not improve walk-forward probability quality, so the feature remains available for
research but is excluded from the selected production model. A time-aware calibration
experiment then selected sigmoid calibration by walk-forward probability metrics, while
keeping the confidence-tier thresholds unchanged because the tier evidence was not yet
stable enough. The next improvement is an optional Streamlit presentation layer over
the same pipeline. That local dashboard now provides recruiter-friendly prediction,
evaluation, and trust views while preserving the CLI as the single prediction path.
