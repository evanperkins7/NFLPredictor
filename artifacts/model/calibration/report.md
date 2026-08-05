# Probability calibration experiment

## Protocol

- Base model: the selected logistic regression with prior eight-game EPA differentials.
- Primary evaluation: walk-forward testing from 2020–2025, with at least four earlier
  training seasons.
- Calibration boundary: sigmoid and isotonic mappings reserve the latest prior season
  from each outer training window. The outer test season is never used to fit either
  the model or its calibration mapping.
- Reference evaluation: train on 2016–2023 and test on 2024–2025.

## Walk-forward results

| Method | Accuracy | Log loss | Brier score | ECE |
| --- | ---: | ---: | ---: | ---: |
| Sigmoid | **63.9%** | **0.644850** | **0.226668** | **0.021376** |
| Raw logistic | 62.0% | 0.644863 | 0.226912 | 0.022544 |
| Isotonic | 62.3% | 0.744864 | 0.231587 | 0.032453 |

Sigmoid calibration has the best primary walk-forward log loss, Brier score, and
expected calibration error, although the log-loss improvement over raw logistic
probabilities is small. Isotonic calibration substantially degrades log loss and is
not selected.

The 2024–2025 reference holdout favors raw logistic probabilities on log loss, Brier
score, and ECE. Because walk-forward evaluation is the project’s primary model-selection
rule, the weekly workflow uses sigmoid calibration while retaining this disagreement as
a limitation.

## Confidence tiers

The existing low/medium/high thresholds remain unchanged. The pooled reliability data
shows reasonable aggregate calibration but does not establish stable enough tier-level
reliability to justify replacing the current communication-oriented cutoffs.

See `summary_results.csv`, `season_results.csv`, `reliability.csv`, and
`reliability.png` in this directory for the complete evidence.
