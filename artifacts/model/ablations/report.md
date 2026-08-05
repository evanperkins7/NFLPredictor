# Step 4: Feature and rolling-window ablations

## Protocol

- Source table: regular-season games from 2016–2025.
- Primary evaluation: walk-forward tests beginning with four seasons of training.
  Each test season is evaluated using only earlier seasons.
- Reference evaluation: the existing fixed holdout, training on 2016–2023 and testing
  on 2024–2025.
- Rolling windows: expanding history, plus the previous 3, 5, or 8 games.
- Models: standardized logistic regression for every experiment.
- Primary selection metrics: log loss and Brier score, with accuracy reported as a
  secondary classification metric.

The complete machine-readable outputs are `season_results.csv` and
`summary_results.csv` in this directory.

## Best walk-forward results

| Rolling window | Feature group | Accuracy | Log loss | Brier score |
| --- | --- | ---: | ---: | ---: |
| 8 | EPA | 62.0% | **0.645** | **0.227** |
| 8 | EPA + neutral pace | 61.7% | 0.645 | 0.227 |
| 8 | EPA + pace | 62.1% | 0.645 | 0.227 |
| 5 | EPA | **63.0%** | 0.650 | 0.229 |
| expanding | all features | 58.3% | 0.671 | 0.238 |

The 8-game EPA-only model is the recommended configuration for the next milestone
because it has the best walk-forward log loss and Brier score. The 5-game EPA-only
model has the best accuracy, so it should remain a documented alternative if the
product later prioritizes winner classification over probability quality. Adding
neutral pace did not improve the selected model's probability quality.

## Findings

1. EPA is the useful signal in this baseline. EPA-only models outperform the
   rest-only and pace-only reference models across the primary evaluation.
2. A finite recent-history window is better than an expanding history for this data.
   The 8-game EPA model improves walk-forward log loss from 0.671 to 0.645 and Brier
   score from 0.238 to 0.227 compared with the current all-feature baseline.
3. Adding neutral pace, the current pace proxy, or rest differential does not improve
   the selected 8-game EPA model's probabilistic metrics.
4. The fixed 2024–2025 holdout gives stronger results than the walk-forward aggregate,
   which is expected because it evaluates a different time slice. It remains a
   reference, not the ablation selection rule.

## Limitations

- The pace feature is still a volume proxy, not neutral-situation tempo.
- The neutral-situation pace feature uses an explicit first-three-quarters,
  one-possession filter and was not retained because it did not improve validation
  metrics.
- The comparison tests several configurations without confidence intervals or a
  correction for multiple comparisons; the selected configuration should be treated
  as a transparent engineering choice, not proof of a universal optimum.
- The baseline still omits week-one games without a documented prior-season cold-start
  policy.
