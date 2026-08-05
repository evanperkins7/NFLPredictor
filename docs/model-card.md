# Model card

## Model

The selected model is a standardized logistic-regression classifier using each team’s
prior eight-game offensive and defensive EPA differentials. It predicts the probability
of a home-team win and is used by `scripts/predict_week.py`.

## Intended use

This model is intended for a reproducible engineering demonstration of time-aware
feature construction, evaluation, and uncertainty communication. It is suitable for
exploring weekly probabilities and model behavior. It is not intended as financial,
betting, or guaranteed decision-making advice.

## Training and evaluation data

- Source: nflverse team statistics and schedules through the 2025 season.
- Scope: completed regular-season games.
- Feature history: prior eight games, shifted to exclude the game being predicted.
- Primary evaluation: walk-forward tests beginning with four training seasons.
- Reference evaluation: train on 2016–2023 and test on 2024–2025.

## Results

| Evaluation | Accuracy | Log loss | Brier score | Rows |
| --- | ---: | ---: | ---: | ---: |
| Selected 8-game EPA, walk-forward | 62.1% | 0.645 | 0.227 | 1,615 |
| Selected 8-game EPA, 2024–2025 reference | 63.6% | 0.636 | 0.223 | 544 |
| Existing expanding all-feature baseline, walk-forward | 58.2% | 0.671 | 0.238 | 1,615 |
| Existing expanding all-feature baseline, 2024–2025 reference | 59.6% | 0.663 | 0.236 | 544 |

The complete ablation tables are in
[`summary_results.csv`](../artifacts/model/ablations/summary_results.csv) and
[`season_results.csv`](../artifacts/model/ablations/season_results.csv).

## Leakage and quality checks

The 2016–2025 feature table contains 2,547 rows and passed structural checks. An
adversarial mutation of `2021_06_HOU_IND` left all 1,281 current-or-earlier protected
rows unchanged. The audit is recorded in
[`leakage_audit.json`](../artifacts/model/leakage_audit.json).

The repository currently has 14 passing tests covering feature construction, temporal
leakage, ablations, and the weekly prediction path.

## Limitations and risks

- The model is a simple baseline and does not include betting lines, injuries, weather,
  roster changes, or other market information.
- Defensive EPA is derived from opponent offensive EPA in the available team-stat
  schema and should be validated against play-by-play before being treated as a final
  defensive metric.
- The pace proxy is play volume, not neutral-situation tempo.
- Week-one cold-start games are omitted from the baseline.
- Confidence tiers are not a substitute for calibration analysis or uncertainty
  intervals.
- The ablation comparison does not include confidence intervals or multiple-comparison
  correction; the selected configuration is a transparent engineering choice, not a
  claim of universal optimality.
