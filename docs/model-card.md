# Model card

## Model

The selected model is a sigmoid-calibrated, standardized logistic-regression classifier
using each team’s prior eight-game offensive and defensive EPA differentials. It predicts
the probability of a home-team win and is used by `scripts/predict_week.py`.

The local Streamlit dashboard in `app.py` displays saved weekly artifacts and the
supporting evaluation evidence. Its refresh action delegates to the same CLI rather
than reimplementing the prediction pipeline.

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
| Selected 8-game EPA, walk-forward | 62.0% | 0.645 | 0.227 | 1,615 |
| Selected 8-game EPA, 2024–2025 reference | 63.6% | 0.636 | 0.223 | 544 |
| Existing expanding all-feature baseline, walk-forward | 58.3% | 0.671 | 0.238 | 1,615 |
| Existing expanding all-feature baseline, 2024–2025 reference | 59.6% | 0.663 | 0.235 | 544 |

The calibration comparison selected sigmoid mapping by primary walk-forward metrics:
log loss 0.644850, Brier score 0.226668, and expected calibration error 0.021376. The
improvement over raw probabilities is small, and the fixed 2024–2025 reference does not
replicate it; see [`calibration report`](../artifacts/model/calibration/report.md).

The complete ablation tables are in
[`summary_results.csv`](../artifacts/model/ablations/summary_results.csv) and
[`season_results.csv`](../artifacts/model/ablations/season_results.csv).

## Leakage and quality checks

The 2016–2025 feature table contains 2,547 rows and passed structural checks. An
adversarial mutation of `2021_06_HOU_IND` left all 1,281 current-or-earlier protected
rows unchanged. The audit is recorded in
[`leakage_audit.json`](../artifacts/model/leakage_audit.json).

The repository currently has 22 passing tests covering feature construction, temporal
leakage, ablations, and the weekly prediction path.

## Limitations and risks

- The model is a simple baseline and does not include betting lines, injuries, weather,
  roster changes, or other market information.
- Defensive EPA is derived from opponent offensive EPA in the available team-stat
  schema and should be validated against play-by-play before being treated as a final
  defensive metric.
- The pace proxy is play volume, not neutral-situation tempo.
- A neutral-situation pace feature was tested using play-by-play but was not retained;
  it did not improve walk-forward probability metrics.
- Week-one cold-start games are omitted from the baseline.
- Confidence tiers are not a substitute for calibration analysis or uncertainty
  intervals. Their thresholds remain communication-oriented because tier-level
  reliability evidence is not stable enough to set new cutoffs.
- The ablation comparison does not include confidence intervals or multiple-comparison
  correction; the selected configuration is a transparent engineering choice, not a
  claim of universal optimality.
