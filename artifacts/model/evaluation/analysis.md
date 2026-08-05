# Model analysis

## Evaluation scope

- Training seasons: 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023
- Test seasons: 2024, 2025
- Test rows: 544
- Accuracy: 59.6%
- Log loss: 0.663
- Brier score: 0.236
- Always-pick-home accuracy: 53.5%

## Performance by season

| Season | Games | Accuracy | Log loss | Brier score | Accuracy vs. home baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | 272 | 61.0% | 0.654 | 0.231 | +7.7% |
| 2025 | 272 | 58.1% | 0.672 | 0.240 | +4.4% |

The model beat the always-pick-home baseline overall by
**6.1%**.

## Coefficients

Coefficients come from standardized inputs, so their magnitudes compare the relative
effect of a one-standard-deviation feature change. They describe association in this
model, not causal impact.

| Feature | Column | Coefficient | Interpretation |
| --- | --- | ---: | --- |
| Offensive EPA differential | `offensive_epa_diff` | 0.494 | strong relationship; a one-standard-deviation increase increases home-win probability. |
| Rest-days differential | `rest_days_diff` | 0.068 | moderate relationship; a one-standard-deviation increase increases home-win probability. |
| Pace-proxy differential | `pace_diff` | -0.023 | small relationship; a one-standard-deviation increase decreases home-win probability. |
| Defensive EPA differential | `defensive_epa_diff` | -0.185 | moderate relationship; a one-standard-deviation increase decreases home-win probability. |

### Feature definitions

- **Offensive EPA differential:** Positive values mean the home team has the stronger prior offensive EPA.
- **Rest-days differential:** Positive values mean the home team has more rest days before kickoff.
- **Pace-proxy differential:** This is attempts plus carries plus sacks suffered, not neutral-situation pace.
- **Defensive EPA differential:** Positive values mean the home team allowed more opponent EPA; lower is better.

## Current interpretation

Offensive EPA is the dominant signal in this baseline. The defensive EPA coefficient
has the expected negative sign because the feature represents opponent EPA allowed,
where lower values are better. Pace contributes very little, which is a useful result:
the current pace proxy may be too crude to add predictive value. Performance declined
from 2024 to 2025, so the next improvement should be tested across seasons rather than
optimized against one holdout.

## Limitations

- Defensive EPA is derived from the opponent's offensive EPA in the same game.
- The pace field is a volume proxy, not neutral-situation tempo.
- Week-one games are omitted because the baseline has no prior-season cold-start policy.
- This evaluation does not include betting lines or market comparisons.
