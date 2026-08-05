"""Generate human-readable interpretation of model evaluation artifacts."""

from __future__ import annotations

import pandas as pd


FEATURE_LABELS = {
    "offensive_epa_diff": "Offensive EPA differential",
    "defensive_epa_diff": "Defensive EPA differential",
    "pace_diff": "Pace-proxy differential",
    "rest_days_diff": "Rest-days differential",
}

FEATURE_NOTES = {
    "offensive_epa_diff": "Positive values mean the home team has the stronger prior offensive EPA.",
    "defensive_epa_diff": "Positive values mean the home team allowed more opponent EPA; lower is better.",
    "pace_diff": "This is attempts plus carries plus sacks suffered, not neutral-situation pace.",
    "rest_days_diff": "Positive values mean the home team has more rest days before kickoff.",
}


def _coefficient_interpretation(feature: str, coefficient: float) -> str:
    magnitude = abs(coefficient)
    if magnitude < 0.05:
        strength = "small"
    elif magnitude < 0.20:
        strength = "moderate"
    else:
        strength = "strong"
    direction = "increases" if coefficient > 0 else "decreases"
    return f"{strength} relationship; a one-standard-deviation increase {direction} home-win probability."


def build_analysis_markdown(
    metrics: dict[str, object], season_metrics: pd.DataFrame, coefficients: pd.DataFrame
) -> str:
    """Build an interpretation report from model outputs."""
    coefficient_rows = []
    for row in coefficients.itertuples(index=False):
        feature = str(row.feature)
        coefficient = float(row.coefficient)
        coefficient_rows.append(
            "| {label} | `{feature}` | {coefficient:.3f} | {interpretation} |".format(
                label=FEATURE_LABELS.get(feature, feature),
                feature=feature,
                coefficient=coefficient,
                interpretation=_coefficient_interpretation(feature, coefficient),
            )
        )

    season_rows = []
    for row in season_metrics.itertuples(index=False):
        improvement = float(row.accuracy - row.home_win_rate)
        season_rows.append(
            f"| {int(row.season)} | {int(row.rows)} | {row.accuracy:.1%} | "
            f"{row.log_loss:.3f} | {row.brier_score:.3f} | {improvement:+.1%} |"
        )

    feature_notes = "\n".join(
        f"- **{FEATURE_LABELS.get(feature, feature)}:** {FEATURE_NOTES.get(feature, 'See the source data dictionary.')}"
        for feature in coefficients["feature"]
    )
    train_seasons = ", ".join(str(season) for season in metrics["train_seasons"])
    test_seasons = ", ".join(str(season) for season in metrics["test_seasons"])
    return f"""# Model analysis

## Evaluation scope

- Training seasons: {train_seasons}
- Test seasons: {test_seasons}
- Test rows: {int(metrics["test_rows"])}
- Accuracy: {float(metrics["accuracy"]):.1%}
- Log loss: {float(metrics["log_loss"]):.3f}
- Brier score: {float(metrics["brier_score"]):.3f}
- Always-pick-home accuracy: {float(metrics["always_home_accuracy"]):.1%}

## Performance by season

| Season | Games | Accuracy | Log loss | Brier score | Accuracy vs. home baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(season_rows)}

The model beat the always-pick-home baseline overall by
**{float(metrics["accuracy"] - metrics["always_home_accuracy"]):.1%}**.

## Coefficients

Coefficients come from standardized inputs, so their magnitudes compare the relative
effect of a one-standard-deviation feature change. They describe association in this
model, not causal impact.

| Feature | Column | Coefficient | Interpretation |
| --- | --- | ---: | --- |
{chr(10).join(coefficient_rows)}

### Feature definitions

{feature_notes}

## Current interpretation

Offensive EPA is the dominant signal in this baseline. The defensive EPA coefficient
has the expected negative sign because the feature represents opponent EPA allowed,
where lower values are better. Pace contributes very little, which is a useful result:
the current pace proxy may be too crude to add predictive value. Performance declined
from 2024 to 2025, so the next improvement should be tested across seasons rather than
optimized against one holdout.

## Limitations

- Defensive EPA is derived from the opponent's offensive EPA in the same completed game,
  then shifted so it is only available to later predictions.
- The pace field is a volume proxy, not neutral-situation tempo.
- Week-one games are omitted because the baseline has no prior-season cold-start policy.
- This evaluation does not include betting lines or market comparisons.
"""
