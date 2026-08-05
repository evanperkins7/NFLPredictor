# Data dictionary

## Game identity and outcomes

| Column | Meaning | Availability |
| --- | --- | --- |
| `game_id` | nflverse game identifier | Before kickoff |
| `season` | NFL season label | Before kickoff |
| `week` | Regular-season week number | Before kickoff |
| `game_date` | Scheduled game date | Before kickoff |
| `home_team` | Home team abbreviation | Before kickoff |
| `away_team` | Away team abbreviation | Before kickoff |
| `home_score` / `away_score` | Final scores | After game; labels only |
| `home_win` | Whether the home team won | After game; label only |
| `home_margin` | Home score minus away score | After game; analysis only |

## Model features

All model features are home-minus-away differentials. Team statistics are shifted by
one game and aggregated over the selected prior-history window before the prediction
row is created.

| Feature | Definition | Interpretation |
| --- | --- | --- |
| `offensive_epa_diff` | Prior rolling offensive EPA differential | Positive favors the home team |
| `defensive_epa_diff` | Prior rolling opponent-offense EPA differential | Lower is better for a team; positive favors the away team |
| `pace_diff` | Prior rolling play-volume proxy differential | Positive means more prior volume for the home team |
| `rest_days_diff` | Days since each team’s prior scheduled game | Positive means more home-team rest |
| `neutral_pace_diff` | Prior rolling count of qualifying neutral offensive plays | Positive means more neutral pace for the home team |

The selected production configuration uses `offensive_epa_diff` and
`defensive_epa_diff` over the previous eight games. Neutral pace counts `run`, `pass`,
and `sack` plays in quarters 1–3 when the score margin is within eight points, excluding
penalties, deleted plays, aborted plays, kneels, and spikes. It was evaluated but not
retained because it did not improve walk-forward probability metrics.

## Outputs

| Column | Meaning |
| --- | --- |
| `home_win_probability` | Sigmoid-calibrated estimated probability that the home team wins |
| `predicted_winner` | Team selected at the 0.5 probability threshold |
| `confidence_tier` | `low`, `medium`, or `high` based on distance from 0.5 |

Confidence tiers use these probability bands: low through 0.55, medium from above
0.55 through 0.70, and high above 0.70. They are communication aids, not statistical
guarantees. The thresholds were retained after calibration evaluation because tier-level
reliability was not stable enough to support evidence-based replacements.

## Missing data and cold starts

The baseline drops rows without complete prior features. Week-one games are therefore
omitted until a prior-season cold-start policy is explicitly added. Upcoming games are
included only when both teams have enough historical information to produce the
selected features.
