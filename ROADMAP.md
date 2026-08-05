# Build roadmap

## Product thesis

Build a weekly prediction system that is reproducible, honest about uncertainty, and
easy for a recruiter to inspect from raw data through evaluation. The resume story
should emphasize time-aware validation and leakage prevention rather than claiming
that a simple model can beat the market.

## Milestones

1. **Data contract** — verify the exact `nflreadpy` team-stats columns in a live
   download and snapshot a small schema report.
2. **Feature table** — materialize pre-game rows for the ten-season window, add a
   true pace measure from play-by-play if the team-stats pace proxy is insufficient,
   and document missing-data/cold-start rules.
3. **Baseline** — fit logistic regression with chronological holdout; report accuracy,
   log loss, Brier score, calibration, and a naive home-team baseline.
4. **Ablations** — compare feature groups and rolling windows; preserve results in a
   reproducible table rather than tuning only for one season.
5. **Weekly workflow** — add a command that refreshes data, produces upcoming-game
   probabilities, and exports a clean CSV plus a seaborn evaluation chart.
6. **Recruiter packaging** — add an architecture diagram, data dictionary, model card,
   limitations, and a short case study with linked evidence.

## Confirmed direction

- Output win probabilities, predicted winner, confidence tiers, and eventually an
  independently trained expected-margin/spread estimate.
- Keep betting lines out of the baseline. Add them later as a separate market-informed
  comparison so the effect of market information is explainable.
- Build a static weekly report first, then a Streamlit interface over the same pipeline.
  This is the recommended learning sequence: the report forces the data and evaluation
  contracts to be correct before UI work begins, while Streamlit adds a useful final
  demo without introducing a separate backend.

## Current assumptions

- “Last ten seasons” means the ten completed season labels supplied to the CLI; for
  the current offseason this will normally be 2016–2025.
- The first baseline uses regular-season games only and omits week-one games until a
  documented prior-season cold-start policy is added.
- The initial `plays` feature is labeled a pace proxy. A true neutral-situation pace
  feature should be added from play-by-play before making strong claims about tempo.
- Confidence tiers are a communication aid, not a statistical guarantee; calibration
  will determine whether the tiers are trustworthy.
- The inspected 2024 team-stats schema exposes passing and rushing EPA rather than
  combined offense/defense EPA. The baseline derives offense from those fields and
  derives defense from opponent offense; this assumption must be documented and
  validated against play-by-play before treating it as a final metric.
