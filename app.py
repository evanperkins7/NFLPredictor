"""Local recruiter-facing dashboard for the NFL weekly predictor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from nfl_predictor.dashboard import (
    load_json,
    load_predictions,
    prediction_paths,
    prediction_runs,
    refresh_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "artifacts" / "model"


@st.cache_data(show_spinner=False)
def cached_predictions(season: int, week: int) -> pd.DataFrame:
    return load_predictions(PROJECT_ROOT, season, week)


@st.cache_data(show_spinner=False)
def cached_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def show_predictions(season: int, week: int) -> None:
    st.subheader(f"{season} Week {week} predictions")
    try:
        predictions = cached_predictions(season, week)
    except FileNotFoundError as error:
        st.info(str(error))
        return
    strongest = predictions.loc[predictions["home_win_probability"].sub(0.5).abs().idxmax()]
    average_probability = predictions["home_win_probability"].mean()
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Games", len(predictions))
    metric_two.metric("Average home-win probability", f"{average_probability:.1%}")
    metric_three.metric("Largest edge", f"{strongest['predicted_winner']} · {max(strongest['home_win_probability'], 1 - strongest['home_win_probability']):.1%}")

    display = predictions.copy()
    display["game_date"] = display["game_date"].dt.strftime("%b %d")
    display["home_win_probability"] = display["home_win_probability"].map("{:.1%}".format)
    st.dataframe(
        display[
            ["game_date", "away_team", "home_team", "home_win_probability", "predicted_winner", "confidence_tier"]
        ],
        width="stretch",
        hide_index=True,
    )
    chart = prediction_paths(PROJECT_ROOT, season, week)["chart"]
    if chart.exists():
        st.image(chart, caption="Calibrated home-win probabilities", width="stretch")


def show_evaluation() -> None:
    st.subheader("Model evaluation")
    calibration = cached_csv(str(MODEL_ROOT / "calibration" / "summary_results.csv"))
    walk_forward = calibration[calibration["evaluation"] == "walk_forward"].set_index("method")
    selected = walk_forward.loc["sigmoid"]
    raw = walk_forward.loc["raw"]
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Walk-forward log loss", f"{selected['log_loss']:.3f}", f"{selected['log_loss'] - raw['log_loss']:.5f} vs raw")
    metric_two.metric("Walk-forward Brier score", f"{selected['brier_score']:.3f}", f"{selected['brier_score'] - raw['brier_score']:.5f} vs raw")
    metric_three.metric("Expected calibration error", f"{selected['expected_calibration_error']:.3f}")
    st.image(
        MODEL_ROOT / "calibration" / "reliability.png",
        caption="Walk-forward reliability comparison",
        width="stretch",
    )
    st.caption("Sigmoid calibration is selected by the primary walk-forward metrics; isotonic calibration overfit.")


def show_trust() -> None:
    st.subheader("Evidence and limitations")
    audit = load_json(MODEL_ROOT / "leakage_audit.json")
    audit_one, audit_two, audit_three = st.columns(3)
    audit_one.metric("Audited feature rows", audit["structural"]["rows"])
    audit_two.metric("Duplicate game rows", audit["structural"]["duplicate_game_rows"])
    audit_three.metric("Leakage audit", "Passed" if audit["passed"] else "Failed")
    st.markdown(
        "The model uses prior eight-game EPA history and sigmoid-calibrated probabilities. "
        "It excludes betting lines, injuries, weather, and a week-one cold-start policy. "
        "Confidence tiers are communication aids, not guarantees."
    )
    ablations = cached_csv(str(MODEL_ROOT / "ablations" / "summary_results.csv"))
    selected = ablations[(ablations["evaluation"] == "walk_forward") & (ablations["rolling_window"].astype(str) == "8")]
    st.dataframe(
        selected.sort_values("log_loss")[["feature_group", "accuracy", "log_loss", "brier_score"]],
        width="stretch",
        hide_index=True,
    )
    st.markdown(
        "[Model card](docs/model-card.md) · [Leakage audit](artifacts/model/leakage_audit.json) · "
        "[Calibration report](artifacts/model/calibration/report.md) · [Case study](docs/case-study.md)"
    )


def main() -> None:
    st.set_page_config(page_title="NFL Weekly Predictor", page_icon="🏈", layout="wide")
    st.markdown(
        """<style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
        [data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.8rem; border-radius: 0.55rem;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("NFL Weekly Predictor")
    st.caption("A reproducible, leakage-aware weekly probability demo with time-aware evaluation.")
    runs = prediction_runs(PROJECT_ROOT)
    default_season, default_week = runs[-1] if runs else (2026, 1)
    with st.sidebar:
        st.header("Prediction run")
        season = st.number_input("Season", min_value=1999, max_value=2100, value=default_season, step=1)
        week = st.number_input("Week", min_value=1, max_value=22, value=default_week, step=1)
        st.divider()
        st.caption("Saved artifacts load by default. Refresh downloads nflverse data and may take a moment.")
        if st.button("Refresh predictions", width="stretch"):
            with st.spinner("Running the verified weekly pipeline…"):
                result = refresh_predictions(PROJECT_ROOT, int(season), int(week))
            if result.returncode == 0:
                cached_predictions.clear()
                st.success("Predictions refreshed.")
            else:
                st.error("Refresh failed. Check the pipeline output below.")
                st.code(result.stderr or result.stdout)

    predictions_tab, evaluation_tab, trust_tab = st.tabs(["Predictions", "Evaluation", "Trust"])
    with predictions_tab:
        show_predictions(int(season), int(week))
    with evaluation_tab:
        show_evaluation()
    with trust_tab:
        show_trust()


if __name__ == "__main__":
    main()
