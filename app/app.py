"""Dive - interface.

Dive is the analysis and prediction program for the PORAF study (NCSEM,
Loughborough). PORAF names the study and its cohort; Dive names this program.

Six screens. Every one of them shows the plain-language answer before the
statistics, and none of them shows a number the underlying model has declined
to stand behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import theme
from dive.config import SETTINGS, ensure_directories
from dive.data_pipeline import build_dataset, load_dataset
from dive.decline_model import fit_decline_model
from dive.group_comparison import run_group_comparison
from dive.advisor import Advisor, brief_from_profile
from dive.metrics import HealthMetric, MetricRegistry
from dive.recommend import profile_from_participant, recommend_for_profile
from dive.trajectory_model import fit_trajectory_model

st.set_page_config(
    page_title="Dive",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.stylesheet(), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached work
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load(path: str | None):
    ensure_directories()
    return build_dataset(path) if path else load_dataset()


@st.cache_resource(show_spinner=False)
def _decline(path: str | None):
    data = _load(path)
    return fit_decline_model(data.substudy, data.wide)


@st.cache_resource(show_spinner=False)
def _trajectory(path: str | None):
    data = _load(path)
    return fit_trajectory_model(data.trajectory_frame())


def active_path() -> str | None:
    return st.session_state.get("data_path")


def data_or_stop():
    try:
        return _load(active_path())
    except FileNotFoundError:
        st.error(
            "No dataset is available yet. Open the Data screen and either generate the "
            "synthetic workbook or upload a file."
        )
        st.stop()


# --------------------------------------------------------------------------
# Shared fragments
# --------------------------------------------------------------------------

def masthead(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="masthead">'
        f'<div class="eyebrow">Dive &middot; PORAF &middot; IL-6 trans-signalling</div>'
        f"<h1>{title}</h1>"
        f'<div class="sub">{subtitle}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def prediction_block(prediction, decimals: int = 1) -> None:
    """Render a prediction, or the refusal and its reasons."""
    if prediction.abstained:
        reasons = "".join(f"<li>{r}</li>" for r in prediction.reasons)
        st.markdown(
            f'<div class="abstain"><div class="t">No prediction issued</div>'
            f"<ul>{reasons}</ul></div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="hero"><div class="v">{prediction.format_value(decimals)}</div>'
        f'<div class="band">95% prediction interval '
        f"<b>{prediction.format_interval(decimals)}</b></div></div>",
        unsafe_allow_html=True,
    )
    for caveat in prediction.caveats:
        st.markdown(f'<div class="caveat">{caveat}</div>', unsafe_allow_html=True)


def case_control_notice() -> None:
    st.markdown(
        '<div class="caveat">The B/T substudy is a <b>case-control sample</b>: 50 grip '
        "decliners and 50 stable participants were selected deliberately. It is not a "
        "random sample of the cohort, so proportions within it do not describe how common "
        "decline is in the population.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Screen 1 - Data
# --------------------------------------------------------------------------

def screen_data() -> None:
    masthead(
        "Data",
        "Load a workbook, run the cleaning pipeline, and see exactly what was read, "
        "what was corrected and what is missing.",
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(theme.section("Source"), unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a PORAF workbook", type=["xlsx", "xlsm", "xls"], label_visibility="collapsed"
        )
        if uploaded is not None:
            target = SETTINGS.paths.upload_dir / uploaded.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded.getbuffer())
            st.session_state["data_path"] = str(target)
            st.cache_resource.clear()
            st.success(f"Loaded {uploaded.name}")

        columns = st.columns(2)
        if columns[0].button("Generate synthetic data", width="stretch"):
            from dive.generate_synthetic_data import generate_workbook

            with st.spinner("Generating..."):
                path, _ = generate_workbook()
            st.session_state["data_path"] = str(path)
            st.cache_resource.clear()
            st.success("Synthetic workbook generated.")
        if columns[1].button("Reset to configured source", width="stretch"):
            st.session_state.pop("data_path", None)
            st.cache_resource.clear()
            st.rerun()

    with right:
        st.markdown(theme.section("Configuration"), unsafe_allow_html=True)
        is_real = SETTINGS.data_source == "real"
        tag = "real" if is_real else ""
        st.markdown(
            f'<span class="source-tag {tag}">DATA_SOURCE = {SETTINGS.data_source}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="note">Change this in <code>settings.local.yaml</code> or by setting '
            f"<code>PORAF_DATA_SOURCE=real</code>. Nothing else needs editing.</div>",
            unsafe_allow_html=True,
        )

    data = data_or_stop()
    report = data.report

    if SETTINGS.data_source != "real":
        st.markdown(
            theme.verdict_block(
                "This is <b>synthetic data</b>. Every result below is a test of whether the "
                "pipeline works, not a finding about IL-6 trans-signalling.",
                tone="caution",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(theme.section("What was loaded"), unsafe_allow_html=True)
    st.markdown(
        theme.stat_row(
            [
                ("Participants", f"{report.n_participants}", ""),
                ("Visit records", f"{report.n_participant_visits}", ""),
                ("B/T samples", f"{report.n_substudy}", ""),
                ("Longitudinal", f"{data.trajectory_frame()['participant_id'].nunique()}", ""),
            ]
        ),
        unsafe_allow_html=True,
    )

    cohort = data.wide
    if "age_group" in cohort.columns:
        counts = cohort["age_group"].value_counts()
        st.markdown(
            f'<div class="note">{counts.get("Older", 0)} older and {counts.get("Younger", 0)} '
            "younger participants. Faller subgroups are classified for the older cohort only; "
            "younger participants were not followed for falls and are left unclassified rather "
            "than defaulted into the Non-Faller group.</div>",
            unsafe_allow_html=True,
        )

    tab_clean, tab_missing, tab_preview = st.tabs(
        ["Cleaning actions", "Missing data", "Cleaned records"]
    )

    with tab_clean:
        st.markdown(
            '<div class="note">Each line is a correction the pipeline applied automatically. '
            "These are the documented quirks of the PORAF workbooks.</div>",
            unsafe_allow_html=True,
        )
        for flag in report.flags:
            st.markdown(f'<div class="caveat">{flag}</div>', unsafe_allow_html=True)

    with tab_missing:
        missing = report.to_frame()
        if missing.empty:
            st.info("No missingness recorded.")
        else:
            figure = go.Figure(
                go.Bar(
                    x=missing["Missing %"],
                    y=missing["Variable"],
                    orientation="h",
                    marker=dict(color=theme.ACCENT, line=dict(width=0)),
                    hovertemplate="%{y}<br>%{x:.1f}% missing<extra></extra>",
                )
            )
            layout = theme.plotly_layout(
                xaxis_title="Missing (%)",
                height=26 * len(missing) + 90,
                showlegend=False,
            )
            layout["yaxis"]["autorange"] = "reversed"
            layout["yaxis"]["showgrid"] = False
            layout["xaxis"]["showgrid"] = True
            layout["xaxis"]["gridcolor"] = theme.RULE
            figure.update_layout(**layout)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
            st.dataframe(missing, width="stretch", hide_index=True)

    with tab_preview:
        show = [
            c for c in (
                "participant_id", "age_years", "sex_label", "age_group", "subgroup",
                "grip_Baseline", "grip_FM", "grip_change_kg", "declined",
                "fried_score", "fried_category", "BT_ratio",
            ) if c in data.wide.columns
        ]
        st.dataframe(data.wide[show].head(200), width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Screen 2 - Group comparison
# --------------------------------------------------------------------------

def screen_group_comparison() -> None:
    masthead(
        "Group comparison",
        "Does the B/T ratio differ between the four faller subgroups? "
        "Kruskal-Wallis, then the same question adjusted for sex.",
    )
    data = data_or_stop()

    if not data.has_bt:
        st.warning("No B/T ratios are available in this dataset.")
        return

    outcome = st.selectbox(
        "Measure", ["BT_ratio", "FME", "IL6_pg_mL", "sIL6R_ng_mL", "sgp130_ng_mL"],
        format_func=lambda v: {
            "BT_ratio": "B/T ratio", "FME": "Free mediator equivalent (FME)",
            "IL6_pg_mL": "IL-6 (pg/mL)", "sIL6R_ng_mL": "sIL-6R (ng/mL)",
            "sgp130_ng_mL": "sgp130 (ng/mL)",
        }[v],
    )

    with st.spinner("Running tests..."):
        result = run_group_comparison(data.substudy, outcome=outcome)

    tone = "null" if not result.is_significant else "neutral"
    st.markdown(theme.verdict_block(result.verdict, tone=tone), unsafe_allow_html=True)
    case_control_notice()

    frame = data.substudy[[outcome, "subgroup"]].dropna()
    present = [g for g in theme.SUBGROUP_ORDER if g in set(frame["subgroup"])]

    figure = go.Figure()
    for group in present:
        values = frame.loc[frame["subgroup"] == group, outcome]
        figure.add_trace(
            go.Box(
                y=values,
                name=f"{group} (n={len(values)})",
                marker=dict(color=theme.SUBGROUP_COLORS[group], size=5, opacity=0.65),
                line=dict(color=theme.SUBGROUP_COLORS[group], width=1.6),
                fillcolor="rgba(0,0,0,0)",
                boxpoints="all", jitter=0.45, pointpos=0,
                hovertemplate="%{y:.4f}<extra></extra>",
            )
        )
    figure.update_layout(
        **theme.plotly_layout(
            yaxis_title=outcome.replace("_", " "),
            height=430,
            showlegend=False,
        )
    )
    figure.update_xaxes(tickangle=0)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    st.markdown(theme.section("Tests"), unsafe_allow_html=True)
    st.dataframe(result.summary_table(), width="stretch", hide_index=True)

    st.markdown(theme.section("By subgroup"), unsafe_allow_html=True)
    st.dataframe(result.descriptives, width="stretch", hide_index=True)

    with st.expander("Technical detail and assumptions"):
        st.markdown(f'<div class="note tech">{result.technical_note}</div>', unsafe_allow_html=True)
        if result.ancova is not None and not result.ancova.empty:
            st.dataframe(result.ancova.round(4), width="stretch")
        for warning in result.warnings:
            st.markdown(f'<div class="caveat">{warning}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Screen 3 - Individual prediction
# --------------------------------------------------------------------------

def screen_individual() -> None:
    masthead(
        "Individual prediction",
        "Enter a participant's characteristics to get a decline risk and a 12-month "
        "function forecast. If the models cannot support a prediction, they say so "
        "instead of producing a number.",
    )
    data = data_or_stop()

    with st.form("individual"):
        row1 = st.columns(4)
        age = row1[0].number_input("Age (years)", 50.0, 100.0, 75.0, 0.5)
        sex_label = row1[1].selectbox("Sex", ["Female", "Male"])
        grip = row1[2].number_input("Baseline grip (kg)", 4.0, 70.0, 24.0, 0.5)
        bt = row1[3].number_input("B/T ratio", 0.05, 2.0, 0.33, 0.01, format="%.3f")

        row2 = st.columns(4)
        horizon = row2[0].slider("Forecast horizon (months)", 3, 24, 12, 1)
        n_prior = row2[1].selectbox("Prior measurements available", [0, 2, 3], index=0,
                                    help="Supplying a participant's own history narrows the interval.")
        submitted = st.form_submit_button("Predict", width="content")

    if not submitted:
        return

    sex = 1.0 if sex_label == "Female" else 0.0
    profile = {
        "age_years": age, "sex": sex, "BT_ratio": bt,
        "log_BT_ratio": float(np.log(bt)) if bt > 0 else np.nan,
    }

    history = None
    if n_prior:
        months = [0.0, 1.5, 6.0][:n_prior]
        history = pd.DataFrame(
            {"elapsed_months": months, "max_grip_kg": [grip + i * -0.4 for i in range(n_prior)]}
        )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(theme.section("Decline risk"), unsafe_allow_html=True)
        with st.spinner("Fitting..."):
            model = _decline(active_path())
        prediction = model.predict_risk(profile)
        prediction_block(prediction, decimals=1)
        st.markdown(
            f'<div class="note">{model.result_.verdict}</div>', unsafe_allow_html=True
        )

    with right:
        st.markdown(theme.section(f"Grip at {horizon} months"), unsafe_allow_html=True)
        with st.spinner("Fitting..."):
            trajectory = _trajectory(active_path())
        forecast = trajectory.forecast(profile, months=float(horizon), observed=history)
        prediction_block(forecast, decimals=1)

    with st.expander("How these numbers were produced"):
        st.markdown(
            f'<div class="note tech">{model.result_.technical_note}</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="note tech">{trajectory.result_.technical_note}</div>',
            unsafe_allow_html=True,
        )
        for warning in model.result_.warnings + trajectory.result_.warnings:
            st.markdown(f'<div class="caveat">{warning}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Screen 4 - Trajectory
# --------------------------------------------------------------------------

def screen_trajectory() -> None:
    masthead(
        "Trajectory",
        "Observed measurements and the projected path ahead, with the band inside which "
        "this individual's future value is expected to fall.",
    )
    data = data_or_stop()
    long = data.trajectory_frame()

    if long.empty:
        st.warning("No participant has enough repeated measurements to plot a trajectory.")
        return

    counts = long.groupby("participant_id").size().sort_values(ascending=False)
    options = list(counts.index)

    left, right = st.columns([2, 1], gap="large")
    participant = left.selectbox(
        "Participant", options,
        format_func=lambda p: f"{p} - {counts[p]} visits",
    )
    horizon = right.slider("Project to (months)", 12, 24, 12, 1)

    history = long[long["participant_id"] == participant].sort_values("elapsed_months")
    with st.spinner("Fitting..."):
        model = _trajectory(active_path())

    profile = {
        "age_years": float(history["age_years"].iloc[0]),
        "sex": float(history["sex"].iloc[0]),
    }
    if not np.isfinite(profile["age_years"]) or not np.isfinite(profile["sex"]):
        st.warning("This participant is missing the age or sex needed to project a trajectory.")
        return

    observed = history[["elapsed_months", "max_grip_kg"]].dropna()
    months = np.arange(0, horizon + 0.5, 0.5)
    projection = model.predict_trajectory(profile, months=months, observed=observed)

    subgroup = history["subgroup"].iloc[0] if "subgroup" in history.columns else None
    colour = theme.SUBGROUP_COLORS.get(subgroup, theme.ACCENT)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(projection["months"]) + list(projection["months"])[::-1],
            y=list(projection["pi_upper"]) + list(projection["pi_lower"])[::-1],
            fill="toself", fillcolor="rgba(0,114,178,0.10)",
            line=dict(width=0), hoverinfo="skip",
            name="95% prediction interval",
        )
    )
    # Split the model line at the last real measurement. Everything to the left
    # is the model's fit to data already in hand; everything to the right is an
    # actual forecast. Drawing them as one line would present the two as
    # equally well supported, which they are not.
    last_observed = float(observed["elapsed_months"].max()) if len(observed) else 0.0
    fitted = projection[projection["months"] <= last_observed]
    ahead = projection[projection["months"] >= last_observed]

    if len(fitted) > 1:
        figure.add_trace(
            go.Scatter(
                x=fitted["months"], y=fitted["predicted"],
                mode="lines", line=dict(color=theme.ACCENT, width=1.6),
                name="Model fit",
                hovertemplate="Month %{x:.1f}<br>Model fit %{y:.1f} kg<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=ahead["months"], y=ahead["predicted"],
            mode="lines", line=dict(color=theme.ACCENT, width=2, dash="dot"),
            name="Projection",
            hovertemplate="Month %{x:.1f}<br>Projected %{y:.1f} kg<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=observed["elapsed_months"], y=observed["max_grip_kg"],
            mode="lines+markers",
            line=dict(color=colour, width=2),
            marker=dict(size=9, color=theme.PAPER, line=dict(color=colour, width=2)),
            name="Measured",
            hovertemplate="Month %{x:.1f}<br>Measured %{y:.1f} kg<extra></extra>",
        )
    )
    if len(observed):
        figure.add_vline(
            x=last_observed,
            line=dict(color=theme.RULE_STRONG, width=1, dash="dash"),
            annotation_text="last measured",
            annotation_position="bottom right",
            annotation_font=dict(size=10.5, color=theme.INK_MUTED),
        )

    figure.update_layout(
        **theme.plotly_layout(
            xaxis_title="Months since baseline",
            yaxis_title="Maximum grip strength (kg)",
            height=460,
        )
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    final = projection.iloc[-1]
    st.markdown(
        theme.verdict_block(
            f"At {horizon} months this participant's grip strength is projected to be "
            f"<b>{final['predicted']:.1f} kg</b>, and is expected to fall between "
            f"<b>{final['pi_lower']:.1f}</b> and <b>{final['pi_upper']:.1f} kg</b>. "
            f"They were last measured at {observed['max_grip_kg'].iloc[-1]:.1f} kg."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="caveat">The shaded band is a <b>prediction interval</b> for this '
        "individual, not a confidence interval for the group average. It is wider on "
        "purpose: it includes the variation between people as well as measurement "
        "error.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Measurements used"):
        st.dataframe(
            history[[c for c in ("visit", "elapsed_months", "max_grip_kg", "gait_speed_ms",
                                 "sppb_total") if c in history.columns]],
            width="stretch", hide_index=True,
        )


# --------------------------------------------------------------------------
# Screen 5 - Recommendations
# --------------------------------------------------------------------------

def screen_recommendations() -> None:
    masthead(
        "Recommendations",
        "Interventions whose published evidence applies to this participant's measured "
        "values. Nothing here is generated: each entry is triggered by a specific "
        "measurement and carries its evidence grade and source.",
    )
    data = data_or_stop()

    mode = st.radio(
        "Profile", ["Select a participant", "Enter values manually"],
        horizontal=True, label_visibility="collapsed",
    )

    if mode == "Select a participant":
        wide = data.wide
        options = wide["participant_id"].tolist()
        participant = st.selectbox("Participant", options)
        row = wide[wide["participant_id"] == participant].iloc[0]
        substudy_row = None
        if not data.substudy.empty:
            matches = data.substudy[data.substudy["participant_id"] == participant]
            if len(matches):
                substudy_row = matches.iloc[0]
        profile = profile_from_participant(row, substudy_row)
    else:
        columns = st.columns(4)
        profile = {
            "sex": 1.0 if columns[0].selectbox("Sex", ["Female", "Male"]) == "Female" else 0.0,
            "age_years": columns[1].number_input("Age", 50.0, 100.0, 76.0, 1.0),
            "max_grip_kg": columns[2].number_input("Grip (kg)", 4.0, 70.0, 18.0, 0.5),
            "gait_speed_ms": columns[3].number_input("Gait speed (m/s)", 0.1, 2.0, 0.85, 0.01),
        }
        columns = st.columns(4)
        profile["sppb_total"] = columns[0].number_input("SPPB total", 0, 12, 9)
        profile["chair_time_s"] = columns[1].number_input("Chair stand (s)", 3.0, 60.0, 13.0, 0.5)
        profile["bmi"] = columns[2].number_input("BMI", 14.0, 50.0, 27.0, 0.1)
        profile["fried_score"] = columns[3].number_input("Fried score", 0, 5, 1)

    result = recommend_for_profile(profile)
    st.markdown(theme.verdict_block(result.headline()), unsafe_allow_html=True)
    st.markdown(
        '<div class="caveat">These are <b>evidence-based options for someone with this '
        "profile</b>. They are not an output of the prediction models and they do not "
        "forecast what will happen to this person.</div>",
        unsafe_allow_html=True,
    )

    for domain, label in (
        ("physical_function", "Physical function"),
        ("body_composition", "Body composition"),
        ("trans_signalling", "IL-6 trans-signalling"),
    ):
        items = result.by_domain(domain)
        if not items:
            continue
        st.markdown(theme.section(label), unsafe_allow_html=True)
        for item in items:
            triggers = "<br>".join(item.triggered_by)
            caution = (
                f'<div class="caution"><b>Caution.</b> {item.caution}</div>' if item.caution else ""
            )
            st.markdown(
                f'<div class="rec">'
                f'<div class="title"><span class="grade grade-{item.grade}">'
                f'{item.grade}</span>{item.title}</div>'
                f'<div class="body">{item.detail}</div>'
                f'<div class="body"><b>Expected effect.</b> {item.effect}</div>'
                f'<div class="trigger">Triggered by &mdash; {triggers}</div>'
                f"{caution}"
                f'<div class="cite">{item.citation}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    if result.evidence_gaps:
        st.markdown(theme.section("Where the evidence does not support a recommendation"),
                    unsafe_allow_html=True)
        for gap in result.evidence_gaps:
            st.markdown(
                f'<div class="rec">'
                f'<div class="title"><span class="grade grade-D">D</span>{gap.title}</div>'
                f'<div class="body">{gap.effect}</div>'
                f'<div class="cite">{gap.citation}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    _evidence_advisor_section(profile, result)

    if result.unassessed:
        with st.expander(f"Not assessed ({len(result.unassessed)})"):
            st.markdown(
                '<div class="note">These checks could not run because the measurement was '
                "missing. A missing measurement is not treated as a normal result.</div>",
                unsafe_allow_html=True,
            )
            for item in result.unassessed:
                st.markdown(f'<div class="caveat">{item}</div>', unsafe_allow_html=True)


def _evidence_advisor_section(profile: dict, curated_result) -> None:
    """Literature-grounded, participant-specific advice via the Claude API."""
    st.markdown(theme.section("Evidence advisor"), unsafe_allow_html=True)
    st.markdown(
        '<div class="note">The library above matches fixed thresholds and returns the same '
        "wording for anyone who crosses them. The advisor retrieves published trials and "
        "reviews for this participant's specific findings, then reasons about what they mean "
        "for <i>this</i> combination of values. It can only cite studies it actually "
        "retrieved, and every identifier is checked against that set before anything is "
        "shown.</div>",
        unsafe_allow_html=True,
    )

    available, why = Advisor.client_available()
    if not available:
        st.markdown(
            f'<div class="abstain"><div class="t">Advisor unavailable</div>'
            f'<ul><li>{why}.</li>'
            "<li>Dive will not write literature-grounded advice without retrieval - "
            "the curated library above still applies.</li></ul></div>",
            unsafe_allow_html=True,
        )
        return

    notes: list[str] = []
    try:
        decline = _decline(active_path())
        notes.append(decline.result_.verdict)
    except Exception:
        pass

    if not st.button("Retrieve evidence and advise"):
        return

    brief = brief_from_profile(profile, curated_result, model_notes=notes)
    with st.spinner("Retrieving studies and reviewing them for this participant..."):
        outcome = Advisor().advise(brief)

    if not outcome.available:
        st.markdown(
            f'<div class="abstain"><div class="t">No advice issued</div>'
            f"<ul><li>{outcome.reason}</li></ul></div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(theme.verdict_block(outcome.headline()), unsafe_allow_html=True)

    if outcome.n_citations_dropped:
        st.markdown(
            f'<div class="caveat"><b>{outcome.n_citations_dropped} citation'
            f"{'s' if outcome.n_citations_dropped != 1 else ''} discarded</b> because they did "
            "not match any retrieved study. This is the fabrication check doing its job; the "
            "recommendations below rest only on verified sources.</div>",
            unsafe_allow_html=True,
        )

    strength_grade = {"strong": "A", "moderate": "B", "limited": "C", "insufficient": "D"}
    for item in outcome.recommendations:
        grade = strength_grade.get(item.evidence_strength, "C")
        sources = "".join(
            f'<div class="cite">{s.short_citation()} &mdash; '
            f'<a href="{s.url}" target="_blank">{s.identifier}</a>. {s.title}.</div>'
            for s in item.studies
        )
        st.markdown(
            f'<div class="rec">'
            f'<div class="title"><span class="grade grade-{grade}">'
            f'{item.evidence_strength.upper()}</span>{item.title}</div>'
            f'<div class="body">{item.rationale}</div>'
            f'<div class="body"><b>For this participant.</b> {item.specific_to_participant}</div>'
            f'<div class="caution"><b>Applicability.</b> {item.applicability_caveat}</div>'
            f"{sources}"
            f"</div>",
            unsafe_allow_html=True,
        )

    if outcome.evidence_gaps:
        st.markdown(theme.section("Gaps the evidence does not cover"), unsafe_allow_html=True)
        for gap in outcome.evidence_gaps:
            st.markdown(f'<div class="caveat">{gap}</div>', unsafe_allow_html=True)

    with st.expander(f"Studies retrieved ({outcome.n_studies_retrieved})"):
        if outcome.literature:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Identifier": s.identifier,
                        "Title": s.title,
                        "Journal": s.journal,
                        "Year": s.year,
                        "Type": ", ".join(s.pub_types[:2]),
                        "Cited by": s.cited_by,
                    }
                    for s in outcome.literature.studies
                ]),
                width="stretch", hide_index=True,
            )
        st.markdown(
            f'<div class="note tech">Model {outcome.model} &middot; '
            f"source {outcome.literature.source_api if outcome.literature else 'n/a'} &middot; "
            f"{outcome.usage.get('input_tokens', '?')} in / "
            f"{outcome.usage.get('output_tokens', '?')} out tokens</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Screen 6 - Learning
# --------------------------------------------------------------------------

def screen_learning() -> None:
    masthead(
        "Training and data sources",
        "Add new data from any source, see what was accepted and what was held back, "
        "and retrain. A new model replaces the current one only if it demonstrably "
        "performs better.",
    )

    from dive.learning import DataSource, ModelRegistry, ingest_file, retrain_and_evaluate
    from dive.learning.ingest import SourceRegistry
    from dive.learning.retrain import detect_drift

    data = data_or_stop()
    registry = SourceRegistry()

    tab_metrics, tab_add, tab_batches, tab_search, tab_retrain, tab_models = st.tabs(
        ["Health metrics", "Add data", "Ingested batches", "Metric value",
         "Retrain", "Model history"]
    )

    with tab_metrics:
        # Named distinctly: `registry` above is the SourceRegistry used by the
        # batches tab, and Streamlit runs every tab body on each rerun.
        metric_registry = MetricRegistry()
        st.markdown(
            '<div class="note">Dive can learn from any health metric, not just the ones it '
            "ships with. Register a measurement here and it becomes ingestable immediately, "
            "with its plausible range enforced. Registering does not put it into a model - "
            "the <b>Metric value</b> tab decides that, on held-out participants.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(metric_registry.to_frame(), width="stretch", hide_index=True)

        with st.form("register_metric"):
            st.markdown("**Register a new metric**")
            row = st.columns(3)
            metric_name = row[0].text_input("Name", placeholder="serum_vitamin_d")
            metric_label = row[1].text_input("Label", placeholder="Serum 25-OH vitamin D")
            metric_unit = row[2].text_input("Unit", placeholder="nmol/L")
            row = st.columns(4)
            metric_domain = row[0].selectbox(
                "Domain",
                ["physical_function", "body_composition", "inflammation",
                 "cognition", "activity", "clinical", "other"],
            )
            metric_min = row[1].number_input("Plausible minimum", value=0.0, step=1.0)
            metric_max = row[2].number_input("Plausible maximum", value=100.0, step=1.0)
            metric_dir = row[3].selectbox("Higher is", ["better", "worse", "neither"])
            metric_aliases = st.text_input(
                "Other spellings (comma separated)", placeholder="vit_d, vitamin_d_25oh",
                help="Column names in incoming files that mean this metric.",
            )
            if st.form_submit_button("Register"):
                try:
                    metric = HealthMetric(
                        name=metric_name,
                        label=metric_label,
                        unit=metric_unit,
                        domain=metric_domain,
                        minimum=metric_min,
                        maximum=metric_max,
                        higher_is_better={"better": True, "worse": False}.get(metric_dir),
                        aliases=tuple(
                            a.strip() for a in metric_aliases.split(",") if a.strip()
                        ),
                    )
                    metric_registry.register(metric)
                    st.success(
                        f"Registered '{metric.name}'. Values outside "
                        f"{metric.range_text()} will be quarantined on ingestion."
                    )
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with tab_search:
        st.markdown(
            '<div class="note">Tests whether any available metric improves prediction on '
            "participants the model has not seen. The catch this guards against: trying many "
            "metrics and keeping the best will find an apparent improvement even when every "
            "one of them is noise. So the outcome labels are shuffled and the <i>whole search</i> "
            "is re-run, which answers the question that matters - how large an improvement "
            "would this search have found if nothing carried signal?</div>",
            unsafe_allow_html=True,
        )
        permutations = st.slider(
            "Permutations of the search", 40, 400, 120, 20,
            help="More permutations give a more precise p-value and take longer.",
        )
        if st.button("Test metric value"):
            from dive.learning.feature_search import search_features

            with st.spinner("Re-running the search on shuffled outcomes..."):
                found = search_features(
                    data.substudy, n_permutations=permutations,
                )
            st.markdown(
                theme.verdict_block(
                    found.verdict, tone="neutral" if found.improved else "null"
                ),
                unsafe_allow_html=True,
            )
            if not found.to_frame().empty:
                st.dataframe(found.to_frame(), width="stretch", hide_index=True)
            st.markdown(
                f'<div class="note tech">Baseline predictors: '
                f"{', '.join(found.base_features)}. Adopted: "
                f"{', '.join(found.adopted_features)}.</div>",
                unsafe_allow_html=True,
            )
            for warning in found.warnings:
                st.markdown(f'<div class="caveat">{warning}</div>', unsafe_allow_html=True)

    with tab_add:
        st.markdown(
            '<div class="note">Column names are matched automatically against the canonical '
            "schema. Rows with implausible values are quarantined with a reason rather than "
            "silently corrected.</div>",
            unsafe_allow_html=True,
        )
        columns = st.columns(3)
        source_id = columns[0].text_input("Source ID", "new_source")
        label = columns[1].text_input("Label", "New data source")
        kind = columns[2].selectbox("Kind", ["cohort", "assay", "wearable", "registry", "external"])
        trust = st.slider("Trust weight", 0.0, 1.0, 1.0, 0.05,
                          help="Weights this source's rows during retraining.")
        uploaded = st.file_uploader("Data file", type=["csv", "tsv", "xlsx", "json", "parquet"])

        if uploaded is not None and st.button("Ingest"):
            target = SETTINGS.paths.upload_dir / uploaded.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded.getbuffer())
            source = DataSource(source_id=source_id, label=label, kind=kind, trust=trust)
            result = ingest_file(target, source)

            st.markdown(theme.verdict_block(result.summary()), unsafe_allow_html=True)
            if result.unmapped_columns:
                st.markdown(
                    f'<div class="caveat">Columns not recognised and therefore ignored: '
                    f"{', '.join(result.unmapped_columns)}</div>",
                    unsafe_allow_html=True,
                )
            if len(result.quarantined):
                st.markdown(theme.section("Quarantined rows"), unsafe_allow_html=True)
                st.dataframe(
                    result.quarantined[["participant_id", "quarantine_reason"]]
                    if "participant_id" in result.quarantined.columns
                    else result.quarantined,
                    width="stretch", hide_index=True,
                )

    with tab_batches:
        batches = registry.list_batches()
        if batches.empty:
            st.info("No external data has been ingested yet.")
        else:
            st.dataframe(batches, width="stretch", hide_index=True)
            combined = registry.load_all()
            if not combined.empty:
                st.markdown(theme.section("Distribution shift"), unsafe_allow_html=True)
                drift = detect_drift(data.substudy, combined)
                if drift.empty:
                    st.info("Not enough overlapping variables to test for drift.")
                else:
                    drifted = int(drift["drifted"].sum())
                    st.markdown(
                        theme.verdict_block(
                            f"{drifted} of {len(drift)} shared variables differ meaningfully "
                            "from the reference cohort."
                            + (
                                " Drift is not automatically a problem, but a model trained on "
                                "one distribution and applied to another will be confidently "
                                "wrong."
                                if drifted
                                else " The incoming data looks distributionally comparable."
                            ),
                            tone="caution" if drifted else "null",
                        ),
                        unsafe_allow_html=True,
                    )
                    st.dataframe(drift.round(4), width="stretch", hide_index=True)

    with tab_retrain:
        st.markdown(
            '<div class="note">Retraining fits a challenger model and compares it with the '
            "current champion on participants neither has seen. The challenger is promoted "
            "only if it beats the champion by a real margin, the margin survives a bootstrap, "
            "and the challenger passes its own permutation test.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Retrain decline model"):
            with st.spinner("Fitting challenger and comparing..."):
                outcome = retrain_and_evaluate(data.substudy, data.wide)
            st.markdown(
                theme.verdict_block(
                    outcome.reason, tone="neutral" if outcome.promoted else "null"
                ),
                unsafe_allow_html=True,
            )
            if not outcome.comparison.empty:
                st.dataframe(outcome.comparison.round(4), width="stretch", hide_index=True)

    with tab_models:
        history = ModelRegistry().history()
        if history.empty:
            st.info("No models have been registered yet.")
        else:
            st.dataframe(history, width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------

SCREENS = {
    "Data": screen_data,
    "Group comparison": screen_group_comparison,
    "Individual prediction": screen_individual,
    "Trajectory": screen_trajectory,
    "Recommendations": screen_recommendations,
    "Training and data": screen_learning,
}


def main() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="font-family:Source Serif 4,Georgia,serif;font-size:1.5rem;'
            'font-weight:600;letter-spacing:-.015em;margin-bottom:.1rem;">Dive</div>'
            '<div style="font-size:.70rem;letter-spacing:.13em;text-transform:uppercase;'
            'color:#86837A;margin-bottom:1.6rem;">PORAF &middot; B/T ratio</div>',
            unsafe_allow_html=True,
        )
        choice = st.radio("Screen", list(SCREENS), label_visibility="collapsed")
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        is_real = SETTINGS.data_source == "real"
        st.markdown(
            f'<div style="font-size:.74rem;color:#86837A;line-height:1.6;">'
            f'<b style="color:{"#B3261E" if is_real else "#55534C"}">'
            f'{"REAL DATA" if is_real else "SYNTHETIC DATA"}</b><br>'
            f'{"Results describe real participants." if is_real else "Results test the pipeline, not the biology."}'
            f"</div>",
            unsafe_allow_html=True,
        )

    SCREENS[choice]()


main()
