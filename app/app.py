"""Dive - interface.

Dive is a clinical research platform for predicting physical function decline
and trajectory from longitudinal cohort data.

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
from datetime import datetime, timezone

from app import theme
from dive.config import SETTINGS, ensure_directories
from dive.data_pipeline import build_dataset, load_dataset
from dive.decline_model import fit_decline_model
from dive.group_comparison import run_group_comparison
from dive.accounts import MIN_PASSWORD_LENGTH, AccountStore, bootstrap_admin
from dive.advisor import Advisor, brief_from_profile
from dive.owner import OwnerConsole, OwnerGate, SourceEditor
from dive.privacy import ALLOWED_CLINICAL_KEYS, EgressGuard, PrivacyPolicy, deidentify_profile
from dive.calibration import CalibrationStore, PhantomReading, ScannerCalibration, precision_from_repeats
from dive.fairness import PROTECTED_ATTRIBUTES, audit_subgroups, reference_population_note, screen_features
from dive.providers import available_providers
from dive.security import (
    ENCRYPTED_SUFFIX,
    AuditLog,
    DataVault,
    UploadValidator,
    ValidationError,
)
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
        f'<div class="eyebrow">Dive &middot; {SETTINGS.deployment_label}</div>'
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
            "Upload a cohort workbook", type=["xlsx", "xlsm", "xls"], label_visibility="collapsed"
        )
        if uploaded is not None:
            # Validate before anything opens the file: size, extension, magic
            # bytes and filename are all checked, and a failure is reported
            # rather than worked around.
            try:
                payload = uploaded.getbuffer().tobytes()
                safe_name = UploadValidator().validate_bytes(payload, uploaded.name)
            except ValidationError as exc:
                st.error(f"Upload rejected: {exc}")
            else:
                # Encrypted at rest when DIVE_ENCRYPTION_KEY is set; written
                # in the clear, and said so below, when it is not.
                written = DataVault().write_file(
                    payload, SETTINGS.paths.upload_dir / safe_name
                )
                st.session_state["data_path"] = str(written)
                st.cache_resource.clear()
                if written.name.endswith(ENCRYPTED_SUFFIX):
                    st.success(f"Loaded {safe_name} - stored encrypted.")
                else:
                    st.success(f"Loaded {safe_name}")
                    st.markdown(
                        '<div class="caveat">Stored unencrypted: DIVE_ENCRYPTION_KEY '
                        "is not set on this deployment.</div>",
                        unsafe_allow_html=True,
                    )

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
            f"<code>DIVE_DATA_SOURCE=real</code>. Nothing else needs editing.</div>",
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
            "These are recurring quirks of hand-maintained clinical spreadsheets.</div>",
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

    (tab_metrics, tab_add, tab_batches, tab_calib, tab_search,
     tab_retrain, tab_auto, tab_models) = st.tabs(
        ["Health metrics", "Add data", "Ingested batches", "Calibration",
         "Metric value", "Retrain", "Autonomous", "Model history"]
    )

    with tab_calib:
        store = CalibrationStore()
        st.markdown(
            '<div class="note">Imaging values are only comparable if the scanner that '
            "produced them is. Register each device, record phantom QC scans, and Dive "
            "flags drift and refuses to pool measurements that are not comparable.</div>",
            unsafe_allow_html=True,
        )
        status = store.status_frame()
        if status.empty:
            st.info("No imaging devices registered yet.")
        else:
            st.dataframe(status, width="stretch", hide_index=True)
            for report in store.drift_reports():
                st.markdown(
                    theme.verdict_block(
                        report.verdict, tone="stop" if report.drifted else "null"
                    ),
                    unsafe_allow_html=True,
                )

        with st.form("register_device"):
            st.markdown("**Register a device**")
            row = st.columns(4)
            device_id = row[0].text_input("Device ID", placeholder="XCT-3000-A")
            modality = row[1].selectbox("Modality", ["pQCT", "DXA", "CT", "MRI"])
            device_site = row[2].text_input("Site", placeholder="Main clinic")
            voxel = row[3].number_input("Voxel size (mm)", 0.0, 5.0, 0.4, 0.05)
            if st.form_submit_button("Register device") and device_id:
                store.register_device(ScannerCalibration(
                    device_id=device_id, modality=modality, site=device_site,
                    voxel_size_mm=voxel or None,
                ))
                st.success(f"Registered {device_id}.")
                st.rerun()

        devices = list(store.devices())
        if devices:
            with st.form("phantom"):
                st.markdown("**Record a phantom QC scan**")
                row = st.columns(3)
                which = row[0].selectbox("Device", devices)
                measured = row[1].number_input("Measured value", 0.0, 10000.0, 100.0, 0.1)
                nominal = row[2].number_input("Nominal value", 0.1, 10000.0, 100.0, 0.1)
                if st.form_submit_button("Record"):
                    store.record_phantom(PhantomReading(which, measured, nominal))
                    st.success("Recorded.")
                    st.rerun()

        st.markdown(theme.section("Least significant change"), unsafe_allow_html=True)
        st.markdown(
            '<div class="note">From repeat scans, the smallest difference that is a real '
            "change rather than machine noise (2.77 x RMS-SD, the ISCD formula). A "
            "follow-up difference smaller than this has not changed.</div>",
            unsafe_allow_html=True,
        )
        row = st.columns(3)
        precision_sd = row[0].number_input("Precision SD", 0.0, 1000.0, 3.7, 0.1)
        mean_value = row[1].number_input("Mean value", 0.1, 10000.0, 220.0, 1.0)
        from dive.calibration import least_significant_change as _lsc

        row[2].markdown(
            theme.stat_row([
                ("CV", f"{precision_sd / mean_value * 100:.2f}", "%"),
                ("LSC", f"{_lsc(precision_sd):.2f}", ""),
            ]),
            unsafe_allow_html=True,
        )

    with tab_auto:
        from dive.learning import AutoLearner

        learner = AutoLearner()
        st.markdown(
            '<div class="note">Dive can run the whole improvement cycle unattended: ingest '
            "what has arrived, check the new data resembles the old, test whether any metric "
            "now earns a place, retrain, and promote only what is demonstrably better. Every "
            "stage can decline to act, and declining is the default. <b>A cycle that changes "
            "nothing is a successful cycle</b> - the failure mode this guards against is a "
            "system that always finds something.</div>",
            unsafe_allow_html=True,
        )
        due, why = learner.due()
        st.markdown(
            f'<div class="caveat">{"Ready to run." if due else why}</div>',
            unsafe_allow_html=True,
        )

        columns = st.columns(2)
        run_dry = columns[0].button("Dry run", width="stretch")
        run_live = columns[1].button("Run cycle now", width="stretch")

        if run_dry or run_live:
            with st.spinner("Running the improvement cycle..."):
                outcome = learner.run_cycle(dry_run=run_dry, force=True)
            st.markdown(
                theme.verdict_block(
                    outcome.summary(),
                    tone="neutral" if outcome.changed else ("caution" if outcome.halted_at else "null"),
                ),
                unsafe_allow_html=True,
            )
            st.dataframe(outcome.to_frame(), width="stretch", hide_index=True)

        history = learner.history()
        if not history.empty:
            st.markdown(theme.section("Recent cycles"), unsafe_allow_html=True)
            st.dataframe(history, width="stretch", hide_index=True)

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
            safe_name = UploadValidator().validate_bytes(
                uploaded.getbuffer().tobytes(), uploaded.name
            )
            target = DataVault().write_file(
                uploaded.getbuffer().tobytes(), SETTINGS.paths.upload_dir / safe_name
            )
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
# Screen 7 - Privacy and security
# --------------------------------------------------------------------------

def screen_privacy() -> None:
    masthead(
        "Privacy and security",
        "What leaves this process, where it goes, and what is recorded about it.",
    )

    policy = SETTINGS.privacy
    guard = EgressGuard(policy, audit_path=SETTINGS.paths.data_dir / "audit.jsonl")

    tab_egress, tab_providers, tab_fair, tab_audit, tab_posture = st.tabs(
        ["What leaves", "Model providers", "Fairness", "Audit log", "Security posture"]
    )

    with tab_fair:
        st.markdown(
            theme.verdict_block(
                "A model can be accurate overall and much worse for one group. The "
                "aggregate number hides that, so Dive measures performance <b>within</b> "
                "each group."
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="note">Race and ethnicity are self-reported social categories, '
            "not biological variables. Dive refuses them as model features by default "
            "and uses them for this audit instead. Clinical algorithms that treated them "
            "as biology - the eGFR race coefficient, spirometry race correction, the VBAC "
            "calculator - all had to be withdrawn or rebuilt. "
            "See <code>docs/OMICS.md</code>.</div>",
            unsafe_allow_html=True,
        )

        data = data_or_stop()
        attributes = [
            c for c in data.wide.columns
            if c in PROTECTED_ATTRIBUTES or c in {"sex_label", "age_group", "subgroup"}
        ]
        if not attributes:
            st.info("No grouping variable is available in this dataset to audit against.")
        else:
            attribute = st.selectbox("Audit across", attributes)
            if st.button("Run fairness audit"):
                with st.spinner("Scoring each subgroup..."):
                    model = _decline(active_path())
                    frame = data.substudy.copy()
                    if "log_BT_ratio" not in frame.columns:
                        frame["log_BT_ratio"] = np.log(frame["BT_ratio"].where(frame["BT_ratio"] > 0))
                    features = model.guard_.feature_names_
                    usable = frame.dropna(subset=list(features) + ["declined"])
                    if attribute not in usable.columns:
                        usable = usable.merge(
                            data.wide[["participant_id", attribute]],
                            on="participant_id", how="left",
                        )
                    usable = usable.dropna(subset=[attribute])
                    if usable.empty:
                        st.warning("No participants have both a prediction and this attribute.")
                    else:
                        usable["predicted_risk"] = model.pipeline_.predict_proba(
                            usable[list(features)].to_numpy(float)
                        )[:, 1]
                        audit = audit_subgroups(usable, attribute)
                        st.markdown(
                            theme.verdict_block(
                                audit.verdict,
                                tone="stop" if audit.concerning else "null",
                            ),
                            unsafe_allow_html=True,
                        )
                        st.dataframe(audit.to_frame(), width="stretch", hide_index=True)
                        for warning in audit.warnings:
                            st.markdown(
                                f'<div class="caveat">{warning}</div>', unsafe_allow_html=True
                            )

        st.markdown(theme.section("Which variables may be model features"), unsafe_allow_html=True)
        from dive.metrics import MetricRegistry as _MR

        allowed, refused = screen_features(_MR().candidate_features())
        st.markdown(
            f'<div class="note">{len(allowed)} eligible; {len(refused)} refused.</div>',
            unsafe_allow_html=True,
        )
        for name, why in refused:
            st.markdown(
                f'<div class="caveat"><b>{name}</b> &mdash; {why}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(theme.section("Reference populations"), unsafe_allow_html=True)
        measure = st.selectbox(
            "Derived score", ["bmd_t_score", "egfr_ml_min", "prs_ebmd", "dnam_age_horvath"]
        )
        st.markdown(
            f'<div class="caveat">{reference_population_note(measure)}</div>',
            unsafe_allow_html=True,
        )

    # ---------------- what leaves ----------------
    with tab_egress:
        st.markdown(
            theme.verdict_block(
                "If Dive sends text to a model provider, that provider processes it. No "
                "setting changes that. The only enforceable control is <b>what is sent</b>, "
                "so that is what this page shows."
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            theme.stat_row([
                ("Egress mode", policy.egress_mode, ""),
                ("Min cell size", str(policy.min_aggregate_n), ""),
                ("Age banding", f"{policy.age_band_width}", "yr"),
                ("Offline only", "yes" if policy.offline_only else "no", ""),
            ]),
            unsafe_allow_html=True,
        )

        st.markdown(theme.section("See it on a real record"), unsafe_allow_html=True)
        st.markdown(
            '<div class="note">Pick a participant and see exactly what would be sent and '
            "what would be withheld. This is the same function the advisor calls.</div>",
            unsafe_allow_html=True,
        )

        data = data_or_stop()
        participant = st.selectbox("Participant", data.wide["participant_id"].tolist())
        row = data.wide[data.wide["participant_id"] == participant].iloc[0]
        raw = {k: v for k, v in row.to_dict().items() if pd.notna(v)}

        clean, dropped = deidentify_profile(raw)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(theme.section(f"Would be sent ({len(clean)})"), unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(
                    [{"Field": k, "Value": v} for k, v in sorted(clean.items())]
                ),
                width="stretch", hide_index=True,
            )
        with right:
            st.markdown(theme.section(f"Withheld ({len(dropped)})"), unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame({"Field": dropped}), width="stretch", hide_index=True
            )

        st.markdown(
            '<div class="caveat">De-identified is not anonymous. An unusual combination of '
            "measurements, cross-referenced against another dataset, remains a remote "
            "re-identification route. Set <code>egress_mode: none</code>, or point the router "
            "at a local model, if that risk is unacceptable for your deployment.</div>",
            unsafe_allow_html=True,
        )

    # ---------------- providers ----------------
    with tab_providers:
        st.markdown(
            '<div class="note">Every external call passes the privacy guard before a '
            "request is made. A blocked decision means the call never happens.</div>",
            unsafe_allow_html=True,
        )
        described = available_providers()
        st.dataframe(
            pd.DataFrame([
                {
                    "Provider": d["provider"],
                    "Model": d["model"],
                    "Usable": "yes" if d["available"] else "no",
                    "Key": d["key_fingerprint"],
                    "Note": d["reason"],
                }
                for d in described
            ]),
            width="stretch", hide_index=True,
        )
        st.markdown(
            '<div class="caveat"><b>Local models are the zero-egress option.</b> Point '
            "<code>DIVE_LOCAL_LLM_URL</code> at an OpenAI-compatible endpoint on your own "
            "hardware and no participant-derived data leaves your network at all.</div>",
            unsafe_allow_html=True,
        )

        st.markdown(theme.section("Test the guard"), unsafe_allow_html=True)
        probe = st.text_area(
            "Paste any text to see whether it would be allowed out",
            value="Summarise the evidence for participant WL042, NHS 943 476 5919.",
            height=90,
        )
        if st.button("Check"):
            decision = guard.review_text(probe, destination="anthropic")
            st.markdown(
                theme.verdict_block(
                    decision.explain(), tone="neutral" if decision.allowed else "stop"
                ),
                unsafe_allow_html=True,
            )

    # ---------------- audit ----------------
    with tab_audit:
        log = AuditLog(SETTINGS.paths.data_dir / "audit.jsonl")
        intact, problems = log.verify()
        st.markdown(
            theme.verdict_block(
                "The audit chain is intact - no entry has been removed or edited since it "
                "was written."
                if intact else
                "The audit chain is BROKEN. Entries have been removed or altered.",
                tone="null" if intact else "stop",
            ),
            unsafe_allow_html=True,
        )
        for problem in problems[:10]:
            st.markdown(f'<div class="caveat">{problem}</div>', unsafe_allow_html=True)

        if st.session_state.get("auth_role") != "admin":
            st.markdown(
                '<div class="note">The chain check above is shown to everyone, because '
                "whether the record is intact is not a secret. The entries themselves "
                "name who signed in, when, and what was sent where, so they are "
                "visible to administrators only.</div>",
                unsafe_allow_html=True,
            )
        else:
            entries = log.tail(100)
            if not entries:
                st.info("No audit entries yet.")
            else:
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Time": e.get("ts", "")[:19].replace("T", " "),
                            "Actor": e.get("actor", ""),
                            "Event": e.get("event", ""),
                            "Allowed": e.get("allowed", ""),
                            "Bytes": e.get("bytes_out", ""),
                            "Payload hash": str(e.get("payload_sha256", ""))[:12],
                        }
                        for e in reversed(entries)
                    ]),
                    width="stretch", hide_index=True,
                )
            st.markdown(
                '<div class="note">Each entry stores the hash of the one before it, so a '
                "deleted or edited line breaks the chain and is detected here. For stronger "
                "guarantees, ship this file to append-only storage on a separate host.</div>",
                unsafe_allow_html=True,
            )

    # ---------------- posture ----------------
    with tab_posture:
        vault = DataVault()
        controls = [
            ("Encryption at rest (uploads)", vault.available,
             "AES-GCM on uploaded cohort files - see the note below for what it "
             "does not cover" if vault.available
             else "DIVE_ENCRYPTION_KEY is not set - uploads are stored in the clear"),
            ("Tamper-evident audit log", intact, "hash-chained, verified on this page"),
            ("Identifier egress blocking", True, "fail-closed scan on every outbound payload"),
            ("Upload validation", True, "size, extension, magic bytes, path traversal, formula injection"),
            ("Password hashing", True, "scrypt with per-password salt and lockout"),
            ("Signed sessions", True, "HMAC-signed, expiring, constant-time verification"),
        ]
        st.dataframe(
            pd.DataFrame([
                {"Control": name, "Active": "yes" if on else "NO", "Detail": detail}
                for name, on, detail in controls
            ]),
            width="stretch", hide_index=True,
        )

        st.markdown(theme.section("What this does not cover"), unsafe_allow_html=True)
        st.markdown(
            '<div class="note">These controls raise the cost of an attack and make a '
            "successful one visible afterwards. They do not make Dive unbreakable, and no "
            "software is. Before real patient data is loaded, the following sit with the "
            "deployment and are not solvable in application code:</div>",
            unsafe_allow_html=True,
        )
        for item in (
            "TLS termination, certificate management and network isolation.",
            "Operating system and dependency patching.",
            "Encryption at rest covers uploaded cohort files only. Everything Dive "
            "derives from them - trained models, ingested batches, the audit log - is "
            "written in the clear, so the volume itself needs to be on an encrypted "
            "disk.",
            "Key custody, rotation and encrypted backups with tested restores.",
            "An independent penetration test and a threat model.",
            "UK GDPR / DPA 2018 obligations, including a DPIA covering the model providers.",
            "If Dive's output is used to guide care rather than research, medical device "
            "regulation almost certainly applies (UKCA / MHRA, EU MDR). That is a "
            "classification question to settle early, because it shapes the whole product.",
        ):
            st.markdown(f'<div class="caveat">{item}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Screen 8 - Owner console
# --------------------------------------------------------------------------

def screen_owner() -> None:
    masthead(
        "Owner console",
        "What has happened, what has changed, what the model has learned, and the "
        "source it all runs on.",
    )

    gate = OwnerGate()

    if not gate.configured:
        st.markdown(
            theme.verdict_block(
                "No owner key is configured on this deployment.", tone="caution"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="note tech">{gate.setup_instructions()}</div>',
            unsafe_allow_html=True,
        )
        return

    if not st.session_state.get("owner_unlocked"):
        st.markdown(
            '<div class="note">The console is locked. Failed attempts are rate-limited '
            "and audited.</div>",
            unsafe_allow_html=True,
        )
        with st.form("owner_unlock"):
            key = st.text_input("Owner key", type="password")
            if st.form_submit_button("Unlock"):
                try:
                    if gate.verify(key):
                        st.session_state["owner_unlocked"] = True
                        st.rerun()
                    else:
                        st.error("That key was not accepted.")
                except ValidationError as exc:
                    st.error(str(exc))
        return

    console = OwnerConsole()
    st.markdown(theme.verdict_block(console.summary()), unsafe_allow_html=True)

    if st.button("Lock console"):
        st.session_state.pop("owner_unlocked", None)
        st.rerun()

    tab_time, tab_learn, tab_source, tab_state = st.tabs(
        ["What happened", "What it learned", "Source", "State"]
    )

    # ---------------- timeline ----------------
    with tab_time:
        st.markdown(
            '<div class="note">One record across every subsystem - data ingestion, '
            "outbound calls, learning cycles, model promotions and source edits. This is "
            "where \"why does it do that now?\" gets answered.</div>",
            unsafe_allow_html=True,
        )
        timeline = console.timeline()
        if timeline.empty:
            st.info("Nothing recorded yet.")
        else:
            sources = sorted(timeline["Source"].unique())
            chosen = st.multiselect("Show", sources, default=sources)
            filtered = timeline[timeline["Source"].isin(chosen)]
            st.dataframe(filtered, width="stretch", hide_index=True, height=520)

    # ---------------- learning ----------------
    with tab_learn:
        st.markdown(theme.section("Model lineage"), unsafe_allow_html=True)
        lineage = console.model_lineage()
        if lineage.empty:
            st.info("No models have been registered yet. Train one from Training and data.")
        else:
            st.dataframe(lineage, width="stretch", hide_index=True)

            curve = console.learning_curve()
            if len(curve) >= 2 and "auc_cv" in curve.columns:
                st.markdown(theme.section("Performance over time"), unsafe_allow_html=True)
                figure = go.Figure()
                figure.add_hline(
                    y=0.5, line=dict(color=theme.RULE_STRONG, width=1, dash="dash"),
                    annotation_text="chance", annotation_position="bottom right",
                    annotation_font=dict(size=10.5, color=theme.INK_MUTED),
                )
                figure.add_trace(go.Scatter(
                    x=list(range(1, len(curve) + 1)), y=curve["auc_cv"],
                    mode="lines+markers", name="Cross-validated AUC",
                    line=dict(color=theme.ACCENT, width=2),
                    marker=dict(size=9, color=theme.PAPER,
                                line=dict(color=theme.ACCENT, width=2)),
                    hovertemplate="Version %{x}<br>AUC %{y:.3f}<extra></extra>",
                ))
                if "auc_holdout" in curve.columns:
                    figure.add_trace(go.Scatter(
                        x=list(range(1, len(curve) + 1)), y=curve["auc_holdout"],
                        mode="lines+markers", name="Held-out AUC",
                        line=dict(color=theme.SUBGROUP_COLORS["Maintained Function"], width=2, dash="dot"),
                        marker=dict(size=8),
                        hovertemplate="Version %{x}<br>Held-out %{y:.3f}<extra></extra>",
                    ))
                figure.update_layout(**theme.plotly_layout(
                    xaxis_title="Model version (oldest first)",
                    yaxis_title="AUC", height=380,
                ))
                st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
                st.markdown(
                    '<div class="caveat">A flat line is the expected result when new data '
                    "carries no new signal. Movement matters only when it clears the "
                    "promotion gate, which is what the lineage table records.</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(theme.section("Metrics that earned a place"), unsafe_allow_html=True)
        adopted = console.adopted_metrics()
        if adopted.empty:
            st.markdown(
                '<div class="note">No metric has yet been adopted into a model. That is '
                "the normal state - a metric is adopted only when it beats a permutation "
                "null built from the whole search.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(adopted, width="stretch", hide_index=True)

        st.markdown(theme.section("Learning cycles"), unsafe_allow_html=True)
        cycles = console.cycle_history()
        if cycles.empty:
            st.info("No autonomous cycles have run.")
        else:
            st.dataframe(cycles, width="stretch", hide_index=True)

    # ---------------- source ----------------
    with tab_source:
        editor = SourceEditor()
        if not editor.enabled:
            st.markdown(
                f'<div class="abstain"><div class="t">Source editing is disabled</div>'
                "<ul><li>Editing source through a web interface is remote code execution "
                "by design. A path allowlist narrows the blast radius; it does not remove "
                "it.</li>"
                "<li>To enable it, set <code>allow_source_editing: true</code> in "
                "<code>settings.yaml</code>.</li>"
                "<li>Leave it off unless Dive is on localhost or a trusted network. "
                "Editing the repository directly is the safer route.</li></ul></div>",
                unsafe_allow_html=True,
            )

        files = editor.list_files()
        if files.empty:
            st.info("No editable files found.")
        else:
            # Default to a substantive file: opening on an empty __init__.py
            # makes the editor look broken.
            options = files["Path"].tolist()
            default = next(
                (i for i, path in enumerate(options) if path == "dive/config.py"),
                int(files["Lines"].astype(int).idxmax()) if len(files) else 0,
            )
            chosen = st.selectbox("File", options, index=min(default, len(options) - 1))
            try:
                content = editor.read(chosen)
            except ValidationError as exc:
                st.error(str(exc))
                content = ""

            edited = st.text_area(
                "Contents", value=content, height=460, label_visibility="collapsed"
            )
            note = st.text_input("What are you changing, and why?", placeholder="e.g. widen the plausible range for CTX")

            columns = st.columns(2)
            if columns[0].button("Save", width="stretch", disabled=not editor.enabled):
                result = editor.write(chosen, edited, note=note)
                if result.written:
                    st.markdown(
                        theme.verdict_block(
                            f"{result.reason} +{result.lines_added} / "
                            f"-{result.lines_removed} lines."
                        ),
                        unsafe_allow_html=True,
                    )
                    if result.diff:
                        with st.expander("Diff"):
                            st.code(result.diff, language="diff")
                else:
                    st.error(result.reason)

            if columns[1].button("Check without saving", width="stretch"):
                valid, problem = SourceEditor.validate(chosen, edited)
                if valid:
                    st.success("Parses cleanly.")
                else:
                    st.error(problem)

        st.markdown(theme.section("Backups"), unsafe_allow_html=True)
        backups = editor.backups()
        if backups.empty:
            st.info("No backups yet. One is taken automatically before every save.")
        else:
            st.dataframe(backups, width="stretch", hide_index=True)
            restore_name = st.selectbox("Restore", backups["Backup"].tolist())
            if st.button("Restore this version", disabled=not editor.enabled):
                result = editor.restore(restore_name)
                st.success(result.reason) if result.written else st.error(result.reason)

    # ---------------- state ----------------
    with tab_state:
        state = console.integrity()
        st.dataframe(
            pd.DataFrame([
                {"Check": "Audit chain intact", "Value": "yes" if state["audit_intact"] else "NO"},
                {"Check": "Audited events", "Value": state["n_entries"]},
                {"Check": "Owner key configured", "Value": "yes" if state["owner_key_configured"] else "no"},
                {"Check": "Source editing", "Value": "ENABLED" if state["source_editing_enabled"] else "disabled"},
                {"Check": "Egress mode", "Value": SETTINGS.privacy.egress_mode},
                {"Check": "Data source", "Value": SETTINGS.data_source},
                {"Check": "Protected features allowed", "Value": "YES" if SETTINGS.allow_protected_features else "no"},
            ]),
            width="stretch", hide_index=True,
        )
        for problem in state["problems"][:10]:
            st.markdown(f'<div class="caveat">{problem}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------

def _accounts() -> AccountStore:
    return AccountStore()


def screen_login() -> None:
    """The only screen an unauthenticated visitor can reach."""
    store = _accounts()

    st.markdown(
        '<div class="masthead">'
        '<div class="eyebrow">Dive</div>'
        "<h1>Sign in</h1>"
        '<div class="sub">This deployment holds clinical data. Access is '
        "logged.</div></div>",
        unsafe_allow_html=True,
    )

    if store.empty:
        created, message = bootstrap_admin(store)
        if created:
            st.markdown(
                theme.verdict_block(
                    "First run - an administrator account has been created.",
                    tone="caution",
                ),
                unsafe_allow_html=True,
            )
            st.code(message)
            st.markdown(
                '<div class="caveat">This is shown once and is not recoverable. '
                "Copy it somewhere safe now, then sign in and change it.</div>",
                unsafe_allow_html=True,
            )
        else:
            # No accounts and none created: usually DIVE_ADMIN_PASSWORD is too
            # short. Saying so beats a sign-in form that nothing can sign in to.
            st.error(f"No accounts exist and one could not be created. {message}")

    with st.form("signin"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            try:
                token = store.authenticate(username, password)
            except ValidationError as exc:
                st.error(str(exc))
            else:
                st.session_state["auth_token"] = token
                st.session_state["auth_user"] = store._normalise(username)
                st.rerun()

    if store.session_secret_is_ephemeral:
        st.markdown(
            '<div class="caveat"><b>DIVE_SESSION_SECRET is not set.</b> Sessions will '
            "be invalidated whenever the server restarts, and will not work at all "
            "across more than one instance. Set it before putting this in front of "
            "real users.</div>",
            unsafe_allow_html=True,
        )


def screen_change_password(username: str) -> None:
    store = _accounts()
    st.markdown(
        '<div class="masthead"><div class="eyebrow">Dive</div>'
        "<h1>Change your password</h1>"
        '<div class="sub">This account still has the password it was created '
        "with. Choose a new one before continuing.</div></div>",
        unsafe_allow_html=True,
    )

    with st.form("change_password"):
        first = st.text_input("New password", type="password")
        second = st.text_input("Confirm new password", type="password")
        if st.form_submit_button("Set password"):
            if first != second:
                st.error("The two passwords do not match.")
            else:
                try:
                    store.set_password(username, first)
                except ValidationError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pop("auth_token", None)
                    st.success("Password changed. Please sign in again.")
                    st.rerun()

    st.markdown(
        f'<div class="note">At least {MIN_PASSWORD_LENGTH} characters. Longer '
        "matters more than complicated - a passphrase of several words beats a "
        "short string of symbols.</div>",
        unsafe_allow_html=True,
    )


def screen_users() -> None:
    """Account administration. Requires manage_users."""
    masthead("Users", "Who can sign in, and what they may do.")

    store = _accounts()
    st.dataframe(store.to_frame(), width="stretch", hide_index=True)

    st.markdown(
        '<div class="note"><b>viewer</b> sees aggregate results only. '
        "<b>analyst</b> additionally sees individual records, runs models and "
        "ingests data. <b>admin</b> additionally manages users, promotes models "
        "and changes the privacy policy.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(theme.section("Add a user"), unsafe_allow_html=True)
    with st.form("add_user"):
        columns = st.columns(3)
        new_username = columns[0].text_input("Username")
        new_password = columns[1].text_input("Temporary password", type="password")
        new_role = columns[2].selectbox("Role", ["viewer", "analyst", "admin"])
        if st.form_submit_button("Create"):
            try:
                store.create(new_username, new_password, new_role, must_change_password=True)
            except ValidationError as exc:
                st.error(str(exc))
            else:
                st.success(f"Created '{new_username}'. They must change the password at first sign-in.")
                st.rerun()

    existing = store.to_frame()
    if not existing.empty:
        st.markdown(theme.section("Change a user"), unsafe_allow_html=True)
        columns = st.columns(3)
        target = columns[0].selectbox("User", existing["User"].tolist())
        changed_role = columns[1].selectbox("Set role to", ["viewer", "analyst", "admin"])
        action = columns[2].selectbox("Or", ["-", "disable", "enable", "delete"])

        if st.button("Apply"):
            try:
                if action == "-":
                    store.set_role(target, changed_role)
                    st.success(f"'{target}' is now {changed_role}.")
                elif action == "delete":
                    store.delete(target)
                    st.success(f"Deleted '{target}'.")
                else:
                    store.set_disabled(target, action == "disable")
                    st.success(f"'{target}' {action}d.")
            except ValidationError as exc:
                st.error(str(exc))
            else:
                st.rerun()


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------

#: The capability a user must hold to see each screen. A user without it does
#: not get a disabled menu entry - the screen is absent, because an entry you
#: cannot use is just an invitation to try.
SCREEN_CAPABILITIES: dict[str, str] = {
    "Data": "view_aggregates",
    "Group comparison": "view_aggregates",
    "Individual prediction": "view_individual",
    "Trajectory": "view_individual",
    "Recommendations": "view_individual",
    "Training and data": "ingest_data",
    "Users": "manage_users",
    "Privacy and security": "view_results",
    "Owner console": "manage_users",      # plus the owner key, checked inside
}


SCREENS = {
    "Data": screen_data,
    "Group comparison": screen_group_comparison,
    "Individual prediction": screen_individual,
    "Trajectory": screen_trajectory,
    "Recommendations": screen_recommendations,
    "Training and data": screen_learning,
    "Users": screen_users,
    "Privacy and security": screen_privacy,
    "Owner console": screen_owner,
}


def main() -> None:
    store = _accounts()

    # ---- authentication gate. Nothing below runs without a valid session. ----
    token = st.session_state.get("auth_token")
    claims = None
    if token:
        try:
            claims = store.verify_token(token)
        except ValidationError as exc:
            st.session_state.pop("auth_token", None)
            st.warning(str(exc))

    if claims is None:
        screen_login()
        return

    st.session_state["auth_role"] = claims["role"]

    account = store._load().get(claims["username"])
    if account is not None and account.must_change_password:
        screen_change_password(claims["username"])
        return

    permitted = [
        name for name, capability in SCREEN_CAPABILITIES.items()
        if name in SCREENS and store.can(token, capability)
    ]
    if not permitted:
        st.error("This account has no permissions assigned. Ask an administrator.")
        return

    with st.sidebar:
        st.markdown(
            '<div style="font-family:Source Serif 4,Georgia,serif;font-size:1.5rem;'
            'font-weight:600;letter-spacing:-.015em;margin-bottom:.1rem;">Dive</div>'
            '<div style="font-size:.70rem;letter-spacing:.13em;text-transform:uppercase;'
            f'color:#86837A;margin-bottom:1.6rem;">{SETTINGS.deployment_label}</div>',
            unsafe_allow_html=True,
        )
        choice = st.radio("Screen", permitted, label_visibility="collapsed")
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

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        expires = datetime.fromtimestamp(claims["expires"], timezone.utc)
        st.markdown(
            f'<div style="font-size:.74rem;color:#86837A;line-height:1.6;">'
            f'Signed in as <b>{claims["username"]}</b> ({claims["role"]})<br>'
            f'Session ends {expires.strftime("%H:%M UTC")}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Sign out", width="stretch"):
            st.session_state.pop("auth_token", None)
            st.session_state.pop("owner_unlocked", None)
            st.rerun()

    SCREENS[choice]()


main()
