"""Synthetic cohort workbook generator.

Produces an Excel workbook with a realistic sheet structure, column layout and
set of data quirks, so the whole pipeline can be built and validated before
real data is available.

Two things matter about the defaults:

1. ``bt_decline_log_or`` and ``bt_slope_per_sd`` default to **zero**. On the
   default synthetic data the B/T ratio carries no signal, so the group
   comparison and decline models must return null results. That is the
   pipeline's sanity check, and it mirrors the real prior (an earlier
   RFU-based analysis on this pathway found nothing: RF CV-AUC 0.42,
   permutation p=0.75).

2. Age, sex and baseline-function effects on grip *are* simulated, because
   they are genuinely well established. They give the validation harness a
   known non-zero truth to recover without manufacturing a fake B/T effect.

Every generated workbook is accompanied by a ``ground_truth.json`` sidecar
recording the exact parameters used, which ``validation.py`` reads back to
check that the pipeline recovers what was simulated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings

__all__ = ["GenerationParams", "generate_workbook", "main"]


# --------------------------------------------------------------------------
# Parameters (= ground truth)
# --------------------------------------------------------------------------

@dataclass
class GenerationParams:
    """Every knob the generator uses. Serialised as the ground-truth sidecar."""

    seed: int = 20260916

    n_older: int = 281
    n_younger: int = 81

    n_bt_substudy: int = 100
    n_bt_decliners: int = 50
    n_bt_stable: int = 50

    # --- Effects the pipeline must recover (validation harness) -----------
    #: Log-odds of grip decline per 1 SD of log B/T ratio. ZERO by default.
    bt_decline_log_or: float = 0.0
    #: Change in grip slope (kg/month) per 1 SD of log B/T ratio. ZERO by default.
    bt_slope_per_sd: float = 0.0
    #: Difference in mean log B/T across faller subgroups (SD units). ZERO by default.
    bt_subgroup_shift: float = 0.0

    #: Genuinely established effects - non-zero, and recoverable.
    age_slope_per_decade: float = -0.035      # kg/month of grip, per decade older
    male_grip_offset_kg: float = 12.5         # men are stronger at baseline
    grip_grand_mean_kg: float = 26.0
    grip_between_sd_kg: float = 7.0           # random intercept SD
    grip_slope_mean: float = -0.045           # kg/month population mean slope
    grip_slope_sd: float = 0.070              # random slope SD
    grip_residual_sd_kg: float = 1.9          # within-visit measurement noise

    # --- Analyte distributions (log-normal, ELISA-plausible) -------------
    il6_log_mean: float = 0.80                # -> median ~2.2 pg/mL
    il6_log_sd: float = 0.65
    sil6r_log_mean: float = 3.70              # -> median ~40 ng/mL
    sil6r_log_sd: float = 0.28
    sgp130_log_mean: float = 5.75             # -> median ~315 ng/mL
    sgp130_log_sd: float = 0.22
    #: Women run higher sIL-6R; keeps the sex-differential literature plausible.
    female_sil6r_log_offset: float = 0.10
    #: Proportion of ELISA wells that fail / fall below detection.
    analyte_missing_rate: float = 0.04

    # --- Structure -------------------------------------------------------
    subgroup_proportions: dict[str, float] = field(
        default_factory=lambda: {"NF": 0.57, "MF": 0.16, "RF": 0.01, "DF": 0.04}
    )
    #: Probability an older participant attends each follow-up visit.
    attendance: dict[str, float] = field(
        default_factory=lambda: {"PostFall": 1.0, "SixMonth": 0.72, "TwelveMonth": 0.63}
    )
    missing_rate_general: float = 0.05
    tug_cannot_complete_rate: float = 0.06

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _punch_holes(rng: np.random.Generator, values: np.ndarray, rate: float) -> np.ndarray:
    """Randomly set a fraction of values to NaN (item non-response)."""
    out = np.asarray(values, dtype=float).copy()
    mask = rng.random(out.shape) < rate
    out[mask] = np.nan
    return out


def _grip_trials(rng: np.random.Generator, max_grip: np.ndarray, sentinel: float) -> np.ndarray:
    """Expand a max grip into 6 plausible trials whose maximum is that value.

    Real grip data is recorded as 6 trials (3 per hand) and the analysis takes
    the max. Missing trials are coded 999 in the source workbook.
    """
    n = len(max_grip)
    trials = np.empty((n, 6))
    for i, peak in enumerate(max_grip):
        if not np.isfinite(peak):
            trials[i, :] = sentinel
            continue
        # Trials sit at or below the max, with the peak placed at random.
        draws = peak - np.abs(rng.normal(0, 1.6, size=6))
        draws[rng.integers(0, 6)] = peak
        draws = np.clip(draws, 2.0, None)
        # Occasional un-recorded trial
        holes = rng.random(6) < 0.03
        draws[holes] = sentinel
        if np.all(holes):
            draws[0] = peak
        trials[i, :] = np.round(draws, 1)
    return trials


def _sppb_from_performance(
    rng: np.random.Generator,
    gait_speed: np.ndarray,
    chair_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive SPPB subscores (0-4 each) from the underlying performance measures.

    Scored to the standard SPPB cutpoints so the subscores are internally
    consistent with gait speed and chair-stand time rather than free-floating.
    """
    gait_score = np.select(
        [gait_speed >= 0.83, gait_speed >= 0.65, gait_speed >= 0.49, gait_speed > 0],
        [4, 3, 2, 1],
        default=0,
    ).astype(float)
    chair_score = np.select(
        [chair_time <= 11.19, chair_time <= 13.69, chair_time <= 16.69, chair_time <= 60],
        [4, 3, 2, 1],
        default=0,
    ).astype(float)
    # Balance correlates with the other two but carries its own variance.
    latent = (gait_score + chair_score) / 2 + rng.normal(0, 0.8, size=len(gait_speed))
    balance_score = np.clip(np.round(latent), 0, 4)

    gait_score[~np.isfinite(gait_speed)] = np.nan
    chair_score[~np.isfinite(chair_time)] = np.nan
    balance_score[~np.isfinite(latent)] = np.nan

    total = gait_score + chair_score + balance_score
    return balance_score, chair_score, gait_score, total


def _encode_mixed_binary(rng: np.random.Generator, values: np.ndarray) -> np.ndarray:
    """Return a column that mixes 0/1 and Yes/No encodings, as the real file does."""
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        if not np.isfinite(v):
            out[i] = np.nan
        elif rng.random() < 0.35:
            out[i] = "Yes" if v == 1 else "No"
        else:
            out[i] = int(v)
    return out


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def generate_workbook(
    params: GenerationParams | None = None,
    output_path: Path | str | None = None,
    settings: Settings | None = None,
) -> tuple[Path, dict]:
    """Generate the synthetic workbook and its ground-truth sidecar.

    Returns ``(workbook_path, ground_truth_dict)``.
    """
    cfg = settings or SETTINGS
    p = params or GenerationParams(seed=cfg.model.random_seed)
    rng = np.random.default_rng(p.seed)

    out_path = Path(output_path) if output_path else cfg.paths.synthetic_workbook
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = p.n_older + p.n_younger
    pids = [f"WL{i:03d}" for i in range(1, n_total + 1)]
    is_older = np.array([True] * p.n_older + [False] * p.n_younger)
    rng.shuffle(is_older)

    # ---------------- demographics ----------------
    age = np.where(
        is_older,
        rng.normal(74.0, 6.5, n_total).clip(cfg.cohort.older_age_threshold, 96),
        rng.normal(43.0, 11.0, n_total).clip(19, cfg.cohort.older_age_threshold - 0.1),
    )
    # 0 = male, 1 = female
    sex = rng.binomial(1, 0.62, n_total)
    female = sex == 1

    baseline_date = date(2023, 1, 9)
    visit_dates = [baseline_date + timedelta(days=int(d)) for d in rng.integers(0, 540, n_total)]
    dob = [vd - timedelta(days=int(a * 365.25)) for vd, a in zip(visit_dates, age)]

    height = np.where(female, rng.normal(1.62, 0.065, n_total), rng.normal(1.755, 0.070, n_total))
    weight = np.where(female, rng.normal(71.0, 14.0, n_total), rng.normal(83.0, 15.0, n_total))
    weight = np.clip(weight, 40, 150)
    bmi = weight / height**2
    waist = 58 + 0.78 * weight + rng.normal(0, 6, n_total)
    calf = 22 + 0.14 * weight + rng.normal(0, 2.2, n_total)

    education = rng.choice([1, 2, 3, 4], n_total, p=[0.34, 0.30, 0.28, 0.08]).astype(float)
    ethnicity = rng.choice([0, 1, 2, 3], n_total, p=[0.88, 0.05, 0.04, 0.03]).astype(float)
    imd = rng.choice(np.arange(1, 11), n_total, p=np.array([0.05,0.05,0.06,0.07,0.09,0.10,0.12,0.15,0.16,0.15])).astype(float)

    # ---------------- analytes (all participants; substudy assayed later) ----
    il6 = np.exp(rng.normal(p.il6_log_mean, p.il6_log_sd, n_total))
    sil6r = np.exp(
        rng.normal(p.sil6r_log_mean, p.sil6r_log_sd, n_total)
        + np.where(female, p.female_sil6r_log_offset, 0.0)
    )
    sgp130 = np.exp(rng.normal(p.sgp130_log_mean, p.sgp130_log_sd, n_total))

    from .biochem import compute_bt_ratio_frame

    bt_frame = compute_bt_ratio_frame(il6, sil6r, sgp130, constants=cfg.biochem)
    log_bt = bt_frame["log_BT_ratio"].to_numpy()
    log_bt_z = (log_bt - np.nanmean(log_bt)) / np.nanstd(log_bt)

    # ---------------- baseline grip and function ----------------
    grip_intercept = (
        p.grip_grand_mean_kg
        + np.where(female, 0.0, p.male_grip_offset_kg)
        - 0.28 * (age - 70.0)
        + rng.normal(0, p.grip_between_sd_kg, n_total)
    )
    grip_intercept = np.clip(grip_intercept, 6.0, 65.0)

    grip_slope = (
        p.grip_slope_mean
        + p.age_slope_per_decade * (age - 70.0) / 10.0
        + p.bt_slope_per_sd * log_bt_z          # ZERO by default
        + rng.normal(0, p.grip_slope_sd, n_total)
    )

    gait_speed_bl = np.clip(1.15 - 0.006 * (age - 70) + rng.normal(0, 0.20, n_total), 0.20, 1.90)
    chair_time_bl = np.clip(10.5 + 0.10 * (age - 70) + rng.normal(0, 3.2, n_total), 5.0, 60.0)
    tug_bl = np.clip(8.0 + 0.11 * (age - 70) + rng.normal(0, 2.8, n_total), 4.5, 39.0)
    moca = np.clip(np.round(27 - 0.07 * (age - 70) + rng.normal(0, 2.4, n_total)), 6, 30)

    # ---------------- faller subgroups (older participants only) ----------
    props = p.subgroup_proportions
    other = max(0.0, 1.0 - sum(props.values()))
    codes = np.array([1, 2, 3, 4, 0])
    probs = np.array([props["NF"], props["MF"], props["RF"], props["DF"], other])
    probs = probs / probs.sum()
    subgroup = np.where(is_older, rng.choice(codes, n_total, p=probs), np.nan).astype(float)
    fall_status = np.where(np.isin(subgroup, [2, 3, 4]), 1.0, np.where(is_older, 0.0, np.nan))

    # ---------------- Fried frailty components ----------------
    frail_risk = 1 / (1 + np.exp(-(-3.1 + 0.055 * (age - 70) - 0.05 * (grip_intercept - 26))))
    fried_weightloss = rng.binomial(1, np.clip(frail_risk * 0.9, 0, 1)).astype(float)
    worn_out = np.clip(np.round(rng.normal(1.6, 1.3, n_total) + 3 * frail_risk), 0, 5)
    tired = np.clip(np.round(rng.normal(1.5, 1.3, n_total) + 3 * frail_risk), 0, 5)

    ipaq_coded = rng.choice([1, 2, 3], n_total, p=[0.27, 0.44, 0.29]).astype(float)
    ipaq_text_map = {1.0: "low", 2.0: "Moderate ", 3.0: "HIGH"}
    # Text column is deliberately messy - the pipeline must prefer the coded one.
    ipaq_text = np.array(
        [ipaq_text_map[c] if rng.random() > 0.12 else "" for c in ipaq_coded], dtype=object
    )

    comorbid_raw = rng.binomial(1, 0.42, n_total).astype(float)
    polypharm_raw = rng.binomial(1, 0.31, n_total).astype(float)

    # ---------------- assemble the Baseline sheet ----------------
    grip_bl = _punch_holes(rng, grip_intercept, p.missing_rate_general * 0.5)
    trials_bl = _grip_trials(rng, grip_bl, cfg.cohort.grip_missing_sentinel)
    gait1 = np.where(np.isfinite(gait_speed_bl), 4.0 / gait_speed_bl, np.nan)
    gait2 = gait1 + np.abs(rng.normal(0, 0.35, n_total))
    tug_recorded = tug_bl.copy()
    cannot = rng.random(n_total) < p.tug_cannot_complete_rate
    tug_recorded[cannot] = cfg.cohort.tug_sentinel_value

    bal_bl, chair_sc_bl, gait_sc_bl, sppb_bl = _sppb_from_performance(rng, gait_speed_bl, chair_time_bl)

    baseline = pd.DataFrame(
        {
            "Participant_ID": pids,
            "Fall_status": fall_status,
            "DOB": dob,
            "Group": np.where(is_older, 0, 1),
            "Date_of_visit": visit_dates,
            "Age_years": np.round(age, 1),
            "Gender": sex.astype(float),
            "Height_m": np.round(height, 3),
            "Weight_kg": np.round(weight, 1),
            "BMI": np.round(bmi, 2),
            "Waist_cm": np.round(waist, 1),
            "Calf_cm": np.round(calf, 1),
        }
    )
    for j in range(6):
        baseline[f"Grip_trial_{j + 1}"] = trials_bl[:, j]
    baseline["Gait_speed1"] = np.round(gait1, 2)
    baseline["Gait_speed2"] = np.round(gait2, 2)
    baseline["Chair_time_s"] = np.round(_punch_holes(rng, chair_time_bl, p.missing_rate_general), 2)
    baseline["TUG_s"] = np.round(tug_recorded, 2)
    baseline["SPPB_Balance"] = bal_bl
    baseline["SPPB_Chair"] = chair_sc_bl
    baseline["SPPB_Gait"] = gait_sc_bl
    baseline["SPPB_Total"] = sppb_bl
    baseline["MoCA"] = moca
    baseline["EDLEV_highest"] = education
    baseline["Ethnic_group"] = ethnicity
    baseline["IMD"] = imd
    baseline["IPAQ_Category"] = ipaq_text
    baseline["IPAQ_Coded_Category"] = ipaq_coded
    baseline["FRIED1_weight_loss"] = fried_weightloss
    baseline["Worn_out"] = worn_out
    baseline["Tired"] = tired
    baseline["Comorbidity_any"] = _encode_mixed_binary(rng, comorbid_raw)
    baseline["Polypharmacy"] = _encode_mixed_binary(rng, polypharm_raw)
    baseline["Faller_Subgroup_code"] = subgroup

    # ---------------- follow-up visits ----------------
    visit_months = {"PostFall": 1.5, "SixMonth": 6.0, "TwelveMonth": 12.0}
    followups: dict[str, pd.DataFrame] = {}
    elapsed_record: dict[str, np.ndarray] = {}
    #: (elapsed_months, grip) for every generated observation, used to record
    #: the REALISED marginal slope. This is the estimand the mixed model
    #: targets, and it is not the same as ``grip_slope_mean``: the post-fall
    #: dip and the DF non-recovery push the population-average trajectory
    #: down on top of the per-participant linear slope. Validating against the
    #: bare parameter would fail a model that is behaving correctly.
    observed_points: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for visit, nominal in visit_months.items():
        # Irregular timing: actual elapsed months jitter around the nominal.
        elapsed = nominal + rng.normal(0, nominal * 0.14, n_total)
        elapsed = np.clip(elapsed, 0.5, nominal * 1.9)
        elapsed_record[visit] = elapsed

        attends = is_older & (rng.random(n_total) < p.attendance[visit])
        if visit == "PostFall":
            attends &= fall_status == 1

        grip_v = grip_intercept + grip_slope * elapsed + rng.normal(0, p.grip_residual_sd_kg, n_total)
        # Post-fall dip for those who fell, proportional to subgroup severity
        if visit == "PostFall":
            dip = np.select(
                [subgroup == 2, subgroup == 3, subgroup == 4],
                [rng.normal(0.4, 0.6, n_total), rng.normal(3.4, 1.0, n_total), rng.normal(4.2, 1.4, n_total)],
                default=0.0,
            )
            grip_v = grip_v - dip
        elif visit in ("SixMonth", "TwelveMonth"):
            # RF recovers back toward baseline; DF does not.
            recov = np.select(
                [subgroup == 3, subgroup == 4],
                [rng.normal(0.0, 0.7, n_total), rng.normal(-2.6, 1.2, n_total)],
                default=0.0,
            )
            grip_v = grip_v + recov

        grip_v = np.where(attends, grip_v, np.nan)
        grip_v = _punch_holes(rng, grip_v, p.missing_rate_general * 0.4)
        grip_v = np.clip(grip_v, 2.0, 70.0)

        gait_v = np.where(attends, gait_speed_bl + rng.normal(-0.01, 0.12, n_total) * (elapsed / 12), np.nan)
        gait_v = np.clip(gait_v, 0.15, 1.95)
        chair_v = np.where(attends, chair_time_bl + rng.normal(0.4, 2.0, n_total) * (elapsed / 12), np.nan)
        chair_v = np.clip(chair_v, 5.0, 60.0)
        tug_v = np.where(attends, tug_bl + rng.normal(0.3, 1.8, n_total) * (elapsed / 12), np.nan)
        tug_v = np.clip(tug_v, 4.0, 39.0)
        cannot_v = attends & (rng.random(n_total) < p.tug_cannot_complete_rate)
        tug_v = np.where(cannot_v, cfg.cohort.tug_sentinel_value, tug_v)

        bal_v, chair_sc_v, gait_sc_v, sppb_v = _sppb_from_performance(rng, gait_v, chair_v)
        trials_v = _grip_trials(rng, grip_v, cfg.cohort.grip_missing_sentinel)

        keep = np.isfinite(grip_v) & np.isfinite(elapsed)
        observed_points.append((elapsed[keep], grip_v[keep], np.asarray(pids)[keep]))

        frame = pd.DataFrame({"Participant_ID": pids})
        frame["Visit_elapsed_months"] = np.where(attends, np.round(elapsed, 2), np.nan)
        frame["Date_of_visit"] = [
            (vd + timedelta(days=float(e) * 30.44)) if a else pd.NaT
            for vd, e, a in zip(visit_dates, elapsed, attends)
        ]
        for j in range(6):
            frame[f"Grip_trial_{j + 1}"] = np.where(attends, trials_v[:, j], np.nan)
        frame["Gait_speed1"] = np.round(np.where(np.isfinite(gait_v), 4.0 / gait_v, np.nan), 2)
        frame["Gait_speed2"] = frame["Gait_speed1"] + np.abs(rng.normal(0, 0.3, n_total))
        frame["Chair_time_s"] = np.round(chair_v, 2)
        frame["TUG_s"] = np.round(tug_v, 2)
        frame["SPPB_Balance"] = bal_v
        frame["SPPB_Chair"] = chair_sc_v
        frame["SPPB_Gait"] = gait_sc_v
        frame["SPPB_Total"] = sppb_v
        frame["Weight_kg"] = np.round(np.where(attends, weight + rng.normal(0, 2.4, n_total), np.nan), 1)

        # --- QUIRK: stored delta columns that must never be trusted --------
        true_delta_sppb = sppb_v - sppb_bl
        corrupt = rng.random(n_total) < 0.22
        # Some rows carry the follow-up value itself instead of the difference,
        # some carry the sign-flipped difference. This is the documented bug.
        stored = np.where(corrupt, sppb_v, true_delta_sppb)
        stored = np.where(corrupt & (rng.random(n_total) < 0.4), -true_delta_sppb, stored)
        frame["Delta_SPPB_Total"] = stored
        bal_corrupt = rng.random(n_total) < 0.22
        frame["Delta_SPPB_Balance"] = np.where(bal_corrupt, bal_v, bal_v - bal_bl)
        # Grip delta happens to be stored correctly - the pipeline recomputes
        # it anyway, and a test asserts the recomputation matches.
        frame["Delta_Max_Grip"] = grip_v - grip_bl

        followups[visit] = frame

    # --- QUIRK: the 12-month sheet has two unrelated ID columns -----------
    twelve = followups["TwelveMonth"]
    stray = [f"LAB{rng.integers(1000, 9999)}" for _ in range(n_total)]
    twelve.insert(0, "Lab_Ref", stray)
    twelve.insert(1, "Batch_ID", [f"B{rng.integers(1, 40):02d}" for _ in range(n_total)])
    # Participant_ID is now at position 2, exactly as in the real workbook.

    # ---------------- FM (final measure) sheet ----------------
    # Final = 12-month preferred, 6-month fallback, per participant.
    twelve_grip = followups["TwelveMonth"][[f"Grip_trial_{j+1}" for j in range(6)]].to_numpy()
    six_grip = followups["SixMonth"][[f"Grip_trial_{j+1}" for j in range(6)]].to_numpy()

    def _max_trial(row: np.ndarray) -> float:
        vals = row[np.isfinite(row) & (row != cfg.cohort.grip_missing_sentinel)]
        return float(vals.max()) if vals.size else np.nan

    twelve_max = np.array([_max_trial(r) for r in twelve_grip])
    six_max = np.array([_max_trial(r) for r in six_grip])
    fm_grip = np.where(np.isfinite(twelve_max), twelve_max, six_max)
    fm_source = np.where(
        np.isfinite(twelve_max), "TwelveMonth", np.where(np.isfinite(six_max), "SixMonth", "")
    )
    fm_elapsed = np.where(
        np.isfinite(twelve_max), elapsed_record["TwelveMonth"], elapsed_record["SixMonth"]
    )
    fm_elapsed = np.where(np.isfinite(fm_grip), fm_elapsed, np.nan)

    bl_max = np.array([_max_trial(r) for r in trials_bl])
    grip_change = fm_grip - bl_max

    fm = pd.DataFrame(
        {
            "Participant_ID": pids,
            "FM_source_visit": fm_source,
            "FM_elapsed_months": np.round(fm_elapsed, 2),
            "Max_Grip_FM": np.round(fm_grip, 1),
            "Max_Grip_Baseline": np.round(bl_max, 1),
            "Grip_change_kg": np.round(grip_change, 2),
        }
    )

    # ---------------- B/T substudy sheet (case-control, n=100) -----------
    eligible = np.isfinite(grip_change) & is_older
    decline_mask = eligible & (grip_change <= -cfg.cohort.grip_decline_threshold_kg)
    stable_mask = eligible & (np.abs(grip_change) <= cfg.cohort.grip_stable_band_kg)

    decline_idx = np.where(decline_mask)[0]
    stable_idx = np.where(stable_mask)[0]
    # "Top" decliners = largest losses; stable sampled at random within the band.
    decline_idx = decline_idx[np.argsort(grip_change[decline_idx])][: p.n_bt_decliners]
    if len(stable_idx) > p.n_bt_stable:
        stable_idx = rng.choice(stable_idx, p.n_bt_stable, replace=False)

    substudy_idx = np.concatenate([decline_idx, stable_idx])
    substudy_case = np.concatenate([np.ones(len(decline_idx)), np.zeros(len(stable_idx))])

    # Apply the (default zero) B/T effect by shifting analytes in cases only.
    il6_sub = il6[substudy_idx].copy()
    sgp130_sub = sgp130[substudy_idx].copy()
    if p.bt_decline_log_or != 0.0 or p.bt_subgroup_shift != 0.0:
        # Higher B/T in cases is produced by lowering free sgp130 buffering.
        shift = (p.bt_decline_log_or + p.bt_subgroup_shift) * 0.5
        sgp130_sub = sgp130_sub * np.exp(-shift * substudy_case * p.sgp130_log_sd)

    sub = pd.DataFrame(
        {
            "Participant_ID": [pids[i] for i in substudy_idx],
            "Sample_group": np.where(substudy_case == 1, "Decliner", "Stable"),
            "Selection_design": "case-control 50/50 (NOT population prevalence)",
            "Age_years": np.round(age[substudy_idx], 1),
            "Gender": sex[substudy_idx].astype(float),
            "Max_Grip_Baseline": np.round(bl_max[substudy_idx], 1),
            "Grip_change_kg": np.round(grip_change[substudy_idx], 2),
            "IL6_pg_mL": np.round(_punch_holes(rng, il6_sub, p.analyte_missing_rate), 3),
            "sIL6R_ng_mL": np.round(_punch_holes(rng, sil6r[substudy_idx], p.analyte_missing_rate), 2),
            "sgp130_ng_mL": np.round(_punch_holes(rng, sgp130_sub, p.analyte_missing_rate), 2),
            "Plate_ID": [f"P{rng.integers(1, 6)}" for _ in range(len(substudy_idx))],
        }
    )
    sub["Faller_Subgroup_code"] = subgroup[substudy_idx]

    # ---------------- write workbook ----------------
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        baseline.to_excel(writer, sheet_name="Baseline", index=False)
        followups["PostFall"].to_excel(writer, sheet_name="PostFall", index=False)
        followups["SixMonth"].to_excel(writer, sheet_name="SixMonth", index=False)
        followups["TwelveMonth"].to_excel(writer, sheet_name="TwelveMonth", index=False)
        fm.to_excel(writer, sheet_name="FM", index=False)
        sub.to_excel(writer, sheet_name="BT_Substudy", index=False)

    # Realised marginal slope: OLS of every generated grip observation on
    # elapsed time, in kg per year. Baseline observations anchor the intercept.
    base_keep = np.isfinite(grip_bl)
    all_t = np.concatenate([np.zeros(int(base_keep.sum()))] + [t for t, _, _ in observed_points])
    all_y = np.concatenate([grip_bl[base_keep]] + [y for _, y, _ in observed_points])
    all_pid = np.concatenate(
        [np.asarray(pids)[base_keep]] + [ids for _, _, ids in observed_points]
    )
    if len(all_t) > 2 and np.ptp(all_t) > 0:
        marginal_slope_per_month = float(np.polyfit(all_t, all_y, 1)[0])
    else:
        marginal_slope_per_month = float("nan")

    # Within-participant slope: the estimand a mixed model with a random
    # intercept actually targets. The pooled OLS slope above is confounded by
    # between-participant differences and by who attends which visit, so it is
    # recorded for reference but is NOT the right validation target.
    within_slope_per_month = float("nan")
    if len(all_t) > 2:
        panel = pd.DataFrame({"pid": all_pid, "t": all_t, "y": all_y})
        grouped = panel.groupby("pid")
        panel["t_dm"] = panel["t"] - grouped["t"].transform("mean")
        panel["y_dm"] = panel["y"] - grouped["y"].transform("mean")
        denominator = float((panel["t_dm"] ** 2).sum())
        if denominator > 0:
            within_slope_per_month = float((panel["t_dm"] * panel["y_dm"]).sum() / denominator)

    ground_truth = {
        "params": asdict(p),
        "workbook": str(out_path),
        "n_participants": int(n_total),
        "n_substudy": int(len(substudy_idx)),
        "n_substudy_cases": int(substudy_case.sum()),
        "realised": {
            "grip_slope_mean": float(np.mean(grip_slope)),
            "grip_slope_sd": float(np.std(grip_slope)),
            "mean_log_bt": float(np.nanmean(log_bt)),
            "sd_log_bt": float(np.nanstd(log_bt)),
            "n_decline_eligible": int(decline_mask.sum()),
            "n_stable_eligible": int(stable_mask.sum()),
            "marginal_slope_per_month": marginal_slope_per_month,
            "marginal_slope_per_year": marginal_slope_per_month * 12.0,
            "within_slope_per_month": within_slope_per_month,
            "within_slope_per_year": within_slope_per_month * 12.0,
            "n_observations": int(len(all_t)),
        },
        "bt_effect_is_null": bool(
            p.bt_decline_log_or == 0.0 and p.bt_slope_per_sd == 0.0 and p.bt_subgroup_shift == 0.0
        ),
    }
    truth_path = out_path.with_name("ground_truth.json")
    truth_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    return out_path, ground_truth


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the synthetic cohort workbook.")
    parser.add_argument("--seed", type=int, default=SETTINGS.model.random_seed)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--bt-decline-log-or",
        type=float,
        default=0.0,
        help="Simulate a non-null B/T effect (validation only - leave at 0 for the sanity check).",
    )
    args = parser.parse_args()

    params = GenerationParams(seed=args.seed, bt_decline_log_or=args.bt_decline_log_or)
    path, truth = generate_workbook(params, output_path=args.out)
    print(f"Wrote {path}")
    print(f"  participants : {truth['n_participants']}")
    print(f"  B/T substudy : {truth['n_substudy']} ({truth['n_substudy_cases']} decliners)")
    print(f"  B/T effect   : {'NULL (sanity-check mode)' if truth['bt_effect_is_null'] else 'NON-NULL'}")


if __name__ == "__main__":
    main()
