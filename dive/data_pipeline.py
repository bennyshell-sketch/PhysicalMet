"""Loading, cleaning and reshaping of cohort workbooks.

Handles quirks that recur across hand-maintained clinical spreadsheets, so that
swapping from synthetic to real data needs no code change:

* the 12-month sheet carries two unrelated ID columns before the real one
* ``TUG_s == 40.0`` is a "could not complete" sentinel, not a duration
* grip trials coded ``999`` are missing, not a 999 kg reading
* stored ``Delta_*`` columns are unreliable and are always recomputed from raw
* comorbidity/medication fields mix ``0/1`` with ``Yes/No``
* ``IPAQ_Category`` free text is unreliable; ``IPAQ_Coded_Category`` wins

Column lookup is alias-based rather than positional wherever possible, with a
positional fallback for the known-ambiguous ID column. Nothing is imputed
silently: a missing input stays missing and is counted in the quality report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .biochem import compute_bt_ratio_frame
from .config import SETTINGS, Settings

__all__ = [
    "CleanedData",
    "QualityReport",
    "load_workbook",
    "build_dataset",
    "load_dataset",
]


# --------------------------------------------------------------------------
# Column alias resolution
# --------------------------------------------------------------------------

#: Canonical name -> candidate spellings seen across source file versions.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "participant_id": ("participant_id", "pid", "id", "wl_id", "studyid", "study_id"),
    "fall_status": ("fall_status", "faller", "fell"),
    "group": ("group", "cohort", "age_group"),
    "age_years": ("age_years", "age", "age_at_visit"),
    "sex": ("gender", "sex", "sex_code"),
    "height_m": ("height_m", "height", "ht_m"),
    "weight_kg": ("weight_kg", "weight", "wt_kg"),
    "bmi": ("bmi", "body_mass_index"),
    "waist_cm": ("waist_cm", "waist"),
    "calf_cm": ("calf_cm", "calf"),
    "date_of_visit": ("date_of_visit", "visit_date", "date"),
    "dob": ("dob", "date_of_birth", "birth_date"),
    "elapsed_months": ("visit_elapsed_months", "elapsed_months", "months_since_baseline"),
    "gait_speed1": ("gait_speed1", "gait_1", "walk1", "walk_time_1"),
    "gait_speed2": ("gait_speed2", "gait_2", "walk2", "walk_time_2"),
    "chair_time_s": ("chair_time_s", "chair", "chair_stand_time", "chair_s"),
    "tug_s": ("tug_s", "tug", "timed_up_and_go"),
    "sppb_balance": ("sppb_balance", "sppb_sbs", "balance_score"),
    "sppb_chair": ("sppb_chair", "chair_score"),
    "sppb_gait": ("sppb_gait", "gait_score"),
    "sppb_total": ("sppb_total", "sppb"),
    "moca": ("moca", "moca_total"),
    "education": ("edlev_highest", "education", "education_level", "edlev"),
    "ethnicity": ("ethnic_group", "ethnicity", "ethnic"),
    "imd": ("imd", "imd_decile"),
    "ipaq_text": ("ipaq_category", "ipaq_cat_text"),
    "ipaq_coded": ("ipaq_coded_category", "coded_category", "ipaq_cat", "ipaq_coded"),
    "fried_weight_loss": ("fried1_weight_loss", "fried1", "weight_loss"),
    "worn_out": ("worn_out", "wornout"),
    "tired": ("tired", "tired_raw"),
    "comorbidity": ("comorbidity_any", "comorbidity", "comorbidities"),
    "polypharmacy": ("polypharmacy", "poly_pharmacy", "medications_5plus"),
    "subgroup_code": ("faller_subgroup_code", "subgroup_code", "faller_subgroup"),
    "il6": ("il6_pg_ml", "il6", "il_6", "il6_pgml"),
    "sil6r": ("sil6r_ng_ml", "sil6r", "sil_6r", "sil6r_ngml"),
    "sgp130": ("sgp130_ng_ml", "sgp130", "sgp_130", "sgp130_ngml"),
    "sample_group": ("sample_group", "case_control_group", "selection_group"),
    "plate_id": ("plate_id", "plate", "elisa_plate"),
    "max_grip_baseline": ("max_grip_baseline", "max_grip_bl", "grip_baseline"),
    "max_grip_fm": ("max_grip_fm", "grip_fm", "max_grip_final"),
    "grip_change": ("grip_change_kg", "grip_change", "delta_max_grip_fm"),
    "fm_source": ("fm_source_visit", "final_visit_source"),
    "fm_elapsed": ("fm_elapsed_months", "final_elapsed_months"),
}


def _norm(name: object) -> str:
    """Normalise a column label for alias matching."""
    text = str(name).strip().lower()
    text = re.sub(r"[^\w]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map canonical names to the actual columns present in ``frame``."""
    lookup = {_norm(c): c for c in frame.columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                resolved[canonical] = lookup[alias]
                break
    return resolved


def _grip_trial_columns(frame: pd.DataFrame) -> list[str]:
    """Find the 6 grip trial columns regardless of naming convention."""
    pattern = re.compile(r"grip.*(st)?_?(\d)|grip_trial_(\d)")
    hits = [c for c in frame.columns if pattern.search(_norm(c)) and "max" not in _norm(c)]
    return hits


# --------------------------------------------------------------------------
# Quality reporting
# --------------------------------------------------------------------------

@dataclass
class QualityReport:
    """What was loaded, what was dropped, and what is missing."""

    source_path: str = ""
    data_source_kind: str = "synthetic"
    n_rows_loaded: dict[str, int] = field(default_factory=dict)
    n_participants: int = 0
    n_participant_visits: int = 0
    n_substudy: int = 0
    flags: list[str] = field(default_factory=list)
    exclusions: list[dict] = field(default_factory=list)
    missingness: dict[str, float] = field(default_factory=dict)
    bt_solver_failures: int = 0

    def add_flag(self, message: str) -> None:
        self.flags.append(message)

    def to_frame(self) -> pd.DataFrame:
        """Missingness as a presentable table."""
        rows = [
            {"Variable": k, "Missing %": round(v * 100, 1)}
            for k, v in sorted(self.missingness.items(), key=lambda kv: -kv[1])
        ]
        return pd.DataFrame(rows)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Source: {self.source_path}",
            f"Kind: {self.data_source_kind}",
            f"Participants: {self.n_participants}",
            f"Participant-visit rows: {self.n_participant_visits}",
            f"B/T substudy samples: {self.n_substudy}",
        ]
        lines.extend(self.flags)
        return lines


# --------------------------------------------------------------------------
# Cleaning primitives
# --------------------------------------------------------------------------

def harmonise_binary(series: pd.Series) -> pd.Series:
    """Coerce a mixed 0/1 and Yes/No column to a clean 0/1 float column."""
    mapping = {
        "yes": 1.0, "y": 1.0, "true": 1.0, "1": 1.0, "1.0": 1.0,
        "no": 0.0, "n": 0.0, "false": 0.0, "0": 0.0, "0.0": 0.0,
    }

    def convert(value: object) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            num = float(value)
            return num if num in (0.0, 1.0) else np.nan
        return mapping.get(str(value).strip().lower(), np.nan)

    return series.map(convert).astype(float)


def clean_tug(series: pd.Series, sentinel: float = 40.0) -> tuple[pd.Series, pd.Series]:
    """Split TUG into a continuous column and a 'could not complete' flag.

    The sentinel value is removed from the continuous column - treating 40.0 s
    as a real duration silently biases every downstream estimate.
    """
    values = pd.to_numeric(series, errors="coerce")
    cannot = values.eq(sentinel)
    continuous = values.mask(cannot)
    return continuous, cannot.astype(float)


def max_grip_from_trials(
    frame: pd.DataFrame,
    trial_columns: Sequence[str],
    missing_sentinel: float = 999.0,
) -> pd.Series:
    """Max of the 6 grip trials, with the 999 sentinel treated as missing."""
    if not trial_columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    trials = frame[list(trial_columns)].apply(pd.to_numeric, errors="coerce")
    trials = trials.mask(trials.eq(missing_sentinel))
    trials = trials.mask(trials <= 0)
    return trials.max(axis=1, skipna=True)


def gait_speed_ms(frame: pd.DataFrame, cols: dict[str, str], distance_m: float = 4.0) -> pd.Series:
    """Gait speed in m/s from the raw walk *times* (which are seconds, not m/s)."""
    candidates = [cols[k] for k in ("gait_speed1", "gait_speed2") if k in cols]
    if not candidates:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    times = frame[candidates].apply(pd.to_numeric, errors="coerce")
    times = times.mask(times <= 0)
    best = times.min(axis=1, skipna=True)
    return distance_m / best


def resolve_participant_id(frame: pd.DataFrame, sheet_name: str, report: QualityReport) -> pd.Series:
    """Find the true participant ID column.

    Follow-up sheets often carry lab or batch references ahead of the real ID
    column, so a naive ``iloc[:, 0]`` picks up the wrong one. Identify the
    column by content (the participant-ID pattern) and fall back to a
    positional index only if the pattern match is ambiguous.
    """
    pattern = re.compile(r"^WL\s*\d{2,4}$", re.IGNORECASE)

    scores: list[tuple[float, str]] = []
    for column in frame.columns:
        series = frame[column].dropna().astype(str).str.strip()
        if series.empty:
            continue
        hit_rate = series.str.match(pattern).mean()
        if hit_rate > 0:
            scores.append((float(hit_rate), column))

    if scores:
        scores.sort(reverse=True)
        best_rate, best_column = scores[0]
        if best_rate >= 0.8:
            position = list(frame.columns).index(best_column)
            if position != 0:
                report.add_flag(
                    f"{sheet_name}: participant ID found at column index {position} "
                    f"('{best_column}'), not column 0 - known workbook quirk handled."
                )
            return frame[best_column].astype(str).str.strip().str.upper()

    cols = resolve_columns(frame)
    if "participant_id" in cols:
        report.add_flag(f"{sheet_name}: fell back to alias-matched ID column.")
        return frame[cols["participant_id"]].astype(str).str.strip().str.upper()

    fallback_index = 2 if sheet_name.lower().startswith("twelve") else 0
    report.add_flag(
        f"{sheet_name}: could not identify ID by content; used documented "
        f"positional fallback (column {fallback_index})."
    )
    return frame.iloc[:, fallback_index].astype(str).str.strip().str.upper()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

#: Canonical visit name -> candidate sheet names.
SHEET_ALIASES: dict[str, tuple[str, ...]] = {
    "Baseline": ("baseline", "baseline_group0", "bl"),
    "PostFall": ("postfall", "post_fall", "postfall_group0", "pf"),
    "SixMonth": ("sixmonth", "sixmonths", "six_month", "sixmonth_group0", "6m"),
    "TwelveMonth": ("twelvemonth", "twelvemonths", "twelve_month", "twelvemonth_group0", "12m"),
    "FM": ("fm", "final", "final_measure", "finalmeasure"),
    "BT_Substudy": ("bt_substudy", "btsubstudy", "bt", "substudy", "elisa", "bt_ratio_candidate_selection"),
}


def load_workbook(path: Path | str, report: QualityReport | None = None) -> dict[str, pd.DataFrame]:
    """Read a cohort workbook and map its sheets onto canonical visit names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {path}\n"
            "Run `python -m dive.generate_synthetic_data` to create the synthetic "
            "workbook, or set DATA_SOURCE=real and drop the real file into data/real/."
        )

    excel = pd.ExcelFile(path)
    available = {_norm(s): s for s in excel.sheet_names}

    sheets: dict[str, pd.DataFrame] = {}
    for canonical, aliases in SHEET_ALIASES.items():
        for alias in aliases:
            if alias in available:
                sheets[canonical] = excel.parse(available[alias])
                break
        else:
            # Partial match as a last resort (real files carry suffixes).
            for norm_name, actual in available.items():
                if any(norm_name.startswith(a) for a in aliases):
                    sheets[canonical] = excel.parse(actual)
                    break

    if report is not None:
        report.n_rows_loaded = {k: len(v) for k, v in sheets.items()}
        missing_sheets = set(SHEET_ALIASES) - set(sheets)
        if missing_sheets:
            report.add_flag(f"Sheets not found (skipped): {', '.join(sorted(missing_sheets))}")

    return sheets


# --------------------------------------------------------------------------
# Visit extraction
# --------------------------------------------------------------------------

#: Nominal months since baseline, used only when no actual date/elapsed is
#: recorded. The trajectory model always prefers the true elapsed time.
NOMINAL_MONTHS: dict[str, float] = {
    "Baseline": 0.0,
    "PostFall": 1.5,
    "SixMonth": 6.0,
    "TwelveMonth": 12.0,
}


def _extract_visit(
    frame: pd.DataFrame,
    visit: str,
    report: QualityReport,
    settings: Settings,
    baseline_dates: pd.Series | None = None,
) -> pd.DataFrame:
    """Reduce one visit sheet to the canonical per-participant-visit record."""
    cols = resolve_columns(frame)
    out = pd.DataFrame(index=frame.index)
    out["participant_id"] = resolve_participant_id(frame, visit, report)
    out["visit"] = visit

    # --- elapsed time (irregular visit timing is the rule, not the exception)
    elapsed = pd.Series(np.nan, index=frame.index, dtype=float)
    if "elapsed_months" in cols:
        elapsed = pd.to_numeric(frame[cols["elapsed_months"]], errors="coerce")
    if elapsed.isna().all() and "date_of_visit" in cols and baseline_dates is not None:
        visit_dates = pd.to_datetime(frame[cols["date_of_visit"]], errors="coerce")
        merged = out[["participant_id"]].copy()
        merged["visit_date"] = visit_dates.to_numpy()
        merged = merged.merge(
            baseline_dates.rename("baseline_date"),
            left_on="participant_id",
            right_index=True,
            how="left",
        )
        elapsed = (
            (merged["visit_date"] - merged["baseline_date"]).dt.days / 30.4375
        ).to_numpy()
        elapsed = pd.Series(elapsed, index=frame.index)
    if visit == "Baseline":
        elapsed = pd.Series(0.0, index=frame.index)
    out["elapsed_months"] = elapsed.fillna(NOMINAL_MONTHS.get(visit, np.nan))
    out["elapsed_is_imputed"] = elapsed.isna().astype(float)

    # --- physical function
    out["max_grip_kg"] = max_grip_from_trials(
        frame, _grip_trial_columns(frame), settings.cohort.grip_missing_sentinel
    )
    out["gait_speed_ms"] = gait_speed_ms(frame, cols)

    if "chair_time_s" in cols:
        out["chair_time_s"] = pd.to_numeric(frame[cols["chair_time_s"]], errors="coerce")
    if "tug_s" in cols:
        tug, cannot = clean_tug(frame[cols["tug_s"]], settings.cohort.tug_sentinel_value)
        out["tug_s"] = tug
        out["tug_cannot_complete"] = cannot
        n_sentinel = int(cannot.sum())
        if n_sentinel:
            report.add_flag(
                f"{visit}: {n_sentinel} TUG values of "
                f"{settings.cohort.tug_sentinel_value} treated as 'could not complete' "
                "and excluded from the continuous variable."
            )

    for canonical, target in (
        ("sppb_balance", "sppb_balance"),
        ("sppb_chair", "sppb_chair"),
        ("sppb_gait", "sppb_gait"),
        ("sppb_total", "sppb_total"),
        ("weight_kg", "weight_kg"),
        ("moca", "moca"),
    ):
        if canonical in cols:
            out[target] = pd.to_numeric(frame[cols[canonical]], errors="coerce")

    # --- stored delta columns are recorded but never used downstream
    stored_deltas = [c for c in frame.columns if _norm(c).startswith("delta_")]
    for column in stored_deltas:
        out[f"stored_{_norm(column)}"] = pd.to_numeric(frame[column], errors="coerce")

    return out


def _classify_subgroup(wide: pd.DataFrame, settings: Settings) -> pd.Series:
    """Faller subgroup from fall status and post-fall grip change.

    1 Non-Faller           fall_status = 0 or missing
    2 Maintained Function  faller, post-fall grip loss < threshold (or no PF grip)
    3 Recovered Function   faller, PF loss >= threshold, back within threshold at FM
    4 Declined Function    faller, PF loss >= threshold, not recovered at FM
    """
    threshold = settings.cohort.grip_decline_threshold_kg
    bl = wide.get("grip_Baseline")
    pf = wide.get("grip_PostFall")
    fm = wide.get("grip_FM")

    codes = pd.Series(np.nan, index=wide.index, dtype=float)
    fall_status = wide.get("fall_status", pd.Series(np.nan, index=wide.index))
    # Faller subgroups are defined for the older cohort only. A participant
    # with no fall-status record is left unclassified, never defaulted to
    # Non-Faller - that would silently inflate the NF group with the younger
    # cohort, who were never followed for falls.
    assessed = fall_status.notna()
    is_faller = fall_status.eq(1.0)

    codes[assessed & ~is_faller] = 1.0

    if bl is None or pf is None:
        codes[is_faller] = 2.0
        return codes

    pf_loss = bl - pf                      # positive = loss
    no_pf = pf.isna()
    codes[is_faller & (no_pf | (pf_loss < threshold))] = 2.0

    severe = is_faller & ~no_pf & (pf_loss >= threshold)
    if fm is not None:
        fm_loss = bl - fm
        codes[severe & fm_loss.notna() & (fm_loss < threshold)] = 3.0
        codes[severe & fm_loss.notna() & (fm_loss >= threshold)] = 4.0
        # Severe post-fall loss with no final visit cannot be adjudicated.
        codes[severe & fm_loss.isna()] = 4.0
    else:
        codes[severe] = 4.0

    return codes


def _fried_score(wide: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Fried 2001 frailty phenotype, five criteria, scored on the published cutpoints."""
    out = pd.DataFrame(index=wide.index)

    out["fried_weight_loss"] = wide.get("fried_weight_loss", pd.Series(np.nan, index=wide.index))

    worn = pd.to_numeric(wide.get("worn_out"), errors="coerce")
    tired = pd.to_numeric(wide.get("tired"), errors="coerce")
    if worn is not None or tired is not None:
        exhaustion = pd.concat([worn, tired], axis=1).max(axis=1, skipna=True)
        out["fried_exhaustion"] = (exhaustion >= 3).astype(float).mask(exhaustion.isna())

    ipaq = pd.to_numeric(wide.get("ipaq_coded"), errors="coerce")
    if ipaq is not None:
        out["fried_low_activity"] = (ipaq == 1).astype(float).mask(ipaq.isna())

    # Slow gait: Fried 15ft cutpoints rescaled to the 4 m course (x 4/4.57).
    height = pd.to_numeric(wide.get("height_m"), errors="coerce")
    sex = pd.to_numeric(wide.get("sex"), errors="coerce")   # 0 = male, 1 = female
    gait = pd.to_numeric(wide.get("gait_speed_Baseline"), errors="coerce")
    if gait is not None and height is not None and sex is not None:
        walk_time = 4.0 / gait.where(gait > 0)
        tall_cut = np.where(sex.eq(1.0), 1.59, 1.73)
        cutoff = np.where(height <= tall_cut, 6.13, 5.25)
        out["fried_slow_gait"] = (walk_time >= cutoff).astype(float).mask(walk_time.isna())

    # Weak grip: Fried sex x BMI cutpoints.
    grip = pd.to_numeric(wide.get("grip_Baseline"), errors="coerce")
    bmi = pd.to_numeric(wide.get("bmi"), errors="coerce")
    if grip is not None and bmi is not None and sex is not None:
        male_cut = np.select([bmi <= 24, bmi <= 26, bmi <= 28], [29.0, 30.0, 30.0], default=32.0)
        female_cut = np.select([bmi <= 23, bmi <= 26, bmi <= 29], [17.0, 17.3, 18.0], default=21.0)
        cut = np.where(sex.eq(1.0), female_cut, male_cut)
        out["fried_weak_grip"] = (grip < cut).astype(float).mask(grip.isna() | bmi.isna())

    criteria = [c for c in out.columns if c.startswith("fried_")]
    out["fried_score"] = out[criteria].sum(axis=1, skipna=True)
    out["fried_n_assessed"] = out[criteria].notna().sum(axis=1)
    # A score is only meaningful if enough criteria were assessable.
    out.loc[out["fried_n_assessed"] < 3, "fried_score"] = np.nan
    out["fried_category"] = pd.cut(
        out["fried_score"], bins=[-0.1, 0.5, 2.5, 5.1], labels=["Robust", "Pre-frail", "Frail"]
    )
    return out


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

@dataclass
class CleanedData:
    """The cleaned dataset in both shapes, plus provenance."""

    long: pd.DataFrame            # one row per participant-visit
    wide: pd.DataFrame            # one row per participant
    substudy: pd.DataFrame        # B/T substudy (case-control, n~100)
    report: QualityReport

    @property
    def has_bt(self) -> bool:
        return not self.substudy.empty and self.substudy["BT_ratio"].notna().any()

    def trajectory_frame(self) -> pd.DataFrame:
        """Long rows usable by the mixed model: real grip at a known time."""
        frame = self.long.dropna(subset=["max_grip_kg", "elapsed_months"]).copy()
        counts = frame.groupby("participant_id")["max_grip_kg"].transform("size")
        return frame[counts >= 2].copy()


def build_dataset(
    path: Path | str | None = None,
    settings: Settings | None = None,
) -> CleanedData:
    """Load, clean and reshape a workbook into the canonical dataset."""
    cfg = settings or SETTINGS
    source = Path(path) if path is not None else cfg.active_workbook()

    report = QualityReport(
        source_path=str(source),
        data_source_kind="real participant data" if cfg.data_source == "real" else "synthetic",
    )
    sheets = load_workbook(source, report)

    if "Baseline" not in sheets:
        raise ValueError(f"No Baseline sheet found in {source}; sheets present: {list(sheets)}")

    baseline_raw = sheets["Baseline"]
    base_cols = resolve_columns(baseline_raw)

    baseline_ids = resolve_participant_id(baseline_raw, "Baseline", report)
    baseline_dates = None
    if "date_of_visit" in base_cols:
        baseline_dates = pd.Series(
            pd.to_datetime(baseline_raw[base_cols["date_of_visit"]], errors="coerce").to_numpy(),
            index=baseline_ids.to_numpy(),
        )
        baseline_dates = baseline_dates[~baseline_dates.index.duplicated()]

    # ---------------- long format ----------------
    visit_frames: list[pd.DataFrame] = []
    for visit in ("Baseline", "PostFall", "SixMonth", "TwelveMonth"):
        if visit in sheets:
            visit_frames.append(
                _extract_visit(sheets[visit], visit, report, cfg, baseline_dates)
            )

    long = pd.concat(visit_frames, ignore_index=True)
    # A visit row with no measurements at all is a non-attendance, not data.
    measure_cols = [
        c for c in ("max_grip_kg", "gait_speed_ms", "chair_time_s", "tug_s", "sppb_total")
        if c in long.columns
    ]
    attended = long[measure_cols].notna().any(axis=1)
    n_empty = int((~attended).sum())
    long = long[attended].copy()
    report.add_flag(f"Removed {n_empty} participant-visit rows with no measurements (non-attendance).")

    # ---------------- participant-level (wide) ----------------
    wide = pd.DataFrame({"participant_id": baseline_ids.to_numpy()}).drop_duplicates("participant_id")
    wide = wide.set_index("participant_id")

    static_map = {
        "age_years": "age_years",
        "sex": "sex",
        "height_m": "height_m",
        "weight_kg": "weight_kg",
        "bmi": "bmi",
        "waist_cm": "waist_cm",
        "calf_cm": "calf_cm",
        "education": "education",
        "ethnicity": "ethnicity",
        "imd": "imd",
        "group": "group",
        "fall_status": "fall_status",
        "ipaq_coded": "ipaq_coded",
        "fried_weight_loss": "fried_weight_loss",
        "worn_out": "worn_out",
        "tired": "tired",
        "moca": "moca",
    }
    indexed_baseline = baseline_raw.copy()
    indexed_baseline.index = baseline_ids.to_numpy()
    for canonical, target in static_map.items():
        if canonical in base_cols:
            series = pd.to_numeric(indexed_baseline[base_cols[canonical]], errors="coerce")
            wide[target] = series[~series.index.duplicated()]

    # Mixed-encoding fields get harmonised, not coerced to NaN.
    for canonical, target in (("comorbidity", "comorbidity_any"), ("polypharmacy", "polypharmacy")):
        if canonical in base_cols:
            series = harmonise_binary(indexed_baseline[base_cols[canonical]])
            wide[target] = series[~series.index.duplicated()]
            report.add_flag(f"Harmonised mixed 0/1 and Yes/No encoding in '{base_cols[canonical]}'.")

    # IPAQ: the coded column wins over the free-text one.
    if "ipaq_text" in base_cols and "ipaq_coded" in base_cols:
        report.add_flag(
            "IPAQ: used IPAQ_Coded_Category; free-text IPAQ_Category ignored as unreliable."
        )

    # Per-visit function measures pivoted onto the participant row.
    for measure, prefix in (
        ("max_grip_kg", "grip"),
        ("gait_speed_ms", "gait_speed"),
        ("sppb_total", "sppb"),
        ("chair_time_s", "chair"),
        ("tug_s", "tug"),
    ):
        if measure not in long.columns:
            continue
        pivot = long.pivot_table(
            index="participant_id", columns="visit", values=measure, aggfunc="first"
        )
        for visit in pivot.columns:
            wide[f"{prefix}_{visit}"] = pivot[visit]

    # ---------------- FM sheet ----------------
    if "FM" in sheets:
        fm_raw = sheets["FM"]
        fm_cols = resolve_columns(fm_raw)
        fm_ids = resolve_participant_id(fm_raw, "FM", report)
        fm = pd.DataFrame(index=fm_ids.to_numpy())
        for canonical, target in (
            ("max_grip_fm", "grip_FM"),
            ("fm_elapsed", "fm_elapsed_months"),
            ("fm_source", "fm_source_visit"),
        ):
            if canonical in fm_cols:
                values = fm_raw[fm_cols[canonical]].to_numpy()
                fm[target] = pd.Series(values, index=fm_ids.to_numpy())
        fm = fm[~fm.index.duplicated()]
        for column in fm.columns:
            wide[column] = fm[column]
    if "grip_FM" not in wide.columns:
        # Derive FM ourselves: 12-month preferred, 6-month fallback.
        twelve = wide.get("grip_TwelveMonth")
        six = wide.get("grip_SixMonth")
        if twelve is not None or six is not None:
            wide["grip_FM"] = twelve.combine_first(six) if six is not None else twelve

    # ---------------- ALWAYS recompute deltas from raw ----------------
    # Stored Delta_* columns are documented as unreliable and are never used.
    for prefix in ("grip", "gait_speed", "sppb", "chair", "tug"):
        baseline_col = f"{prefix}_Baseline"
        if baseline_col not in wide.columns:
            continue
        for visit in ("PostFall", "SixMonth", "TwelveMonth", "FM"):
            target = f"{prefix}_{visit}"
            if target in wide.columns:
                wide[f"delta_{prefix}_{visit}"] = wide[target] - wide[baseline_col]
    report.add_flag(
        "All delta variables recomputed as (follow-up - baseline) from raw values; "
        "stored Delta_* columns were read for audit but never used in analysis."
    )

    # ---------------- decline outcome ----------------
    threshold = cfg.cohort.grip_decline_threshold_kg
    if "delta_grip_FM" in wide.columns:
        wide["grip_change_kg"] = wide["delta_grip_FM"]
        wide["declined"] = (wide["grip_change_kg"] <= -threshold).astype(float)
        wide.loc[wide["grip_change_kg"].isna(), "declined"] = np.nan
        wide["stable"] = (wide["grip_change_kg"].abs() <= cfg.cohort.grip_stable_band_kg).astype(float)
        wide.loc[wide["grip_change_kg"].isna(), "stable"] = np.nan

    # ---------------- subgroups and frailty ----------------
    if "subgroup_code" in base_cols:
        stored = pd.to_numeric(indexed_baseline[base_cols["subgroup_code"]], errors="coerce")
        wide["subgroup_code_stored"] = stored[~stored.index.duplicated()]
    wide["subgroup_code"] = _classify_subgroup(wide, cfg)
    wide["subgroup"] = wide["subgroup_code"].map(cfg.cohort.subgroup_labels)
    wide["subgroup_abbrev"] = wide["subgroup_code"].map(cfg.cohort.subgroup_abbrev)
    report.add_flag("Faller subgroups recomputed from fall status and post-fall grip change.")

    fried = _fried_score(wide, cfg)
    for column in fried.columns:
        wide[column] = fried[column]

    wide["age_group"] = np.where(
        wide["age_years"] >= cfg.cohort.older_age_threshold, "Older", "Younger"
    )
    wide["sex_label"] = wide["sex"].map({0.0: "Male", 1.0: "Female"})

    # ---------------- B/T substudy ----------------
    substudy = pd.DataFrame()
    if "BT_Substudy" in sheets:
        raw = sheets["BT_Substudy"]
        sub_cols = resolve_columns(raw)
        sub_ids = resolve_participant_id(raw, "BT_Substudy", report)
        substudy = pd.DataFrame({"participant_id": sub_ids.to_numpy()})

        for canonical, target in (
            ("il6", "IL6_pg_mL"),
            ("sil6r", "sIL6R_ng_mL"),
            ("sgp130", "sgp130_ng_mL"),
        ):
            substudy[target] = (
                pd.to_numeric(raw[sub_cols[canonical]], errors="coerce").to_numpy()
                if canonical in sub_cols
                else np.nan
            )
        if "sample_group" in sub_cols:
            substudy["sample_group"] = raw[sub_cols["sample_group"]].to_numpy()
        if "plate_id" in sub_cols:
            substudy["plate_id"] = raw[sub_cols["plate_id"]].to_numpy()

        bt = compute_bt_ratio_frame(
            substudy["IL6_pg_mL"], substudy["sIL6R_ng_mL"], substudy["sgp130_ng_mL"],
            constants=cfg.biochem,
        )
        substudy = pd.concat([substudy, bt], axis=1)

        n_failed = int(((~substudy["bt_converged"]) & substudy["IL6_pg_mL"].notna()).sum())
        report.bt_solver_failures = n_failed
        n_incomputable = int(substudy["BT_ratio"].isna().sum())
        report.n_substudy = int(substudy["BT_ratio"].notna().sum())
        if n_incomputable:
            report.add_flag(
                f"B/T ratio not computable for {n_incomputable} of {len(substudy)} samples "
                "(missing or non-positive analyte); left as missing, not imputed."
            )
            report.exclusions.append(
                {"reason": "incomplete ELISA triplet", "n": n_incomputable, "stage": "B/T computation"}
            )

        # Merge participant covariates onto the substudy rows.
        merge_cols = [
            c for c in (
                "age_years", "sex", "sex_label", "bmi", "grip_Baseline", "grip_FM",
                "grip_change_kg", "declined", "subgroup", "subgroup_code", "subgroup_abbrev",
                "fried_score", "fried_category", "gait_speed_Baseline", "sppb_Baseline",
            ) if c in wide.columns
        ]
        substudy = substudy.merge(
            wide[merge_cols], left_on="participant_id", right_index=True, how="left"
        )
        substudy["case_control_design"] = True

        # Attach B/T back to the participant table for the trajectory model.
        bt_map = substudy.set_index("participant_id")["BT_ratio"]
        bt_map = bt_map[~bt_map.index.duplicated()]
        wide["BT_ratio"] = bt_map
        wide["log_BT_ratio"] = np.log(wide["BT_ratio"].where(wide["BT_ratio"] > 0))
        fme_map = substudy.set_index("participant_id")["FME"]
        wide["FME"] = fme_map[~fme_map.index.duplicated()]

    # ---------------- quality report ----------------
    report.n_participants = int(len(wide))
    report.n_participant_visits = int(len(long))
    quality_vars = [
        c for c in (
            "age_years", "sex", "bmi", "grip_Baseline", "grip_PostFall", "grip_SixMonth",
            "grip_TwelveMonth", "grip_FM", "gait_speed_Baseline", "sppb_Baseline",
            "tug_Baseline", "fried_score", "imd", "education", "BT_ratio",
        ) if c in wide.columns
    ]
    report.missingness = {c: float(wide[c].isna().mean()) for c in quality_vars}

    wide = wide.reset_index().rename(columns={"index": "participant_id"})
    if "participant_id" not in wide.columns:
        wide = wide.rename(columns={wide.columns[0]: "participant_id"})

    # Merge participant covariates onto the long frame for modelling.
    long = long.merge(
        wide[[c for c in (
            "participant_id", "age_years", "sex", "sex_label", "bmi", "subgroup",
            "subgroup_code", "subgroup_abbrev", "BT_ratio", "log_BT_ratio", "FME",
            "declined", "grip_Baseline", "fried_score", "age_group",
        ) if c in wide.columns]],
        on="participant_id",
        how="left",
    )

    return CleanedData(long=long, wide=wide, substudy=substudy, report=report)


def load_dataset(settings: Settings | None = None) -> CleanedData:
    """Build the dataset from the configured source, generating synthetic data if absent."""
    cfg = settings or SETTINGS
    source = cfg.active_workbook()
    if not source.exists() and cfg.data_source == "synthetic":
        from .generate_synthetic_data import generate_workbook

        generate_workbook(settings=cfg)
    return build_dataset(source, cfg)
