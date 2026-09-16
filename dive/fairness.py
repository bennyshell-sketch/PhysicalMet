"""Ethnicity, ancestry and geography: auditing the model, not powering it.

This module exists because the obvious thing to do with ethnicity and race in a
predictive model is the thing that has repeatedly caused documented harm.

Clinical algorithms that used race as a biological input have had to be
withdrawn or rebuilt, and the pattern is consistent: a social category was
treated as a biological one, the model encoded the consequences of inequity as
though they were properties of patients, and the people it disadvantaged were
the ones already disadvantaged.

* **Kidney function.** The 2009 CKD-EPI equation raised estimated GFR for Black
  patients via a race coefficient. Higher eGFR means later referral, later
  transplant listing. The NKF-ASN Task Force recommended a race-free equation in
  2021. (Delgado C, et al. J Am Soc Nephrol 2021;32(12):2994-3015.)
* **Lung function.** "Race correction" in spirometry normalised lower measured
  values as expected, masking disease. The American Thoracic Society moved to
  race-neutral equations in 2023. (Bhakta NR, et al. Am J Respir Crit Care Med
  2023;207(8):978-995.)
* **Obstetrics.** The MFMU VBAC calculator lowered predicted success for Black
  and Hispanic women; race and ethnicity were removed in the 2021 revision.
  (Grobman WA, et al. Am J Obstet Gynecol 2021;225(6):664.e1-664.e7.)
* **Devices, not just models.** Pulse oximeters overestimate oxygen saturation
  in patients with darker skin, causing missed hypoxaemia. (Sjoding MW, et al.
  N Engl J Med 2020;383(25):2477-2478.)

So Dive draws three lines.

**Race and ethnicity are not biology.** They are self-reported social
categories whose meaning depends on the classification system used and the
society doing the classifying. They are excluded from model features by
default. Overriding that is possible, requires an explicit flag, and is logged.

**Genetic ancestry is a different variable and is allowed.** Continuous ancestry
principal components are measurable, are not a social category, and matter for
specific variant effects and polygenic score transfer. They live in the
``genetic`` domain and are ordinary candidate features.

**Geography and deprivation are context, not cause.** They belong in
reference-range selection and in fairness auditing, and as explicitly labelled
proxies for social determinants - never as though postcode caused frailty.

What this module then does is the useful part: it **audits**. A model can be
accurate overall and much worse for one group, and the only way to know is to
measure it per group. ``audit_subgroups`` does that, with small-cell
suppression so the audit does not itself disclose individuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PROTECTED_ATTRIBUTES",
    "CONTEXT_ATTRIBUTES",
    "SubgroupResult",
    "FairnessAudit",
    "audit_subgroups",
    "screen_features",
    "reference_population_note",
]


#: Self-reported social categories. Excluded from model features by default.
PROTECTED_ATTRIBUTES: frozenset[str] = frozenset({
    "ethnicity_self_reported", "ethnicity", "ethnic_group", "race",
    "ethnicity_classification", "preferred_language", "migration_generation",
    "religion", "sexual_orientation", "disability_status",
})

#: Geography and social position. Legitimate for reference ranges and auditing;
#: usable as a predictor only when explicitly labelled a social-determinant
#: proxy, never as a biological mechanism.
CONTEXT_ATTRIBUTES: frozenset[str] = frozenset({
    "country_iso3", "region_code", "latitude_band", "urban_rural",
    "deprivation_decile", "healthcare_access_index",
})

#: Measurable, continuous, and not a social category.
ANCESTRY_ATTRIBUTES: frozenset[str] = frozenset({
    "genetic_ancestry_pc1", "genetic_ancestry_pc2", "genetic_ancestry_pc3",
})


# --------------------------------------------------------------------------
# Feature screening
# --------------------------------------------------------------------------

def screen_features(
    candidates: Iterable[str],
    allow_protected: bool = False,
    allow_context: bool = True,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split candidate features into allowed and refused, with reasons.

    Called by the feature search before anything is fitted, so a protected
    attribute cannot become a predictor by accident.
    """
    allowed: list[str] = []
    refused: list[tuple[str, str]] = []

    for name in candidates:
        key = str(name).strip().lower()
        if key in PROTECTED_ATTRIBUTES and not allow_protected:
            refused.append((
                str(name),
                "self-reported social category, excluded from model features by default. "
                "Race and ethnicity are not biological variables, and clinical algorithms "
                "that treated them as such have caused documented harm. Use genetic "
                "ancestry components if you need a biological axis, or set "
                "allow_protected=True and state the justification.",
            ))
        elif key in CONTEXT_ATTRIBUTES and not allow_context:
            refused.append((str(name), "geographic or social-position variable, excluded by configuration"))
        else:
            allowed.append(str(name))

    return allowed, refused


# --------------------------------------------------------------------------
# Subgroup auditing
# --------------------------------------------------------------------------

@dataclass
class SubgroupResult:
    group: str
    n: int
    n_events: int
    auc: float = float("nan")
    calibration_slope: float = float("nan")
    mean_predicted: float = float("nan")
    observed_rate: float = float("nan")
    suppressed: bool = False
    note: str = ""

    @property
    def calibration_gap(self) -> float:
        return self.mean_predicted - self.observed_rate


@dataclass
class FairnessAudit:
    attribute: str
    subgroups: list[SubgroupResult] = field(default_factory=list)
    auc_range: float = float("nan")
    worst_group: str = ""
    best_group: str = ""
    max_calibration_gap: float = float("nan")
    n_suppressed: int = 0
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def concerning(self) -> bool:
        return bool(np.isfinite(self.auc_range) and self.auc_range >= 0.10)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for result in self.subgroups:
            rows.append({
                "Group": result.group,
                "n": result.n if not result.suppressed else "<5",
                "Events": result.n_events if not result.suppressed else "-",
                "AUC": "-" if not np.isfinite(result.auc) else f"{result.auc:.3f}",
                "Predicted": "-" if not np.isfinite(result.mean_predicted) else f"{result.mean_predicted:.3f}",
                "Observed": "-" if not np.isfinite(result.observed_rate) else f"{result.observed_rate:.3f}",
                "Note": result.note,
            })
        return pd.DataFrame(rows)


def audit_subgroups(
    data: pd.DataFrame,
    attribute: str,
    outcome: str = "declined",
    prediction: str = "predicted_risk",
    min_cell: int = 5,
    concerning_auc_range: float = 0.10,
) -> FairnessAudit:
    """Measure model performance separately within each subgroup.

    A model can be accurate overall and substantially worse for one group; the
    aggregate number hides it. Groups smaller than ``min_cell`` are suppressed
    rather than reported, because a performance figure computed on three people
    is both meaningless and disclosive.
    """
    audit = FairnessAudit(attribute=attribute)

    missing = [c for c in (attribute, outcome, prediction) if c not in data.columns]
    if missing:
        audit.verdict = f"Cannot audit: column(s) not present - {', '.join(missing)}."
        return audit

    frame = data[[attribute, outcome, prediction]].dropna()
    if frame.empty:
        audit.verdict = "Cannot audit: no complete rows."
        return audit

    from sklearn.metrics import roc_auc_score

    results: list[SubgroupResult] = []
    for name, chunk in frame.groupby(attribute, observed=True):
        n = len(chunk)
        y = chunk[outcome].to_numpy(dtype=float)
        p = chunk[prediction].to_numpy(dtype=float)
        n_events = int(np.nansum(y))

        if n < min_cell:
            results.append(SubgroupResult(
                str(name), n, n_events, suppressed=True,
                note=f"suppressed: fewer than {min_cell} participants",
            ))
            audit.n_suppressed += 1
            continue

        result = SubgroupResult(str(name), n, n_events)
        result.mean_predicted = float(np.nanmean(p))
        result.observed_rate = float(np.nanmean(y))

        if len(np.unique(y[np.isfinite(y)])) > 1:
            result.auc = float(roc_auc_score(y, p))
        else:
            result.note = "only one outcome class present; AUC undefined"

        # Calibration slope: regress the outcome on the logit of the prediction.
        try:
            clipped = np.clip(p, 1e-6, 1 - 1e-6)
            logit = np.log(clipped / (1 - clipped))
            if np.std(logit) > 0 and len(np.unique(y)) > 1:
                result.calibration_slope = float(np.polyfit(logit, y, 1)[0])
        except (ValueError, np.linalg.LinAlgError):
            pass

        results.append(result)

    audit.subgroups = sorted(results, key=lambda r: (-r.n, r.group))

    scored = [r for r in results if np.isfinite(r.auc)]
    if len(scored) >= 2:
        best = max(scored, key=lambda r: r.auc)
        worst = min(scored, key=lambda r: r.auc)
        audit.best_group, audit.worst_group = best.group, worst.group
        audit.auc_range = float(best.auc - worst.auc)
        audit.max_calibration_gap = float(max(abs(r.calibration_gap) for r in scored))

        if audit.auc_range >= concerning_auc_range:
            audit.verdict = (
                f"The model performs unevenly across {attribute}. It discriminates best "
                f"for '{best.group}' (AUC {best.auc:.2f}) and worst for '{worst.group}' "
                f"(AUC {worst.auc:.2f}), a gap of {audit.auc_range:.2f}. A model this "
                "uneven should not be deployed across these groups without investigating "
                "why - the usual causes are unequal representation in training and "
                "measurements that behave differently between groups."
            )
        else:
            audit.verdict = (
                f"Performance is comparable across {attribute}: AUC ranges from "
                f"{worst.auc:.2f} to {best.auc:.2f} (gap {audit.auc_range:.2f}). That is "
                "reassuring but not conclusive - small subgroups give imprecise estimates."
            )
    elif scored:
        audit.verdict = (
            f"Only one subgroup of {attribute} was large enough to score, so performance "
            "cannot be compared across groups."
        )
    else:
        audit.verdict = f"No subgroup of {attribute} was large enough to score."

    if audit.n_suppressed:
        audit.warnings.append(
            f"{audit.n_suppressed} subgroup(s) suppressed for having fewer than "
            f"{min_cell} participants. Absence from this table is not evidence that "
            "the model works for them - it is evidence that you cannot tell."
        )

    return audit


# --------------------------------------------------------------------------
# Reference populations
# --------------------------------------------------------------------------

def reference_population_note(measure: str = "bmd_t_score") -> str:
    """Explain which reference population a derived score depends on.

    This is where geography and ancestry legitimately belong: not as predictors,
    but as determinants of the reference database a T-score or a predicted
    normal is computed against. Change the reference and you change who is
    diagnosed, without any patient changing.
    """
    notes = {
        "bmd_t_score": (
            "A T-score is a standard deviation score against a young-adult reference "
            "population, so the reference database determines the number. The ISCD "
            "convention is a uniform young white female reference (NHANES III femoral "
            "neck) for women, applied across groups, precisely to avoid the trap of "
            "ethnicity-specific thresholds shifting who qualifies for treatment. If "
            "your deployment uses a different reference database, T-scores are not "
            "comparable with published cutoffs and that must be stated."
        ),
        "egfr_ml_min": (
            "Use the race-free CKD-EPI 2021 equation. Earlier equations included a "
            "coefficient that raised estimated GFR for Black patients, delaying "
            "referral and transplant listing."
        ),
        "prs_ebmd": (
            "Polygenic scores transfer poorly across genetic ancestries - typically "
            "losing most of their predictive accuracy outside the ancestry group they "
            "were derived in, which for most published scores is European. A PRS "
            "applied across a diverse cohort will look predictive in one group and "
            "near-useless in another, and the aggregate number hides it."
        ),
        "dnam_age_horvath": (
            "Epigenetic clocks are calibrated on specific populations and array "
            "platforms. Age acceleration should be computed as a residual within your "
            "own cohort rather than taken from a vendor report."
        ),
    }
    return notes.get(
        measure,
        "No reference-population note is recorded for this measure. If it is a "
        "standardised or predicted-normal score, establish which population it was "
        "standardised against before comparing it with published thresholds.",
    )
