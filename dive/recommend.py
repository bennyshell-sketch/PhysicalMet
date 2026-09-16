"""Evidence-matched recommendations.

This engine does not generate advice. It holds a curated library of
interventions (``knowledge/interventions.yaml``), each with explicit trigger
conditions, an evidence grade and a citation, and it surfaces the entries
whose triggers the participant's *measured* values actually fire.

Three consequences follow, all deliberate:

* A recommendation can only appear if a real measurement triggered it. Nothing
  is suggested on a hunch or by analogy.
* Every recommendation carries its evidence grade and source, so a grade A
  meta-analysis is never displayed as though it carried the same weight as a
  mechanistic inference.
* Where the honest answer is "there is no good evidence", the library says so
  (grade D) and the engine shows it as a statement of absence, never as a
  suggestion. The B/T ratio itself is the main such case: nothing has been
  shown to shift it.

Recommendations are explicitly separated from model predictions. They describe
what the published literature supports for someone with this profile; they are
not an output of the decline or trajectory models, and they carry no claim
about what will happen to this individual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .config import SETTINGS, Settings

__all__ = [
    "Recommendation",
    "RecommendationSet",
    "load_knowledge_base",
    "recommend_for_profile",
]

KNOWLEDGE_PATH = Path(__file__).parent / "knowledge" / "interventions.yaml"

#: Ordering used when ranking. Grade D is not a recommendation at all and is
#: separated out before ranking happens.
GRADE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}

GRADE_DESCRIPTION = {
    "A": "Consistent evidence from multiple randomised trials or meta-analyses",
    "B": "Single good randomised trial, or consistent prospective observational evidence",
    "C": "Mechanistic, cross-sectional or expert-consensus evidence only",
    "D": "Insufficient or conflicting evidence",
}


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

@dataclass
class Recommendation:
    id: str
    title: str
    domain: str
    grade: str
    detail: str
    effect: str
    citation: str
    triggered_by: list[str] = field(default_factory=list)
    caution: str | None = None

    @property
    def is_evidence_gap(self) -> bool:
        return self.grade == "D"

    def grade_label(self) -> str:
        return f"Grade {self.grade} - {GRADE_DESCRIPTION.get(self.grade, '')}"


@dataclass
class RecommendationSet:
    recommendations: list[Recommendation] = field(default_factory=list)
    evidence_gaps: list[Recommendation] = field(default_factory=list)
    unassessed: list[str] = field(default_factory=list)
    profile_summary: dict[str, Any] = field(default_factory=dict)

    def by_domain(self, domain: str) -> list[Recommendation]:
        return [r for r in self.recommendations if r.domain == domain]

    def to_frame(self) -> pd.DataFrame:
        if not self.recommendations:
            return pd.DataFrame(columns=["Recommendation", "Domain", "Evidence", "Triggered by"])
        return pd.DataFrame(
            [
                {
                    "Recommendation": r.title,
                    "Domain": r.domain.replace("_", " ").title(),
                    "Evidence": f"Grade {r.grade}",
                    "Triggered by": "; ".join(r.triggered_by),
                }
                for r in self.recommendations
            ]
        )

    def headline(self) -> str:
        if not self.recommendations:
            return (
                "No recommendations were triggered. On the measurements supplied, this "
                "participant does not meet the threshold for any intervention in the "
                "evidence library."
            )
        n_a = sum(1 for r in self.recommendations if r.grade == "A")
        return (
            f"{len(self.recommendations)} evidence-based recommendation"
            f"{'s' if len(self.recommendations) != 1 else ''} triggered"
            + (f", {n_a} supported by grade A evidence." if n_a else ".")
        )


# --------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_knowledge_base(path: str | None = None) -> dict:
    """Load and cache the curated intervention library."""
    target = Path(path) if path else KNOWLEDGE_PATH
    if not target.exists():
        raise FileNotFoundError(f"Intervention knowledge base not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# --------------------------------------------------------------------------
# Trigger evaluation
# --------------------------------------------------------------------------

def _sex_key(profile: dict) -> str | None:
    """Sex is coded 0 = male, 1 = female."""
    value = profile.get("sex")
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        label = str(profile.get("sex_label", "")).strip().lower()
        return label if label in {"male", "female"} else None
    return "female" if float(value) == 1.0 else "male"


def _evaluate_trigger(trigger: dict, profile: dict) -> tuple[bool, str | None, str | None]:
    """Evaluate one trigger.

    Returns ``(fired, description, unassessed_reason)``. A trigger that cannot
    be evaluated because the measurement is absent does NOT fire - it is
    reported as unassessed, so a missing measurement never silently becomes a
    negative finding.
    """
    variable = trigger["variable"]
    raw = profile.get(variable)

    if raw is None or (isinstance(raw, (int, float)) and not np.isfinite(float(raw))):
        return False, None, f"{variable} was not measured"

    operator = trigger["operator"]
    rationale = trigger.get("rationale", "")

    if operator == "is_present":
        return True, f"{variable} available ({rationale})", None

    value = float(raw)

    if operator in {"below_sex_cutoff", "above_sex_cutoff"}:
        sex = _sex_key(profile)
        if sex is None:
            return False, None, f"{variable} needs sex to apply its cutoff, and sex is missing"
        cutoff = float(trigger["cutoffs"][sex])
        fired = value < cutoff if operator == "below_sex_cutoff" else value > cutoff
        direction = "below" if operator == "below_sex_cutoff" else "above"
        if fired:
            return True, f"{variable} {value:.4g} is {direction} the {sex} cutoff of {cutoff:.4g} ({rationale})", None
        return False, None, None

    threshold = float(trigger["value"])
    comparisons = {
        "<": value < threshold,
        "<=": value <= threshold,
        ">": value > threshold,
        ">=": value >= threshold,
        "==": value == threshold,
    }
    if operator not in comparisons:
        return False, None, f"unknown operator '{operator}' for {variable}"

    if comparisons[operator]:
        return True, f"{variable} {value:.4g} {operator} {threshold:.4g} ({rationale})", None
    return False, None, None


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def recommend_for_profile(
    profile: dict | pd.Series,
    domains: Sequence[str] | None = None,
    knowledge_path: str | None = None,
    settings: Settings | None = None,
) -> RecommendationSet:
    """Match a participant profile against the evidence library.

    ``profile`` is a mapping of measured values (grip, gait speed, SPPB, BMI,
    waist, VAT, ASMI, IL-6, B/T ratio, Fried score, sex). Absent measurements
    are reported as unassessed rather than treated as normal.
    """
    if isinstance(profile, pd.Series):
        profile = profile.to_dict()
    profile = {k: v for k, v in profile.items()}

    knowledge = load_knowledge_base(knowledge_path)
    fired: list[Recommendation] = []
    gaps: list[Recommendation] = []
    unassessed: set[str] = set()

    for entry in knowledge.get("interventions", []):
        if domains and entry["domain"] not in domains:
            continue

        reasons: list[str] = []
        for trigger in entry.get("triggers", []):
            triggered, description, missing = _evaluate_trigger(trigger, profile)
            if triggered and description:
                reasons.append(description)
            if missing:
                unassessed.add(missing)

        if not reasons:
            continue

        recommendation = Recommendation(
            id=entry["id"],
            title=entry["title"],
            domain=entry["domain"],
            grade=entry["grade"],
            detail=entry.get("detail", "").strip(),
            effect=entry.get("effect", "").strip(),
            citation=entry.get("citation", "").strip(),
            triggered_by=reasons,
            caution=(entry.get("caution") or "").strip() or None,
        )
        (gaps if recommendation.is_evidence_gap else fired).append(recommendation)

    fired.sort(key=lambda r: (GRADE_RANK.get(r.grade, 9), r.title))

    measured = {
        k: v for k, v in profile.items()
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v))
    }

    return RecommendationSet(
        recommendations=fired,
        evidence_gaps=gaps,
        unassessed=sorted(unassessed),
        profile_summary=measured,
    )


def profile_from_participant(
    wide_row: pd.Series, substudy_row: pd.Series | None = None
) -> dict:
    """Build a recommendation profile from a cleaned participant record."""
    profile: dict[str, Any] = {}

    mapping = {
        "sex": "sex",
        "sex_label": "sex_label",
        "age_years": "age_years",
        "bmi": "bmi",
        "waist_cm": "waist_cm",
        "fried_score": "fried_score",
        "grip_Baseline": "max_grip_kg",
        "gait_speed_Baseline": "gait_speed_ms",
        "sppb_Baseline": "sppb_total",
        "chair_Baseline": "chair_time_s",
        "asmi": "asmi",
        "vat_mass_g": "vat_mass_g",
        "fat_mass_pct": "fat_mass_pct",
    }
    for source, target in mapping.items():
        if source in wide_row.index and pd.notna(wide_row[source]):
            profile[target] = wide_row[source]

    if substudy_row is not None:
        for source, target in (
            ("IL6_pg_mL", "IL6_pg_mL"),
            ("sIL6R_ng_mL", "sIL6R_ng_mL"),
            ("sgp130_ng_mL", "sgp130_ng_mL"),
            ("BT_ratio", "BT_ratio"),
            ("FME", "FME"),
        ):
            if source in substudy_row.index and pd.notna(substudy_row[source]):
                profile[target] = substudy_row[source]

    return profile
