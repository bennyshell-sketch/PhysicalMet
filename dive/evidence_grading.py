"""Formal evidence appraisal: OCEBM, GRADE, RoB 2 and AMSTAR-2.

A citation is not evidence. A randomised trial with an unconcealed allocation
sequence and 40% attrition is a citation; what it supports is a different
question, and the published appraisal frameworks exist to answer it.

The distinction this module is built around:

**Derivable.** The OCEBM Level of Evidence follows mechanically from study
design - a systematic review of randomised trials is Level 1 whatever it
concludes. Dive derives that automatically from the publication type recorded
against the retrieved study, and shows its reasoning.

**Not derivable.** GRADE certainty, Cochrane RoB 2 and AMSTAR-2 require someone
to read the full paper and make judgements: was allocation concealed, is the
effect estimate precise enough, were outcomes selectively reported. These
cannot be inferred from an abstract, and a language model asked to produce them
will produce something plausible and unfounded - the same failure as a
fabricated citation, wearing a rating.

So Dive stores them as **explicitly unassessed** until a human enters them, and
an unassessed rating never displays as a favourable one. ``appraise`` returns
what is known and names what is not, which is the honest state of most
literature most of the time.

References for the frameworks implemented here:
* OCEBM Levels of Evidence Working Group. The Oxford 2011 Levels of Evidence.
* Guyatt GH, et al. GRADE guidelines: 1. Introduction. J Clin Epidemiol 2011;64(4):383-94.
* Sterne JAC, et al. RoB 2: a revised tool for assessing risk of bias in
  randomised trials. BMJ 2019;366:l4898.
* Shea BJ, et al. AMSTAR 2: a critical appraisal tool for systematic reviews.
  BMJ 2017;358:j4008.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

__all__ = [
    "ocebm_level",
    "GRADEAssessment",
    "RoB2Assessment",
    "AMSTAR2Assessment",
    "StudyAppraisal",
    "appraise",
    "GRADE_DOWNGRADE_DOMAINS",
    "GRADE_UPGRADE_DOMAINS",
    "ROB2_DOMAINS",
    "AMSTAR2_CRITICAL_ITEMS",
]

Certainty = Literal["high", "moderate", "low", "very low", "not assessed"]
RiskJudgement = Literal["low", "some concerns", "high", "not assessed"]


# --------------------------------------------------------------------------
# OCEBM - derivable from design
# --------------------------------------------------------------------------

#: Oxford 2011 Levels of Evidence for treatment benefits. Level 1 is the
#: strongest. Mapped from the publication types the literature APIs return.
_OCEBM_BY_TYPE: list[tuple[frozenset[str], int, str]] = [
    (frozenset({"meta-analysis", "systematic review"}), 1,
     "systematic review or meta-analysis of trials"),
    (frozenset({"randomized controlled trial", "randomised controlled trial"}), 2,
     "randomised controlled trial"),
    (frozenset({"clinical trial", "controlled clinical trial", "multicenter study"}), 3,
     "non-randomised controlled study"),
    (frozenset({"cohort studies", "observational study", "prospective studies"}), 3,
     "controlled cohort or follow-up study"),
    (frozenset({"case-control studies", "cross-sectional studies", "case reports"}), 4,
     "case-control, cross-sectional or case series"),
    (frozenset({"review", "editorial", "comment", "letter"}), 5,
     "narrative review or commentary - mechanism-based reasoning at best"),
]

OCEBM_DESCRIPTIONS: dict[int, str] = {
    1: "Level 1 - systematic review of randomised trials",
    2: "Level 2 - randomised trial, or observational study with a dramatic effect",
    3: "Level 3 - non-randomised controlled cohort or follow-up study",
    4: "Level 4 - case series, case-control, or historically controlled study",
    5: "Level 5 - mechanism-based reasoning",
}


def ocebm_level(pub_types: Iterable[str]) -> tuple[int, str]:
    """Derive the OCEBM level from publication type.

    Returns ``(level, reasoning)``. The strongest matching design wins: a paper
    tagged both "systematic review" and "review" is a systematic review.
    """
    normalised = {str(t).strip().lower() for t in pub_types}
    if not normalised:
        return 5, "no publication type recorded, so no design can be established"

    for types, level, description in _OCEBM_BY_TYPE:
        if normalised & types:
            return level, f"{description} ({', '.join(sorted(normalised & types))})"

    return 5, f"publication type(s) {', '.join(sorted(normalised))} map to no controlled design"


# --------------------------------------------------------------------------
# GRADE - requires appraisal
# --------------------------------------------------------------------------

#: The five reasons to rate certainty down.
GRADE_DOWNGRADE_DOMAINS: tuple[str, ...] = (
    "risk_of_bias",
    "inconsistency",
    "indirectness",
    "imprecision",
    "publication_bias",
)

#: The three reasons observational evidence may be rated up.
GRADE_UPGRADE_DOMAINS: tuple[str, ...] = (
    "large_effect",
    "dose_response",
    "plausible_confounding_would_reduce_effect",
)

_CERTAINTY_ORDER: list[Certainty] = ["very low", "low", "moderate", "high"]


@dataclass
class GRADEAssessment:
    """GRADE certainty for a body of evidence about one outcome.

    Starts High for randomised evidence and Low for observational, then moves
    on the recorded domains. Nothing is assumed: an unrecorded domain does not
    count as "no concern", it counts as unassessed, and certainty stays
    ``not assessed`` until someone has actually looked.
    """

    outcome: str = ""
    design: Literal["randomised", "observational", "unknown"] = "unknown"
    #: domain -> 0 (no concern), -1 (serious), -2 (very serious)
    downgrades: dict[str, int] = field(default_factory=dict)
    #: domain -> 0 or +1
    upgrades: dict[str, int] = field(default_factory=dict)
    assessed_by: str = ""
    assessed_utc: str = ""
    notes: str = ""

    @property
    def assessed(self) -> bool:
        return bool(self.assessed_by and self.downgrades)

    def certainty(self) -> Certainty:
        if not self.assessed:
            return "not assessed"
        if self.design == "randomised":
            index = _CERTAINTY_ORDER.index("high")
        elif self.design == "observational":
            index = _CERTAINTY_ORDER.index("low")
        else:
            return "not assessed"

        index += sum(self.downgrades.get(d, 0) for d in GRADE_DOWNGRADE_DOMAINS)
        if self.design == "observational":
            index += sum(self.upgrades.get(d, 0) for d in GRADE_UPGRADE_DOMAINS)

        return _CERTAINTY_ORDER[max(0, min(index, len(_CERTAINTY_ORDER) - 1))]

    def explain(self) -> str:
        if not self.assessed:
            return (
                "GRADE certainty has not been assessed. It requires reading the full "
                "text and judging risk of bias, inconsistency, indirectness, imprecision "
                "and publication bias - it cannot be inferred from an abstract."
            )
        concerns = [
            f"{d.replace('_', ' ')} ({'very serious' if v <= -2 else 'serious'})"
            for d, v in self.downgrades.items() if v < 0
        ]
        raised = [d.replace("_", " ") for d, v in self.upgrades.items() if v > 0]
        parts = [f"Starting from {self.design} evidence, certainty is {self.certainty().upper()}"]
        if concerns:
            parts.append("rated down for " + ", ".join(concerns))
        if raised:
            parts.append("rated up for " + ", ".join(raised))
        return ". ".join(parts) + "."


# --------------------------------------------------------------------------
# Cochrane RoB 2 - requires appraisal
# --------------------------------------------------------------------------

ROB2_DOMAINS: dict[str, str] = {
    "randomisation": "Bias arising from the randomisation process",
    "deviations": "Bias due to deviations from intended interventions",
    "missing_data": "Bias due to missing outcome data",
    "outcome_measurement": "Bias in measurement of the outcome",
    "selective_reporting": "Bias in selection of the reported result",
}


@dataclass
class RoB2Assessment:
    """Risk of bias in a randomised trial, per domain."""

    judgements: dict[str, RiskJudgement] = field(default_factory=dict)
    assessed_by: str = ""
    assessed_utc: str = ""
    notes: str = ""

    @property
    def assessed(self) -> bool:
        return bool(self.assessed_by) and set(self.judgements) >= set(ROB2_DOMAINS)

    def overall(self) -> RiskJudgement:
        """RoB 2 overall judgement: the worst domain drives the result."""
        if not self.assessed:
            return "not assessed"
        values = [self.judgements.get(d, "not assessed") for d in ROB2_DOMAINS]
        if any(v == "high" for v in values):
            return "high"
        if any(v == "some concerns" for v in values):
            return "some concerns"
        return "low"

    def explain(self) -> str:
        if not self.assessed:
            missing = sorted(set(ROB2_DOMAINS) - set(self.judgements))
            return (
                "Risk of bias has not been fully assessed"
                + (f"; unassessed domains: {', '.join(missing)}" if missing else "")
                + "."
            )
        worst = [d for d, v in self.judgements.items() if v == self.overall()]
        return (
            f"Overall risk of bias is {self.overall().upper()}, driven by "
            + ", ".join(ROB2_DOMAINS[d].lower() for d in worst if d in ROB2_DOMAINS)
            + "."
        )


# --------------------------------------------------------------------------
# AMSTAR-2 - requires appraisal
# --------------------------------------------------------------------------

#: The seven items AMSTAR-2 designates critical. A flaw in any of these caps
#: overall confidence regardless of how the other nine score.
AMSTAR2_CRITICAL_ITEMS: tuple[int, ...] = (2, 4, 7, 9, 11, 13, 15)

AMSTAR2_ITEM_LABELS: dict[int, str] = {
    2: "Protocol registered before commencement",
    4: "Adequacy of the literature search",
    7: "Justification for excluding individual studies",
    9: "Risk of bias assessment of included studies",
    11: "Appropriateness of meta-analytical methods",
    13: "Consideration of risk of bias when interpreting results",
    15: "Assessment of publication bias",
}


@dataclass
class AMSTAR2Assessment:
    """Confidence in the results of a systematic review."""

    #: item number -> True (yes / partial yes) or False (no)
    items: dict[int, bool] = field(default_factory=dict)
    assessed_by: str = ""
    assessed_utc: str = ""
    notes: str = ""

    @property
    def assessed(self) -> bool:
        return bool(self.assessed_by) and set(AMSTAR2_CRITICAL_ITEMS) <= set(self.items)

    def critical_flaws(self) -> list[int]:
        return sorted(i for i in AMSTAR2_CRITICAL_ITEMS if self.items.get(i) is False)

    def non_critical_weaknesses(self) -> list[int]:
        return sorted(
            i for i, ok in self.items.items()
            if ok is False and i not in AMSTAR2_CRITICAL_ITEMS
        )

    def confidence(self) -> Literal["high", "moderate", "low", "critically low", "not assessed"]:
        if not self.assessed:
            return "not assessed"
        critical = len(self.critical_flaws())
        weaknesses = len(self.non_critical_weaknesses())
        if critical > 1:
            return "critically low"
        if critical == 1:
            return "low"
        if weaknesses > 1:
            return "moderate"
        return "high"

    def explain(self) -> str:
        if not self.assessed:
            return (
                "AMSTAR-2 confidence has not been assessed; the seven critical items "
                "require reading the review in full."
            )
        flaws = self.critical_flaws()
        if flaws:
            named = "; ".join(AMSTAR2_ITEM_LABELS.get(i, f"item {i}") for i in flaws)
            return (
                f"Confidence in this review is {self.confidence().upper()} - critical "
                f"flaw(s) in: {named}."
            )
        return (
            f"Confidence in this review is {self.confidence().upper()} with no critical "
            f"flaws and {len(self.non_critical_weaknesses())} non-critical weakness(es)."
        )


# --------------------------------------------------------------------------
# Combined appraisal
# --------------------------------------------------------------------------

@dataclass
class StudyAppraisal:
    identifier: str
    level: int
    level_label: str
    level_reasoning: str
    grade: GRADEAssessment = field(default_factory=GRADEAssessment)
    rob2: RoB2Assessment = field(default_factory=RoB2Assessment)
    amstar2: AMSTAR2Assessment = field(default_factory=AMSTAR2Assessment)

    @property
    def fully_appraised(self) -> bool:
        return self.grade.assessed and (self.rob2.assessed or self.amstar2.assessed)

    def headline(self) -> str:
        parts = [self.level_label]
        if self.grade.assessed:
            parts.append(f"GRADE {self.grade.certainty()}")
        if self.rob2.assessed:
            parts.append(f"RoB 2 {self.rob2.overall()}")
        if self.amstar2.assessed:
            parts.append(f"AMSTAR-2 {self.amstar2.confidence()}")
        return " | ".join(parts)

    def outstanding(self) -> list[str]:
        """Appraisals a human still needs to do."""
        missing: list[str] = []
        if not self.grade.assessed:
            missing.append("GRADE certainty")
        if self.level <= 2 and not self.rob2.assessed:
            missing.append("Cochrane RoB 2 (randomised evidence)")
        if self.level == 1 and not self.amstar2.assessed:
            missing.append("AMSTAR-2 (systematic review)")
        return missing


def appraise(
    identifier: str,
    pub_types: Iterable[str],
    grade: GRADEAssessment | None = None,
    rob2: RoB2Assessment | None = None,
    amstar2: AMSTAR2Assessment | None = None,
) -> StudyAppraisal:
    """Derive what design allows, and record what still requires judgement."""
    level, reasoning = ocebm_level(pub_types)
    return StudyAppraisal(
        identifier=identifier,
        level=level,
        level_label=OCEBM_DESCRIPTIONS[level],
        level_reasoning=reasoning,
        grade=grade or GRADEAssessment(),
        rob2=rob2 or RoB2Assessment(),
        amstar2=amstar2 or AMSTAR2Assessment(),
    )
