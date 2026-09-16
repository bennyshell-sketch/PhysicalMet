"""Participant-specific recommendations grounded in retrieved literature.

This module uses the Claude API to do one job: read evidence that has already
been *retrieved* and work out what it means for one specific person, given
their measured values. It is deliberately not allowed to do two other things.

**It does not supply citations.** Every study it may cite is retrieved first by
``dive.literature`` and passed in. Afterwards, every identifier it returns is
checked against that retrieved set and anything else is discarded. A model
asked to recall references produces ones that look flawless and do not exist;
the only reliable defence is to make the model incapable of introducing a
reference the code did not already hold.

**It does not supply facts about the participant.** Every measurement comes
from the pipeline and is passed in as a fixed block. The model reasons about
those numbers; it never produces one. It is also told not to forecast - the
decline and trajectory models own prediction, including the decision to abstain,
and an advisor that volunteered a prognosis would be bypassing that.

What it adds over the curated library in ``dive.recommend`` is specificity.
The curated library fires on thresholds and returns the same wording for anyone
who crosses them. This returns reasoning about *this* combination of values,
and it says when the retrieved evidence does not cover the person in front of
it - an older adult with severe frailty is not well represented by a trial that
recruited robust 65-year-olds, and it is expected to say so.

**It never sees an identifier.** The profile is de-identified by
``dive.privacy`` before it reaches the prompt - no participant ID, no dates, no
name, age in 5-year bands - and the resulting text is scanned again before any
provider is contacted. The interface shows exactly which fields were withheld.
Setting ``egress_mode="none"`` disables profile egress entirely; pointing the
router at a local model removes third-party processing altogether.

Requires a provider key (``ANTHROPIC_API_KEY`` or ``GEMINI_API_KEY``, or a
local endpoint) and network access to Europe PMC or PubMed. Without either, it
reports itself unavailable and Dive falls back to the curated library rather
than guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import json
from typing import Any, Sequence

from .config import SETTINGS, Settings
from .literature import LiteratureClient, LiteratureResult, Study
from .privacy import EgressGuard, PrivacyPolicy
from .providers import ProviderRouter

__all__ = ["ParticipantBrief", "AdvisorRecommendation", "AdvisorResult", "Advisor"]

MODEL = "claude-opus-5"

#: Search terms attached to each finding, used to retrieve relevant evidence.
FINDING_SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "low_grip": ("grip strength", "sarcopenia", "resistance training", "older adults"),
    "slow_gait": ("gait speed", "mobility limitation", "exercise", "older adults"),
    "low_sppb": ("short physical performance battery", "physical function", "exercise"),
    "slow_chair": ("chair stand", "lower limb power", "older adults"),
    "frail": ("frailty", "frail older adults", "multicomponent exercise"),
    "high_vat": ("visceral adipose tissue", "exercise", "older adults"),
    "low_asmi": ("appendicular lean mass", "sarcopenia", "protein supplementation"),
    "obesity": ("sarcopenic obesity", "weight loss", "older adults", "lean mass"),
    "high_il6": ("interleukin-6", "inflammation", "exercise training", "older adults"),
    "bt_ratio": ("IL-6 trans-signalling", "soluble gp130", "soluble IL-6 receptor"),
}


# --------------------------------------------------------------------------
# Inputs and outputs
# --------------------------------------------------------------------------

@dataclass
class ParticipantBrief:
    """The fixed facts about one participant. Assembled from the pipeline only."""

    measurements: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)          # keys of FINDING_SEARCH_TERMS
    finding_details: list[str] = field(default_factory=list)   # human-readable trigger text
    age_years: float | None = None
    sex_label: str | None = None
    model_notes: list[str] = field(default_factory=list)       # incl. any abstentions
    unassessed: list[str] = field(default_factory=list)

    def search_terms(self) -> list[str]:
        terms: list[str] = []
        for finding in self.findings:
            terms.extend(FINDING_SEARCH_TERMS.get(finding, ()))
        seen: set[str] = set()
        ordered = [t for t in terms if not (t in seen or seen.add(t))]
        return ordered or ["physical function", "older adults", "exercise"]

    def as_prompt_block(self, safe_profile: dict[str, Any] | None = None) -> str:
        """Render the participant block.

        ``safe_profile`` is the de-identified profile returned by the egress
        guard. When supplied it is the ONLY source of values - the raw
        measurements never reach the prompt.
        """
        source = safe_profile if safe_profile is not None else self.measurements
        lines = ["PARTICIPANT DATA (de-identified measured values - treat as authoritative)"]
        for name, value in sorted(source.items()):
            unit = self.units.get(name, "")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f"- {name}: {value:g}{(' ' + unit) if unit else ''}")
            else:
                lines.append(f"- {name}: {value}")

        if self.finding_details:
            lines.append("")
            lines.append("THRESHOLDS THIS PARTICIPANT CROSSES")
            lines.extend(f"- {d}" for d in self.finding_details)

        if self.unassessed:
            lines.append("")
            lines.append("NOT MEASURED (do not assume these are normal)")
            lines.extend(f"- {u}" for u in self.unassessed)

        if self.model_notes:
            lines.append("")
            lines.append("WHAT DIVE'S OWN PREDICTION MODELS REPORT")
            lines.extend(f"- {n}" for n in self.model_notes)
        return "\n".join(lines)


@dataclass
class AdvisorRecommendation:
    title: str
    rationale: str
    specific_to_participant: str
    evidence_strength: str
    applicability_caveat: str
    citation_ids: list[str] = field(default_factory=list)
    studies: list[Study] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)


@dataclass
class AdvisorResult:
    recommendations: list[AdvisorRecommendation] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    available: bool = True
    reason: str = ""
    literature: LiteratureResult | None = None
    n_studies_retrieved: int = 0
    n_citations_dropped: int = 0
    model: str = MODEL
    provider: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    #: Fields removed before anything left this process.
    withheld_fields: list[str] = field(default_factory=list)
    blocked_by_guard: bool = False

    def headline(self) -> str:
        if not self.available:
            return f"Literature-grounded advice unavailable - {self.reason}"
        if not self.recommendations:
            return (
                "No recommendation could be grounded in the retrieved evidence for this "
                "participant's profile."
            )
        return (
            f"{len(self.recommendations)} recommendation"
            f"{'s' if len(self.recommendations) != 1 else ''} drawn from "
            f"{self.n_studies_retrieved} retrieved studies, every citation verified."
        )


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the evidence advisor inside Dive, a research tool used by a PhD \
researcher studying physical function decline in older adults.

Your job: read the retrieved studies supplied below and work out what they imply \
for the ONE specific participant described, given their measured values.

Absolute rules:

1. CITATIONS. You may cite ONLY the studies in the RETRIEVED EVIDENCE block, using \
their exact bracketed identifiers (for example PMID:12345678). You must not cite, \
mention, or allude to any other paper, author, trial name, or guideline - not even \
one you are confident exists. If the retrieved evidence does not support a point, \
leave the point out. Every recommendation must carry at least one identifier from \
the block.

2. PARTICIPANT FACTS. Every number you state about the participant must appear in \
the PARTICIPANT DATA block. Do not estimate, infer, convert, or invent a \
measurement. If something was not measured, treat it as unknown - never as normal.

3. NO PREDICTION. Do not forecast this person's outcome, risk, or trajectory. \
Dive's statistical models own prediction, and they abstain when they cannot \
support an estimate. Recommend actions and explain the evidence; do not \
prognosticate.

4. SPECIFICITY. Generic advice is not useful here. Tie each recommendation to \
this participant's actual values, and state plainly where the retrieved trials \
enrolled people unlike them - age, baseline function, frailty, comorbidity - so \
the researcher can judge transferability. This applicability caveat is required, \
not optional.

5. HONEST GAPS. If the retrieved evidence does not cover something that matters \
for this participant, record it in evidence_gaps. An accurate "the evidence does \
not address this" is more valuable than a confident extrapolation.

Write for a researcher: precise, plain, no hedging filler and no marketing tone."""


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short imperative name for the intervention.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "What the retrieved evidence shows, and its magnitude.",
                    },
                    "specific_to_participant": {
                        "type": "string",
                        "description": (
                            "Why this applies to THIS participant, referencing their "
                            "measured values from the PARTICIPANT DATA block."
                        ),
                    },
                    "evidence_strength": {
                        "type": "string",
                        "enum": ["strong", "moderate", "limited", "insufficient"],
                    },
                    "applicability_caveat": {
                        "type": "string",
                        "description": (
                            "How the trial populations differ from this participant."
                        ),
                    },
                    "citation_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Identifiers taken verbatim from RETRIEVED EVIDENCE.",
                    },
                },
                "required": [
                    "title", "rationale", "specific_to_participant",
                    "evidence_strength", "applicability_caveat", "citation_ids",
                ],
                "additionalProperties": False,
            },
        },
        "evidence_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Questions this participant raises that the evidence does not answer.",
        },
    },
    "required": ["recommendations", "evidence_gaps"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Advisor
# --------------------------------------------------------------------------

class Advisor:
    """Retrieval-grounded, participant-specific evidence synthesis."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str = MODEL,
        literature_client: LiteratureClient | None = None,
        max_studies: int = 12,
        min_year: int = 2005,
        router: ProviderRouter | None = None,
        guard: EgressGuard | None = None,
    ):
        self.cfg = settings or SETTINGS
        self.model = model
        self.literature = literature_client or LiteratureClient()
        self.max_studies = max_studies
        self.min_year = min_year
        self.guard = guard or EgressGuard(
            self.cfg.privacy, audit_path=self.cfg.paths.data_dir / "audit.jsonl"
        )
        self.router = router or ProviderRouter(guard=self.guard, settings=self.cfg)

    # -- availability -----------------------------------------------------

    @staticmethod
    def client_available() -> tuple[bool, str]:
        """Is at least one model provider usable right now?"""
        from .providers import available_providers

        described = available_providers()
        usable = [d for d in described if d["available"]]
        if usable:
            return True, ""
        return False, "; ".join(f"{d['provider']}: {d['reason']}" for d in described)

    # -- main -------------------------------------------------------------

    def advise(
        self,
        brief: ParticipantBrief,
        extra_terms: Sequence[str] = (),
        max_tokens: int = 16000,
    ) -> AdvisorResult:
        """Retrieve evidence, then reason about it for this participant."""
        ok, reason = self.client_available()
        if not ok:
            return AdvisorResult(available=False, reason=reason)

        terms = list(brief.search_terms()) + list(extra_terms)
        retrieved = self.literature.search(
            terms[:6], max_results=self.max_studies, min_year=self.min_year
        )

        if not retrieved.available or not retrieved.studies:
            return AdvisorResult(
                available=False,
                reason=(
                    "no studies could be retrieved, so there is nothing to ground advice in "
                    f"({retrieved.reason or 'no matching records'}). Dive will not generate "
                    "recommendations from recall."
                ),
                literature=retrieved,
            )

        # --- privacy boundary: de-identify before the profile is written into
        # any prompt. `withheld` is surfaced so the user sees what was removed.
        safe_profile, decision = self.guard.review_profile(
            brief.measurements | (
                {"age_years": brief.age_years} if brief.age_years is not None else {}
            ) | ({"sex": brief.sex_label} if brief.sex_label else {}),
            destination=self.router.order[0] if self.router.order else "anthropic",
        )
        if not decision.allowed:
            return AdvisorResult(
                available=False,
                reason=(
                    "the privacy guard did not permit this profile to leave the process: "
                    + "; ".join(decision.reasons)
                ),
                literature=retrieved,
                blocked_by_guard=True,
                withheld_fields=decision.dropped_fields,
            )

        evidence_block = "\n\n".join(s.as_prompt_record() for s in retrieved.studies)
        user_content = (
            f"{brief.as_prompt_block(safe_profile)}\n\n"
            f"RETRIEVED EVIDENCE ({len(retrieved.studies)} studies - the only sources you "
            f"may cite)\n\n{evidence_block}\n\n"
            "Produce recommendations for this participant, grounded only in the studies above."
        )

        response = self.router.complete(
            system=SYSTEM_PROMPT, user=user_content,
            schema=RESPONSE_SCHEMA, max_tokens=max_tokens,
        )
        if not response.ok:
            return AdvisorResult(
                available=False,
                reason=response.reason,
                literature=retrieved,
                blocked_by_guard=response.blocked_by_guard,
                withheld_fields=decision.dropped_fields,
            )

        payload = response.parsed if response.parsed is not None else _safe_json(response.text)
        if payload is None:
            return AdvisorResult(
                available=False,
                reason="the model's response was not valid JSON",
                literature=retrieved,
                withheld_fields=decision.dropped_fields,
            )

        result = self._verify(payload, retrieved, response)
        result.withheld_fields = decision.dropped_fields
        return result

    # -- verification -----------------------------------------------------

    def _verify(
        self, payload: dict, retrieved: LiteratureResult, response: Any
    ) -> AdvisorResult:
        """Drop anything not traceable to a retrieved study."""
        lookup = retrieved.by_identifier()
        recommendations: list[AdvisorRecommendation] = []
        dropped_total = 0

        for item in payload.get("recommendations", []):
            verified, unverifiable = LiteratureClient.verify_identifiers(
                item.get("citation_ids", []), retrieved
            )
            dropped_total += len(unverifiable)

            # A recommendation with no verifiable source is not a recommendation.
            if not verified:
                continue

            recommendations.append(
                AdvisorRecommendation(
                    title=str(item.get("title", "")).strip(),
                    rationale=str(item.get("rationale", "")).strip(),
                    specific_to_participant=str(item.get("specific_to_participant", "")).strip(),
                    evidence_strength=str(item.get("evidence_strength", "limited")),
                    applicability_caveat=str(item.get("applicability_caveat", "")).strip(),
                    citation_ids=verified,
                    studies=[lookup[v] for v in verified if v in lookup],
                    dropped_citations=unverifiable,
                )
            )

        return AdvisorResult(
            recommendations=recommendations,
            evidence_gaps=[str(g) for g in payload.get("evidence_gaps", [])],
            available=True,
            literature=retrieved,
            n_studies_retrieved=len(retrieved.studies),
            n_citations_dropped=dropped_total,
            model=getattr(response, "model", self.model),
            provider=getattr(response, "provider", ""),
            usage=getattr(response, "usage", {}) or {},
        )


# --------------------------------------------------------------------------
# Brief construction
# --------------------------------------------------------------------------

def brief_from_profile(
    profile: dict,
    recommendation_set: Any = None,
    model_notes: Sequence[str] = (),
) -> ParticipantBrief:
    """Build a ParticipantBrief from a measured profile and curated triggers.

    The findings drive literature retrieval, so they come from the same
    threshold logic the curated library uses - not from anything generated.
    """
    from .recommend import recommend_for_profile

    result = recommendation_set or recommend_for_profile(profile)

    #: Curated intervention id -> advisor finding key.
    finding_for = {
        "prt_strength": "low_grip",
        "multicomponent_falls": "frail",
        "chair_stand_focus": "slow_chair",
        "protein_intake": "low_grip",
        "vitamin_d": "frail",
        "aerobic_vat": "high_vat",
        "combined_training_lean_mass": "low_asmi",
        "exercise_il6": "high_il6",
        "adiposity_il6": "high_il6",
        "bt_ratio_direct": "bt_ratio",
    }

    findings: list[str] = []
    details: list[str] = []
    for item in list(result.recommendations) + list(result.evidence_gaps):
        key = finding_for.get(item.id)
        if key and key not in findings:
            findings.append(key)
        details.extend(item.triggered_by)

    units = {
        "max_grip_kg": "kg", "gait_speed_ms": "m/s", "sppb_total": "of 12",
        "chair_time_s": "s", "tug_s": "s", "bmi": "kg/m2", "waist_cm": "cm",
        "vat_mass_g": "g", "asmi": "kg/m2", "IL6_pg_mL": "pg/mL",
        "sIL6R_ng_mL": "ng/mL", "sgp130_ng_mL": "ng/mL", "fried_score": "of 5",
    }
    measurements = {
        k: float(v) for k, v in profile.items()
        if isinstance(v, (int, float)) and k not in {"sex", "age_years"} and v == v
    }

    return ParticipantBrief(
        measurements=measurements,
        units=units,
        findings=findings,
        finding_details=sorted(set(details)),
        age_years=profile.get("age_years"),
        sex_label=profile.get("sex_label")
        or ({1.0: "female", 0.0: "male"}.get(profile.get("sex"))),
        model_notes=list(model_notes),
        unassessed=list(getattr(result, "unassessed", [])),
    )



def _safe_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
