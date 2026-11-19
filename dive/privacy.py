"""The data boundary: what may leave this process, and what may never.

Start from the honest position. If Dive sends text to Claude or Gemini, that
provider processes the text. There is no configuration, contract or header that
makes a provider unable to see what you send them. So the only enforceable
guarantee is about **what is sent in the first place**, and that is what this
module controls.

Four egress modes, set by ``privacy.egress_mode``:

``topic_only`` (default)
    Only a **clinical question** leaves - the kind of thing a clinician would
    type into PubMed. No values, no profile, no record. The provider is asked
    what the evidence says about an intervention in a population; Dive then
    matches that evidence to the participant locally. The provider never learns
    that a specific person exists, let alone anything about them. See the note
    on blind computation below.

``none``
    Nothing derived from participant records leaves the process. External
    models can still be used for literature synthesis on generic clinical
    topics, but they receive no profile at all. Maximum privacy, least
    specificity.

``deidentified``
    A profile may leave, stripped to clinical values only: no identifiers, no
    dates, no free text, age in 5-year bands, measurements rounded. This is the
    shape of data routinely shared for research, and it is what the audit log
    will show byte for byte.

``full``
    Refused. The mode exists so that the configuration surface is explicit, and
    ``EgressGuard`` raises rather than honouring it. There is no legitimate
    reason to send raw clinical records to a general-purpose LLM API.

Enforcement is fail-closed at every step. A field that is not on the allowlist
is dropped, not passed through on the assumption it is harmless. A payload that
still matches an identifier pattern after redaction is blocked outright rather
than sent with a warning. If the scanner itself raises, the egress is blocked.

**On "let the provider analyse it without knowing what it is".** That is
computation on data the computer cannot read - fully homomorphic encryption, or
a hardware enclave the vendor cannot inspect. For a general-purpose LLM API it
does not exist: the model has to process plaintext to reason about it, and FHE
for transformer inference remains orders of magnitude too slow to be a product.
Any vendor claiming otherwise is describing a *retention* policy, not a
*processing* one.

What is achievable is to make the question carry no patient information in the
first place. That is what ``topic_only`` does, and it reaches the same goal by a
different route: the provider answers a literature question, Dive does the
participant-specific reasoning locally, and there is nothing to keep private
because nothing private was sent. For deployments that also need the model
weights kept off third-party hardware, run a local model - the only arrangement
where no data leaves your network at all.

What this module does **not** claim: that de-identified clinical values are
mathematically impossible to re-identify. A sufficiently unusual combination of
measurements, cross-referenced against an external dataset, is a real (if
remote) re-identification vector. ``topic_only``, ``none`` and a locally hosted
model each reduce that to zero, and all three are supported.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

__all__ = [
    "EgressMode",
    "EgressDecision",
    "EgressGuard",
    "PrivacyPolicy",
    "deidentify_profile",
    "build_topic_query",
    "FINDING_TOPICS",
    "scan_for_identifiers",
    "redact",
]

EgressMode = Literal["topic_only", "none", "deidentified", "full"]


# --------------------------------------------------------------------------
# Identifier detection
# --------------------------------------------------------------------------

#: Patterns that must never appear in an outbound payload. Written to
#: over-match rather than under-match: a false positive costs one blocked
#: request, a false negative costs a disclosure.
IDENTIFIER_PATTERNS: dict[str, re.Pattern] = {
    "nhs_number": re.compile(r"\b\d{3}[ -]?\d{3}[ -]?\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "uk_phone": re.compile(r"\b(?:\+44|0)\s?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b"),
    "uk_postcode": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE),
    "date_dmy": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    "date_iso": re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b"),
    "participant_id": re.compile(r"\b[A-Z]{2,4}[-_]?\d{3,6}\b"),
    "ni_number": re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b", re.IGNORECASE),
    "mrn": re.compile(r"\b(?:MRN|HOSP(?:ITAL)?[ _-]?(?:NO|NUM|ID))\W{0,3}\w+\b", re.IGNORECASE),
    "long_digit_run": re.compile(r"\b\d{9,}\b"),
}

#: Keys whose *value* is presumed identifying regardless of content.
IDENTIFYING_KEYS: frozenset[str] = frozenset({
    "participant_id", "pid", "id", "subject_id", "nhs_number", "mrn", "name",
    "first_name", "last_name", "surname", "forename", "dob", "date_of_birth",
    "address", "postcode", "email", "phone", "telephone", "ni_number",
    "date_of_visit", "visit_date", "date", "accession", "accession_number",
    "study_uid", "series_uid", "sop_uid", "patient_name", "patient_id",
    "referrer", "clinician", "consultant", "gp", "site", "hospital", "ward",
    "notes", "free_text", "comment", "comments", "history",
    # Quasi-identifiers. Ethnicity plus a region plus an unusual clinical
    # profile can single somebody out even with no name attached, and a rare
    # genotype is identifying on its own.
    "ethnicity_self_reported", "ethnicity", "ethnic_group", "race",
    "ethnicity_classification", "preferred_language", "migration_generation",
    "country_iso3", "country", "region_code", "region", "latitude_band",
    "urban_rural", "deprivation_decile", "imd",
    "genetic_ancestry_pc1", "genetic_ancestry_pc2", "genetic_ancestry_pc3",
    "actn3_r577x_dosage", "vdr_fokI_dosage", "col1a1_sp1_dosage",
    "apoe_e4_dosage", "fto_rs9939609_dosage",
})

#: Clinical value fields permitted to leave in ``deidentified`` mode. Anything
#: not named here is dropped. Adding to this list is a deliberate act.
#:
#: Note what is deliberately absent: ethnicity, country, region, deprivation and
#: any genotype. Individually they look innocuous; together with a clinical
#: profile they are among the strongest re-identification routes there is, and a
#: rare genotype is identifying by itself.
ALLOWED_CLINICAL_KEYS: frozenset[str] = frozenset({
    # demographics, coarsened
    "age_band", "sex",
    # physical function
    "max_grip_kg", "gait_speed_ms", "sppb_total", "sppb_balance", "sppb_chair",
    "sppb_gait", "chair_time_s", "tug_s", "tug_cannot_complete",
    # body composition
    "bmi", "waist_cm", "calf_cm", "fat_mass_pct", "asmi", "vat_mass_g",
    "lean_mass_kg", "bone_mineral_density", "bmd_t_score", "bmd_z_score",
    "trabecular_bone_score", "cortical_thickness_mm",
    # imaging-derived
    "ct_muscle_area_cm2", "ct_muscle_attenuation_hu", "ct_vat_area_cm2",
    "ct_sat_area_cm2", "ct_imat_area_cm2", "vertebral_ct_density_hu",
    # inflammation / biomarkers
    "IL6_pg_mL", "sIL6R_ng_mL", "sgp130_ng_mL", "BT_ratio", "FME",
    "crp_mg_L", "albumin_g_L", "haemoglobin_g_L", "serum_vitamin_d",
    # frailty, cognition, activity
    "fried_score", "moca", "ipaq_coded", "daily_step_count",
})


def scan_for_identifiers(payload: str) -> dict[str, list[str]]:
    """Return every identifier-like match found in ``payload``.

    Used as a last line of defence *after* structured de-identification, to
    catch identifiers that arrived inside an unexpected field.
    """
    hits: dict[str, list[str]] = {}
    for name, pattern in IDENTIFIER_PATTERNS.items():
        found = pattern.findall(payload)
        if found:
            # findall returns tuples when the pattern has groups.
            flattened = [m if isinstance(m, str) else "".join(m) for m in found]
            hits[name] = sorted(set(flattened))[:5]
    return hits


def redact(text: str, replacement: str = "[REDACTED]") -> str:
    """Replace every identifier-like span. For logs, not for egress.

    Egress is blocked on a hit rather than redacted - silently rewriting a
    payload that contained an identifier hides the fact that something
    upstream put it there.
    """
    for pattern in IDENTIFIER_PATTERNS.values():
        text = pattern.sub(replacement, text)
    return text


# --------------------------------------------------------------------------
# De-identification
# --------------------------------------------------------------------------

def age_band(age: float, width: int = 5, cap: int = 90) -> str:
    """Band an age. Ages at or above ``cap`` collapse into one open band.

    The cap matters: the oldest participants are the most identifiable, and a
    single 97-year-old in a cohort is effectively named by their age alone.
    """
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if value >= cap:
        return f"{cap}+"
    low = int(value // width * width)
    return f"{low}-{low + width - 1}"


def deidentify_profile(
    profile: dict[str, Any],
    allowed_keys: Iterable[str] | None = None,
    round_to: int = 2,
    age_band_width: int = 5,
) -> tuple[dict[str, Any], list[str]]:
    """Reduce a participant profile to permitted clinical values.

    Returns ``(clean_profile, dropped_keys)``. Everything not explicitly
    allowed is dropped and reported - the caller can then show the user exactly
    what was withheld.
    """
    allowed = frozenset(allowed_keys) if allowed_keys is not None else ALLOWED_CLINICAL_KEYS
    clean: dict[str, Any] = {}
    dropped: list[str] = []

    for key, value in profile.items():
        lowered = str(key).strip().lower()

        # Age becomes a band, never an exact value.
        if lowered in {"age", "age_years", "age_at_visit"}:
            if value is not None and value == value:
                clean["age_band"] = age_band(value, age_band_width)
            continue

        if lowered in IDENTIFYING_KEYS:
            dropped.append(str(key))
            continue

        if key not in allowed and lowered not in allowed:
            dropped.append(str(key))
            continue

        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, (int, float)):
            if value != value:      # NaN
                continue
            clean[key] = round(float(value), round_to)
        elif isinstance(value, str):
            # Only short, controlled vocabulary strings survive.
            text = value.strip()
            if len(text) <= 24 and not scan_for_identifiers(text):
                clean[key] = text
            else:
                dropped.append(str(key))
        else:
            dropped.append(str(key))

    return clean, sorted(set(dropped))


# --------------------------------------------------------------------------
# Topic-only queries
# --------------------------------------------------------------------------

#: Finding key -> the clinical topic it maps to. Values never appear: a topic
#: is a category, not a measurement, and two participants who cross the same
#: threshold produce the identical query.
FINDING_TOPICS: dict[str, str] = {
    "low_grip": "low grip strength and sarcopenia",
    "slow_gait": "slow gait speed and mobility limitation",
    "low_sppb": "impaired short physical performance battery score",
    "slow_chair": "impaired sit-to-stand performance",
    "frail": "physical frailty phenotype",
    "high_vat": "excess visceral adiposity",
    "low_asmi": "low appendicular lean mass",
    "obesity": "sarcopenic obesity",
    "high_il6": "elevated circulating interleukin-6",
    "bt_ratio": "interleukin-6 trans-signalling",
    "low_bmd": "low bone mineral density and osteoporosis",
    "high_bmat": "bone marrow adiposity",
    "low_cortical": "cortical bone loss and reduced bone strength",
}

#: Broad population descriptors. Deliberately coarse - a literature query does
#: not need a patient's age to be useful, and including one is the thin end of
#: sending a profile.
POPULATIONS: dict[str, str] = {
    "older": "older adults",
    "older_female": "older women",
    "older_male": "older men",
    "adult": "adults",
}


def build_topic_query(
    findings: Sequence[str], population: str = "older", max_topics: int = 4
) -> tuple[str, list[str]]:
    """Turn threshold crossings into a literature question carrying no data.

    Returns ``(query, unmapped_findings)``. The result is indistinguishable
    from what a clinician would type into PubMed, and is identical for every
    participant who crosses the same thresholds - which is precisely why it
    discloses nothing about any of them.
    """
    topics: list[str] = []
    unmapped: list[str] = []
    for finding in findings:
        topic = FINDING_TOPICS.get(finding)
        if topic is None:
            unmapped.append(finding)
        elif topic not in topics:
            topics.append(topic)

    if not topics:
        topics = ["physical function decline"]

    who = POPULATIONS.get(population, POPULATIONS["older"])
    subject = "; ".join(topics[:max_topics])
    query = (
        f"What does the published evidence show about interventions for {subject} "
        f"in {who}?"
    )
    return query, unmapped


# --------------------------------------------------------------------------
# Policy and guard
# --------------------------------------------------------------------------

@dataclass
class PrivacyPolicy:
    egress_mode: EgressMode = "topic_only"
    #: Minimum cohort size behind any aggregate statistic that leaves.
    min_aggregate_n: int = 5
    age_band_width: int = 5
    round_to: int = 2
    #: Refuse egress entirely when the destination is not on this list.
    allowed_destinations: tuple[str, ...] = ("anthropic", "google", "openai", "local")
    #: Write every egress decision to the audit log.
    audit: bool = True
    #: Hard kill switch. When True, no external call is permitted at all.
    offline_only: bool = False


@dataclass
class EgressDecision:
    allowed: bool
    payload: str = ""
    reasons: list[str] = field(default_factory=list)
    dropped_fields: list[str] = field(default_factory=list)
    destination: str = ""
    mode: EgressMode = "deidentified"
    payload_sha256: str = ""
    bytes_out: int = 0

    def explain(self) -> str:
        if self.allowed:
            return (
                f"Permitted: {self.bytes_out} bytes to {self.destination} in "
                f"'{self.mode}' mode"
                + (f"; {len(self.dropped_fields)} field(s) withheld" if self.dropped_fields else "")
            )
        return "Blocked: " + "; ".join(self.reasons)


class EgressGuard:
    """The single chokepoint every outbound model call must pass through.

    Nothing in Dive calls an external API directly. ``dive.providers`` routes
    every request through :meth:`review`, and a blocked decision means the call
    is never made - not made and discarded.
    """

    def __init__(self, policy: PrivacyPolicy | None = None, audit_path: Path | None = None):
        self.policy = policy or PrivacyPolicy()
        self.audit_path = audit_path

    # -- review -----------------------------------------------------------

    def review_profile(
        self, profile: dict[str, Any], destination: str
    ) -> tuple[dict[str, Any], EgressDecision]:
        """De-identify a participant profile and decide whether it may leave."""
        decision = EgressDecision(
            allowed=False, destination=destination, mode=self.policy.egress_mode
        )

        if self.policy.offline_only:
            decision.reasons.append("offline_only is set; no external call is permitted")
            self._audit(decision)
            return {}, decision

        if destination not in self.policy.allowed_destinations:
            decision.reasons.append(
                f"destination '{destination}' is not in the allowed list "
                f"{list(self.policy.allowed_destinations)}"
            )
            self._audit(decision)
            return {}, decision

        if self.policy.egress_mode == "full":
            decision.reasons.append(
                "egress_mode 'full' is refused by design: raw clinical records are never "
                "sent to a general-purpose model API"
            )
            self._audit(decision)
            return {}, decision

        if self.policy.egress_mode == "none":
            decision.reasons.append(
                "egress_mode 'none': no participant-derived data may leave this process"
            )
            self._audit(decision)
            return {}, decision

        if self.policy.egress_mode == "topic_only":
            decision.reasons.append(
                "egress_mode 'topic_only': no profile leaves this process. Only a generic "
                "clinical question is sent, and the participant-specific reasoning is done "
                "locally against the evidence that comes back."
            )
            self._audit(decision)
            return {}, decision

        clean, dropped = deidentify_profile(
            profile, round_to=self.policy.round_to, age_band_width=self.policy.age_band_width
        )
        decision.dropped_fields = dropped

        # Fail-closed re-scan of the de-identified result.
        try:
            residual = scan_for_identifiers(json.dumps(clean, default=str))
        except Exception as exc:
            decision.reasons.append(f"identifier scan failed, blocking ({exc})")
            self._audit(decision)
            return {}, decision

        if residual:
            decision.reasons.append(
                "identifier-like content survived de-identification: "
                + ", ".join(sorted(residual))
            )
            self._audit(decision)
            return {}, decision

        if not clean:
            decision.reasons.append("nothing remained after de-identification")
            self._audit(decision)
            return {}, decision

        decision.allowed = True
        decision.payload = json.dumps(clean, sort_keys=True)
        decision.payload_sha256 = hashlib.sha256(decision.payload.encode()).hexdigest()
        decision.bytes_out = len(decision.payload.encode())
        self._audit(decision)
        return clean, decision

    def review_text(self, text: str, destination: str) -> EgressDecision:
        """Check free text (a prompt, a search query) before it leaves."""
        decision = EgressDecision(
            allowed=False, destination=destination, mode=self.policy.egress_mode
        )

        if self.policy.offline_only:
            decision.reasons.append("offline_only is set; no external call is permitted")
        elif destination not in self.policy.allowed_destinations:
            decision.reasons.append(f"destination '{destination}' is not allowed")
        else:
            try:
                hits = scan_for_identifiers(text)
            except Exception as exc:
                decision.reasons.append(f"identifier scan failed, blocking ({exc})")
                self._audit(decision)
                return decision
            if hits:
                decision.reasons.append(
                    "outbound text contains identifier-like content: "
                    + ", ".join(f"{k} ({len(v)})" for k, v in sorted(hits.items()))
                )
            else:
                decision.allowed = True
                decision.payload_sha256 = hashlib.sha256(text.encode()).hexdigest()
                decision.bytes_out = len(text.encode())

        self._audit(decision)
        return decision

    def check_aggregate(self, n: int) -> tuple[bool, str]:
        """Small-cell suppression for any aggregate statistic that leaves."""
        if n < self.policy.min_aggregate_n:
            return False, (
                f"aggregate covers only {n} participants; the minimum for disclosure is "
                f"{self.policy.min_aggregate_n} (small-cell suppression)"
            )
        return True, ""

    # -- audit ------------------------------------------------------------

    def _audit(self, decision: EgressDecision) -> None:
        if not self.policy.audit or self.audit_path is None:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "egress",
            "destination": decision.destination,
            "mode": decision.mode,
            "allowed": decision.allowed,
            "bytes_out": decision.bytes_out,
            "payload_sha256": decision.payload_sha256,
            "dropped_fields": decision.dropped_fields,
            "reasons": decision.reasons,
        }
        try:
            from .security import AuditLog

            AuditLog(self.audit_path).append(entry)
        except Exception:
            # Auditing must never crash the caller, but a failed write is
            # itself notable - surface it on the decision.
            decision.reasons.append("audit write failed")
