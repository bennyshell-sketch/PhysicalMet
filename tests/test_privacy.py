"""Nothing identifying may leave the process. These tests are the contract."""

import json

import pytest

from dive.privacy import (
    ALLOWED_CLINICAL_KEYS,
    EgressGuard,
    PrivacyPolicy,
    age_band,
    deidentify_profile,
    redact,
    scan_for_identifiers,
)

IDENTIFIED = {
    "participant_id": "WL042",
    "name": "Jane Smith",
    "dob": "1947-03-12",
    "nhs_number": "943 476 5919",
    "email": "jane.smith@example.nhs.uk",
    "postcode": "LE11 3TU",
    "date_of_visit": "2024-06-01",
    "notes": "Lives alone at 12 Elm Road; son is next of kin",
    "age_years": 79.4,
    "sex": 1.0,
    "max_grip_kg": 13.5,
    "bmd_t_score": -2.6,
    "ct_muscle_area_cm2": 88.4,
}


# --- de-identification ----------------------------------------------------

def test_a_topic_query_carries_no_participant_data():
    """Two participants crossing the same thresholds produce identical queries."""
    from dive.privacy import build_topic_query

    first, _ = build_topic_query(["low_grip", "frail"], "older_female")
    second, _ = build_topic_query(["low_grip", "frail"], "older_female")
    assert first == second
    for forbidden in ("13.5", "WL042", "79", "Jane"):
        assert forbidden not in first


def test_quasi_identifiers_never_leave():
    """Ethnicity, region, deprivation and genotype are re-identification routes."""
    clean, dropped = deidentify_profile({
        "ethnicity_self_reported": "South Asian", "country_iso3": "GBR",
        "region_code": "UKF2", "deprivation_decile": 2, "apoe_e4_dosage": 2,
        "genetic_ancestry_pc1": 0.04, "max_grip_kg": 13.5,
    })
    assert clean == {"max_grip_kg": 13.5}
    for field in ("ethnicity_self_reported", "country_iso3", "deprivation_decile",
                  "apoe_e4_dosage", "genetic_ancestry_pc1"):
        assert field in dropped


def test_every_identifier_is_removed():
    clean, dropped = deidentify_profile(IDENTIFIED)
    serialised = json.dumps(clean)
    for forbidden in ("WL042", "Jane", "1947", "943", "example.nhs.uk", "LE11", "Elm Road"):
        assert forbidden not in serialised
    for key in ("participant_id", "name", "dob", "nhs_number", "email", "notes"):
        assert key in dropped


def test_exact_age_never_survives_only_a_band():
    clean, _ = deidentify_profile(IDENTIFIED)
    assert "age_years" not in clean
    assert clean["age_band"] == "75-79"
    assert 79.4 not in clean.values()


def test_the_oldest_ages_collapse_into_one_open_band():
    """A lone 97-year-old is identified by their age alone."""
    assert age_band(97) == "90+"
    assert age_band(91) == "90+"
    assert age_band(89) == "85-89"


def test_clinical_values_do_survive():
    clean, _ = deidentify_profile(IDENTIFIED)
    assert clean["max_grip_kg"] == 13.5
    assert clean["bmd_t_score"] == -2.6
    assert clean["ct_muscle_area_cm2"] == 88.4


def test_an_unknown_field_is_dropped_not_passed_through():
    """Fail-closed: the allowlist is the rule, not a suggestion."""
    clean, dropped = deidentify_profile({"some_new_biomarker": 42.0, "max_grip_kg": 30.0})
    assert "some_new_biomarker" not in clean
    assert "some_new_biomarker" in dropped
    assert clean["max_grip_kg"] == 30.0


def test_long_or_identifying_free_text_is_dropped():
    clean, dropped = deidentify_profile({"sex": "female", "notes": "seen at LE11 3TU"})
    assert clean.get("sex") == "female"
    assert "notes" in dropped


# --- scanning -------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("NHS 943 476 5919", "nhs_number"),
    ("contact jane@example.com", "email"),
    ("participant WL042 attended", "participant_id"),
    ("born 12/03/1947", "date_dmy"),
    ("visit 2024-06-01", "date_iso"),
    ("lives at LE11 3TU", "uk_postcode"),
])
def test_scanner_catches_each_identifier_type(text, expected):
    assert expected in scan_for_identifiers(text)


def test_scanner_is_quiet_on_clinical_text():
    text = "grip strength 13.5 kg, gait speed 0.62 m/s, T-score -2.6"
    assert scan_for_identifiers(text) == {}


def test_redact_replaces_rather_than_removes():
    out = redact("patient WL042 on 12/03/1947")
    assert "WL042" not in out and "1947" not in out
    assert "[REDACTED]" in out


# --- the guard ------------------------------------------------------------

def test_topic_only_is_the_default_and_blocks_profile_egress():
    """The default mode sends a literature question, never a profile."""
    guard = EgressGuard(PrivacyPolicy())
    assert PrivacyPolicy().egress_mode == "topic_only"
    _, decision = guard.review_profile(IDENTIFIED, "anthropic")
    assert not decision.allowed
    assert "topic_only" in decision.reasons[0]


def test_guard_permits_a_deidentified_profile():
    guard = EgressGuard(PrivacyPolicy(egress_mode="deidentified"))
    clean, decision = guard.review_profile(IDENTIFIED, "anthropic")
    assert decision.allowed
    assert "participant_id" in decision.dropped_fields
    assert clean["age_band"] == "75-79"


def test_egress_mode_none_blocks_every_profile():
    guard = EgressGuard(PrivacyPolicy(egress_mode="none"))
    _, decision = guard.review_profile(IDENTIFIED, "anthropic")
    assert not decision.allowed


def test_egress_mode_full_is_refused_by_design():
    """The mode exists to be explicit; the guard still refuses it."""
    guard = EgressGuard(PrivacyPolicy(egress_mode="full"))
    _, decision = guard.review_profile(IDENTIFIED, "anthropic")
    assert not decision.allowed
    assert any("refused by design" in r for r in decision.reasons)


def test_offline_only_is_an_absolute_kill_switch():
    guard = EgressGuard(PrivacyPolicy(offline_only=True))
    assert not guard.review_profile(IDENTIFIED, "anthropic")[1].allowed
    assert not guard.review_text("grip strength older adults", "anthropic").allowed


def test_an_unlisted_destination_is_blocked():
    guard = EgressGuard(PrivacyPolicy(allowed_destinations=("anthropic",)))
    assert not guard.review_profile(IDENTIFIED, "google")[1].allowed


def test_free_text_carrying_an_identifier_is_blocked_not_redacted():
    """Silently rewriting the payload would hide that something upstream leaked."""
    guard = EgressGuard(PrivacyPolicy())
    decision = guard.review_text("summarise for patient WL042, NHS 943 476 5919", "anthropic")
    assert not decision.allowed
    assert decision.payload == ""


def test_small_cells_are_suppressed():
    guard = EgressGuard(PrivacyPolicy(min_aggregate_n=5))
    assert not guard.check_aggregate(3)[0]
    assert guard.check_aggregate(12)[0]


def test_guard_writes_an_audit_record(tmp_path):
    from dive.security import AuditLog

    audit = tmp_path / "audit.jsonl"
    guard = EgressGuard(PrivacyPolicy(egress_mode="deidentified"), audit_path=audit)
    guard.review_profile(IDENTIFIED, "anthropic")

    entries = AuditLog(audit).tail()
    assert entries and entries[0]["event"] == "egress"
    # The audit records a hash of what left, never the payload itself.
    assert "payload" not in entries[0]
    assert entries[0]["payload_sha256"]


def test_allowlist_contains_no_identifying_key():
    from dive.privacy import IDENTIFYING_KEYS

    assert not (ALLOWED_CLINICAL_KEYS & IDENTIFYING_KEYS)
