"""Recommendations must be triggered by measurements and never invented."""

import pytest

from dive.recommend import load_knowledge_base, recommend_for_profile

FRAIL = {
    "sex": 1.0, "age_years": 79, "max_grip_kg": 13.5, "gait_speed_ms": 0.62,
    "sppb_total": 7, "chair_time_s": 17.2, "bmi": 31.4, "waist_cm": 96,
    "vat_mass_g": 2100, "asmi": 5.1, "fried_score": 3, "IL6_pg_mL": 4.8, "BT_ratio": 0.41,
}
ROBUST = {
    "sex": 0.0, "age_years": 66, "max_grip_kg": 44.0, "gait_speed_ms": 1.32,
    "sppb_total": 12, "chair_time_s": 8.1, "bmi": 24.2, "waist_cm": 89, "fried_score": 0,
}


def test_every_entry_carries_a_citation_and_a_grade():
    knowledge = load_knowledge_base()
    for entry in knowledge["interventions"]:
        assert entry["citation"].strip()
        assert entry["grade"] in {"A", "B", "C", "D"}
        assert entry["triggers"]


def test_a_robust_participant_triggers_nothing():
    result = recommend_for_profile(ROBUST)
    assert result.recommendations == []


def test_every_recommendation_names_the_measurement_that_fired_it():
    result = recommend_for_profile(FRAIL)
    assert result.recommendations
    for item in result.recommendations:
        assert item.triggered_by
        for reason in item.triggered_by:
            assert any(ch.isdigit() for ch in reason)


def test_recommendations_are_ranked_by_evidence_grade():
    grades = [r.grade for r in recommend_for_profile(FRAIL).recommendations]
    assert grades == sorted(grades)


def test_bt_ratio_is_reported_as_an_evidence_gap_not_as_advice():
    result = recommend_for_profile(FRAIL)
    gap_ids = {g.id for g in result.evidence_gaps}
    assert "bt_ratio_direct" in gap_ids
    assert "bt_ratio_direct" not in {r.id for r in result.recommendations}
    gap = next(g for g in result.evidence_gaps if g.id == "bt_ratio_direct")
    assert "INSUFFICIENT EVIDENCE" in gap.effect


def test_sex_specific_cutoffs_are_applied_by_sex():
    # 20 kg is below the male cutoff (27) but above the female cutoff (16).
    male = recommend_for_profile({"sex": 0.0, "max_grip_kg": 20.0})
    female = recommend_for_profile({"sex": 1.0, "max_grip_kg": 20.0})
    assert "prt_strength" in {r.id for r in male.recommendations}
    assert "prt_strength" not in {r.id for r in female.recommendations}


def test_a_missing_measurement_is_unassessed_not_treated_as_normal():
    result = recommend_for_profile({"sex": 1.0, "age_years": 75})
    assert result.unassessed
    assert any("max_grip_kg" in u for u in result.unassessed)
