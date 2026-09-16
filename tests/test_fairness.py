"""Ethnicity and race audit the model; they do not power it."""

import numpy as np
import pandas as pd
import pytest

from dive.fairness import (
    ANCESTRY_ATTRIBUTES,
    CONTEXT_ATTRIBUTES,
    PROTECTED_ATTRIBUTES,
    audit_subgroups,
    reference_population_note,
    screen_features,
)


def test_protected_attributes_are_refused_as_features():
    allowed, refused = screen_features(
        ["max_grip_kg", "ethnicity_self_reported", "race", "preferred_language"]
    )
    assert allowed == ["max_grip_kg"]
    assert {name for name, _ in refused} == {
        "ethnicity_self_reported", "race", "preferred_language"
    }
    assert all("not biological" in why for _, why in refused)


def test_genetic_ancestry_is_allowed_because_it_is_not_a_social_category():
    allowed, refused = screen_features(list(ANCESTRY_ATTRIBUTES))
    assert set(allowed) == ANCESTRY_ATTRIBUTES
    assert refused == []


def test_protected_use_requires_an_explicit_override():
    allowed, refused = screen_features(["ethnicity_self_reported"], allow_protected=True)
    assert allowed == ["ethnicity_self_reported"]
    assert refused == []


def test_context_attributes_are_allowed_but_separable():
    allowed, _ = screen_features(list(CONTEXT_ATTRIBUTES))
    assert set(allowed) == CONTEXT_ATTRIBUTES
    allowed, refused = screen_features(["deprivation_decile"], allow_context=False)
    assert allowed == [] and refused


def test_protected_and_ancestry_sets_do_not_overlap():
    assert not (PROTECTED_ATTRIBUTES & ANCESTRY_ATTRIBUTES)


# --- auditing -------------------------------------------------------------

def _frame(n=400, uneven=True, seed=0):
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], n, p=[0.8, 0.2])
    y = rng.binomial(1, 0.3, n)
    if uneven:
        predicted = np.where(group == "A", y * 0.5 + rng.uniform(0, 0.4, n), rng.uniform(0, 0.8, n))
    else:
        predicted = y * 0.5 + rng.uniform(0, 0.4, n)
    return pd.DataFrame({"grp": group, "declined": y, "predicted_risk": predicted})


def test_an_uneven_model_is_flagged():
    audit = audit_subgroups(_frame(uneven=True), "grp")
    assert audit.concerning
    assert audit.auc_range >= 0.10
    assert "unevenly" in audit.verdict


def test_an_even_model_is_not_flagged():
    audit = audit_subgroups(_frame(uneven=False), "grp")
    assert not audit.concerning
    assert "comparable" in audit.verdict


def test_small_subgroups_are_suppressed_not_reported():
    frame = _frame(n=200)
    frame.loc[frame.index[:3], "grp"] = "C"      # a three-person subgroup
    audit = audit_subgroups(frame, "grp", min_cell=5)

    tiny = [s for s in audit.subgroups if s.group == "C"]
    assert tiny and tiny[0].suppressed
    assert audit.n_suppressed == 1
    # The displayed table must not reveal the exact count.
    assert "<5" in audit.to_frame()["n"].astype(str).tolist()


def test_suppression_is_explained_as_ignorance_not_reassurance():
    frame = _frame(n=200)
    frame.loc[frame.index[:3], "grp"] = "C"
    audit = audit_subgroups(frame, "grp", min_cell=5)
    assert any("not evidence that" in w for w in audit.warnings)


def test_auditing_a_missing_column_fails_clearly():
    audit = audit_subgroups(_frame(), "nonexistent")
    assert "not present" in audit.verdict
    assert audit.subgroups == []


# --- reference populations ------------------------------------------------

@pytest.mark.parametrize("measure,keyword", [
    ("bmd_t_score", "reference"),
    ("egfr_ml_min", "race-free"),
    ("prs_ebmd", "ancestr"),
    ("dnam_age_horvath", "calibrat"),
])
def test_reference_notes_exist_for_the_scores_that_depend_on_them(measure, keyword):
    """Each derived score names the population choice its value depends on."""
    note = reference_population_note(measure)
    assert len(note) > 80
    assert keyword in note.lower()


def test_an_unknown_measure_gets_an_honest_default():
    assert "No reference-population note" in reference_population_note("made_up_measure")
