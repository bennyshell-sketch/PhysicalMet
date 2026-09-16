"""The cleaning pipeline must handle every documented workbook quirk."""

import numpy as np
import pandas as pd
import pytest

from poraf_bt.config import SETTINGS
from poraf_bt.data_pipeline import (
    build_dataset,
    clean_tug,
    harmonise_binary,
    max_grip_from_trials,
    resolve_participant_id,
)
from poraf_bt.data_pipeline import QualityReport


def test_tug_sentinel_is_split_out_not_averaged_in():
    series = pd.Series([8.0, 40.0, 12.5, 40.0])
    continuous, cannot = clean_tug(series, sentinel=40.0)
    assert continuous.isna().sum() == 2
    assert cannot.sum() == 2
    assert continuous.max() == 12.5


def test_mixed_yes_no_and_binary_encodings_harmonise():
    series = pd.Series(["Yes", "no", 1, 0, "YES", None, "n/a"])
    result = harmonise_binary(series)
    assert list(result[:5]) == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert np.isnan(result.iloc[5]) and np.isnan(result.iloc[6])


def test_grip_999_sentinel_is_missing_not_a_reading():
    frame = pd.DataFrame({"Grip_trial_1": [30.0, 999.0], "Grip_trial_2": [999.0, 999.0]})
    result = max_grip_from_trials(frame, ["Grip_trial_1", "Grip_trial_2"], 999.0)
    assert result.iloc[0] == 30.0
    assert np.isnan(result.iloc[1])


def test_participant_id_found_by_content_not_position():
    """The 12-month sheet has two stray ID columns before the real one."""
    frame = pd.DataFrame(
        {
            "Lab_Ref": ["LAB1", "LAB2", "LAB3"],
            "Batch_ID": ["B1", "B2", "B3"],
            "Participant_ID": ["WL001", "WL002", "WL003"],
        }
    )
    ids = resolve_participant_id(frame, "TwelveMonth", QualityReport())
    assert list(ids) == ["WL001", "WL002", "WL003"]


def test_deltas_are_recomputed_not_read_from_stored_columns(workbook, dataset):
    raw = pd.read_excel(workbook, sheet_name="TwelveMonth")
    baseline = pd.read_excel(workbook, sheet_name="Baseline").set_index("Participant_ID")["SPPB_Total"]
    truth = raw.set_index("Participant_ID")["SPPB_Total"] - baseline
    stored = raw.set_index("Participant_ID")["Delta_SPPB_Total"]

    both = stored.notna() & truth.notna()
    # The fixture deliberately corrupts some stored deltas.
    assert (~np.isclose(stored[both], truth[both])).sum() > 0

    wide = dataset.wide.set_index("participant_id")
    recomputed = wide["delta_sppb_TwelveMonth"]
    common = recomputed.index.intersection(truth.index)
    mask = recomputed[common].notna() & truth[common].notna()
    assert np.allclose(recomputed[common][mask], truth[common][mask])


def test_no_tug_sentinel_survives_into_the_continuous_column(dataset):
    assert not (dataset.long["tug_s"] == SETTINGS.cohort.tug_sentinel_value).any()


def test_younger_participants_are_not_defaulted_to_non_faller(dataset):
    younger = dataset.wide[dataset.wide["age_group"] == "Younger"]
    assert younger["subgroup_code"].isna().all()


def test_substudy_preserves_the_case_control_structure(dataset):
    assert len(dataset.substudy) == SETTINGS.cohort.n_bt_substudy
    counts = dataset.substudy["sample_group"].value_counts()
    assert counts["Decliner"] == SETTINGS.cohort.n_bt_decliners
    assert counts["Stable"] == SETTINGS.cohort.n_bt_stable


def test_incomplete_elisa_triplets_stay_missing(dataset):
    incomplete = dataset.substudy[
        dataset.substudy[["IL6_pg_mL", "sIL6R_ng_mL", "sgp130_ng_mL"]].isna().any(axis=1)
    ]
    assert incomplete["BT_ratio"].isna().all()


def test_long_format_has_one_row_per_participant_visit(dataset):
    assert not dataset.long.duplicated(subset=["participant_id", "visit"]).any()
