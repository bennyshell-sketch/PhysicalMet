"""Ingestion must quarantine bad data; promotion must be earned."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poraf_bt.learning.ingest import (
    DataSource,
    SourceRegistry,
    ingest_file,
    map_columns,
    validate_frame,
)
from poraf_bt.learning.registry import ModelRegistry
from poraf_bt.learning.retrain import detect_drift, retrain_and_evaluate


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the registries at a temp directory so tests never touch real data."""
    from dataclasses import replace

    from poraf_bt import config

    paths = replace(config.SETTINGS.paths, data_dir=tmp_path, model_dir=tmp_path / "models")
    settings = replace(config.SETTINGS, paths=paths)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    return settings


def test_aliases_map_onto_the_canonical_schema():
    frame = pd.DataFrame(columns=["subject_id", "age", "gender", "handgrip", "nonsense_col"])
    mapping, unmapped = map_columns(frame)
    assert mapping["subject_id"] == "participant_id"
    assert mapping["age"] == "age_years"
    assert mapping["handgrip"] == "max_grip_kg"
    assert unmapped == ["nonsense_col"]


def test_implausible_values_are_quarantined_with_a_reason():
    frame = pd.DataFrame(
        {
            "participant_id": ["WL001", "WL002", "WL003"],
            "max_grip_kg": [30.0, 250.0, 28.0],   # 250 kg is a unit error
            "age_years": [72.0, 75.0, 250.0],     # 250 y is impossible
        }
    )
    accepted, quarantined, _ = validate_frame(frame)
    assert len(accepted) == 1
    assert len(quarantined) == 2
    assert all(quarantined["quarantine_reason"].str.len() > 0)


def test_out_of_range_values_are_never_silently_clipped():
    frame = pd.DataFrame({"participant_id": ["WL001"], "max_grip_kg": [250.0]})
    accepted, quarantined, _ = validate_frame(frame)
    assert accepted.empty
    # The original value survives in quarantine rather than being coerced to 90.
    assert quarantined["max_grip_kg"].iloc[0] == 250.0


def test_a_row_missing_a_required_field_is_held_back():
    frame = pd.DataFrame({"participant_id": [None, "WL002"], "max_grip_kg": [30.0, 28.0]})
    accepted, quarantined, _ = validate_frame(frame)
    assert len(accepted) == 1
    assert "required field" in quarantined["quarantine_reason"].iloc[0]


def test_ingestion_writes_a_traceable_manifest(tmp_path, isolated):
    source_file = tmp_path / "partner.csv"
    pd.DataFrame(
        {"subject_id": [f"WL{i:03d}" for i in range(1, 21)],
         "age": np.linspace(65, 85, 20),
         "handgrip": np.linspace(18, 38, 20)}
    ).to_csv(source_file, index=False)

    source = DataSource(source_id="partner_a", label="Partner A", kind="cohort", trust=0.8)
    result = ingest_file(source_file, source, settings=isolated)

    assert result.n_accepted == 20
    batches = SourceRegistry(isolated).list_batches()
    assert len(batches) == 1
    assert batches["source"].iloc[0] == "Partner A"
    assert batches["trust"].iloc[0] == 0.8


def test_drift_detection_flags_a_shifted_distribution():
    reference = pd.DataFrame({"max_grip_kg": np.random.default_rng(0).normal(28, 6, 300)})
    shifted = pd.DataFrame({"max_grip_kg": np.random.default_rng(1).normal(40, 6, 300)})
    drift = detect_drift(reference, shifted)
    assert bool(drift.loc[drift["variable"] == "max_grip_kg", "drifted"].iloc[0])


def test_drift_detection_is_quiet_when_nothing_shifted():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"max_grip_kg": rng.normal(28, 6, 400)})
    same = pd.DataFrame({"max_grip_kg": rng.normal(28, 6, 400)})
    drift = detect_drift(reference, same)
    assert not bool(drift.loc[drift["variable"] == "max_grip_kg", "drifted"].iloc[0])


def test_a_model_that_cannot_beat_chance_is_never_promoted(dataset, isolated):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = retrain_and_evaluate(
            dataset.substudy, dataset.wide, settings=isolated,
            registry=ModelRegistry(isolated),
        )
    assert not outcome.promoted
    assert "chance" in outcome.reason.lower()


def test_registration_alone_does_not_make_a_champion(isolated):
    registry = ModelRegistry(isolated)
    version = registry.register(
        model={"dummy": True}, model_kind="decline",
        metrics={"auc_cv": 0.9}, n_training=100,
    )
    assert not version.is_champion
    assert registry.champion("decline") is None

    registry.promote(version.version_id, reason="test")
    assert registry.champion("decline").version_id == version.version_id


def test_promoting_a_new_version_demotes_the_incumbent(isolated):
    registry = ModelRegistry(isolated)
    first = registry.register({"a": 1}, "decline", {"auc_cv": 0.7}, 100)
    second = registry.register({"b": 2}, "decline", {"auc_cv": 0.8}, 100)
    registry.promote(first.version_id)
    registry.promote(second.version_id)

    champions = [e for e in registry._load() if e.get("is_champion")]
    assert len(champions) == 1
    assert champions[0]["version_id"] == second.version_id
