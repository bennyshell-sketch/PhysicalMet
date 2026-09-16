"""Registering a metric makes it eligible; only evidence makes it adopted."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dive.learning.feature_search import search_features
from dive.learning.ingest import DataSource, effective_schema, ingest_file, map_columns
from dive.metrics import CORE_METRICS, HealthMetric, MetricRegistry, slugify


@pytest.fixture
def registry(tmp_path):
    from dive import config

    paths = replace(config.SETTINGS.paths, data_dir=tmp_path)
    settings = replace(config.SETTINGS, paths=paths)
    return MetricRegistry(settings), settings


def test_names_are_slugified():
    metric = HealthMetric("Serum 25-OH Vitamin D!", "")
    assert metric.name == "serum_25_oh_vitamin_d"
    assert metric.label == "Serum 25 Oh Vitamin D"


def test_an_inverted_range_is_rejected_at_definition():
    with pytest.raises(ValueError, match="minimum"):
        HealthMetric("bad", "Bad", minimum=100, maximum=10)


def test_core_metrics_cannot_be_silently_redefined(registry):
    store, _ = registry
    with pytest.raises(ValueError, match="core metric"):
        store.register(HealthMetric("max_grip_kg", "Grip", minimum=0, maximum=1000))


def test_registering_twice_needs_an_explicit_overwrite(registry):
    store, _ = registry
    store.register(HealthMetric("novel_marker", "Novel", minimum=0, maximum=10))
    with pytest.raises(ValueError, match="already registered"):
        store.register(HealthMetric("novel_marker", "Novel", minimum=0, maximum=20))
    store.register(HealthMetric("novel_marker", "Novel", minimum=0, maximum=20), overwrite=True)
    assert store.get("novel_marker").maximum == 20


def test_a_registered_metric_joins_the_ingestion_schema(registry):
    store, settings = registry
    store.register(HealthMetric(
        "serum_vitamin_d", "Vitamin D", "nmol/L", "clinical", 5, 300, True, ("vit_d",)
    ))
    schema = effective_schema(settings)
    assert "serum_vitamin_d" in schema
    assert schema["serum_vitamin_d"].maximum == 300

    frame = pd.DataFrame({"subject_id": ["WL001"], "vit_d": [55.0]})
    mapping, unmapped = map_columns(frame, settings=settings)
    assert mapping["vit_d"] == "serum_vitamin_d"
    assert unmapped == []


def test_a_registered_metrics_range_is_enforced_on_ingestion(tmp_path, registry):
    store, settings = registry
    store.register(HealthMetric(
        "daily_step_count", "Steps", "steps/day", "activity", 0, 40000, True, ("steps",)
    ))
    source_file = tmp_path / "wearable.csv"
    pd.DataFrame({
        "subject_id": ["WL001", "WL002"],
        "steps": [8000.0, 450000.0],      # second is a unit error
    }).to_csv(source_file, index=False)

    result = ingest_file(
        source_file, DataSource(source_id="w", label="Wearable"), settings=settings
    )
    assert result.n_accepted == 1
    assert result.n_quarantined == 1
    assert "daily_step_count" in result.quarantined["quarantine_reason"].iloc[0]


def test_registration_alone_does_not_adopt_a_metric(registry):
    store, _ = registry
    metric = store.register(HealthMetric("novel_marker", "Novel", minimum=0, maximum=10))
    assert metric.adopted_by == []


# --------------------------------------------------------------------------
# Feature search
# --------------------------------------------------------------------------

def _frame(n=140, seed=0, signal=0.0):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.5, n)
    return pd.DataFrame({
        "participant_id": [f"WL{i:03d}" for i in range(n)],
        "declined": y.astype(float),
        "age_years": rng.normal(75, 6, n),
        "sex": rng.binomial(1, 0.6, n).astype(float),
        "log_BT_ratio": rng.normal(-1.1, 0.2, n),
        "candidate_a": y * signal + rng.normal(0, 1, n),
        "candidate_b": rng.normal(0, 1, n),
        "candidate_c": rng.normal(0, 1, n),
    })


def test_noise_metrics_are_not_adopted():
    """The central guard: searching over noise must not yield an adoption."""
    result = search_features(
        _frame(signal=0.0), candidates=["candidate_a", "candidate_b", "candidate_c"],
        n_permutations=60,
    )
    assert result.adopted_features == result.base_features
    assert result.best_candidate is None


def test_a_genuinely_informative_metric_is_adopted():
    result = search_features(
        _frame(signal=1.6, seed=5), candidates=["candidate_a", "candidate_b", "candidate_c"],
        n_permutations=60,
    )
    assert result.best_candidate == "candidate_a"
    assert "candidate_a" in result.adopted_features
    assert result.selection_p_value < 0.05


def test_a_sparsely_measured_metric_is_skipped_not_guessed():
    frame = _frame(signal=1.6, seed=5)
    frame.loc[frame.index[:110], "candidate_a"] = np.nan   # ~21% coverage
    result = search_features(
        frame, candidates=["candidate_a"], n_permutations=0, min_coverage=0.6
    )
    candidate = next(c for c in result.candidates if c.name == "candidate_a")
    assert not candidate.evaluated
    assert "coverage" in candidate.skip_reason or "measured on only" in candidate.skip_reason
    assert result.adopted_features == result.base_features


def test_the_selection_null_accounts_for_the_number_of_candidates():
    """More candidates tried means a larger improvement is needed to pass."""
    frame = _frame(signal=0.0, seed=11)
    rng = np.random.default_rng(2)
    for i in range(8):
        frame[f"extra_{i}"] = rng.normal(0, 1, len(frame))
    candidates = ["candidate_a", "candidate_b", "candidate_c"] + [f"extra_{i}" for i in range(8)]
    result = search_features(frame, candidates=candidates, n_permutations=60)
    assert result.adopted_features == result.base_features
    assert np.isfinite(result.selection_p_value)
