"""The autonomous cycle must be able to decline to act, at every stage."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dive.learning.autolearn import AutoLearner


@pytest.fixture
def learner(tmp_path):
    from dive import config

    paths = replace(config.SETTINGS.paths, data_dir=tmp_path, model_dir=tmp_path / "models")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    settings = replace(config.SETTINGS, paths=paths)
    return AutoLearner(settings=settings, min_hours_between_runs=12.0)


def test_a_cycle_halts_when_there_is_too_little_new_data(learner):
    result = learner.run_cycle(force=True)
    assert result.halted_at == "ingest"
    assert not result.changed


def test_the_rate_limiter_blocks_a_second_immediate_run(learner):
    learner.run_cycle(force=True)
    due, reason = learner.due()
    assert not due
    assert "next in" in reason


def test_repeated_failures_pause_automatic_learning(learner):
    state = learner._state()
    state["consecutive_failures"] = 3
    state["last_run_utc"] = None
    learner._save_state(state)

    due, reason = learner.due()
    assert not due
    assert "paused" in reason


def test_a_large_distribution_shift_halts_the_cycle(learner, tmp_path):
    from dive.data_pipeline import load_dataset
    from dive.learning.ingest import DataSource, ingest_file

    data = load_dataset()
    rng = np.random.default_rng(1)
    n = 120
    source = tmp_path / "drifted.csv"
    pd.DataFrame({
        "subject_id": [f"WL{i:03d}" for i in range(500, 500 + n)],
        "age": rng.normal(45, 4, n),          # a much younger population
    }).to_csv(source, index=False)

    ingest_file(
        source, DataSource(source_id="site_b", label="Site B"), settings=learner.cfg
    )
    result = learner.run_cycle(force=True)
    assert result.halted_at == "drift"
    assert "different population" in dict(
        (s.name, s.summary) for s in result.stages
    )["drift"]


def test_a_dry_run_changes_nothing(learner, tmp_path):
    from dive.learning.ingest import DataSource, ingest_file

    rng = np.random.default_rng(0)
    n = 60
    source = tmp_path / "similar.csv"
    pd.DataFrame({
        "subject_id": [f"WL{i:03d}" for i in range(600, 600 + n)],
        "age": rng.normal(75, 6, n),
    }).to_csv(source, index=False)
    ingest_file(source, DataSource(source_id="site_c", label="Site C"), settings=learner.cfg)

    result = learner.run_cycle(force=True, dry_run=True)
    assert result.dry_run
    assert not any(s.acted for s in result.stages)


def test_every_cycle_is_recorded(learner):
    learner.run_cycle(force=True)
    history = learner.history()
    assert len(history) == 1
    assert history["Outcome"].iloc[0] in {"halted", "changed", "no change"}
