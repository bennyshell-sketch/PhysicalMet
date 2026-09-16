"""Models must be honest about what they can and cannot predict."""

import warnings

import numpy as np
import pandas as pd
import pytest

from poraf_bt.config import SETTINGS
from poraf_bt.decline_model import fit_decline_model, prior_correction_offset
from poraf_bt.group_comparison import assert_null_on_synthetic, run_group_comparison
from poraf_bt.trajectory_model import fit_trajectory_model
from poraf_bt.validation import assert_no_participant_leakage, participant_aware_folds


@pytest.fixture(scope="module")
def decline(dataset):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_decline_model(dataset.substudy, dataset.wide, run_sensitivity=False)


@pytest.fixture(scope="module")
def trajectory(dataset):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_trajectory_model(dataset.trajectory_frame())


# --- null in, null out ----------------------------------------------------

def test_group_comparison_is_null_on_null_data(dataset):
    result = run_group_comparison(dataset.substudy)
    assert_null_on_synthetic(result)
    assert not result.is_significant


def test_decline_model_does_not_beat_chance_on_null_data(decline):
    assert decline.result_.auc_permutation_p > 0.05
    assert not decline.result_.is_predictive


def test_model_abstains_rather_than_issuing_unsupported_risk(decline):
    prediction = decline.predict_risk({"log_BT_ratio": -1.1, "age_years": 75, "sex": 1})
    assert prediction.abstained
    assert prediction.reasons


# --- case-control handling ------------------------------------------------

def test_prior_correction_arithmetic():
    # A 50/50 sample mapped onto a 25% prevalence population shifts by log(3).
    assert prior_correction_offset(0.5, 0.25) == pytest.approx(np.log(3.0))
    assert prior_correction_offset(0.5, 0.5) == pytest.approx(0.0)


def test_case_control_caveat_is_always_present(decline):
    assert any("case-control" in w.lower() for w in decline.result_.warnings)


# --- extrapolation --------------------------------------------------------

def test_model_refuses_profiles_outside_its_training_range(dataset):
    """Even a model that passes validation must refuse impossible covariates."""
    from poraf_bt.decline_model import DeclineModel

    settings = SETTINGS
    model = DeclineModel(settings=settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(dataset.substudy)
    inside, messages = model.guard_.check({"log_BT_ratio": -1.1, "age_years": 250.0, "sex": 1.0})
    assert not inside
    assert any("age_years" in m for m in messages)


# --- trajectory -----------------------------------------------------------

def test_trajectory_uses_elapsed_time_not_visit_index(trajectory):
    frame = trajectory.training_frame_
    # Time is stored in years internally; visits are irregular, so the distinct
    # count must far exceed the four nominal visits.
    assert frame["time"].nunique() > 20


def test_prediction_interval_is_wider_than_confidence_interval(trajectory):
    projection = trajectory.predict_trajectory({"age_years": 75.0, "sex": 1.0}, months=[12.0])
    row = projection.iloc[0]
    assert row["pi_upper"] - row["pi_lower"] > row["ci_upper"] - row["ci_lower"]


def test_history_narrows_the_interval(trajectory):
    profile = {"age_years": 75.0, "sex": 1.0}
    history = pd.DataFrame(
        {"elapsed_months": [0.0, 1.6, 6.2], "max_grip_kg": [24.0, 23.1, 22.5]}
    )
    without = trajectory.predict_trajectory(profile, months=[12.0]).iloc[0]
    with_history = trajectory.predict_trajectory(profile, months=[12.0], observed=history).iloc[0]
    assert (with_history["pi_upper"] - with_history["pi_lower"]) < (
        without["pi_upper"] - without["pi_lower"]
    )


def test_variance_components_are_physiologically_plausible(trajectory):
    """A random-slope SD of several kg per month would be nonsense."""
    result = trajectory.result_
    assert 0 < result.residual_sd < 6.0
    intercept_sd = result.random_effect_sd.get("Group", np.nan)
    assert 2.0 < intercept_sd < 15.0


def test_forecast_refuses_beyond_the_supported_horizon(trajectory):
    prediction = trajectory.forecast({"age_years": 75.0, "sex": 1.0}, months=120.0)
    assert prediction.abstained


# --- leakage --------------------------------------------------------------

def test_participant_aware_folds_do_not_leak(dataset):
    ids = dataset.long["participant_id"]
    folds = participant_aware_folds(ids, n_splits=5, seed=3)
    assert_no_participant_leakage(ids, folds)


def test_naive_row_split_would_leak(dataset):
    ids = dataset.long["participant_id"].to_numpy()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(ids))
    half = len(order) // 2
    with pytest.raises(AssertionError):
        assert_no_participant_leakage(ids, [(order[:half], order[half:])])
