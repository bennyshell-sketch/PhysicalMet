"""The equilibrium solver must be exactly right; everything downstream rests on it."""

import numpy as np
import pytest

from poraf_bt.biochem import (
    bt_ratio,
    free_mediator_equivalent,
    ng_per_ml_to_nM,
    pg_per_ml_to_nM,
    solve_equilibrium,
    to_molar_nM,
)
from poraf_bt.config import SETTINGS

PHYSIOLOGICAL = [(2.0, 40.0, 350.0), (0.5, 20.0, 500.0), (30.0, 80.0, 180.0), (8.2, 55.0, 280.0)]


@pytest.mark.parametrize("il6,sil6r,sgp130", PHYSIOLOGICAL)
def test_mass_is_conserved(il6, sil6r, sgp130):
    solution = solve_equilibrium(il6, sil6r, sgp130)
    total_il6, total_sil6r, total_sgp130 = (float(x) for x in to_molar_nM(il6, sil6r, sgp130))

    assert solution.converged
    assert solution.free_il6_nM + solution.binary_nM + solution.ternary_nM == pytest.approx(total_il6, rel=1e-9)
    assert solution.free_sil6r_nM + solution.binary_nM + solution.ternary_nM == pytest.approx(total_sil6r, rel=1e-9)
    assert solution.free_sgp130_nM + solution.ternary_nM == pytest.approx(total_sgp130, rel=1e-9)


@pytest.mark.parametrize("il6,sil6r,sgp130", PHYSIOLOGICAL)
def test_bt_matches_analytic_identity(il6, sil6r, sgp130):
    """B/T collapses to Kd2 / [free sgp130]; the numerical solve must agree."""
    solution = solve_equilibrium(il6, sil6r, sgp130)
    expected = SETTINGS.biochem.kd2_nM / solution.free_sgp130_nM
    assert solution.bt_ratio == pytest.approx(expected, rel=1e-9)


def test_more_sgp130_buffers_more_so_bt_falls():
    """Raising sgp130 must lower the B/T ratio - the direction is the whole point."""
    low = bt_ratio(4.0, 45.0, 200.0)
    high = bt_ratio(4.0, 45.0, 500.0)
    assert high < low


def test_free_species_never_exceed_totals():
    solution = solve_equilibrium(50.0, 90.0, 150.0)
    total_il6, total_sil6r, total_sgp130 = (float(x) for x in to_molar_nM(50.0, 90.0, 150.0))
    assert 0 < solution.free_il6_nM <= total_il6
    assert 0 < solution.free_sil6r_nM <= total_sil6r
    assert 0 < solution.free_sgp130_nM <= total_sgp130


@pytest.mark.parametrize("bad", [(0.0, 40.0, 350.0), (2.0, -1.0, 350.0), (np.nan, 40.0, 350.0)])
def test_invalid_input_raises_rather_than_inventing_a_number(bad):
    with pytest.raises(ValueError):
        solve_equilibrium(*bad)


def test_bt_ratio_wrapper_returns_nan_not_a_guess():
    assert np.isnan(bt_ratio(np.nan, 40.0, 350.0))


def test_unit_conversions():
    # 21 kDa IL-6 at 21 pg/mL is exactly 1 pM = 0.001 nM
    assert pg_per_ml_to_nM(21.0, 21_000.0) == pytest.approx(1e-3)
    # 50 kDa sIL-6R at 50 ng/mL is exactly 1 nM
    assert ng_per_ml_to_nM(50.0, 50_000.0) == pytest.approx(1.0)


def test_fme_is_a_plain_molar_ratio():
    # 100 kDa sgp130 at 100 ng/mL = 1 nM; 50 kDa sIL-6R at 50 ng/mL = 1 nM
    assert free_mediator_equivalent(50.0, 100.0) == pytest.approx(1.0)
