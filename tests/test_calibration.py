"""Imaging numbers are only usable if the scanner that produced them is."""

import numpy as np
import pytest

from dive.calibration import (
    PhantomReading,
    ScannerCalibration,
    assert_bmat_comparable,
    assert_comparable,
    least_significant_change,
    precision_from_repeats,
)


def test_least_significant_change_uses_the_iscd_multiplier():
    assert least_significant_change(1.0) == pytest.approx(2.77, abs=1e-9)
    assert least_significant_change(4.0) == pytest.approx(11.08, abs=1e-9)


def test_precision_pools_variances_not_standard_deviations():
    rng = np.random.default_rng(0)
    repeats = [list(rng.normal(220, 4.0, 3)) for _ in range(40)]
    result = precision_from_repeats(repeats, "pqct_trabecular_vbmd")

    assert 3.0 < result.rms_sd < 5.0        # recovers the simulated SD of 4
    assert 1.0 < result.cv_percent < 3.0
    assert result.lsc == pytest.approx(2.77 * result.rms_sd)


def test_precision_needs_actual_repeats():
    with pytest.raises(ValueError, match="two usable repeats"):
        precision_from_repeats([[220.0], [218.0]])


def test_a_stable_device_is_in_control():
    rng = np.random.default_rng(1)
    calibration = ScannerCalibration("XCT-A")
    readings = [
        PhantomReading("XCT-A", 100.0 + rng.normal(0, 0.3), 100.0, taken_utc=f"2026-01-{i:02d}")
        for i in range(1, 11)
    ]
    report = calibration.check_drift(readings)
    assert not report.drifted
    assert "in control" in report.verdict


def test_a_shifted_phantom_value_is_flagged():
    calibration = ScannerCalibration("XCT-A", drift_tolerance=0.01)
    readings = [PhantomReading("XCT-A", 100.0, 100.0, taken_utc="2026-01-01")]
    readings.append(PhantomReading("XCT-A", 102.5, 100.0, taken_utc="2026-01-02"))
    report = calibration.check_drift(readings)
    assert report.drifted
    assert "outside" in report.verdict


def test_a_run_on_one_side_is_flagged_even_within_tolerance():
    """Six consecutive small positive deviations is a shift, not noise."""
    calibration = ScannerCalibration("XCT-A", drift_tolerance=0.02, run_length_limit=5)
    readings = [
        PhantomReading("XCT-A", 100.5, 100.0, taken_utc=f"2026-01-{i:02d}")
        for i in range(1, 8)
    ]
    report = calibration.check_drift(readings)
    assert report.drifted
    assert any("consecutive" in r for r in report.reasons)


def test_a_device_with_no_qc_scans_is_not_trusted():
    report = ScannerCalibration("XCT-B").check_drift([])
    assert "cannot be trusted" in report.verdict


def test_uncalibrated_devices_are_not_poolable():
    calibrations = {
        "A": ScannerCalibration("A", voxel_size_mm=0.4),
        "B": ScannerCalibration("B", voxel_size_mm=0.4),
    }
    comparable, problems = assert_comparable("pqct_trabecular_vbmd", ["A", "B"], calibrations)
    assert not comparable
    assert any("cross-calibrated" in p for p in problems)


def test_different_voxel_sizes_are_not_poolable():
    calibrations = {
        "A": ScannerCalibration("A", voxel_size_mm=0.4, cross_calibrated_against="A"),
        "B": ScannerCalibration("B", voxel_size_mm=0.5, cross_calibrated_against="A"),
    }
    comparable, problems = assert_comparable("pqct_cortical_vbmd", ["A", "B"], calibrations)
    assert not comparable
    assert any("voxel" in p for p in problems)


def test_a_single_device_is_trivially_comparable():
    comparable, problems = assert_comparable("x", ["A"], {"A": ScannerCalibration("A")})
    assert comparable and problems == []


def test_cross_calibration_is_identity_until_derived():
    calibration = ScannerCalibration("A")
    assert calibration.apply_cross_calibration(180.0) == 180.0
    assert not calibration.is_cross_calibrated()


def test_marrow_fat_methods_are_not_pooled():
    comparable, problems = assert_bmat_comparable(["bmat_pdff_pct", "vertebral_ct_density_hu"])
    assert not comparable
    assert "inverse proxy" in problems[0]


def test_one_marrow_method_alone_is_fine():
    comparable, _ = assert_bmat_comparable(["bmat_pdff_pct"])
    assert comparable
