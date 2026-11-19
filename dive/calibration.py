"""Scanner calibration, precision and cross-device comparability.

Densitometry numbers are not self-describing. A trabecular vBMD of 180 mg/cm3
means one thing on a phantom-calibrated machine that has not drifted, and
something else on one that has. Feeding both into the same model as if they
were the same measurement is a quiet way to learn the scanner instead of the
patient - and in a multi-site deployment that is exactly what happens.

This module holds the three things that make an imaging measurement usable:

**Calibration.** A phantom of known density is scanned on a schedule. Drift in
the measured phantom value is drift in every patient value taken that day.
``ScannerCalibration.check_drift`` applies the standard approach: a shift
beyond a set fraction of the nominal, or a run of consecutive readings on one
side of it, flags the device.

**Precision.** Repeat scans give a coefficient of variation, from which the
**least significant change** follows as ``2.77 x RMS-SD`` - the ISCD formula
for the smallest difference that is a real change rather than measurement
noise at 95% confidence. Dive uses it to refuse to call a follow-up difference
a change when it is inside the machine's own error.

**Comparability.** Values from different devices, and marrow fat fractions from
different modalities, are not interchangeable. ``assert_comparable`` refuses to
pool them rather than silently concatenating. Cross-calibration coefficients
can be supplied when a site has actually derived them; they are never assumed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings

__all__ = [
    "PhantomReading",
    "DriftReport",
    "PrecisionResult",
    "ScannerCalibration",
    "CalibrationStore",
    "least_significant_change",
    "precision_from_repeats",
    "assert_bmat_comparable",
    "assert_comparable",
    "BMAT_MODALITIES",
]

#: The ISCD multiplier converting a precision SD into the least significant
#: change at 95% confidence for a paired measurement: 1.96 * sqrt(2) = 2.77.
LSC_MULTIPLIER = 2.77

#: Marrow fat fraction methods. Values from different methods, or different
#: anatomical sites, are NOT interchangeable and Dive will not pool them.
BMAT_MODALITIES: dict[str, str] = {
    "bmat_pdff_pct": "MR spectroscopy proton density fat fraction (reference method)",
    "bmat_ff_mri_pct": "Chemical-shift-encoded water-fat MRI fat fraction",
    "bmat_volume_cm3": "Segmented marrow adipose tissue volume",
    "vertebral_ct_density_hu": "CT attenuation (an inverse proxy, not a fat fraction)",
}


# --------------------------------------------------------------------------
# Phantom readings and drift
# --------------------------------------------------------------------------

@dataclass
class PhantomReading:
    """One quality-control scan of a phantom of known density."""

    device_id: str
    measured: float
    nominal: float
    taken_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    phantom_id: str = ""
    operator: str = ""
    note: str = ""

    @property
    def deviation(self) -> float:
        return self.measured - self.nominal

    @property
    def relative_deviation(self) -> float:
        return self.deviation / self.nominal if self.nominal else float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    device_id: str
    n_readings: int
    mean_relative_deviation: float
    latest_relative_deviation: float
    consecutive_same_side: int
    drifted: bool
    reasons: list[str] = field(default_factory=list)
    verdict: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Device": self.device_id,
            "QC scans": self.n_readings,
            "Mean deviation": f"{self.mean_relative_deviation * 100:+.2f}%",
            "Latest deviation": f"{self.latest_relative_deviation * 100:+.2f}%",
            "Run on one side": self.consecutive_same_side,
            "Status": "DRIFTED" if self.drifted else "in control",
        }])


@dataclass
class PrecisionResult:
    """Short-term precision from repeat scans, and what it implies."""

    variable: str
    n_subjects: int
    n_repeats: int
    rms_sd: float
    mean_value: float
    cv_percent: float
    lsc: float

    def describe(self) -> str:
        return (
            f"Precision for {self.variable}: CV {self.cv_percent:.2f}% "
            f"(RMS-SD {self.rms_sd:.4g}) from {self.n_subjects} participants scanned "
            f"{self.n_repeats} times. Least significant change is {self.lsc:.4g} - a "
            "follow-up difference smaller than this cannot be distinguished from "
            "measurement error."
        )


def least_significant_change(rms_sd: float, confidence: float = 0.95) -> float:
    """Smallest real change detectable, from the precision SD.

    At 95% confidence for a paired measurement this is the standard
    ``2.77 x RMS-SD``. Other confidence levels scale the z accordingly.
    """
    if confidence == 0.95:
        multiplier = LSC_MULTIPLIER
    else:
        from scipy.stats import norm

        multiplier = float(norm.ppf(0.5 + confidence / 2) * math.sqrt(2))
    return float(multiplier * rms_sd)


def precision_from_repeats(
    values: Sequence[Sequence[float]], variable: str = "measurement"
) -> PrecisionResult:
    """Root-mean-square SD across participants each scanned several times.

    ``values`` is one sequence of repeat measurements per participant. RMS-SD
    (not the mean SD) is the correct pooling: precision studies pool variances,
    and averaging SDs understates the error.
    """
    usable = [np.asarray(v, dtype=float) for v in values if len(v) >= 2]
    usable = [v[np.isfinite(v)] for v in usable]
    usable = [v for v in usable if len(v) >= 2]
    if not usable:
        raise ValueError("Precision needs at least one participant with two usable repeats.")

    variances = [float(np.var(v, ddof=1)) for v in usable]
    rms_sd = float(np.sqrt(np.mean(variances)))
    mean_value = float(np.mean([np.mean(v) for v in usable]))
    cv = (rms_sd / mean_value * 100) if mean_value else float("nan")

    return PrecisionResult(
        variable=variable,
        n_subjects=len(usable),
        n_repeats=int(round(float(np.mean([len(v) for v in usable])))),
        rms_sd=rms_sd,
        mean_value=mean_value,
        cv_percent=cv,
        lsc=least_significant_change(rms_sd),
    )


# --------------------------------------------------------------------------
# Scanner calibration
# --------------------------------------------------------------------------

@dataclass
class ScannerCalibration:
    """Calibration state for one imaging device."""

    device_id: str
    modality: str = "pQCT"            # pQCT | DXA | CT | MRI
    site: str = ""
    #: Fractional deviation of the phantom value that counts as drift.
    drift_tolerance: float = 0.01     # 1% - the usual densitometry QC limit
    #: Consecutive same-side readings that count as a shift even within tolerance.
    run_length_limit: int = 5
    #: Multiplicative and additive terms to bring this device onto the
    #: reference device's scale. Identity until a site derives them.
    cross_calibration_slope: float = 1.0
    cross_calibration_intercept: float = 0.0
    cross_calibrated_against: str = ""
    #: Acquisition settings that must match for values to be comparable.
    voxel_size_mm: float | None = None
    scan_speed_mm_s: float | None = None
    analysis_mode: str = ""

    def apply_cross_calibration(self, value: float) -> float:
        """Map a value onto the reference device's scale.

        Identity unless a slope and intercept have actually been derived at
        this site - Dive never invents a conversion.
        """
        return self.cross_calibration_slope * value + self.cross_calibration_intercept

    def is_cross_calibrated(self) -> bool:
        return bool(self.cross_calibrated_against)

    def check_drift(self, readings: Sequence[PhantomReading]) -> DriftReport:
        """Assess whether this device's phantom values have moved."""
        mine = [r for r in readings if r.device_id == self.device_id]
        mine.sort(key=lambda r: r.taken_utc)

        if not mine:
            return DriftReport(
                self.device_id, 0, float("nan"), float("nan"), 0, False,
                ["no quality-control scans recorded"],
                "No phantom scans have been recorded for this device, so its values "
                "cannot be trusted for longitudinal comparison.",
            )

        deviations = np.array([r.relative_deviation for r in mine], dtype=float)
        deviations = deviations[np.isfinite(deviations)]
        if deviations.size == 0:
            return DriftReport(
                self.device_id, len(mine), float("nan"), float("nan"), 0, True,
                ["phantom readings are unusable (zero or missing nominal value)"],
                "Phantom readings could not be interpreted.",
            )

        latest = float(deviations[-1])
        mean_dev = float(np.mean(deviations))

        # Longest trailing run on one side of nominal.
        run = 0
        sign = np.sign(latest)
        for value in deviations[::-1]:
            if np.sign(value) == sign and sign != 0:
                run += 1
            else:
                break

        reasons: list[str] = []
        if abs(latest) > self.drift_tolerance:
            reasons.append(
                f"latest phantom reading is {latest * 100:+.2f}% from nominal, outside "
                f"the {self.drift_tolerance * 100:.1f}% tolerance"
            )
        if abs(mean_dev) > self.drift_tolerance:
            reasons.append(
                f"mean phantom deviation is {mean_dev * 100:+.2f}%, outside tolerance"
            )
        if run >= self.run_length_limit:
            reasons.append(
                f"{run} consecutive readings on the same side of nominal - a systematic "
                "shift even though individual readings are within tolerance"
            )

        drifted = bool(reasons)
        verdict = (
            f"Device {self.device_id} has drifted. "
            + "; ".join(reasons)
            + ". Measurements taken since the last in-control scan should be treated as "
              "suspect until the device is recalibrated."
            if drifted else
            f"Device {self.device_id} is in control: latest phantom reading "
            f"{latest * 100:+.2f}% from nominal across {len(mine)} QC scans."
        )

        return DriftReport(
            self.device_id, len(mine), mean_dev, latest, run, drifted, reasons, verdict
        )

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Comparability
# --------------------------------------------------------------------------

def assert_comparable(
    variable: str,
    devices: Iterable[str],
    calibrations: dict[str, ScannerCalibration],
) -> tuple[bool, list[str]]:
    """Can values of ``variable`` from these devices be pooled?

    Returns ``(comparable, problems)``. Pooling incomparable measurements is
    how a model ends up learning the scanner rather than the patient, so the
    default answer for an uncalibrated mix is no.
    """
    problems: list[str] = []
    device_list = sorted(set(devices))

    if len(device_list) <= 1:
        return True, []

    known = [d for d in device_list if d in calibrations]
    unknown = [d for d in device_list if d not in calibrations]
    if unknown:
        problems.append(
            f"no calibration record for device(s): {', '.join(unknown)}"
        )

    modalities = {calibrations[d].modality for d in known}
    if len(modalities) > 1:
        problems.append(
            f"values come from different modalities ({', '.join(sorted(modalities))}); "
            "these measure different physical quantities and must not be pooled"
        )

    voxels = {calibrations[d].voxel_size_mm for d in known if calibrations[d].voxel_size_mm}
    if len(voxels) > 1:
        problems.append(
            f"different voxel sizes in use ({', '.join(f'{v:g}mm' for v in sorted(voxels))}); "
            "partial-volume effects differ, which biases cortical measures in particular"
        )

    uncalibrated = [d for d in known if not calibrations[d].is_cross_calibrated()]
    if len(known) > 1 and len(uncalibrated) == len(known):
        problems.append(
            f"none of {', '.join(known)} has been cross-calibrated against a reference "
            "device; a site-derived slope and intercept are required before pooling"
        )

    return not problems, problems


def assert_bmat_comparable(variables: Iterable[str]) -> tuple[bool, list[str]]:
    """Marrow fat measured by different methods must not be pooled.

    MR spectroscopy PDFF, water-fat imaging fat fraction, segmented volume and
    CT attenuation are four different quantities. They correlate; they are not
    the same number, and CT attenuation moves in the opposite direction.
    """
    present = [v for v in variables if v in BMAT_MODALITIES]
    if len(present) <= 1:
        return True, []
    return False, [
        "marrow adiposity is present in more than one form ("
        + "; ".join(f"{v}: {BMAT_MODALITIES[v]}" for v in present)
        + "). These are different quantities - CT attenuation is an inverse proxy, not a "
          "fat fraction - and Dive will not convert between them. Model them separately, "
          "or choose one and treat the rest as missing."
    ]


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

class CalibrationStore:
    """Persisted calibration records and phantom QC history."""

    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or SETTINGS
        self.path = self.cfg.paths.data_dir / "calibration.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"devices": {}, "phantom_readings": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"devices": {}, "phantom_readings": []}

    def _save(self, payload: dict) -> None:
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def register_device(self, calibration: ScannerCalibration) -> ScannerCalibration:
        payload = self._load()
        payload["devices"][calibration.device_id] = calibration.as_dict()
        self._save(payload)
        return calibration

    def devices(self) -> dict[str, ScannerCalibration]:
        payload = self._load()
        return {
            device_id: ScannerCalibration(**record)
            for device_id, record in payload.get("devices", {}).items()
        }

    def record_phantom(self, reading: PhantomReading) -> PhantomReading:
        payload = self._load()
        payload.setdefault("phantom_readings", []).append(reading.as_dict())
        self._save(payload)
        return reading

    def phantom_readings(self, device_id: str | None = None) -> list[PhantomReading]:
        payload = self._load()
        readings = [PhantomReading(**r) for r in payload.get("phantom_readings", [])]
        return [r for r in readings if device_id is None or r.device_id == device_id]

    def drift_reports(self) -> list[DriftReport]:
        readings = self.phantom_readings()
        return [cal.check_drift(readings) for cal in self.devices().values()]

    def status_frame(self) -> pd.DataFrame:
        reports = self.drift_reports()
        if not reports:
            return pd.DataFrame(columns=["Device", "QC scans", "Latest deviation", "Status"])
        return pd.concat([r.to_frame() for r in reports], ignore_index=True)
