"""Registry of health metrics Dive can learn from.

Dive ships with a core schema covering physical function, body composition,
imaging-derived measures (DEXA bone density and CT body composition), routine
bloods, frailty, cognition and activity. Anything else is registered here at
runtime and becomes available to ingestion, validation and model fitting
without a code change.

A note on imaging. Dive consumes **derived measurements** - the numbers a
radiologist, a DEXA report or a segmentation tool produces (T-score, L3
skeletal muscle area, muscle attenuation in Hounsfield units) - not raw DICOM
pixel data. That is a deliberate boundary. Learning directly from images needs
sample sizes in the thousands and a segmentation pipeline that is itself a
regulated piece of software; consuming validated derived measures gets most of
the signal at a fraction of the risk, and keeps every predictor something a
clinician can interrogate.

Registering a metric does **not** put it into a model. It makes the metric
*eligible*. Whether it earns a place is decided by
``dive.learning.feature_search``, which tests it on held-out participants and
adopts it only if it genuinely improves out-of-sample prediction. That split
matters: a system that automatically fits every new variable it is handed will
find structure in noise, and with enough candidate variables it will find it
reliably.

Every metric carries a plausible range. That range is not a preference about
what the data should look like - it is the unit-error detector. A grip of
250 kg, a haemoglobin of 1400 g/L and a step count of 900,000 are all data
entry or unit-conversion faults, and they are quarantined rather than learned.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from .config import SETTINGS, Settings

__all__ = ["HealthMetric", "MetricRegistry", "CORE_METRICS", "slugify"]

MetricDomain = Literal[
    "physical_function",
    "body_composition",
    "imaging",
    "inflammation",
    "cognition",
    "activity",
    "clinical",
    "other",
]


def slugify(name: str) -> str:
    """Normalise a metric name into a safe column identifier."""
    text = re.sub(r"[^\w]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


@dataclass
class HealthMetric:
    """One measurable quantity Dive can ingest and potentially learn from."""

    name: str
    label: str
    unit: str = ""
    domain: MetricDomain = "other"
    minimum: float | None = None
    maximum: float | None = None
    #: True if higher values indicate better health. Used only for phrasing
    #: recommendations - never for modelling, where direction is estimated.
    higher_is_better: bool | None = None
    aliases: tuple[str, ...] = ()
    description: str = ""
    source: str = ""
    added_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    #: Set by the feature search once the metric has been shown to help.
    adopted_by: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = slugify(self.name)
        if not self.label:
            self.label = self.name.replace("_", " ").title()
        if self.minimum is not None and self.maximum is not None and self.minimum >= self.maximum:
            raise ValueError(
                f"Metric '{self.name}': minimum ({self.minimum}) must be below "
                f"maximum ({self.maximum})."
            )

    def plausible(self, value: float) -> bool:
        if value != value:  # NaN
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True

    def range_text(self) -> str:
        if self.minimum is None and self.maximum is None:
            return "unbounded"
        low = "-inf" if self.minimum is None else f"{self.minimum:g}"
        high = "+inf" if self.maximum is None else f"{self.maximum:g}"
        return f"{low} to {high}{(' ' + self.unit) if self.unit else ''}"

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["aliases"] = list(self.aliases)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "HealthMetric":
        payload = dict(payload)
        payload["aliases"] = tuple(payload.get("aliases", ()))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


#: Metrics Dive understands out of the box. Ranges are physiological
#: plausibility bounds, deliberately generous - they catch unit errors, not
#: unusual participants.
CORE_METRICS: tuple[HealthMetric, ...] = (
    HealthMetric("max_grip_kg", "Maximum grip strength", "kg", "physical_function",
                 2, 90, True, ("grip", "max_grip", "handgrip", "grip_kg")),
    HealthMetric("gait_speed_ms", "Gait speed", "m/s", "physical_function",
                 0.10, 2.50, True, ("gait_speed", "walk_speed")),
    HealthMetric("sppb_total", "SPPB total", "points", "physical_function",
                 0, 12, True, ("sppb",)),
    HealthMetric("chair_time_s", "Five chair rises", "s", "physical_function",
                 3, 120, False, ("chair", "sts_time")),
    HealthMetric("tug_s", "Timed Up and Go", "s", "physical_function",
                 3, 39.9, False, ("tug", "timed_up_go")),
    HealthMetric("bmi", "Body mass index", "kg/m2", "body_composition",
                 12, 60, None, ("body_mass_index",)),
    HealthMetric("waist_cm", "Waist circumference", "cm", "body_composition",
                 50, 200, False, ("waist",)),
    HealthMetric("fat_mass_pct", "Body fat", "%", "body_composition",
                 3, 70, False, ("body_fat_pct", "fat_pct")),
    HealthMetric("asmi", "Appendicular skeletal muscle index", "kg/m2", "body_composition",
                 2, 15, True, ("almi", "appendicular_lean_mass_index")),
    HealthMetric("vat_mass_g", "Visceral adipose tissue", "g", "body_composition",
                 0, 8000, False, ("vat", "visceral_fat_g")),
    HealthMetric("IL6_pg_mL", "Interleukin-6", "pg/mL", "inflammation",
                 0.01, 1000, False, ("il6", "il_6", "interleukin_6")),
    HealthMetric("sIL6R_ng_mL", "Soluble IL-6 receptor", "ng/mL", "inflammation",
                 1, 500, None, ("sil6r", "soluble_il6r")),
    HealthMetric("sgp130_ng_mL", "Soluble gp130", "ng/mL", "inflammation",
                 10, 2000, None, ("sgp130", "soluble_gp130")),
    HealthMetric("BT_ratio", "B/T ratio", "", "inflammation",
                 0.001, 100, None, ("bt_ratio", "b_t_ratio")),
    HealthMetric("fried_score", "Fried frailty score", "criteria", "clinical",
                 0, 5, False, ("frailty_score",)),

    # --- DEXA-derived -----------------------------------------------------
    HealthMetric("bmd_total_hip", "Bone mineral density, total hip", "g/cm2", "imaging",
                 0.2, 1.8, True, ("bmd_hip", "total_hip_bmd"),
                 "DEXA areal bone mineral density at the total hip"),
    HealthMetric("bmd_femoral_neck", "Bone mineral density, femoral neck", "g/cm2", "imaging",
                 0.2, 1.8, True, ("bmd_neck", "femoral_neck_bmd")),
    HealthMetric("bmd_lumbar_spine", "Bone mineral density, lumbar spine", "g/cm2", "imaging",
                 0.2, 2.2, True, ("bmd_spine", "lumbar_spine_bmd", "l1_l4_bmd")),
    HealthMetric("bmd_t_score", "Bone density T-score", "SD", "imaging",
                 -6.0, 4.0, True, ("t_score", "dexa_t_score"),
                 "T-score at the lowest measured site; <= -2.5 meets the WHO osteoporosis threshold"),
    HealthMetric("bmd_z_score", "Bone density Z-score", "SD", "imaging",
                 -6.0, 4.0, True, ("z_score", "dexa_z_score")),
    HealthMetric("trabecular_bone_score", "Trabecular bone score", "", "imaging",
                 0.5, 2.0, True, ("tbs",), "DEXA texture index of trabecular microarchitecture"),
    HealthMetric("lean_mass_kg", "Total lean mass", "kg", "imaging",
                 15, 100, True, ("total_lean_mass", "lean_mass")),
    HealthMetric("appendicular_lean_mass_kg", "Appendicular lean mass", "kg", "imaging",
                 5, 50, True, ("alm", "appendicular_lean_mass")),

    # --- CT-derived -------------------------------------------------------
    HealthMetric("ct_muscle_area_cm2", "CT skeletal muscle area (L3)", "cm2", "imaging",
                 40, 400, True, ("l3_smi_area", "skeletal_muscle_area", "smi_area"),
                 "Cross-sectional skeletal muscle area at the third lumbar vertebra"),
    HealthMetric("ct_muscle_attenuation_hu", "CT muscle attenuation", "HU", "imaging",
                 -10, 80, True, ("muscle_hu", "muscle_radiodensity", "smd"),
                 "Lower attenuation indicates greater intramuscular fat (myosteatosis)"),
    HealthMetric("ct_vat_area_cm2", "CT visceral adipose area (L3)", "cm2", "imaging",
                 0, 600, False, ("vat_area", "visceral_fat_area")),
    HealthMetric("ct_sat_area_cm2", "CT subcutaneous adipose area (L3)", "cm2", "imaging",
                 0, 900, None, ("sat_area", "subcutaneous_fat_area")),
    HealthMetric("ct_imat_area_cm2", "CT intermuscular adipose area", "cm2", "imaging",
                 0, 120, False, ("imat_area", "intermuscular_fat_area")),
    HealthMetric("vertebral_ct_density_hu", "Vertebral trabecular attenuation", "HU", "imaging",
                 20, 350, True, ("vertebral_hu", "l1_hu", "opportunistic_ct_bmd"),
                 "Opportunistic CT screening measure; < 100 HU suggests osteoporosis"),
    HealthMetric("aortic_calcification_score", "Aortic calcification score", "", "imaging",
                 0, 30, False, ("aac_score", "aortic_calcium")),

    # --- routine bloods ---------------------------------------------------
    HealthMetric("crp_mg_L", "C-reactive protein", "mg/L", "inflammation",
                 0.01, 400, False, ("crp", "c_reactive_protein")),
    HealthMetric("albumin_g_L", "Serum albumin", "g/L", "clinical",
                 10, 60, True, ("albumin",)),
    HealthMetric("haemoglobin_g_L", "Haemoglobin", "g/L", "clinical",
                 40, 220, True, ("hb", "hemoglobin", "haemoglobin")),
    HealthMetric("serum_vitamin_d", "Serum 25-OH vitamin D", "nmol/L", "clinical",
                 5, 400, True, ("vit_d", "vitamin_d_25oh", "25ohd")),
    HealthMetric("daily_step_count", "Daily step count", "steps/day", "activity",
                 0, 40000, True, ("steps", "step_count")),
    HealthMetric("moca", "MoCA", "points", "cognition", 0, 30, True, ("moca_total",)),
    HealthMetric("ipaq_coded", "IPAQ category", "1-3", "activity",
                 1, 3, True, ("ipaq_coded_category", "coded_category")),
)


class MetricRegistry:
    """Persisted registry of core plus user-registered metrics."""

    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or SETTINGS
        self.path = self.cfg.paths.data_dir / "metrics.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- persistence ------------------------------------------------------

    def _load_custom(self) -> dict[str, HealthMetric]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {m["name"]: HealthMetric.from_dict(m) for m in payload}

    def _save_custom(self, metrics: dict[str, HealthMetric]) -> None:
        self.path.write_text(
            json.dumps([m.as_dict() for m in metrics.values()], indent=2), encoding="utf-8"
        )

    # -- access -----------------------------------------------------------

    def all(self) -> dict[str, HealthMetric]:
        """Core metrics plus registered ones. Custom entries win on collision."""
        metrics = {m.name: m for m in CORE_METRICS}
        metrics.update(self._load_custom())
        return metrics

    def custom(self) -> dict[str, HealthMetric]:
        return self._load_custom()

    def get(self, name: str) -> HealthMetric | None:
        return self.all().get(slugify(name))

    def is_core(self, name: str) -> bool:
        return slugify(name) in {m.name for m in CORE_METRICS}

    # -- registration -----------------------------------------------------

    def register(self, metric: HealthMetric, overwrite: bool = False) -> HealthMetric:
        """Add a metric to the registry.

        Core metrics cannot be overwritten - redefining the plausible range of
        grip strength from the interface would silently change what counts as
        valid data in every past and future batch.
        """
        if self.is_core(metric.name) and not overwrite:
            raise ValueError(
                f"'{metric.name}' is a core metric and cannot be redefined. "
                "Register it under a different name if you mean a different measurement."
            )
        custom = self._load_custom()
        if metric.name in custom and not overwrite:
            raise ValueError(
                f"Metric '{metric.name}' is already registered. Pass overwrite=True to replace it."
            )
        custom[metric.name] = metric
        self._save_custom(custom)
        return metric

    def remove(self, name: str) -> bool:
        custom = self._load_custom()
        removed = custom.pop(slugify(name), None)
        if removed is not None:
            self._save_custom(custom)
        return removed is not None

    def mark_adopted(self, name: str, model_kind: str) -> None:
        """Record that a model has adopted this metric as a predictor."""
        custom = self._load_custom()
        metric = custom.get(slugify(name))
        if metric is None:
            return
        if model_kind not in metric.adopted_by:
            metric.adopted_by.append(model_kind)
            self._save_custom(custom)

    # -- integration ------------------------------------------------------

    def alias_index(self) -> dict[str, str]:
        """Map every accepted spelling to its canonical metric name."""
        index: dict[str, str] = {}
        for metric in self.all().values():
            index[slugify(metric.name)] = metric.name
            for alias in metric.aliases:
                index[slugify(alias)] = metric.name
        return index

    def candidate_features(self, exclude: Iterable[str] = ()) -> list[str]:
        """Metric names eligible to be tested as model predictors."""
        skip = {slugify(x) for x in exclude}
        return [name for name in self.all() if name not in skip]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for metric in self.all().values():
            rows.append(
                {
                    "Metric": metric.label,
                    "Name": metric.name,
                    "Unit": metric.unit,
                    "Domain": metric.domain.replace("_", " ").title(),
                    "Plausible range": metric.range_text(),
                    "Origin": "core" if self.is_core(metric.name) else "registered",
                    "Adopted by": ", ".join(metric.adopted_by) or "-",
                }
            )
        return pd.DataFrame(rows).sort_values(["Domain", "Metric"]).reset_index(drop=True)
