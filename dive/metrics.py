"""Registry of health metrics Dive can learn from.

Dive ships with a core schema covering physical function, body composition,
imaging-derived measures (DEXA bone density and CT body composition), routine
bloods, frailty, cognition and activity. Anything else is registered here at
runtime and becomes available to ingestion, validation and model fitting
without a code change.

A note on imaging. Dive consumes **derived measurements** - the numbers a
radiologist, a DEXA or pQCT report, an MR spectroscopy fit or a segmentation
tool produces (T-score, trabecular vBMD, stress-strain index, marrow proton
density fat fraction, L3 skeletal muscle area) - not raw DICOM pixel data. That is a deliberate boundary. Learning directly from images needs
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
    "biochemical",
    "microbiological",
    "histological",
    "anatomical",
    "genetic",
    "epigenetic",
    "context",          # geography, health system, deprivation - see note below
    "demographic",      # self-reported social categories - NOT biology
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

    # --- pQCT: volumetric density, geometry and strength ------------------
    # Site convention: 4% of limb length = distal, trabecular-rich; 38%/66% =
    # diaphyseal, cortical. Ranges are generous plausibility bounds covering
    # both the radius and the tibia across adult ages.
    HealthMetric("pqct_total_vbmd_4pct", "pQCT total vBMD (4% site)", "mg/cm3", "imaging",
                 40, 600, True, ("total_vbmd_4", "trab_site_total_density"),
                 "Volumetric density of the whole distal cross-section"),
    HealthMetric("pqct_trabecular_vbmd", "pQCT trabecular vBMD", "mg/cm3", "imaging",
                 20, 400, True, ("trab_vbmd", "trabecular_density", "trbd"),
                 "Trabecular volumetric density at the 4% site"),
    HealthMetric("pqct_cortical_vbmd", "pQCT cortical vBMD", "mg/cm3", "imaging",
                 600, 1400, True, ("cort_vbmd", "cortical_density", "crtbd"),
                 "Cortical volumetric density; falls with increasing porosity"),
    HealthMetric("pqct_cortical_thickness_mm", "pQCT cortical thickness", "mm", "imaging",
                 0.2, 12.0, True, ("cort_thickness", "crtthk")),
    HealthMetric("pqct_cortical_area_mm2", "pQCT cortical area", "mm2", "imaging",
                 20, 700, True, ("cort_area", "crtar")),
    HealthMetric("pqct_total_area_mm2", "pQCT total bone area", "mm2", "imaging",
                 50, 1600, None, ("total_bone_area", "ttar")),
    HealthMetric("pqct_periosteal_circ_mm", "pQCT periosteal circumference", "mm", "imaging",
                 20, 160, None, ("periosteal_circumference", "peri_c")),
    HealthMetric("pqct_endosteal_circ_mm", "pQCT endosteal circumference", "mm", "imaging",
                 10, 140, False, ("endosteal_circumference", "endo_c")),
    HealthMetric("pqct_ssi_mm3", "pQCT stress-strain index (polar)", "mm3", "imaging",
                 50, 4000, True, ("ssi", "ssi_polar", "stress_strain_index"),
                 "Density-weighted section modulus; the pQCT proxy for bone bending strength"),
    HealthMetric("pqct_bsi", "pQCT bone strength index", "mg2/cm4", "imaging",
                 0, 3.0, True, ("bsi", "bone_strength_index")),
    HealthMetric("pqct_muscle_csa_mm2", "pQCT muscle cross-sectional area", "mm2", "imaging",
                 1000, 12000, True, ("muscle_csa", "mcsa")),
    HealthMetric("pqct_fat_csa_mm2", "pQCT fat cross-sectional area", "mm2", "imaging",
                 100, 12000, False, ("fat_csa", "fcsa")),
    HealthMetric("pqct_muscle_density", "pQCT muscle density", "mg/cm3", "imaging",
                 40, 100, True, ("muscle_vbmd", "muscle_radiodensity_pqct"),
                 "Lower values indicate greater intramuscular fat"),

    # --- bone marrow adiposity -------------------------------------------
    # MR spectroscopy PDFF is the reference method; water-fat imaging and CT
    # attenuation are the practical alternatives. All three are stored
    # separately because they are NOT interchangeable - see
    # dive/calibration.py for why Dive refuses to convert between them.
    HealthMetric("bmat_pdff_pct", "Marrow fat fraction (MRS PDFF)", "%", "imaging",
                 0, 100, False, ("marrow_pdff", "bmat_pdff", "mrs_pdff"),
                 "Proton density fat fraction by MR spectroscopy - the reference method"),
    HealthMetric("bmat_ff_mri_pct", "Marrow fat fraction (water-fat MRI)", "%", "imaging",
                 0, 100, False, ("marrow_ff_mri", "dixon_ff", "water_fat_ff"),
                 "Chemical-shift-encoded water-fat imaging fat fraction"),
    HealthMetric("bmat_volume_cm3", "Marrow adipose tissue volume", "cm3", "imaging",
                 0, 500, False, ("mat_volume", "marrow_fat_volume")),
    HealthMetric("bmat_site", "Marrow adiposity site", "", "imaging",
                 None, None, None, ("bmat_region", "marrow_site"),
                 "Anatomical site: lumbar spine, femoral neck, tibia - values are "
                 "site-specific and must not be pooled across sites"),

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

    # --- biochemical: bone turnover -------------------------------------
    # P1NP and CTX are the IOF/IFCC reference markers for formation and
    # resorption respectively. Both are strongly affected by fasting state and
    # time of day; record collection conditions or the value is not comparable.
    HealthMetric("p1np_ug_L", "Procollagen type 1 N-propeptide (P1NP)", "ug/L", "biochemical",
                 1, 500, None, ("p1np", "pinp"), "Reference marker of bone formation"),
    HealthMetric("ctx_ng_L", "C-telopeptide of type 1 collagen (CTX)", "ng/L", "biochemical",
                 20, 3000, False, ("ctx", "ctx1", "beta_ctx"),
                 "Reference marker of bone resorption; fasting morning sample required"),
    HealthMetric("bone_alp_ug_L", "Bone-specific alkaline phosphatase", "ug/L", "biochemical",
                 2, 150, None, ("bone_alp", "balp")),
    HealthMetric("osteocalcin_ug_L", "Osteocalcin", "ug/L", "biochemical",
                 1, 150, None, ("osteocalcin", "bgp")),
    HealthMetric("pth_pmol_L", "Parathyroid hormone", "pmol/L", "biochemical",
                 0.5, 100, None, ("pth", "parathyroid_hormone")),
    HealthMetric("sclerostin_pmol_L", "Sclerostin", "pmol/L", "biochemical",
                 5, 200, False, ("sclerostin",)),

    # --- biochemical: endocrine and metabolic ---------------------------
    HealthMetric("igf1_nmol_L", "Insulin-like growth factor 1", "nmol/L", "biochemical",
                 1, 80, True, ("igf1", "igf_1")),
    HealthMetric("testosterone_nmol_L", "Total testosterone", "nmol/L", "biochemical",
                 0.1, 60, None, ("testosterone", "total_testosterone")),
    HealthMetric("oestradiol_pmol_L", "Oestradiol", "pmol/L", "biochemical",
                 5, 2000, None, ("oestradiol", "estradiol", "e2")),
    HealthMetric("shbg_nmol_L", "Sex hormone binding globulin", "nmol/L", "biochemical",
                 5, 250, None, ("shbg",)),
    HealthMetric("cortisol_nmol_L", "Morning cortisol", "nmol/L", "biochemical",
                 50, 1500, False, ("cortisol",)),
    HealthMetric("hba1c_mmol_mol", "HbA1c", "mmol/mol", "biochemical",
                 15, 200, False, ("hba1c", "a1c"), "IFCC units; 48 mmol/mol = 6.5%"),
    HealthMetric("fasting_glucose_mmol_L", "Fasting glucose", "mmol/L", "biochemical",
                 2, 35, False, ("glucose", "fasting_glucose")),
    HealthMetric("homa_ir", "HOMA-IR", "", "biochemical",
                 0.1, 30, False, ("homa", "homa_ir_index")),
    HealthMetric("tsh_mU_L", "Thyroid stimulating hormone", "mU/L", "biochemical",
                 0.01, 100, None, ("tsh",)),

    # --- biochemical: renal, nutritional, muscle -------------------------
    HealthMetric("egfr_ml_min", "Estimated GFR", "mL/min/1.73m2", "biochemical",
                 3, 160, True, ("egfr", "gfr"),
                 "Use a race-free equation (CKD-EPI 2021); see dive/fairness.py"),
    HealthMetric("cystatin_c_mg_L", "Cystatin C", "mg/L", "biochemical",
                 0.3, 8, False, ("cystatin_c",)),
    HealthMetric("creatinine_umol_L", "Serum creatinine", "umol/L", "biochemical",
                 20, 1500, None, ("creatinine",)),
    HealthMetric("creatine_kinase_U_L", "Creatine kinase", "U/L", "biochemical",
                 10, 5000, None, ("ck", "creatine_kinase")),
    HealthMetric("prealbumin_g_L", "Prealbumin (transthyretin)", "g/L", "biochemical",
                 0.02, 0.6, True, ("prealbumin", "transthyretin")),
    HealthMetric("ferritin_ug_L", "Ferritin", "ug/L", "biochemical",
                 1, 3000, None, ("ferritin",)),
    HealthMetric("vitamin_b12_pmol_L", "Vitamin B12", "pmol/L", "biochemical",
                 20, 2000, True, ("b12", "vitamin_b12")),
    HealthMetric("gdf15_pg_mL", "Growth differentiation factor 15", "pg/mL", "biochemical",
                 100, 20000, False, ("gdf15", "gdf_15")),
    HealthMetric("tnf_alpha_pg_mL", "Tumour necrosis factor alpha", "pg/mL", "biochemical",
                 0.1, 500, False, ("tnfa", "tnf_alpha")),

    # --- microbiological -------------------------------------------------
    # Diversity indices are computed from a taxonomic table; the pipeline
    # (16S region or shotgun, rarefaction depth, reference database) must be
    # recorded, because indices are not comparable across pipelines.
    HealthMetric("gut_shannon_diversity", "Gut microbiome Shannon diversity", "", "microbiological",
                 0, 8, True, ("shannon", "shannon_index", "alpha_shannon")),
    HealthMetric("gut_simpson_diversity", "Gut microbiome Simpson index", "", "microbiological",
                 0, 1, True, ("simpson", "simpson_index")),
    HealthMetric("gut_chao1", "Gut microbiome Chao1 richness", "taxa", "microbiological",
                 1, 5000, True, ("chao1", "richness")),
    HealthMetric("gut_observed_asvs", "Observed ASVs", "count", "microbiological",
                 1, 5000, True, ("observed_asvs", "observed_otus")),
    HealthMetric("firmicutes_bacteroidetes_ratio", "Firmicutes:Bacteroidetes ratio", "",
                 "microbiological", 0.01, 100, None, ("fb_ratio", "f_b_ratio")),
    HealthMetric("faecal_butyrate_umol_g", "Faecal butyrate", "umol/g", "microbiological",
                 0, 200, True, ("butyrate", "faecal_butyrate")),
    HealthMetric("faecal_calprotectin_ug_g", "Faecal calprotectin", "ug/g", "microbiological",
                 5, 6000, False, ("calprotectin",)),
    HealthMetric("faecal_scfa_total_umol_g", "Total short-chain fatty acids", "umol/g",
                 "microbiological", 0, 500, True, ("scfa_total", "total_scfa")),
    HealthMetric("lps_binding_protein_ug_mL", "LPS-binding protein", "ug/mL", "microbiological",
                 0.5, 200, False, ("lbp", "lps_binding_protein")),

    # --- histological ----------------------------------------------------
    # Muscle biopsy morphometry, and bone histomorphometry in the standardised
    # ASBMR nomenclature.
    HealthMetric("type1_fibre_pct", "Type I fibre proportion", "%", "histological",
                 5, 95, None, ("type_1_fibres", "slow_fibre_pct")),
    HealthMetric("type2a_fibre_pct", "Type IIa fibre proportion", "%", "histological",
                 2, 90, None, ("type_2a_fibres",)),
    HealthMetric("type2x_fibre_pct", "Type IIx fibre proportion", "%", "histological",
                 0, 70, False, ("type_2x_fibres",)),
    HealthMetric("type2_fibre_csa_um2", "Type II fibre cross-sectional area", "um2",
                 "histological", 500, 12000, True, ("type_2_csa", "fast_fibre_csa")),
    HealthMetric("capillary_to_fibre_ratio", "Capillary-to-fibre ratio", "", "histological",
                 0.5, 5, True, ("c_f_ratio", "capillary_fibre_ratio")),
    HealthMetric("satellite_cells_per_fibre", "Satellite cells per fibre", "", "histological",
                 0, 1.5, True, ("satellite_cell_density",)),
    HealthMetric("bone_volume_fraction", "Bone volume fraction (BV/TV)", "%", "histological",
                 1, 60, True, ("bv_tv", "bone_volume_over_total_volume")),
    HealthMetric("trabecular_thickness_um", "Trabecular thickness (Tb.Th)", "um",
                 "histological", 20, 300, True, ("tb_th",)),
    HealthMetric("trabecular_number_per_mm", "Trabecular number (Tb.N)", "/mm",
                 "histological", 0.2, 5, True, ("tb_n",)),
    HealthMetric("trabecular_separation_um", "Trabecular separation (Tb.Sp)", "um",
                 "histological", 100, 3000, False, ("tb_sp",)),
    HealthMetric("mineral_apposition_rate", "Mineral apposition rate (MAR)", "um/day",
                 "histological", 0.1, 3, True, ("mar",)),
    HealthMetric("bone_formation_rate", "Bone formation rate (BFR/BS)", "um3/um2/year",
                 "histological", 0, 200, True, ("bfr_bs", "bfr")),
    HealthMetric("osteoid_surface_pct", "Osteoid surface (OS/BS)", "%", "histological",
                 0, 100, None, ("os_bs", "osteoid_surface")),

    # --- gross anatomical -------------------------------------------------
    HealthMetric("tibia_length_mm", "Tibia length", "mm", "anatomical",
                 250, 500, None, ("tibial_length",),
                 "Required to locate pQCT measurement sites by percentage"),
    HealthMetric("forearm_length_mm", "Forearm length", "mm", "anatomical",
                 180, 350, None, ("radius_length", "ulna_length")),
    HealthMetric("femur_neck_axis_length_mm", "Hip axis length", "mm", "anatomical",
                 80, 140, False, ("hal", "hip_axis_length"),
                 "Longer hip axis length is associated with higher hip fracture risk"),
    HealthMetric("arm_span_cm", "Arm span", "cm", "anatomical",
                 120, 220, None, ("armspan",)),
    HealthMetric("kyphosis_angle_deg", "Thoracic kyphosis angle", "degrees", "anatomical",
                 10, 110, False, ("cobb_angle", "kyphosis")),
    HealthMetric("mid_thigh_circumference_cm", "Mid-thigh circumference", "cm", "anatomical",
                 25, 90, True, ("thigh_circumference",)),
    HealthMetric("hip_circumference_cm", "Hip circumference", "cm", "anatomical",
                 60, 180, None, ("hip_circ",)),
    HealthMetric("knee_extension_rom_deg", "Knee extension range of motion", "degrees",
                 "anatomical", 0, 160, True, ("knee_rom",)),

    # --- genetic ----------------------------------------------------------
    # Genotypes are stored as allele dosage (0/1/2 copies of the effect allele)
    # and polygenic scores as standardised values. Dive asserts NOTHING about
    # what any of these predict: whether a variant carries signal in YOUR
    # cohort is decided by dive/learning/feature_search.py against a
    # permutation null, not by literature claim. See docs/OMICS.md.
    HealthMetric("actn3_r577x_dosage", "ACTN3 R577X (rs1815739) dosage", "alleles", "genetic",
                 0, 2, None, ("actn3", "rs1815739")),
    HealthMetric("vdr_fokI_dosage", "VDR FokI (rs2228570) dosage", "alleles", "genetic",
                 0, 2, None, ("vdr_foki", "rs2228570")),
    HealthMetric("col1a1_sp1_dosage", "COL1A1 Sp1 (rs1800012) dosage", "alleles", "genetic",
                 0, 2, None, ("col1a1", "rs1800012")),
    HealthMetric("apoe_e4_dosage", "APOE e4 allele dosage", "alleles", "genetic",
                 0, 2, None, ("apoe4", "apoe_e4")),
    HealthMetric("fto_rs9939609_dosage", "FTO rs9939609 dosage", "alleles", "genetic",
                 0, 2, None, ("fto", "rs9939609")),
    HealthMetric("prs_ebmd", "Polygenic score, estimated BMD", "SD", "genetic",
                 -6, 6, True, ("prs_bmd", "ebmd_prs"),
                 "Standardised within the cohort; PRS transfer across ancestries is poor"),
    HealthMetric("prs_grip_strength", "Polygenic score, grip strength", "SD", "genetic",
                 -6, 6, True, ("prs_grip",)),
    HealthMetric("prs_lean_mass", "Polygenic score, lean mass", "SD", "genetic",
                 -6, 6, True, ("prs_lean",)),
    HealthMetric("genetic_ancestry_pc1", "Genetic ancestry principal component 1", "",
                 "genetic", -1, 1, None, ("ancestry_pc1", "pc1"),
                 "Continuous genetic ancestry. This is NOT race or ethnicity - see "
                 "dive/fairness.py for why the distinction matters"),
    HealthMetric("genetic_ancestry_pc2", "Genetic ancestry principal component 2", "",
                 "genetic", -1, 1, None, ("ancestry_pc2", "pc2")),
    HealthMetric("genetic_ancestry_pc3", "Genetic ancestry principal component 3", "",
                 "genetic", -1, 1, None, ("ancestry_pc3", "pc3")),

    # --- epigenetic -------------------------------------------------------
    # Epigenetic clocks. Age acceleration is the residual of predicted age on
    # chronological age, so it must be computed within the cohort, not taken
    # from a vendor report calibrated on a different population.
    HealthMetric("dnam_age_horvath", "DNAm age (Horvath)", "years", "epigenetic",
                 0, 130, False, ("horvath_age", "dnam_horvath")),
    HealthMetric("dnam_age_hannum", "DNAm age (Hannum)", "years", "epigenetic",
                 0, 130, False, ("hannum_age",)),
    HealthMetric("dnam_phenoage", "DNAm PhenoAge", "years", "epigenetic",
                 0, 130, False, ("phenoage",)),
    HealthMetric("dnam_grimage", "DNAm GrimAge", "years", "epigenetic",
                 0, 130, False, ("grimage",)),
    HealthMetric("dunedin_pace", "DunedinPACE rate of ageing", "years/year", "epigenetic",
                 0.4, 2.5, False, ("pace", "dunedinpace")),
    HealthMetric("age_acceleration_years", "Epigenetic age acceleration", "years",
                 "epigenetic", -30, 30, False, ("age_accel", "epigenetic_age_acceleration"),
                 "Residual of DNAm age on chronological age, computed within this cohort"),
    HealthMetric("telomere_length_kb", "Leukocyte telomere length", "kb", "epigenetic",
                 2, 15, True, ("telomere_length", "ltl")),
    HealthMetric("methylation_array_platform", "Methylation array platform", "", "epigenetic",
                 None, None, None, ("array_platform", "methylation_platform"),
                 "450K / EPIC / EPICv2 - clocks are not comparable across platforms"),

    # --- context: geography, system, deprivation -------------------------
    # Legitimate uses are reference-range selection, assay standardisation and
    # fairness auditing. These are NOT biological predictors; see
    # dive/fairness.py, which excludes them from models by default.
    HealthMetric("country_iso3", "Country (ISO 3166-1 alpha-3)", "", "context",
                 None, None, None, ("country", "iso3")),
    HealthMetric("region_code", "Sub-national region", "", "context",
                 None, None, None, ("region", "nuts_code", "state")),
    HealthMetric("latitude_band", "Latitude band", "", "context",
                 None, None, None, ("lat_band",),
                 "Coarse band, relevant to vitamin D synthesis; never a coordinate"),
    HealthMetric("urban_rural", "Urban or rural", "", "context",
                 None, None, None, ("settlement_type",)),
    HealthMetric("deprivation_decile", "Area deprivation decile", "1-10", "context",
                 1, 10, True, ("imd", "imd_decile", "adi_decile", "seifa"),
                 "1 = most deprived. Index differs by country and is not comparable across them"),
    HealthMetric("healthcare_access_index", "Healthcare access index", "", "context",
                 0, 100, True, ("hcai", "access_index")),

    # --- demographic: self-reported social categories --------------------
    # Recorded for FAIRNESS AUDITING, not prediction. Dive excludes these from
    # model features by default and requires an explicit, logged override.
    # Race and ethnicity are social constructs, not biological variables, and
    # clinical algorithms that used them as biology have caused documented
    # harm. See dive/fairness.py.
    HealthMetric("ethnicity_self_reported", "Ethnicity (self-reported)", "", "demographic",
                 None, None, None, ("ethnicity", "ethnic_group", "race"),
                 "Self-reported social category. Used to AUDIT model fairness, "
                 "not as a predictor"),
    HealthMetric("ethnicity_classification", "Ethnicity classification system", "",
                 "demographic", None, None, None, ("ethnicity_system",),
                 "e.g. ONS-2021, OMB-1997 - categories are not comparable across systems"),
    HealthMetric("preferred_language", "Preferred language", "", "demographic",
                 None, None, None, ("language",)),
    HealthMetric("migration_generation", "Migration generation", "", "demographic",
                 None, None, None, ("generation_status",)),
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
