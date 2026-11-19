"""Central configuration for Dive.

This module is the **single swap-in point** for real data. To move the whole
pipeline from synthetic to real data, change ``DATA_SOURCE`` (here, or via the
``DIVE_DATA_SOURCE`` environment variable, or via ``settings.local.yaml``).
Nothing else in the codebase needs editing.

Resolution order, lowest priority first:
    1. The defaults written in this file
    2. ``settings.yaml`` in the project root, if present
    3. ``settings.local.yaml`` in the project root, if present (gitignored)
    4. Environment variables prefixed ``DIVE_``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard requirement in practice
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DataSource = Literal["synthetic", "real"]


# --------------------------------------------------------------------------
# Biochemistry constants
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BiochemConstants:
    """Constants for the two-step IL-6 trans-signalling equilibrium.

    Dissociation constants are in **nanomolar**. Molecular weights are in
    daltons and are used to convert assay units (pg/mL, ng/mL) into molarity
    before any equilibrium solving happens.
    """

    kd1_nM: float = 10.0   # IL-6 + sIL-6R  <-> binary complex
    kd2_nM: float = 1.0    # binary + sgp130 <-> ternary complex

    mw_il6_da: float = 21_000.0
    mw_sil6r_da: float = 50_000.0
    mw_sgp130_da: float = 100_000.0

    # Assay units as reported by the ELISA plates
    il6_unit: str = "pg/mL"
    sil6r_unit: str = "ng/mL"
    sgp130_unit: str = "ng/mL"


# --------------------------------------------------------------------------
# Cohort definitions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CohortConfig:
    """Cohort structure and clinical thresholds.

    ``older_age_threshold`` is 65 years: an older participant is aged 65 or
    above. The cohort sizes below describe the reference cohort the synthetic
    generator reproduces; they are configurable per deployment.
    """

    older_age_threshold: float = 65.0

    n_older: int = 281
    n_younger: int = 81

    # Grip-decline longitudinal cohort (used for the trajectory model)
    n_grip_cohort: int = 188
    n_grip_maintained: int = 121
    n_grip_declined: int = 67

    # B/T substudy: case-control, NOT a prevalence sample
    n_bt_substudy: int = 100
    n_bt_decliners: int = 50
    n_bt_stable: int = 50

    #: Grip loss (kg, baseline -> follow-up measure) that defines "decline".
    grip_decline_threshold_kg: float = 2.0
    #: Absolute grip change (kg) within which a participant counts as "stable".
    grip_stable_band_kg: float = 1.0

    #: TUG value used in the source workbook to mean "could not complete".
    tug_sentinel_value: float = 40.0
    #: Grip value used in the source workbook to mean "missing".
    grip_missing_sentinel: float = 999.0

    visits: tuple[str, ...] = ("Baseline", "PostFall", "SixMonth", "TwelveMonth", "FM")

    subgroup_labels: dict[int, str] = field(
        default_factory=lambda: {
            1: "Non-Faller",
            2: "Maintained Function",
            3: "Recovered Function",
            4: "Declined Function",
        }
    )
    subgroup_abbrev: dict[int, str] = field(
        default_factory=lambda: {1: "NF", 2: "MF", 3: "RF", 4: "DF"}
    )


# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Modelling and validation settings.

    The defaults are deliberately conservative. ``conformal_alpha=0.05`` gives
    95% prediction intervals; ``abstain_*`` thresholds control when the system
    declines to issue an individual prediction rather than guessing.
    """

    random_seed: int = 20260916

    # Primary decline model
    logistic_penalty: Literal["l2", "none"] = "l2"
    logistic_C_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    n_bootstrap: int = 2000

    # Cross-validation
    cv_folds: int = 10
    cv_repeats: int = 20
    #: Use leave-one-out instead of k-fold when n is at or below this.
    loo_threshold_n: int = 60

    # Conformal prediction (distribution-free interval coverage)
    conformal_alpha: float = 0.05

    # Abstention guardrails - see dive/guardrails.py
    #: Refuse to predict if a covariate sits outside this quantile range of
    #: the training data (Mahalanobis/extrapolation check).
    abstain_extrapolation_quantile: float = 0.99
    #: Refuse to predict if the conformal interval is wider than this many
    #: units of the training-set outcome SD.
    abstain_interval_sd_multiple: float = 3.0
    #: Minimum training n before individual predictions are released at all.
    abstain_min_training_n: int = 40

    # Sensitivity / robustness models (never the headline result)
    enable_gradient_boosting: bool = True
    enable_shap: bool = True

    # Trajectory model
    trajectory_horizon_months: float = 12.0
    trajectory_max_horizon_months: float = 24.0


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    synthetic_dir: Path = PROJECT_ROOT / "data" / "synthetic"
    real_dir: Path = PROJECT_ROOT / "data" / "real"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    model_dir: Path = PROJECT_ROOT / "data" / "models"
    output_dir: Path = PROJECT_ROOT / "outputs"

    synthetic_workbook_name: str = "dive_synthetic_cohort.xlsx"
    #: Filenames Dive looks for in ``real_dir``. Override per deployment and
    #: flip ``data_source`` to "real"; no other code changes.
    real_bt_workbook_name: str = "cohort_biomarkers.xlsx"
    real_master_workbook_name: str = "cohort_master.xlsx"

    def rerooted(self, data_dir: Path) -> "PathConfig":
        """Move the writable tree under ``data_dir``.

        A hosted deployment gets one writable volume, so a single setting has
        to move everything Dive writes. Each subdirectory follows only if it
        is still sitting at its default position under the old data directory;
        one that was pointed somewhere else deliberately is left where it is,
        so an explicit override is never silently overwritten.

        ``output_dir`` follows too, even though its default sits beside the
        data directory rather than inside it - in a container anything outside
        the volume disappears on restart.
        """
        data_dir = Path(data_dir)
        moves: dict[str, Any] = {"data_dir": data_dir}
        for field_name, leaf in (
            ("synthetic_dir", "synthetic"),
            ("real_dir", "real"),
            ("upload_dir", "uploads"),
            ("model_dir", "models"),
        ):
            if getattr(self, field_name) == self.data_dir / leaf:
                moves[field_name] = data_dir / leaf
        if self.output_dir == self.project_root / "outputs":
            moves["output_dir"] = data_dir / "outputs"
        return replace(self, **moves)

    @property
    def synthetic_workbook(self) -> Path:
        return self.synthetic_dir / self.synthetic_workbook_name

    @property
    def real_bt_workbook(self) -> Path:
        return self.real_dir / self.real_bt_workbook_name

    @property
    def real_master_workbook(self) -> Path:
        return self.real_dir / self.real_master_workbook_name


# --------------------------------------------------------------------------
# Privacy
# --------------------------------------------------------------------------

def _default_privacy():
    """Imported lazily: privacy imports security, which must not import config."""
    from .privacy import PrivacyPolicy

    return PrivacyPolicy()


# --------------------------------------------------------------------------
# Top-level settings object
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    #: THE SWAP-IN POINT. "synthetic" | "real"
    data_source: DataSource = "synthetic"

    #: Shown in the interface masthead. Set per deployment or per study.
    deployment_label: str = "Clinical prediction platform"

    biochem: BiochemConstants = field(default_factory=BiochemConstants)
    privacy: "PrivacyPolicy" = field(default_factory=lambda: _default_privacy())
    cohort: CohortConfig = field(default_factory=CohortConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    #: When True, every individual prediction carries a provenance record and
    #: predictions outside the model's competence are withheld, not guessed.
    strict_evidence_mode: bool = True

    #: Allow the owner console to write to project source files. OFF by
    #: default: editing source through a web interface is remote code
    #: execution by design, and no path filter changes that. See dive/owner.py.
    allow_source_editing: bool = False

    #: Allow self-reported ethnicity or race to be used as a model FEATURE.
    #: Off by default and logged when enabled - see dive/fairness.py for why
    #: this is not a neutral configuration choice.
    allow_protected_features: bool = False

    def active_workbook(self) -> Path:
        """Return the workbook the pipeline should read, given ``data_source``."""
        if self.data_source == "real":
            return self.paths.real_bt_workbook
        return self.paths.synthetic_workbook

    def describe_source(self) -> str:
        path = self.active_workbook()
        kind = "REAL participant data" if self.data_source == "real" else "SYNTHETIC data"
        return f"{kind} - {path}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_ENV_PREFIX = "DIVE_"

_SCALAR_ENV_MAP: dict[str, tuple[str, ...]] = {
    "DATA_SOURCE": ("data_source",),
    "DEPLOYMENT_LABEL": ("deployment_label",),
    "STRICT_EVIDENCE_MODE": ("strict_evidence_mode",),
    "ALLOW_PROTECTED_FEATURES": ("allow_protected_features",),
    "ALLOW_SOURCE_EDITING": ("allow_source_editing",),
    "RANDOM_SEED": ("model", "random_seed"),
    "OLDER_AGE_THRESHOLD": ("cohort", "older_age_threshold"),
    "CONFORMAL_ALPHA": ("model", "conformal_alpha"),
    "EGRESS_MODE": ("privacy", "egress_mode"),
    "OFFLINE_ONLY": ("privacy", "offline_only"),
    # Relocates the whole writable tree - see PathConfig.rerooted.
    "DATA_DIR": ("paths", "data_dir"),
    "OUTPUT_DIR": ("paths", "output_dir"),
}


def _coerce(raw: str) -> Any:
    low = raw.strip().lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """Apply a nested dict of overrides onto a frozen Settings tree."""
    top: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(settings, key):
            continue
        current = getattr(settings, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            sub = {k: v for k, v in value.items() if hasattr(current, k)}
            # Paths need to stay Path objects
            for k, v in list(sub.items()):
                if isinstance(getattr(current, k), Path):
                    sub[k] = Path(v)
            # Moving the data directory has to move what lives under it,
            # otherwise a hosted deployment writes half its state to the
            # volume and half to a container filesystem that is thrown away.
            if isinstance(current, PathConfig) and "data_dir" in sub:
                current = current.rerooted(sub.pop("data_dir"))
            top[key] = replace(current, **sub) if sub else current
        else:
            top[key] = value
    return replace(settings, **top) if top else settings


def _env_overrides() -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for suffix, path in _SCALAR_ENV_MAP.items():
        raw = os.environ.get(_ENV_PREFIX + suffix)
        if raw is None:
            continue
        value = _coerce(raw)
        cursor = nested
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
    return nested


def load_settings(config_file: Path | str | None = None) -> Settings:
    """Build the effective Settings object."""
    settings = Settings()

    candidates: list[Path] = []
    if config_file is not None:
        candidates.append(Path(config_file))
    else:
        candidates.extend([PROJECT_ROOT / "settings.yaml", PROJECT_ROOT / "settings.local.yaml"])

    for candidate in candidates:
        if candidate.exists() and yaml is not None:
            with candidate.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            if isinstance(payload, dict):
                settings = _apply_overrides(settings, payload)

    env = _env_overrides()
    if env:
        settings = _apply_overrides(settings, env)

    return settings


#: Module-level singleton used throughout the pipeline.
SETTINGS = load_settings()

# Convenience aliases so callers can write ``from dive.config import DATA_SOURCE``
DATA_SOURCE: DataSource = SETTINGS.data_source


def ensure_directories(settings: Settings | None = None) -> None:
    """Create the data/output directories the pipeline writes into."""
    cfg = settings or SETTINGS
    for directory in (
        cfg.paths.data_dir,
        cfg.paths.synthetic_dir,
        cfg.paths.real_dir,
        cfg.paths.upload_dir,
        cfg.paths.model_dir,
        cfg.paths.output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
