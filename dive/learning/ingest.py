"""Multi-source data ingestion.

New data arrives in shapes nobody planned for: a collaborator's CSV, a second
assay plate, another site's SPSS export, a DEXA report, a wearable summary. This module
maps any of those onto the canonical schema and - critically - refuses
the rows it cannot map.

The design rule is **quarantine, never coerce**. A row with a grip strength of
250 kg or a B/T ratio of 40 is not silently clipped into range; it is held back
with a reason attached. Data that silently deforms on the way in is how a
model learns something false and then states it confidently.

Every accepted batch is written to the ingestion store with a manifest
recording where it came from, what was dropped and why, so any later model can
be traced back to the exact rows it learned from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import SETTINGS, Settings

__all__ = [
    "FieldSpec",
    "CANONICAL_SCHEMA",
    "effective_schema",
    "DataSource",
    "IngestResult",
    "SourceRegistry",
    "ingest_file",
]


# --------------------------------------------------------------------------
# Canonical schema
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    """One canonical field: its type, plausible range, and accepted aliases."""

    name: str
    dtype: str                      # "float" | "int" | "str" | "date"
    minimum: float | None = None
    maximum: float | None = None
    aliases: tuple[str, ...] = ()
    required: bool = False
    description: str = ""

    def plausible(self, value: float) -> bool:
        if not np.isfinite(value):
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True


#: Ranges are *physiological plausibility* bounds, deliberately generous. They
#: exist to catch unit errors and data corruption (a weight of 6.55 kg, a
#: 12-month weight of 805), not to enforce a narrow expectation.
CANONICAL_SCHEMA: dict[str, FieldSpec] = {
    f.name: f
    for f in [
        FieldSpec("participant_id", "str", aliases=("pid", "id", "wl_id", "subject_id"), required=True,
                  description="Pseudonymous participant identifier"),
        FieldSpec("visit", "str", aliases=("timepoint", "wave", "visit_name"),
                  description="Baseline / PostFall / SixMonth / TwelveMonth / FM"),
        FieldSpec("elapsed_months", "float", 0, 240, ("months_since_baseline", "time_months"),
                  description="Actual elapsed months since baseline"),
        FieldSpec("age_years", "float", 18, 110, ("age",), description="Age at visit"),
        FieldSpec("sex", "float", 0, 1, ("gender", "sex_code"), description="0 = male, 1 = female"),
        FieldSpec("height_m", "float", 1.20, 2.20, ("height",)),
        FieldSpec("weight_kg", "float", 30, 200, ("weight",)),
        FieldSpec("bmi", "float", 12, 60, ("body_mass_index",)),
        FieldSpec("max_grip_kg", "float", 2, 90, ("grip", "max_grip", "grip_kg", "handgrip")),
        FieldSpec("gait_speed_ms", "float", 0.10, 2.50, ("gait_speed", "walk_speed")),
        FieldSpec("chair_time_s", "float", 3, 120, ("chair", "sts_time")),
        FieldSpec("tug_s", "float", 3, 39.9, ("tug", "timed_up_go"),
                  description="Continuous only; 40.0 is a cannot-complete sentinel"),
        FieldSpec("sppb_total", "float", 0, 12, ("sppb",)),
        FieldSpec("sppb_balance", "float", 0, 4), FieldSpec("sppb_chair", "float", 0, 4),
        FieldSpec("sppb_gait", "float", 0, 4),
        FieldSpec("IL6_pg_mL", "float", 0.01, 1000, ("il6", "il_6", "interleukin_6")),
        FieldSpec("sIL6R_ng_mL", "float", 1, 500, ("sil6r", "sil_6r", "soluble_il6r")),
        FieldSpec("sgp130_ng_mL", "float", 10, 2000, ("sgp130", "sgp_130", "soluble_gp130")),
        FieldSpec("BT_ratio", "float", 0.001, 100, ("bt_ratio", "b_t_ratio")),
        FieldSpec("fat_mass_pct", "float", 3, 70, ("body_fat_pct", "fat_pct", "total_region_fat")),
        FieldSpec("asmi", "float", 2, 15, ("appendicular_lean_mass_index", "almi")),
        FieldSpec("vat_mass_g", "float", 0, 8000, ("visceral_fat_g", "vat")),
        FieldSpec("fried_score", "float", 0, 5, ("frailty_score",)),
        FieldSpec("ipaq_coded", "float", 1, 3, ("ipaq_coded_category", "coded_category")),
        FieldSpec("declined", "float", 0, 1, ("decline", "grip_decline")),
        FieldSpec("source_id", "str", description="Which data source this row came from"),
    ]
}


def _normalise(name: object) -> str:
    import re

    text = str(name).strip().lower()
    text = re.sub(r"[^\w]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def effective_schema(settings: Settings | None = None) -> dict[str, FieldSpec]:
    """The canonical schema plus every registered health metric.

    This is what makes Dive extensible: a metric registered through
    ``dive.metrics.MetricRegistry`` becomes ingestable immediately, with its
    own plausible range enforced, without any change to this module.
    """
    from ..metrics import MetricRegistry

    schema = dict(CANONICAL_SCHEMA)
    for metric in MetricRegistry(settings).all().values():
        if metric.name in schema:
            continue
        schema[metric.name] = FieldSpec(
            name=metric.name,
            dtype="float",
            minimum=metric.minimum,
            maximum=metric.maximum,
            aliases=tuple(metric.aliases),
            description=metric.description or metric.label,
        )
    return schema


def build_alias_index(settings: Settings | None = None) -> dict[str, str]:
    """Reverse index from any accepted spelling to its canonical field name."""
    index: dict[str, str] = {}
    for spec in effective_schema(settings).values():
        index[_normalise(spec.name)] = spec.name
        for alias in spec.aliases:
            index[_normalise(alias)] = spec.name
    return index


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

@dataclass
class DataSource:
    """Provenance for one contributing dataset."""

    source_id: str
    label: str
    kind: str = "cohort"              # cohort | assay | wearable | registry | external
    description: str = ""
    trust: float = 1.0                # 0-1; weights the row in retraining
    added_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    column_map: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestResult:
    """What happened to one batch."""

    source_id: str
    accepted: pd.DataFrame
    quarantined: pd.DataFrame
    mapped_columns: dict[str, str]
    unmapped_columns: list[str]
    n_input_rows: int
    issues: list[str] = field(default_factory=list)
    batch_id: str = ""

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    @property
    def n_quarantined(self) -> int:
        return len(self.quarantined)

    def summary(self) -> str:
        return (
            f"{self.n_accepted} of {self.n_input_rows} rows accepted from "
            f"'{self.source_id}'; {self.n_quarantined} quarantined; "
            f"{len(self.unmapped_columns)} columns unmapped."
        )


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

def _read_any(path: Path, sheet: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def map_columns(
    frame: pd.DataFrame,
    explicit: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Map incoming columns onto canonical names.

    ``explicit`` takes precedence and lets a user resolve anything the alias
    index cannot. Returns ``(mapping, unmapped)`` where mapping is
    ``{incoming_column: canonical_name}``.
    """
    index = build_alias_index(settings)
    mapping: dict[str, str] = {}
    unmapped: list[str] = []

    explicit = {_normalise(k): v for k, v in (explicit or {}).items()}

    for column in frame.columns:
        key = _normalise(column)
        if key in explicit:
            mapping[column] = explicit[key]
        elif key in index:
            mapping[column] = index[key]
        else:
            unmapped.append(str(column))
    return mapping, unmapped


def validate_frame(
    frame: pd.DataFrame, settings: Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split a canonical-named frame into accepted and quarantined rows.

    A row is quarantined if any present value falls outside its field's
    plausibility range, or if a required field is missing. The reason is
    recorded on the quarantined row rather than discarded.
    """
    schema = effective_schema(settings)
    issues: list[str] = []
    working = frame.copy()
    reasons: list[list[str]] = [[] for _ in range(len(working))]

    for column in working.columns:
        spec = schema.get(column)
        if spec is None or spec.dtype not in {"float", "int"}:
            continue
        values = pd.to_numeric(working[column], errors="coerce")
        coerced = values.isna() & working[column].notna()
        if coerced.any():
            issues.append(f"{column}: {int(coerced.sum())} value(s) were not numeric.")
        present = values.notna()
        if spec.minimum is not None or spec.maximum is not None:
            out_of_range = present & ~values.apply(lambda v: spec.plausible(v) if np.isfinite(v) else True)
            for position in np.where(out_of_range.to_numpy())[0]:
                reasons[position].append(
                    f"{column}={values.iloc[position]:.4g} outside plausible range "
                    f"[{spec.minimum}, {spec.maximum}]"
                )
        working[column] = values

    for name, spec in schema.items():
        if not spec.required:
            continue
        if name not in working.columns:
            issues.append(f"Required field '{name}' is absent from this source.")
            continue
        missing = working[name].isna() | (working[name].astype(str).str.strip() == "")
        for position in np.where(missing.to_numpy())[0]:
            reasons[position].append(f"required field '{name}' is missing")

    flags = np.array([len(r) > 0 for r in reasons])
    accepted = working[~flags].copy()
    quarantined = working[flags].copy()
    if len(quarantined):
        quarantined["quarantine_reason"] = ["; ".join(r) for r, f in zip(reasons, flags) if f]

    return accepted, quarantined, issues


def ingest_file(
    path: Path | str,
    source: DataSource,
    sheet: str | None = None,
    settings: Settings | None = None,
    store: bool = True,
) -> IngestResult:
    """Read, map, validate and (optionally) store one incoming file."""
    cfg = settings or SETTINGS
    path = Path(path)
    raw = _read_any(path, sheet)

    mapping, unmapped = map_columns(raw, source.column_map, settings=cfg)
    renamed = raw.rename(columns=mapping)
    # Keep only schema columns; unmapped ones are reported, not smuggled in.
    schema = effective_schema(cfg)
    canonical = renamed[[c for c in renamed.columns if c in schema]].copy()

    accepted, quarantined, issues = validate_frame(canonical, settings=cfg)
    accepted["source_id"] = source.source_id
    if len(quarantined):
        quarantined["source_id"] = source.source_id

    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    batch_id = f"{source.source_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{digest}"

    result = IngestResult(
        source_id=source.source_id,
        accepted=accepted,
        quarantined=quarantined,
        mapped_columns=mapping,
        unmapped_columns=unmapped,
        n_input_rows=len(raw),
        issues=issues,
        batch_id=batch_id,
    )

    if store:
        SourceRegistry(cfg).store_batch(source, result, origin=str(path), digest=digest)

    return result


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

class SourceRegistry:
    """On-disk store of ingested batches and their manifests."""

    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or SETTINGS
        self.root = self.cfg.paths.data_dir / "ingested"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    # -- manifest ---------------------------------------------------------

    def _load_manifest(self) -> list[dict]:
        if not self.manifest_path.exists():
            return []
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save_manifest(self, entries: list[dict]) -> None:
        self.manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def store_batch(
        self, source: DataSource, result: IngestResult, origin: str = "", digest: str = ""
    ) -> Path:
        batch_dir = self.root / result.batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        if len(result.accepted):
            result.accepted.to_parquet(batch_dir / "accepted.parquet", index=False)
        if len(result.quarantined):
            result.quarantined.to_parquet(batch_dir / "quarantined.parquet", index=False)

        entry = {
            "batch_id": result.batch_id,
            "source": source.as_dict(),
            "origin": origin,
            "sha256_prefix": digest,
            "ingested_utc": datetime.now(timezone.utc).isoformat(),
            "n_input_rows": result.n_input_rows,
            "n_accepted": result.n_accepted,
            "n_quarantined": result.n_quarantined,
            "mapped_columns": result.mapped_columns,
            "unmapped_columns": result.unmapped_columns,
            "issues": result.issues,
        }
        entries = self._load_manifest()
        entries.append(entry)
        self._save_manifest(entries)
        return batch_dir

    # -- retrieval --------------------------------------------------------

    def list_batches(self) -> pd.DataFrame:
        entries = self._load_manifest()
        if not entries:
            return pd.DataFrame(
                columns=["batch_id", "source", "kind", "n_accepted", "n_quarantined", "ingested_utc"]
            )
        return pd.DataFrame(
            [
                {
                    "batch_id": e["batch_id"],
                    "source": e["source"]["label"],
                    "kind": e["source"].get("kind", ""),
                    "trust": e["source"].get("trust", 1.0),
                    "n_accepted": e["n_accepted"],
                    "n_quarantined": e["n_quarantined"],
                    "ingested_utc": e["ingested_utc"],
                }
                for e in entries
            ]
        )

    def load_all(self, kinds: Sequence[str] | None = None) -> pd.DataFrame:
        """Concatenate every accepted batch into one training frame."""
        frames: list[pd.DataFrame] = []
        for entry in self._load_manifest():
            if kinds and entry["source"].get("kind") not in kinds:
                continue
            path = self.root / entry["batch_id"] / "accepted.parquet"
            if path.exists():
                frame = pd.read_parquet(path)
                frame["_batch_id"] = entry["batch_id"]
                frame["_trust"] = entry["source"].get("trust", 1.0)
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def quarantine_report(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for entry in self._load_manifest():
            path = self.root / entry["batch_id"] / "quarantined.parquet"
            if path.exists():
                frame = pd.read_parquet(path)
                frame["_batch_id"] = entry["batch_id"]
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
