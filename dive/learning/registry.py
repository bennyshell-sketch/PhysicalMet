"""Model registry: versioning, provenance, and champion/challenger promotion.

Every fitted model is stored with the metrics it earned, the data it saw and
the code path that produced it. Nothing becomes the active ("champion") model
by being newest - it has to beat the incumbent on held-out data first.

That gate is what makes continual learning safe. A system that retrains on
whatever arrives and always adopts the new fit will happily drift into
predicting noise from a small or skewed batch. Here the incumbent stays until
a challenger demonstrably outperforms it.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import SETTINGS, Settings

__all__ = ["ModelVersion", "ModelRegistry"]


@dataclass
class ModelVersion:
    """One stored model version and everything needed to judge it."""

    version_id: str
    model_kind: str                       # "decline" | "trajectory"
    created_utc: str
    metrics: dict[str, float] = field(default_factory=dict)
    n_training: int = 0
    n_participants: int = 0
    features: list[str] = field(default_factory=list)
    data_batches: list[str] = field(default_factory=list)
    notes: str = ""
    is_champion: bool = False
    promoted_utc: str | None = None
    parent_version: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def primary_metric(self) -> float:
        """The metric promotion is judged on, per model kind."""
        if self.model_kind == "decline":
            return float(self.metrics.get("auc_cv", float("nan")))
        return -float(self.metrics.get("rmse_holdout", float("inf")))


class ModelRegistry:
    """File-backed registry under ``data/models``."""

    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or SETTINGS
        self.root = self.cfg.paths.model_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"

    # -- index ------------------------------------------------------------

    def _load(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save(self, entries: list[dict]) -> None:
        self.index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    # -- registration -----------------------------------------------------

    def register(
        self,
        model: Any,
        model_kind: str,
        metrics: dict[str, float],
        n_training: int,
        features: list[str] | None = None,
        n_participants: int = 0,
        data_batches: list[str] | None = None,
        notes: str = "",
        parent_version: str | None = None,
    ) -> ModelVersion:
        """Store a fitted model. Registration never promotes it."""
        stamp = datetime.now(timezone.utc)
        version_id = f"{model_kind}-{stamp:%Y%m%dT%H%M%S%f}"[:-3]

        version = ModelVersion(
            version_id=version_id,
            model_kind=model_kind,
            created_utc=stamp.isoformat(),
            metrics={k: float(v) for k, v in metrics.items()},
            n_training=n_training,
            n_participants=n_participants,
            features=list(features or []),
            data_batches=list(data_batches or []),
            notes=notes,
            parent_version=parent_version,
        )

        path = self.root / f"{version_id}.pkl"
        try:
            with path.open("wb") as handle:
                pickle.dump(model, handle)
        except (pickle.PicklingError, TypeError, AttributeError) as exc:
            version.notes += f" [model object not serialisable: {exc}]"

        entries = self._load()
        entries.append(version.as_dict())
        self._save(entries)
        return version

    def load_model(self, version_id: str) -> Any:
        path = self.root / f"{version_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No stored artefact for version {version_id}")
        with path.open("rb") as handle:
            return pickle.load(handle)

    # -- champion ---------------------------------------------------------

    def champion(self, model_kind: str) -> ModelVersion | None:
        for entry in self._load():
            if entry["model_kind"] == model_kind and entry.get("is_champion"):
                return ModelVersion(**entry)
        return None

    def promote(self, version_id: str, reason: str = "") -> ModelVersion:
        """Make a version the champion, demoting the incumbent of its kind."""
        entries = self._load()
        target = next((e for e in entries if e["version_id"] == version_id), None)
        if target is None:
            raise KeyError(f"Unknown version: {version_id}")

        for entry in entries:
            if entry["model_kind"] == target["model_kind"]:
                entry["is_champion"] = False
        target["is_champion"] = True
        target["promoted_utc"] = datetime.now(timezone.utc).isoformat()
        if reason:
            target["notes"] = (target.get("notes", "") + f" | promoted: {reason}").strip(" |")

        self._save(entries)
        return ModelVersion(**target)

    def history(self, model_kind: str | None = None) -> pd.DataFrame:
        entries = self._load()
        if model_kind:
            entries = [e for e in entries if e["model_kind"] == model_kind]
        if not entries:
            return pd.DataFrame(
                columns=["version_id", "model_kind", "created_utc", "n_training", "is_champion"]
            )
        rows = []
        for entry in sorted(entries, key=lambda e: e["created_utc"], reverse=True):
            row = {
                "version_id": entry["version_id"],
                "model_kind": entry["model_kind"],
                "created_utc": entry["created_utc"],
                "n_training": entry["n_training"],
                "champion": "yes" if entry.get("is_champion") else "",
            }
            row.update({k: round(v, 4) for k, v in entry.get("metrics", {}).items()})
            rows.append(row)
        return pd.DataFrame(rows)
