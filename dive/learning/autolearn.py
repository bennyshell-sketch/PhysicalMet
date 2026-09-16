"""The autonomous learning cycle.

Dive can improve itself without being asked: ingest what has arrived, check
whether the new data looks like the old data, test whether any metric now earns
a place, retrain, and promote the result only if it is demonstrably better.

The word doing the work is *only*. An unsupervised loop that retrains on
whatever arrives and adopts whatever scores highest does not get better over
time - it drifts, confidently, and the drift is invisible because the same loop
that caused it also reports on it. Every stage here can therefore decline to
act, and declining is the default:

* **Ingest** quarantines implausible rows instead of clipping them.
* **Drift** is reported, and a large distribution shift *pauses* the cycle
  rather than training through it - data from a different population is a
  reason to look, not to fit faster.
* **Feature search** must beat a permutation null built from the whole search.
* **Retraining** must beat the incumbent on held-out participants, survive a
  bootstrap, and pass its own permutation test.
* **Promotion** is refused outright for a model that cannot beat chance.

A cycle that changes nothing is a successful cycle. The failure mode this
design is built around is a system that always finds something.

Every run is written to the audit log with its inputs, decisions and reasons,
so the question "why does the model do that now?" has an answer.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import SETTINGS, Settings

__all__ = ["CycleStage", "CycleResult", "AutoLearner"]


@dataclass
class CycleStage:
    name: str
    acted: bool
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    halted: bool = False

    def line(self) -> str:
        mark = "changed" if self.acted else ("HALTED" if self.halted else "no change")
        return f"[{mark}] {self.name}: {self.summary}"


@dataclass
class CycleResult:
    started_utc: str
    finished_utc: str = ""
    stages: list[CycleStage] = field(default_factory=list)
    changed: bool = False
    halted_at: str | None = None
    dry_run: bool = False

    def add(self, stage: CycleStage) -> CycleStage:
        self.stages.append(stage)
        if stage.acted:
            self.changed = True
        if stage.halted:
            self.halted_at = stage.name
        return stage

    def summary(self) -> str:
        if self.halted_at:
            stage = next(s for s in self.stages if s.name == self.halted_at)
            return f"Cycle halted at '{self.halted_at}': {stage.summary}"
        if not self.changed:
            return (
                "Cycle completed with no changes. Nothing in the new data justified "
                "altering the models, which is the expected outcome most of the time."
            )
        changed = [s.name for s in self.stages if s.acted]
        return f"Cycle completed. Changed: {', '.join(changed)}."

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "Stage": s.name,
                "Outcome": "changed" if s.acted else ("HALTED" if s.halted else "no change"),
                "Detail": s.summary,
            }
            for s in self.stages
        ])

    def as_dict(self) -> dict:
        return {
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "changed": self.changed,
            "halted_at": self.halted_at,
            "dry_run": self.dry_run,
            "stages": [asdict(s) for s in self.stages],
        }


class AutoLearner:
    """Runs the gated improvement cycle, on a schedule or on demand."""

    def __init__(
        self,
        settings: Settings | None = None,
        min_hours_between_runs: float = 12.0,
        min_new_rows: int = 25,
        drift_halt_fraction: float = 0.5,
        min_delta_auc: float = 0.02,
        feature_permutations: int = 150,
    ):
        self.cfg = settings or SETTINGS
        self.min_hours_between_runs = min_hours_between_runs
        self.min_new_rows = min_new_rows
        self.drift_halt_fraction = drift_halt_fraction
        self.min_delta_auc = min_delta_auc
        self.feature_permutations = feature_permutations
        self.state_path = self.cfg.paths.data_dir / "autolearn_state.json"
        self.audit_path = self.cfg.paths.data_dir / "audit.jsonl"

    # -- state ------------------------------------------------------------

    def _state(self) -> dict:
        if not self.state_path.exists():
            return {"runs": [], "last_run_utc": None, "consecutive_failures": 0}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"runs": [], "last_run_utc": None, "consecutive_failures": 0}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def due(self) -> tuple[bool, str]:
        """Is a cycle allowed to run now?"""
        state = self._state()

        if state.get("consecutive_failures", 0) >= 3:
            return False, (
                "Three consecutive cycles failed. Automatic learning is paused until "
                "someone looks at why - repeatedly retrying a broken pipeline is how a "
                "loop quietly degrades a working model."
            )

        last = state.get("last_run_utc")
        if last:
            try:
                elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(last)
            except ValueError:
                return True, ""
            if elapsed < timedelta(hours=self.min_hours_between_runs):
                remaining = self.min_hours_between_runs - elapsed.total_seconds() / 3600
                return False, f"Last run was {elapsed.total_seconds()/3600:.1f}h ago; next in {remaining:.1f}h."
        return True, ""

    # -- cycle ------------------------------------------------------------

    def run_cycle(self, dry_run: bool = False, force: bool = False) -> CycleResult:
        """Run one full improvement cycle."""
        from ..data_pipeline import load_dataset
        from ..learning.feature_search import search_features
        from ..learning.ingest import SourceRegistry
        from ..learning.registry import ModelRegistry
        from ..learning.retrain import detect_drift, retrain_and_evaluate

        result = CycleResult(
            started_utc=datetime.now(timezone.utc).isoformat(), dry_run=dry_run
        )

        if not force:
            allowed, why = self.due()
            if not allowed:
                result.add(CycleStage("schedule", False, why, halted=True))
                result.finished_utc = datetime.now(timezone.utc).isoformat()
                return result

        try:
            data = load_dataset(self.cfg)
        except Exception as exc:
            result.add(CycleStage("load", False, f"could not load the dataset: {exc}", halted=True))
            self._finish(result, failed=True)
            return result

        # ---------------- 1. new data ----------------
        sources = SourceRegistry(self.cfg)
        incoming = sources.load_all()
        n_new = len(incoming)
        stage = result.add(CycleStage(
            "ingest", False,
            f"{n_new} row(s) available from external sources"
            + (f"; below the {self.min_new_rows}-row minimum for a learning cycle"
               if n_new < self.min_new_rows else ""),
            {"n_rows": n_new},
        ))
        if n_new < self.min_new_rows:
            stage.halted = True
            result.halted_at = "ingest"
            self._finish(result)
            return result

        # ---------------- 2. drift ----------------
        drift = detect_drift(data.substudy, incoming)
        n_drifted = int(drift["drifted"].sum()) if not drift.empty else 0
        n_tested = len(drift)
        fraction = n_drifted / n_tested if n_tested else 0.0

        drift_stage = result.add(CycleStage(
            "drift", False,
            f"{n_drifted} of {n_tested} shared variables shifted"
            + (
                f" ({fraction:.0%} - above the {self.drift_halt_fraction:.0%} halt threshold)"
                if fraction >= self.drift_halt_fraction else ""
            ),
            {"n_drifted": n_drifted, "n_tested": n_tested, "fraction": fraction},
        ))
        if n_tested and fraction >= self.drift_halt_fraction:
            drift_stage.halted = True
            drift_stage.summary += (
                ". Cycle halted: this looks like a different population, and fitting "
                "through that produces a model that is confident about the wrong cohort."
            )
            result.halted_at = "drift"
            self._finish(result)
            return result

        # ---------------- 3. feature value ----------------
        try:
            found = search_features(
                data.substudy, settings=self.cfg,
                min_delta_auc=self.min_delta_auc,
                n_permutations=self.feature_permutations,
            )
            result.add(CycleStage(
                "feature_search", bool(found.improved) and not dry_run,
                found.verdict,
                {
                    "base_auc": found.base_auc,
                    "best_candidate": found.best_candidate,
                    "delta_auc": found.delta_auc,
                    "selection_p": found.selection_p_value,
                    "adopted": found.adopted_features,
                },
            ))
            features = tuple(found.adopted_features)
        except Exception as exc:
            result.add(CycleStage("feature_search", False, f"feature search failed: {exc}"))
            features = ("log_BT_ratio", "age_years", "sex")

        # ---------------- 4. retrain, gated ----------------
        if dry_run:
            result.add(CycleStage(
                "retrain", False,
                "dry run: the challenger was not fitted and nothing was promoted",
            ))
            self._finish(result)
            return result

        try:
            outcome = retrain_and_evaluate(
                data.substudy, data.wide, features=features,
                settings=self.cfg, registry=ModelRegistry(self.cfg),
                min_improvement=self.min_delta_auc,
            )
            result.add(CycleStage(
                "retrain", bool(outcome.promoted), outcome.reason,
                {
                    "promoted": outcome.promoted,
                    "challenger": getattr(outcome.challenger, "version_id", None),
                    "champion": getattr(outcome.champion, "version_id", None),
                },
            ))
        except Exception as exc:
            result.add(CycleStage("retrain", False, f"retraining failed: {exc}", halted=True))
            self._finish(result, failed=True)
            return result

        self._finish(result)
        return result

    # -- bookkeeping ------------------------------------------------------

    def _finish(self, result: CycleResult, failed: bool = False) -> None:
        result.finished_utc = datetime.now(timezone.utc).isoformat()

        state = self._state()
        state["last_run_utc"] = result.finished_utc
        state["consecutive_failures"] = (
            state.get("consecutive_failures", 0) + 1 if failed else 0
        )
        state.setdefault("runs", []).append(result.as_dict())
        state["runs"] = state["runs"][-50:]
        self._save_state(state)

        try:
            from ..security import AuditLog

            AuditLog(self.audit_path).append(
                {
                    "event": "autolearn_cycle",
                    "changed": result.changed,
                    "halted_at": result.halted_at,
                    "dry_run": result.dry_run,
                    "stages": [
                        {"name": s.name, "acted": s.acted, "summary": s.summary}
                        for s in result.stages
                    ],
                },
                actor="autolearn",
            )
        except Exception:
            pass

    def history(self, limit: int = 20) -> pd.DataFrame:
        runs = self._state().get("runs", [])[-limit:]
        if not runs:
            return pd.DataFrame(columns=["Started", "Outcome", "Halted at"])
        return pd.DataFrame([
            {
                "Started": r.get("started_utc", "")[:19].replace("T", " "),
                "Outcome": (
                    "halted" if r.get("halted_at")
                    else ("changed" if r.get("changed") else "no change")
                ),
                "Halted at": r.get("halted_at") or "-",
                "Dry run": "yes" if r.get("dry_run") else "",
            }
            for r in reversed(runs)
        ])
