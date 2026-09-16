"""The owner console: what happened, what changed, and what the model learned.

One key, held by one person, unlocking four things the ordinary interface does
not expose:

* a single timeline merging the audit log, ingestion, learning cycles and model
  promotions, so "why does it do that now?" has one place to look
* model lineage - every version, its metrics, and which one is live
* the learning record - which metrics were tested, which were adopted, which
  were rejected and on what evidence
* source editing, so the owner can change the code that produces all of it

A straight warning about the last one. **Editing source from a web interface is
remote code execution by design.** Anyone who obtains the owner key can write
arbitrary Python into the application. No amount of path filtering changes
that; it narrows the blast radius, it does not remove it.

So it ships **disabled**. Turning it on is a deliberate act
(``owner.allow_source_editing: true``), and even then:

* writes are confined to an allowlist of directories - never ``.git``,
  ``.venv``, ``data/``, or any settings file holding secrets
* Python is parsed before it is written; a file that would not import is
  rejected rather than saved and discovered at the next restart
* every write takes a timestamped backup first, and any version can be restored
* every read, write and restore is audited with before-and-after hashes
* nothing is executed. Changes take effect at the next restart, which means a
  bad edit is a failed start rather than a live compromise

If Dive is reachable from anything other than localhost or a trusted network,
leave source editing off and edit the repository the normal way.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import hmac
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .config import SETTINGS, Settings
from .security import AuditLog, PasswordHasher, ValidationError

__all__ = [
    "OwnerGate",
    "OwnerConsole",
    "SourceEditor",
    "TimelineEntry",
    "EDITABLE_ROOTS",
    "PROTECTED_PATHS",
]

#: Directories the owner may edit. Everything else is refused.
EDITABLE_ROOTS: tuple[str, ...] = ("dive", "app", "tests", "docs")

#: Never editable, whatever the allowlist says. Secrets, version control,
#: the environment itself, and stored participant data.
PROTECTED_PATHS: tuple[str, ...] = (
    ".git", ".venv", "venv", "data", "node_modules", "__pycache__",
    "settings.local.yaml", ".env", "audit.jsonl",
)

#: Extensions the editor will open. A binary blob is not a source file.
EDITABLE_SUFFIXES: frozenset[str] = frozenset({
    ".py", ".yaml", ".yml", ".md", ".toml", ".cfg", ".txt", ".css", ".html", ".json",
})

MAX_SOURCE_BYTES = 2 * 1024 * 1024


# --------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------

class OwnerGate:
    """Authenticates the owner against a stored hash.

    The raw key is never stored - only a scrypt hash, in
    ``DIVE_OWNER_KEY_HASH``. Losing the key means generating a new one, which
    is the correct trade: a recoverable owner key is a key someone else can
    recover.
    """

    HASH_ENV = "DIVE_OWNER_KEY_HASH"

    def __init__(
        self,
        settings: Settings | None = None,
        max_attempts: int = 5,
        lockout_seconds: int = 900,
    ):
        self.cfg = settings or SETTINGS
        self.hasher = PasswordHasher()
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_seconds
        self._failures: list[float] = []
        self.audit = AuditLog(self.cfg.paths.data_dir / "audit.jsonl")

    # -- setup ------------------------------------------------------------

    @staticmethod
    def generate_key() -> str:
        """A fresh owner key. Shown once, stored by you, never by Dive."""
        import secrets

        raw = secrets.token_urlsafe(32)
        return f"dive_owner_{raw}"

    def hash_key(self, key: str) -> str:
        """Hash a key for ``DIVE_OWNER_KEY_HASH``."""
        if len(key) < 20:
            raise ValidationError("An owner key must be at least 20 characters.")
        return self.hasher.hash(key)

    @property
    def configured(self) -> bool:
        return bool(os.environ.get(self.HASH_ENV))

    def setup_instructions(self) -> str:
        return (
            "No owner key is configured. Generate one:\n\n"
            "    python -c \"from dive.owner import OwnerGate; "
            "g=OwnerGate(); k=g.generate_key(); print('KEY :', k); "
            "print('HASH:', g.hash_key(k))\"\n\n"
            "Keep the KEY somewhere safe - it is shown once and Dive never stores it. "
            f"Put the HASH in your environment as {self.HASH_ENV}."
        )

    # -- authentication ---------------------------------------------------

    def _locked_out(self) -> tuple[bool, float]:
        now = time.time()
        self._failures = [t for t in self._failures if now - t < self.lockout_seconds]
        if len(self._failures) >= self.max_attempts:
            return True, self.lockout_seconds - (now - self._failures[0])
        return False, 0.0

    def verify(self, key: str) -> bool:
        """Check a key. Failures are rate-limited and audited."""
        locked, remaining = self._locked_out()
        if locked:
            self.audit.append(
                {"event": "owner_auth", "outcome": "locked_out"}, actor="owner-gate"
            )
            raise ValidationError(
                f"Too many failed attempts. Locked for {int(remaining / 60) + 1} more minute(s)."
            )

        stored = os.environ.get(self.HASH_ENV, "")
        if not stored:
            raise ValidationError("No owner key is configured on this deployment.")

        ok = self.hasher.verify(key, stored)
        self._failures.append(time.time()) if not ok else self._failures.clear()
        self.audit.append(
            {"event": "owner_auth", "outcome": "granted" if ok else "denied"},
            actor="owner-gate",
        )
        return ok


# --------------------------------------------------------------------------
# Source editing
# --------------------------------------------------------------------------

@dataclass
class EditResult:
    path: str
    written: bool
    reason: str = ""
    backup: str | None = None
    diff: str = ""
    lines_added: int = 0
    lines_removed: int = 0


class SourceEditor:
    """Reads and writes project source, under an allowlist and with backups."""

    def __init__(self, settings: Settings | None = None, enabled: bool | None = None):
        self.cfg = settings or SETTINGS
        self.root = self.cfg.paths.project_root
        self.enabled = (
            enabled if enabled is not None
            else getattr(self.cfg, "allow_source_editing", False)
        )
        self.backup_dir = self.cfg.paths.data_dir / "source_backups"
        self.audit = AuditLog(self.cfg.paths.data_dir / "audit.jsonl")

    # -- path safety ------------------------------------------------------

    def resolve(self, relative: str) -> Path:
        """Resolve a repository-relative path, or raise.

        Rejects traversal, symlinks pointing outside the project, protected
        directories and unexpected file types. A rejection is a refusal, not a
        sanitisation - a path containing ``..`` is not a typo.
        """
        candidate = (self.root / str(relative).strip()).resolve()

        try:
            inside = candidate.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValidationError(
                f"'{relative}' resolves outside the project directory."
            ) from exc

        parts = inside.parts
        if not parts:
            raise ValidationError("No path given.")
        if parts[0] not in EDITABLE_ROOTS:
            raise ValidationError(
                f"'{inside}' is outside the editable roots "
                f"({', '.join(EDITABLE_ROOTS)})."
            )
        for part in parts:
            if part in PROTECTED_PATHS or part.startswith("."):
                raise ValidationError(f"'{inside}' touches a protected path ('{part}').")
        if candidate.suffix.lower() not in EDITABLE_SUFFIXES:
            raise ValidationError(
                f"'{candidate.suffix}' is not an editable file type "
                f"({', '.join(sorted(EDITABLE_SUFFIXES))})."
            )
        return candidate

    def list_files(self) -> pd.DataFrame:
        """Every editable file, with size and last-modified time."""
        rows = []
        for root_name in EDITABLE_ROOTS:
            root = self.root / root_name
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in EDITABLE_SUFFIXES:
                    continue
                relative = path.relative_to(self.root)
                if any(p in PROTECTED_PATHS or p.startswith(".") for p in relative.parts):
                    continue
                stat = path.stat()
                rows.append({
                    "Path": str(relative),
                    "Lines": sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")),
                    "Size (KB)": round(stat.st_size / 1024, 1),
                    "Modified": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).strftime("%Y-%m-%d %H:%M"),
                })
        return pd.DataFrame(rows)

    # -- read / write -----------------------------------------------------

    def read(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.exists():
            raise ValidationError(f"'{relative}' does not exist.")
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValidationError(f"'{relative}' is too large to open here.")
        content = path.read_text(encoding="utf-8")
        self.audit.append({"event": "source_read", "path": relative}, actor="owner")
        return content

    @staticmethod
    def validate(relative: str, content: str) -> tuple[bool, str]:
        """Check a file parses before it is written.

        A syntax error saved now is a failed start later, discovered by whoever
        restarts the service rather than by the person who caused it.
        """
        suffix = Path(relative).suffix.lower()
        if suffix == ".py":
            try:
                ast.parse(content)
            except SyntaxError as exc:
                return False, f"Python syntax error on line {exc.lineno}: {exc.msg}"
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml

                yaml.safe_load(content)
            except Exception as exc:
                return False, f"YAML is not valid: {exc}"
        elif suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                return False, f"JSON is not valid: {exc}"
        return True, ""

    def write(self, relative: str, content: str, note: str = "") -> EditResult:
        """Validate, back up, then write. Any failure writes nothing."""
        if not self.enabled:
            return EditResult(
                relative, False,
                "Source editing is disabled. Set allow_source_editing: true in "
                "settings.yaml to enable it, having read why it is off by default.",
            )

        path = self.resolve(relative)
        if len(content.encode()) > MAX_SOURCE_BYTES:
            return EditResult(relative, False, "Content exceeds the size limit.")

        valid, problem = self.validate(relative, content)
        if not valid:
            return EditResult(relative, False, f"Refused: {problem}")

        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        if previous == content:
            return EditResult(relative, False, "No change - the content is identical.")

        diff_lines = list(difflib.unified_diff(
            previous.splitlines(keepends=True), content.splitlines(keepends=True),
            fromfile=f"a/{relative}", tofile=f"b/{relative}",
        ))
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

        backup_path: Path | None = None
        if path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup_path = self.backup_dir / f"{relative.replace('/', '__')}.{stamp}.bak"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        self.audit.append({
            "event": "source_write",
            "path": relative,
            "note": note,
            "sha256_before": hashlib.sha256(previous.encode()).hexdigest(),
            "sha256_after": hashlib.sha256(content.encode()).hexdigest(),
            "lines_added": added,
            "lines_removed": removed,
            "backup": str(backup_path.name) if backup_path else None,
        }, actor="owner")

        return EditResult(
            relative, True,
            "Written. The change takes effect at the next restart - nothing was executed.",
            str(backup_path) if backup_path else None,
            "".join(diff_lines), added, removed,
        )

    # -- backups ----------------------------------------------------------

    def backups(self, relative: str | None = None) -> pd.DataFrame:
        if not self.backup_dir.exists():
            return pd.DataFrame(columns=["Backup", "Original", "Taken", "Size (KB)"])
        rows = []
        for path in sorted(self.backup_dir.glob("*.bak"), reverse=True):
            original = path.name.rsplit(".", 2)[0].replace("__", "/")
            if relative and original != relative:
                continue
            rows.append({
                "Backup": path.name,
                "Original": original,
                "Taken": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).strftime("%Y-%m-%d %H:%M"),
                "Size (KB)": round(path.stat().st_size / 1024, 1),
            })
        return pd.DataFrame(rows)

    def restore(self, backup_name: str) -> EditResult:
        """Put a backed-up version back, taking a backup of the current one."""
        backup_path = self.backup_dir / Path(backup_name).name
        if not backup_path.exists() or backup_path.parent != self.backup_dir:
            return EditResult(backup_name, False, "No such backup.")
        original = backup_path.name.rsplit(".", 2)[0].replace("__", "/")
        return self.write(
            original, backup_path.read_text(encoding="utf-8"),
            note=f"restored from {backup_path.name}",
        )


# --------------------------------------------------------------------------
# Timeline and lineage
# --------------------------------------------------------------------------

@dataclass
class TimelineEntry:
    when: str
    source: str
    event: str
    detail: str
    actor: str = ""


class OwnerConsole:
    """Everything the owner needs to see, assembled from what Dive records."""

    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or SETTINGS
        self.audit = AuditLog(self.cfg.paths.data_dir / "audit.jsonl")
        self.editor = SourceEditor(self.cfg)

    # -- what happened ----------------------------------------------------

    def timeline(self, limit: int = 300) -> pd.DataFrame:
        """One chronological record across every subsystem."""
        entries: list[TimelineEntry] = []

        for record in self.audit.tail(limit):
            event = str(record.get("event", "event"))
            detail = ""
            if event == "egress":
                allowed = record.get("allowed")
                detail = (
                    f"{'permitted' if allowed else 'BLOCKED'} to "
                    f"{record.get('destination', '?')}, {record.get('bytes_out', 0)} bytes"
                )
                if not allowed and record.get("reasons"):
                    detail += f" - {record['reasons'][0][:90]}"
            elif event == "source_write":
                detail = (
                    f"{record.get('path')} (+{record.get('lines_added', 0)} "
                    f"-{record.get('lines_removed', 0)})"
                )
            elif event == "autolearn_cycle":
                detail = (
                    "halted at " + str(record.get("halted_at"))
                    if record.get("halted_at")
                    else ("changed something" if record.get("changed") else "no change")
                )
            elif event == "owner_auth":
                detail = str(record.get("outcome", ""))
            else:
                detail = ", ".join(
                    f"{k}={v}" for k, v in record.items()
                    if k not in {"ts", "actor", "event", "prev_sha256", "entry_sha256"}
                )[:120]

            entries.append(TimelineEntry(
                str(record.get("ts", ""))[:19].replace("T", " "),
                "audit", event, detail, str(record.get("actor", "")),
            ))

        # Ingestion batches
        try:
            from .learning.ingest import SourceRegistry

            for _, row in SourceRegistry(self.cfg).list_batches().iterrows():
                entries.append(TimelineEntry(
                    str(row["ingested_utc"])[:19].replace("T", " "),
                    "data", "batch_ingested",
                    f"{row['source']}: {row['n_accepted']} accepted, "
                    f"{row['n_quarantined']} quarantined",
                ))
        except Exception:
            pass

        # Model versions
        try:
            from .learning.registry import ModelRegistry

            for _, row in ModelRegistry(self.cfg).history().iterrows():
                metrics = ", ".join(
                    f"{k}={v}" for k, v in row.items()
                    if k not in {"version_id", "model_kind", "created_utc", "n_training", "champion"}
                    and pd.notna(v)
                )
                entries.append(TimelineEntry(
                    str(row["created_utc"])[:19].replace("T", " "),
                    "model",
                    "model_promoted" if row.get("champion") == "yes" else "model_registered",
                    f"{row['version_id']} (n={row['n_training']}) {metrics}".strip(),
                ))
        except Exception:
            pass

        if not entries:
            return pd.DataFrame(columns=["When", "Source", "Event", "Detail", "Actor"])

        frame = pd.DataFrame([
            {"When": e.when, "Source": e.source, "Event": e.event,
             "Detail": e.detail, "Actor": e.actor}
            for e in entries
        ])
        return frame.sort_values("When", ascending=False).reset_index(drop=True)

    # -- what the model learned -------------------------------------------

    def model_lineage(self) -> pd.DataFrame:
        from .learning.registry import ModelRegistry

        return ModelRegistry(self.cfg).history()

    def learning_curve(self) -> pd.DataFrame:
        """Held-out performance of each model version, oldest first."""
        history = self.model_lineage()
        if history.empty:
            return pd.DataFrame(columns=["created_utc", "auc_cv", "n_training"])
        columns = [
            c for c in ("created_utc", "n_training", "auc_cv", "auc_holdout",
                        "permutation_p", "brier", "champion")
            if c in history.columns
        ]
        return history[columns].sort_values("created_utc").reset_index(drop=True)

    def cycle_history(self) -> pd.DataFrame:
        from .learning.autolearn import AutoLearner

        return AutoLearner(self.cfg).history(limit=50)

    def adopted_metrics(self) -> pd.DataFrame:
        """Metrics that have earned a place in a model, and when."""
        from .metrics import MetricRegistry

        frame = MetricRegistry(self.cfg).to_frame()
        adopted = frame[frame["Adopted by"] != "-"]
        return adopted.reset_index(drop=True)

    # -- integrity --------------------------------------------------------

    def integrity(self) -> dict[str, Any]:
        intact, problems = self.audit.verify()
        return {
            "audit_intact": intact,
            "problems": problems,
            "n_entries": len(self.audit.tail(100000)),
            "source_editing_enabled": self.editor.enabled,
            "owner_key_configured": OwnerGate(self.cfg).configured,
        }

    def summary(self) -> str:
        state = self.integrity()
        lineage = self.model_lineage()
        champions = lineage[lineage.get("champion", "") == "yes"] if not lineage.empty else lineage

        parts = [
            f"{state['n_entries']} audited events.",
            "Audit chain intact." if state["audit_intact"]
            else "AUDIT CHAIN BROKEN - entries have been removed or altered.",
            f"{len(lineage)} model version(s) registered"
            + (f", live: {champions['version_id'].iloc[0]}" if len(champions) else ", none promoted")
            + ".",
            "Source editing is ENABLED." if state["source_editing_enabled"]
            else "Source editing is disabled.",
        ]
        return " ".join(parts)
