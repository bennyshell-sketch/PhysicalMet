"""The owner console: one key, a full record, and source editing that fails safe."""

import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from dive.owner import (
    EDITABLE_ROOTS,
    PROTECTED_PATHS,
    OwnerConsole,
    OwnerGate,
    SourceEditor,
)
from dive.security import AuditLog, ValidationError


@pytest.fixture
def project(tmp_path):
    """A throwaway copy of the project, so tests never touch the real source."""
    from dive import config

    for directory in ("dive", "docs"):
        shutil.copytree(config.SETTINGS.paths.project_root / directory, tmp_path / directory)
    (tmp_path / "data").mkdir(exist_ok=True)
    paths = replace(config.SETTINGS.paths, project_root=tmp_path, data_dir=tmp_path / "data")
    return replace(config.SETTINGS, paths=paths)


# --- the key --------------------------------------------------------------

def test_a_generated_key_verifies_and_a_wrong_one_does_not(project, monkeypatch):
    gate = OwnerGate(project)
    key = gate.generate_key()
    monkeypatch.setenv(OwnerGate.HASH_ENV, gate.hash_key(key))

    assert OwnerGate(project).verify(key)
    assert not OwnerGate(project).verify("dive_owner_" + "x" * 32)


def test_the_raw_key_is_never_stored(project, monkeypatch):
    gate = OwnerGate(project)
    key = gate.generate_key()
    stored = gate.hash_key(key)
    monkeypatch.setenv(OwnerGate.HASH_ENV, stored)

    assert key not in stored
    assert stored.startswith("scrypt$")


def test_a_short_key_is_refused():
    with pytest.raises(ValidationError, match="20 characters"):
        OwnerGate().hash_key("tooshort")


def test_repeated_failures_lock_the_console(project, monkeypatch):
    gate = OwnerGate(project, max_attempts=3)
    key = gate.generate_key()
    monkeypatch.setenv(OwnerGate.HASH_ENV, gate.hash_key(key))

    for _ in range(3):
        assert not gate.verify("dive_owner_" + "x" * 32)
    with pytest.raises(ValidationError, match="Locked"):
        gate.verify(key)


def test_authentication_attempts_are_audited(project, monkeypatch):
    gate = OwnerGate(project)
    key = gate.generate_key()
    monkeypatch.setenv(OwnerGate.HASH_ENV, gate.hash_key(key))
    gate.verify(key)
    gate.verify("dive_owner_" + "x" * 32)

    events = [e for e in AuditLog(project.paths.data_dir / "audit.jsonl").tail()
              if e.get("event") == "owner_auth"]
    assert {e["outcome"] for e in events} == {"granted", "denied"}


def test_an_unconfigured_deployment_says_how_to_set_up(project, monkeypatch):
    monkeypatch.delenv(OwnerGate.HASH_ENV, raising=False)
    gate = OwnerGate(project)
    assert not gate.configured
    assert "DIVE_OWNER_KEY_HASH" in gate.setup_instructions()
    with pytest.raises(ValidationError, match="No owner key"):
        gate.verify("anything-at-all-here")


# --- path safety ----------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "../../etc/passwd", ".git/config", "data/real/cohort.xlsx",
    "settings.local.yaml", "dive/../.env", ".venv/bin/python",
])
def test_paths_outside_the_allowlist_are_refused(project, bad):
    with pytest.raises(ValidationError):
        SourceEditor(project, enabled=True).resolve(bad)


def test_non_source_file_types_are_refused(project):
    with pytest.raises(ValidationError, match="not an editable file type"):
        SourceEditor(project, enabled=True).resolve("dive/config.pyc")


def test_an_allowed_path_resolves(project):
    resolved = SourceEditor(project, enabled=True).resolve("dive/config.py")
    assert resolved.name == "config.py"


def test_protected_and_editable_sets_do_not_overlap():
    assert not (set(EDITABLE_ROOTS) & set(PROTECTED_PATHS))


# --- writing --------------------------------------------------------------

def test_editing_is_disabled_by_default(project):
    result = SourceEditor(project).write("docs/OMICS.md", "# replaced")
    assert not result.written
    assert "disabled" in result.reason


def test_broken_python_is_refused_and_the_file_is_untouched(project):
    editor = SourceEditor(project, enabled=True)
    before = (project.paths.project_root / "dive/config.py").read_text()

    result = editor.write("dive/config.py", "def broken(:\n    pass")
    assert not result.written
    assert "syntax error" in result.reason
    assert (project.paths.project_root / "dive/config.py").read_text() == before


def test_broken_yaml_is_refused(project):
    editor = SourceEditor(project, enabled=True)
    result = editor.write("dive/knowledge/interventions.yaml", "bad:\n  - [unclosed")
    assert not result.written
    assert "YAML" in result.reason


def test_a_good_edit_is_written_with_a_backup(project):
    editor = SourceEditor(project, enabled=True)
    original = editor.read("docs/OMICS.md")

    result = editor.write("docs/OMICS.md", original + "\n\n## Added\n", note="test")
    assert result.written
    assert result.lines_added > 0
    assert result.backup and Path(result.backup).exists()
    assert "## Added" in editor.read("docs/OMICS.md")


def test_an_identical_write_is_a_no_op(project):
    editor = SourceEditor(project, enabled=True)
    content = editor.read("docs/OMICS.md")
    result = editor.write("docs/OMICS.md", content)
    assert not result.written
    assert "identical" in result.reason


def test_a_backup_can_be_restored(project):
    editor = SourceEditor(project, enabled=True)
    original = editor.read("docs/OMICS.md")
    editor.write("docs/OMICS.md", original + "\n\n## Added\n")

    backup = editor.backups().iloc[0]["Backup"]
    assert editor.restore(backup).written
    assert "## Added" not in editor.read("docs/OMICS.md")


def test_restoring_an_unknown_backup_fails_cleanly(project):
    assert not SourceEditor(project, enabled=True).restore("nonexistent.bak").written


def test_every_write_is_audited_with_before_and_after_hashes(project):
    editor = SourceEditor(project, enabled=True)
    editor.write("docs/OMICS.md", editor.read("docs/OMICS.md") + "\n## X\n", note="why")

    entries = [e for e in AuditLog(project.paths.data_dir / "audit.jsonl").tail()
               if e.get("event") == "source_write"]
    assert entries
    entry = entries[-1]
    assert entry["sha256_before"] != entry["sha256_after"]
    assert entry["note"] == "why"
    assert entry["actor"] == "owner"


def test_validation_can_be_run_without_writing():
    valid, problem = SourceEditor.validate("x.py", "def ok():\n    return 1\n")
    assert valid and problem == ""
    valid, problem = SourceEditor.validate("x.py", "def broken(:")
    assert not valid and "syntax error" in problem


# --- console --------------------------------------------------------------

def test_the_console_reports_integrity(project):
    state = OwnerConsole(project).integrity()
    assert set(state) >= {"audit_intact", "source_editing_enabled", "owner_key_configured"}
    assert state["audit_intact"]


def test_the_timeline_merges_subsystems(project):
    console = OwnerConsole(project)
    AuditLog(project.paths.data_dir / "audit.jsonl").append(
        {"event": "egress", "destination": "anthropic", "allowed": False,
         "reasons": ["blocked for testing"], "bytes_out": 0}
    )
    timeline = console.timeline()
    assert not timeline.empty
    assert "egress" in timeline["Event"].tolist()
    assert any("BLOCKED" in d for d in timeline["Detail"])


def test_the_summary_is_readable(project):
    summary = OwnerConsole(project).summary()
    assert "audited events" in summary
    assert "Source editing" in summary
