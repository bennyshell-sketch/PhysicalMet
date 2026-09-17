"""The hosted deployment: the parts that are easy to get quietly wrong.

A container that writes its accounts file inside its own filesystem looks
perfectly healthy and loses every user on restart. A published port bypasses
the reverse proxy. A build-context rule that excludes the entrypoint fails only
at run time, in production. None of that shows up in an ordinary test run, so
it is checked here.
"""

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from dive.config import PathConfig, Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]


# --- where Dive writes ----------------------------------------------------

def test_moving_the_data_directory_moves_everything_under_it(monkeypatch, tmp_path):
    """A hosted deployment has one writable volume; one setting has to move it all."""
    monkeypatch.setenv("DIVE_DATA_DIR", str(tmp_path / "vol"))
    paths = load_settings().paths

    assert paths.data_dir == tmp_path / "vol"
    for attribute in ("synthetic_dir", "real_dir", "upload_dir", "model_dir", "output_dir"):
        moved = getattr(paths, attribute)
        assert moved.is_relative_to(tmp_path / "vol"), f"{attribute} was left behind at {moved}"


def test_an_explicitly_placed_directory_is_not_dragged_along(tmp_path):
    """Someone who pointed one directory somewhere on purpose meant it."""
    elsewhere = tmp_path / "kept-where-it-was"
    pinned = replace(PathConfig(), real_dir=elsewhere)

    moved = pinned.rerooted(tmp_path / "vol")

    assert moved.real_dir == elsewhere
    assert moved.model_dir == tmp_path / "vol" / "models"


def test_the_default_layout_is_unchanged_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("DIVE_DATA_DIR", raising=False)
    monkeypatch.delenv("DIVE_OUTPUT_DIR", raising=False)
    paths = load_settings().paths

    assert paths.data_dir == paths.project_root / "data"
    assert paths.model_dir == paths.project_root / "data" / "models"


def test_the_output_directory_can_still_be_set_on_its_own(monkeypatch, tmp_path):
    monkeypatch.setenv("DIVE_DATA_DIR", str(tmp_path / "vol"))
    monkeypatch.setenv("DIVE_OUTPUT_DIR", str(tmp_path / "reports"))
    paths = load_settings().paths

    assert paths.data_dir == tmp_path / "vol"
    assert paths.output_dir == tmp_path / "reports"


# --- the container's command line -----------------------------------------

ENTRYPOINT = ROOT / "docker-entrypoint.sh"


def _entrypoint_arguments(**environment: str) -> str:
    """Run the entrypoint with `exec` replaced by `echo`, to see what it built."""
    script = ENTRYPOINT.read_text(encoding="utf-8").replace('exec "$@"', 'echo "$@"')
    result = subprocess.run(
        ["sh", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **environment},
        check=True,
    )
    return result.stdout


def test_a_public_hostname_turns_on_the_host_allow_list():
    """Without this, anyone who can reach the port can open a session."""
    arguments = _entrypoint_arguments(DIVE_PUBLIC_HOSTNAME="dive.example.org")

    assert "--server.allowedHosts=dive.example.org" in arguments
    assert "--browser.serverAddress=dive.example.org" in arguments


def test_the_startup_address_lookup_is_always_pinned():
    """Headless Streamlit otherwise asks the internet for its own address."""
    for arguments in (
        _entrypoint_arguments(DIVE_PUBLIC_HOSTNAME="dive.example.org"),
        _entrypoint_arguments(),
    ):
        assert "--browser.serverAddress=" in arguments


def test_cross_site_protections_are_on_in_every_configuration():
    for arguments in (
        _entrypoint_arguments(DIVE_PUBLIC_HOSTNAME="dive.example.org"),
        _entrypoint_arguments(),
    ):
        assert "--server.enableXsrfProtection=true" in arguments
        assert "--server.enableCORS=true" in arguments
        assert "--browser.gatherUsageStats=false" in arguments


# --- the build and the compose file ---------------------------------------

def test_the_entrypoint_survives_the_build_context_rules():
    """.dockerignore excludes *.sh; the entrypoint has to be excepted by name."""
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "!docker-entrypoint.sh" in [line.strip() for line in ignore]
    assert ignore.index("*.sh") < ignore.index("!docker-entrypoint.sh")


def test_the_image_does_not_run_as_root():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER dive" in dockerfile
    assert dockerfile.index("USER dive") < dockerfile.index("ENTRYPOINT")


def test_no_participant_data_is_baked_into_the_image():
    """The image goes to a registry; the cohort must not go with it."""
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "data/" in ignore
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY data" not in dockerfile


def test_the_application_port_is_not_published():
    """Only the proxy should be reachable; the app listens on the internal network."""
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    app_section = compose.split("proxy:")[0]
    assert "expose:" in app_section
    assert "ports:" not in app_section


def test_the_required_secrets_fail_the_deployment_rather_than_defaulting():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for variable in ("DIVE_SESSION_SECRET", "DIVE_ENCRYPTION_KEY", "DIVE_PUBLIC_HOSTNAME"):
        assert f"${{{variable}:?" in compose, f"{variable} would silently default"


def test_source_editing_is_off_in_a_hosted_deployment():
    """Editing source through a web form is remote code execution by design."""
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert 'DIVE_ALLOW_SOURCE_EDITING: "false"' in compose
    assert Settings().allow_source_editing is False


def test_the_environment_template_carries_no_real_secret():
    """.env.example is committed; anything filled in would be a leaked secret."""
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        name, _, value = line.partition("=")
        if name.strip().endswith(("SECRET", "KEY", "PASSWORD", "KEY_HASH")):
            assert value.strip() == "", f"{name} has a value in .env.example"


# --- what is actually on disk ---------------------------------------------

@pytest.fixture
def keyed(monkeypatch):
    """A deployment with an encryption key configured."""
    from dive.security import DataVault

    monkeypatch.setenv("DIVE_ENCRYPTION_KEY", DataVault.generate_key())
    return DataVault()


def test_an_uploaded_workbook_is_unreadable_on_disk(keyed, tmp_path):
    from dive.generate_synthetic_data import generate_workbook

    source, _ = generate_workbook(output_path=tmp_path / "cohort.xlsx")
    payload = Path(source).read_bytes()

    written = keyed.write_file(payload, tmp_path / "uploaded.xlsx")

    assert written.name.endswith(".enc")
    on_disk = written.read_bytes()
    assert on_disk != payload
    assert not on_disk.startswith(b"PK"), "an unencrypted xlsx is a zip and starts with PK"


def test_an_encrypted_workbook_loads_exactly_like_a_plain_one(keyed, tmp_path):
    from dive.data_pipeline import build_dataset
    from dive.generate_synthetic_data import generate_workbook

    source, _ = generate_workbook(output_path=tmp_path / "cohort.xlsx")
    encrypted = keyed.write_file(Path(source).read_bytes(), tmp_path / "same.xlsx")

    plain = build_dataset(Path(source))
    through_vault = build_dataset(encrypted)

    assert through_vault.wide.shape == plain.wide.shape
    assert set(through_vault.wide["participant_id"]) == set(plain.wide["participant_id"])


def test_an_encrypted_csv_ingests_like_a_plain_one(keyed, tmp_path):
    import pandas as pd

    from dive.learning.ingest import DataSource, ingest_file

    frame = pd.DataFrame({
        "participant_id": ["P1", "P2"],
        "age": [70, 72],
        "grip_strength_kg": [24.0, 28.5],
    })
    plain_path = tmp_path / "batch.csv"
    frame.to_csv(plain_path, index=False)
    encrypted = keyed.write_file(plain_path.read_bytes(), tmp_path / "secret.csv")

    source = DataSource(source_id="s1", label="Test", kind="cohort")
    plain = ingest_file(plain_path, source, store=False)
    through_vault = ingest_file(encrypted, source, store=False)

    assert through_vault.n_input_rows == plain.n_input_rows == 2
    # The batch id digests the plaintext, so the same data gives the same id
    # however it happens to be stored.
    assert through_vault.batch_id.split("-")[-1] == plain.batch_id.split("-")[-1]


def test_without_a_key_the_file_is_written_in_the_clear_and_named_honestly(monkeypatch, tmp_path):
    """Silently failing to encrypt would be worse than not encrypting."""
    from dive.security import DataVault

    monkeypatch.delenv("DIVE_ENCRYPTION_KEY", raising=False)
    written = DataVault().write_file(b"cohort bytes", tmp_path / "plain.xlsx")

    assert written.name == "plain.xlsx"
    assert written.read_bytes() == b"cohort bytes"


def test_a_tampered_encrypted_file_is_refused_rather_than_half_read(keyed, tmp_path):
    from dive.security import ValidationError

    written = keyed.write_file(b"cohort bytes", tmp_path / "cohort.xlsx")
    blob = bytearray(written.read_bytes())
    blob[-1] ^= 0xFF
    written.write_bytes(bytes(blob))

    with pytest.raises(ValidationError, match="failed authentication"):
        keyed.read_file(written)


# --- who can see what ------------------------------------------------------

def _screen_capabilities() -> dict[str, str]:
    import re

    source = (ROOT / "app" / "app.py").read_text(encoding="utf-8")
    block = re.search(r"SCREEN_CAPABILITIES.*?\n\}", source, re.S).group(0)
    return dict(re.findall(r'"([^"]+)": "(\w+)"', block))


def _screens_for(role: str) -> set[str]:
    from dive.security import ROLE_CAPABILITIES

    capabilities = ROLE_CAPABILITIES[role]
    return {name for name, needed in _screen_capabilities().items() if needed in capabilities}


def test_a_viewer_sees_no_individual_level_screen():
    """Individual predictions are identifiable clinical inferences, not summaries."""
    viewer = _screens_for("viewer")

    assert "Individual prediction" not in viewer
    assert "Trajectory" not in viewer
    assert "Recommendations" not in viewer
    assert "Group comparison" in viewer


def test_only_an_administrator_reaches_the_owner_console_door():
    for role in ("viewer", "analyst"):
        assert "Owner console" not in _screens_for(role)
    assert "Owner console" in _screens_for("admin")


def test_only_an_administrator_manages_users():
    for role in ("viewer", "analyst"):
        assert "Users" not in _screens_for(role)
    assert "Users" in _screens_for("admin")


def test_every_screen_is_behind_a_capability():
    """A screen with no entry would be reachable by anyone who signs in."""
    import re

    source = (ROOT / "app" / "app.py").read_text(encoding="utf-8")
    registry = re.search(r"SCREENS: dict.*?\n\}", source, re.S)
    if registry is None:
        registry = re.search(r"SCREENS = \{.*?\n\}", source, re.S)
    listed = set(re.findall(r'"([^"]+)": screen_', registry.group(0)))

    assert listed, "could not find the screen registry"
    assert listed <= set(_screen_capabilities()), (
        f"screens with no capability: {listed - set(_screen_capabilities())}"
    )


def test_the_base_image_can_come_from_an_internal_mirror():
    """Deployments that may not pull from Docker Hub should not have to fork the file."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG PYTHON_IMAGE=python:3.12-slim" in dockerfile
    assert "FROM ${PYTHON_IMAGE}" in dockerfile
