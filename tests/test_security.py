"""Security controls. These raise the cost of an attack; none make Dive unbreakable."""

import base64
import os

import pytest

from dive.security import (
    AuditLog,
    DataVault,
    PasswordHasher,
    SecretRef,
    SessionManager,
    UploadValidator,
    ValidationError,
)


# --- audit ----------------------------------------------------------------

def test_audit_chain_verifies_when_intact(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.append({"event": "test", "n": i})
    intact, problems = log.verify()
    assert intact and problems == []


def test_deleting_an_audit_line_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(5):
        log.append({"event": "test", "n": i})

    lines = path.read_text().splitlines()
    del lines[2]
    path.write_text("\n".join(lines) + "\n")

    intact, problems = log.verify()
    assert not intact
    assert any("chain broken" in p for p in problems)


def test_editing_an_audit_record_breaks_its_hash(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append({"event": "login", "outcome": "denied"})

    tampered = path.read_text().replace('"denied"', '"granted"')
    path.write_text(tampered)

    intact, problems = log.verify()
    assert not intact
    assert any("does not match its hash" in p for p in problems)


def test_identifiers_are_redacted_before_being_written(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append({"event": "x", "note": "patient WL042 nhs 943 476 5919"})
    stored = log.tail(1)[0]["note"]
    assert "WL042" not in stored and "943" not in stored


# --- encryption -----------------------------------------------------------

@pytest.fixture
def vault(monkeypatch):
    monkeypatch.setenv(DataVault.KEY_ENV, DataVault.generate_key())
    return DataVault()


def test_encryption_roundtrip(vault):
    blob = vault.encrypt(b"cohort data")
    assert b"cohort data" not in blob
    assert vault.decrypt(blob) == b"cohort data"


def test_tampered_ciphertext_is_rejected(vault):
    blob = bytearray(vault.encrypt(b"cohort data"))
    blob[-1] ^= 0xFF
    with pytest.raises(Exception):
        vault.decrypt(bytes(blob))


def test_encryption_without_a_key_fails_loudly(monkeypatch):
    monkeypatch.delenv(DataVault.KEY_ENV, raising=False)
    with pytest.raises(ValidationError, match="No encryption key"):
        DataVault().encrypt(b"data")


def test_a_malformed_key_is_rejected_at_construction(monkeypatch):
    monkeypatch.setenv(DataVault.KEY_ENV, base64.urlsafe_b64encode(b"tooshort").decode())
    with pytest.raises(ValidationError, match="32 bytes"):
        DataVault()


def test_file_encryption_roundtrip(tmp_path, vault):
    source = tmp_path / "cohort.csv"
    source.write_bytes(b"participant_id,grip\nWL001,30\n")
    encrypted = vault.encrypt_file(source)
    assert b"WL001" not in encrypted.read_bytes()

    restored = vault.decrypt_file(encrypted, tmp_path / "out.csv")
    assert restored.read_bytes() == source.read_bytes()


# --- passwords and sessions -----------------------------------------------

def test_password_hashing_is_salted_and_verifies():
    hasher = PasswordHasher()
    a, b = hasher.hash("correct-horse-battery"), hasher.hash("correct-horse-battery")
    assert a != b                                  # per-password salt
    assert hasher.verify("correct-horse-battery", a)
    assert not hasher.verify("wrong-password-here", a)


def test_short_passwords_are_refused():
    with pytest.raises(ValidationError, match="12 characters"):
        PasswordHasher().hash("short")


def test_a_forged_or_edited_token_is_rejected():
    manager = SessionManager()
    manager.create_user("ben", "correct-horse-battery", role="viewer")
    token = manager.authenticate("ben", "correct-horse-battery")

    body, _ = token.split(".")
    with pytest.raises(ValidationError, match="signature"):
        manager.verify_token(f"{body}.{'f' * 64}")


def test_role_escalation_via_the_token_is_impossible():
    manager = SessionManager()
    manager.create_user("vera", "viewer-password-123", role="viewer")
    token = manager.authenticate("vera", "viewer-password-123")

    body_b64, _ = token.split(".")
    body = base64.urlsafe_b64decode(body_b64).decode().replace("viewer", "admin")
    forged = f"{base64.urlsafe_b64encode(body.encode()).decode()}.{token.split('.')[1]}"
    with pytest.raises(ValidationError):
        manager.verify_token(forged)


def test_capabilities_follow_the_role():
    manager = SessionManager()
    manager.create_user("vera", "viewer-password-123", role="viewer")
    manager.create_user("ada", "analyst-password-12", role="analyst")
    manager.create_user("ben", "admin-password-1234", role="admin")

    viewer = manager.authenticate("vera", "viewer-password-123")
    analyst = manager.authenticate("ada", "analyst-password-12")
    admin = manager.authenticate("ben", "admin-password-1234")

    assert manager.can(viewer, "view_results")
    assert not manager.can(viewer, "view_individual")
    assert manager.can(analyst, "run_models")
    assert not manager.can(analyst, "change_privacy_policy")
    assert manager.can(admin, "change_privacy_policy")


def test_repeated_failures_lock_the_account():
    manager = SessionManager(max_failed_attempts=3, lockout_minutes=15)
    manager.create_user("vera", "viewer-password-123", role="viewer")

    for _ in range(3):
        with pytest.raises(ValidationError):
            manager.authenticate("vera", "wrong-password-xx")

    with pytest.raises(ValidationError, match="locked"):
        manager.authenticate("vera", "viewer-password-123")


def test_an_expired_session_is_rejected():
    manager = SessionManager(session_minutes=-1)
    token = manager.issue_token("ben", "admin")
    with pytest.raises(ValidationError, match="expired"):
        manager.verify_token(token)


def test_unknown_username_does_not_reveal_itself():
    manager = SessionManager()
    manager.create_user("ben", "correct-horse-battery", role="viewer")
    with pytest.raises(ValidationError) as unknown:
        manager.authenticate("nobody", "whatever-password")
    with pytest.raises(ValidationError) as wrong:
        manager.authenticate("ben", "wrong-password-xxx")
    assert str(unknown.value) == str(wrong.value)


# --- secrets --------------------------------------------------------------

def test_a_secret_never_appears_in_repr(monkeypatch):
    monkeypatch.setenv("TEST_SECRET", "sk-ant-super-secret-value")
    ref = SecretRef("TEST_SECRET")
    assert "super-secret" not in repr(ref)
    assert "super-secret" not in str(ref)
    assert "super-secret" not in ref.fingerprint()
    assert ref.resolve() == "sk-ant-super-secret-value"


# --- uploads --------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "../../etc/passwd", "..\\windows\\system32", "sub/dir/file.xlsx",
    ".hidden.xlsx", "", "a" * 300 + ".xlsx",
])
def test_dangerous_filenames_are_rejected(bad):
    with pytest.raises(ValidationError):
        UploadValidator().validate_filename(bad)


def test_disallowed_extensions_are_rejected():
    with pytest.raises(ValidationError, match="not accepted"):
        UploadValidator().validate_filename("payload.exe")


def test_contents_must_match_the_extension():
    with pytest.raises(ValidationError, match="do not match"):
        UploadValidator().validate_bytes(b"this is not a zip", "cohort.xlsx")


def test_oversized_uploads_are_rejected():
    validator = UploadValidator(max_bytes=10)
    with pytest.raises(ValidationError, match="limit"):
        validator.validate_bytes(b"PK\x03\x04" + b"x" * 100, "cohort.xlsx")


def test_a_valid_xlsx_passes():
    assert UploadValidator().validate_bytes(b"PK\x03\x04rest", "cohort.xlsx") == "cohort.xlsx"


@pytest.mark.parametrize("payload", ["=cmd|'/c calc'!A1", "+1+1", "-1+1", "@SUM(A1)"])
def test_formula_injection_is_neutralised(payload):
    assert UploadValidator.neutralise_formula(payload).startswith("'")


def test_ordinary_text_is_left_alone():
    assert UploadValidator.neutralise_formula("Non-Faller") == "Non-Faller"
