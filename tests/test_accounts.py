"""Persistent accounts: the login surface of a hosted deployment.

These tests are written against the things that actually go wrong on an
internet-facing sign-in - lockout that forgets itself, tokens that outlive the
password they were issued for, an admin account deleted by accident - rather
than against the happy path, which is the easy part.
"""

from dataclasses import replace

import pytest

from dive.accounts import MIN_PASSWORD_LENGTH, AccountStore, bootstrap_admin
from dive.security import ValidationError

GOOD = "correct-horse-battery-staple"
OTHER = "another-long-enough-passphrase"


@pytest.fixture
def store(tmp_path, monkeypatch):
    from dive import config

    monkeypatch.setenv("DIVE_SESSION_SECRET", "test-secret-not-used-anywhere-real")
    monkeypatch.delenv("DIVE_ADMIN_PASSWORD", raising=False)
    paths = replace(config.SETTINGS.paths, data_dir=tmp_path)
    return AccountStore(replace(config.SETTINGS, paths=paths))


# --- passwords ------------------------------------------------------------

def test_a_short_password_is_refused_not_merely_discouraged(store):
    """The interface advises 12 characters; the store has to be what enforces it."""
    with pytest.raises(ValidationError, match=str(MIN_PASSWORD_LENGTH)):
        store.create("clinician", "short", role="analyst")
    assert not store.exists("clinician")


def test_the_password_cannot_be_the_username(store):
    with pytest.raises(ValidationError, match="not be the username"):
        store.create("averylongusername", "averylongusername")


def test_changing_to_a_short_password_is_refused_too(store):
    store.create("clinician", GOOD, role="analyst")
    with pytest.raises(ValidationError):
        store.set_password("clinician", "abc")
    # The old password still works, so a rejected change cannot lock anyone out.
    assert store.authenticate("clinician", GOOD)


def test_the_password_is_never_stored_in_recoverable_form(store, tmp_path):
    store.create("clinician", GOOD, role="analyst")
    raw = (tmp_path / "accounts.json").read_text(encoding="utf-8")
    assert GOOD not in raw
    assert "scrypt" in raw or "$" in raw


# --- authentication -------------------------------------------------------

def test_a_wrong_password_and_an_unknown_user_fail_identically(store):
    """Distinguishable failures tell an attacker which usernames are real."""
    store.create("clinician", GOOD, role="analyst")

    with pytest.raises(ValidationError) as wrong:
        store.authenticate("clinician", OTHER)
    with pytest.raises(ValidationError) as unknown:
        store.authenticate("nobody-here", OTHER)

    assert str(wrong.value) == str(unknown.value)


def test_lockout_survives_a_restart(store, tmp_path, monkeypatch):
    """Restarting the process must not hand an attacker a fresh budget of guesses."""
    from dive import config

    store.create("clinician", GOOD, role="analyst")
    for _ in range(store.max_failed_attempts):
        with pytest.raises(ValidationError):
            store.authenticate("clinician", "wrong-but-long-enough")

    paths = replace(config.SETTINGS.paths, data_dir=tmp_path)
    reopened = AccountStore(replace(config.SETTINGS, paths=paths))
    with pytest.raises(ValidationError, match="locked"):
        reopened.authenticate("clinician", GOOD)


def test_a_disabled_account_cannot_sign_in(store):
    store.create("clinician", GOOD, role="analyst")
    store.set_disabled("clinician", True)
    with pytest.raises(ValidationError):
        store.authenticate("clinician", GOOD)


def test_usernames_are_case_insensitive(store):
    store.create("Clinician", GOOD, role="analyst")
    assert store.authenticate("CLINICIAN", GOOD)


# --- tokens ---------------------------------------------------------------

def test_a_tampered_token_is_rejected(store):
    store.create("clinician", GOOD, role="analyst")
    token = store.authenticate("clinician", GOOD)
    body, signature = token.split(".")

    with pytest.raises(ValidationError, match="signature"):
        store.verify_token(f"{body}.{'0' * len(signature)}")
    with pytest.raises(ValidationError):
        store.verify_token("not-a-token")


def test_the_role_comes_from_storage_not_from_the_token(store):
    """A forged claim is worthless if the server re-reads the account."""
    store.create("clinician", GOOD, role="analyst")
    token = store.authenticate("clinician", GOOD)
    store.set_role("clinician", "viewer")

    assert store.verify_token(token)["role"] == "viewer"
    assert not store.can(token, "ingest_data")


def test_disabling_an_account_ends_its_live_session(store):
    store.create("clinician", GOOD, role="analyst")
    token = store.authenticate("clinician", GOOD)
    store.set_disabled("clinician", True)

    with pytest.raises(ValidationError, match="no longer active"):
        store.verify_token(token)


def test_changing_the_password_ends_sessions_opened_with_the_old_one(store):
    """The moment a leaked password is changed, tokens from it must stop working."""
    store.create("clinician", GOOD, role="analyst")
    stolen = store.authenticate("clinician", GOOD)
    assert store.verify_token(stolen)

    store.set_password("clinician", OTHER)
    with pytest.raises(ValidationError, match="password was changed"):
        store.verify_token(stolen)


def test_an_expired_token_is_rejected(store):
    store.create("clinician", GOOD, role="analyst")
    store.session_minutes = -1
    token = store.authenticate("clinician", GOOD)
    with pytest.raises(ValidationError, match="expired"):
        store.verify_token(token)


def test_a_token_signed_with_another_secret_is_rejected(store, tmp_path, monkeypatch):
    """Two deployments must not accept each other's sessions."""
    from dive import config

    store.create("clinician", GOOD, role="analyst")
    token = store.authenticate("clinician", GOOD)

    monkeypatch.setenv("DIVE_SESSION_SECRET", "a-completely-different-secret")
    paths = replace(config.SETTINGS.paths, data_dir=tmp_path)
    elsewhere = AccountStore(replace(config.SETTINGS, paths=paths))
    with pytest.raises(ValidationError, match="signature"):
        elsewhere.verify_token(token)


# --- capabilities ---------------------------------------------------------

def test_roles_carry_the_capabilities_the_screens_ask_for(store):
    store.create("looker", GOOD, role="viewer")
    store.create("worker", GOOD, role="analyst")
    store.create("boss", GOOD, role="admin")

    viewer = store.authenticate("looker", GOOD)
    analyst = store.authenticate("worker", GOOD)
    admin = store.authenticate("boss", GOOD)

    assert store.can(viewer, "view_aggregates")
    assert not store.can(viewer, "view_individual")
    assert store.can(analyst, "view_individual")
    assert not store.can(analyst, "manage_users")
    assert store.can(admin, "manage_users")


# --- management -----------------------------------------------------------

def test_the_last_active_admin_cannot_be_deleted(store):
    store.create("boss", GOOD, role="admin")
    with pytest.raises(ValidationError, match="only active admin"):
        store.delete("boss")

    store.create("deputy", GOOD, role="admin")
    store.delete("boss")
    assert not store.exists("boss")


def test_a_disabled_admin_does_not_count_as_cover(store):
    """Deleting the working admin while the spare is disabled is a lock-out."""
    store.create("boss", GOOD, role="admin")
    store.create("deputy", GOOD, role="admin")
    store.set_disabled("deputy", True)

    with pytest.raises(ValidationError, match="only active admin"):
        store.delete("boss")


def test_usernames_are_validated(store):
    for bad in ("ab", "has space", "Has/Slash", "x" * 65):
        with pytest.raises(ValidationError):
            store.create(bad, GOOD)


def test_duplicate_users_are_refused(store):
    store.create("clinician", GOOD)
    with pytest.raises(ValidationError, match="already exists"):
        store.create("clinician", OTHER)


# --- bootstrap ------------------------------------------------------------

def test_bootstrap_generates_a_password_once_and_forces_a_change(store):
    created, message = bootstrap_admin(store)
    assert created
    assert "password:" in message

    password = message.split("password:")[1].split()[0]
    token = store.authenticate("admin", password)
    assert store.verify_token(token)["role"] == "admin"
    assert store._load()["admin"].must_change_password

    # Second call must not create another admin or reveal anything.
    created_again, _ = bootstrap_admin(store)
    assert not created_again


def test_bootstrap_uses_the_configured_password_without_printing_it(store, monkeypatch):
    monkeypatch.setenv("DIVE_ADMIN_PASSWORD", "a-configured-admin-passphrase")
    created, message = bootstrap_admin(store)

    assert created
    assert "a-configured-admin-passphrase" not in message
    assert store.authenticate("admin", "a-configured-admin-passphrase")


def test_an_ephemeral_session_secret_is_reported(tmp_path, monkeypatch):
    from dive import config

    monkeypatch.delenv("DIVE_SESSION_SECRET", raising=False)
    paths = replace(config.SETTINGS.paths, data_dir=tmp_path)
    ephemeral = AccountStore(replace(config.SETTINGS, paths=paths))
    assert ephemeral.session_secret_is_ephemeral

    monkeypatch.setenv("DIVE_SESSION_SECRET", "x" * 32)
    assert not AccountStore(replace(config.SETTINGS, paths=paths)).session_secret_is_ephemeral
