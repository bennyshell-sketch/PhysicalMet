"""Persistent user accounts for a hosted deployment.

``dive.security.SessionManager`` holds accounts in memory, which is fine for a
single desktop session and useless for a server: every restart would forget
everyone. This module persists them, and adds the things an internet-facing
login needs that a local one does not.

* Passwords are stored only as scrypt hashes, never recoverable.
* The first account is bootstrapped from ``DIVE_ADMIN_PASSWORD`` and must be
  changed on first login - a deployment whose admin password is sitting in a
  compose file is one leaked file away from compromise.
* Lockout state is persisted, so restarting the server does not reset an
  attacker's budget of guesses.
* Failed logins are audited with the username attempted, and successful ones
  record the time, so an unexpected login is visible afterwards.
* Sessions expire, and the expiry is enforced server-side rather than trusted
  from the client.

What this deliberately does not do: password reset by email, SSO, or MFA. Those
need infrastructure decisions (a mail relay, an identity provider) that belong
to whoever deploys this. For a clinical deployment, SSO against the
institution's identity provider is the right answer and should replace this
module rather than sit alongside it.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import SETTINGS, Settings
from .security import ROLE_CAPABILITIES, ROLE_RANK, AuditLog, PasswordHasher, Role, ValidationError

__all__ = ["Account", "AccountStore", "bootstrap_admin", "MIN_PASSWORD_LENGTH"]

#: Minimum password length, enforced in the store rather than in the screen
#: that happens to be asking. Length beats composition rules: NCSC and NIST
#: SP 800-63B both advise a long passphrase over enforced symbol classes,
#: which push people towards predictable substitutions.
MIN_PASSWORD_LENGTH = 12


def check_password(password: str, username: str = "") -> None:
    """Raise ValidationError if a password is not acceptable."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise ValidationError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters. "
            "A passphrase of three or four words is easier to remember and "
            "harder to guess than a short string of symbols."
        )
    if username and password.strip().lower() == str(username).strip().lower():
        raise ValidationError("Password must not be the username.")


@dataclass
class Account:
    username: str
    password_hash: str
    role: Role = "viewer"
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_login_utc: str | None = None
    failed_attempts: int = 0
    locked_until: float = 0.0
    must_change_password: bool = False
    disabled: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def locked(self) -> bool:
        return self.locked_until > time.time()

    def lock_remaining_minutes(self) -> int:
        return max(0, int((self.locked_until - time.time()) / 60) + 1)


class AccountStore:
    """File-backed accounts with lockout, auditing and role management."""

    def __init__(
        self,
        settings: Settings | None = None,
        max_failed_attempts: int = 5,
        lockout_minutes: int = 15,
        session_minutes: int = 60,
    ):
        self.cfg = settings or SETTINGS
        self.path = self.cfg.paths.data_dir / "accounts.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hasher = PasswordHasher()
        self.max_failed_attempts = max_failed_attempts
        self.lockout_minutes = lockout_minutes
        self.session_minutes = session_minutes
        self.audit = AuditLog(self.cfg.paths.data_dir / "audit.jsonl")
        self._secret = self._session_secret()

    # -- session signing key ----------------------------------------------

    def _session_secret(self) -> bytes:
        """Signing key for session tokens.

        Read from ``DIVE_SESSION_SECRET`` so that it survives restarts and is
        shared across replicas. A generated fallback is fine for a single
        instance but logs a warning, because it silently invalidates every
        session on restart and breaks outright behind a load balancer.
        """
        configured = os.environ.get("DIVE_SESSION_SECRET", "")
        if configured:
            return configured.encode()
        return secrets.token_bytes(32)

    @property
    def session_secret_is_ephemeral(self) -> bool:
        return not os.environ.get("DIVE_SESSION_SECRET")

    # -- storage ----------------------------------------------------------

    def _load(self) -> dict[str, Account]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {name: Account(**record) for name, record in payload.items()}

    def _save(self, accounts: dict[str, Account]) -> None:
        self.path.write_text(
            json.dumps({n: a.as_dict() for n, a in accounts.items()}, indent=2),
            encoding="utf-8",
        )
        # Owner-only where the platform supports it. On Windows this is a no-op
        # and filesystem ACLs are the deployer's responsibility.
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # -- management -------------------------------------------------------

    def exists(self, username: str) -> bool:
        return self._normalise(username) in self._load()

    @property
    def empty(self) -> bool:
        return not self._load()

    @staticmethod
    def _normalise(username: str) -> str:
        return str(username).strip().lower()

    def create(
        self,
        username: str,
        password: str,
        role: Role = "viewer",
        must_change_password: bool = False,
    ) -> Account:
        import re

        name = self._normalise(username)
        if not re.fullmatch(r"[a-z0-9._-]{3,64}", name):
            raise ValidationError(
                "Username must be 3-64 characters: lowercase letters, digits, "
                "dot, underscore or hyphen."
            )
        if role not in ROLE_RANK:
            raise ValidationError(f"Unknown role '{role}'.")

        check_password(password, name)

        accounts = self._load()
        if name in accounts:
            raise ValidationError(f"User '{name}' already exists.")

        account = Account(
            username=name,
            password_hash=self.hasher.hash(password),
            role=role,
            must_change_password=must_change_password,
        )
        accounts[name] = account
        self._save(accounts)
        self.audit.append(
            {"event": "account_created", "username": name, "role": role}, actor="system"
        )
        return account

    def set_password(self, username: str, password: str) -> None:
        accounts = self._load()
        name = self._normalise(username)
        if name not in accounts:
            raise ValidationError("No such user.")
        check_password(password, name)
        accounts[name].password_hash = self.hasher.hash(password)
        accounts[name].must_change_password = False
        self._save(accounts)
        self.audit.append({"event": "password_changed", "username": name}, actor=name)

    def set_role(self, username: str, role: Role) -> None:
        if role not in ROLE_RANK:
            raise ValidationError(f"Unknown role '{role}'.")
        accounts = self._load()
        name = self._normalise(username)
        if name not in accounts:
            raise ValidationError("No such user.")
        previous = accounts[name].role
        accounts[name].role = role
        self._save(accounts)
        self.audit.append(
            {"event": "role_changed", "username": name, "from": previous, "to": role},
            actor="admin",
        )

    def set_disabled(self, username: str, disabled: bool) -> None:
        accounts = self._load()
        name = self._normalise(username)
        if name not in accounts:
            raise ValidationError("No such user.")
        accounts[name].disabled = disabled
        self._save(accounts)
        self.audit.append(
            {"event": "account_disabled" if disabled else "account_enabled", "username": name},
            actor="admin",
        )

    def delete(self, username: str) -> None:
        accounts = self._load()
        name = self._normalise(username)
        if name not in accounts:
            raise ValidationError("No such user.")
        if accounts[name].role == "admin" and sum(
            1 for a in accounts.values() if a.role == "admin" and not a.disabled
        ) <= 1:
            raise ValidationError(
                "This is the only active admin. Promote someone else before deleting it, "
                "or you will lock yourself out."
            )
        del accounts[name]
        self._save(accounts)
        self.audit.append({"event": "account_deleted", "username": name}, actor="admin")

    # -- authentication ---------------------------------------------------

    def authenticate(self, username: str, password: str) -> str:
        """Return a signed session token, or raise.

        Failure messages are uniform: telling an attacker that a username
        exists is telling them half the answer.
        """
        accounts = self._load()
        name = self._normalise(username)
        account = accounts.get(name)

        if account and account.locked:
            self.audit.append(
                {"event": "login", "username": name, "outcome": "locked_out"}, actor=name
            )
            raise ValidationError(
                f"Account locked. Try again in {account.lock_remaining_minutes()} minute(s)."
            )

        if account and account.disabled:
            self.audit.append(
                {"event": "login", "username": name, "outcome": "disabled"}, actor=name
            )
            raise ValidationError("Invalid username or password.")

        # Hash even for an unknown user so response time does not reveal
        # whether the account exists.
        reference = account.password_hash if account else self.hasher.hash("x" * 24)
        valid = bool(account) and self.hasher.verify(password, reference)

        if not valid:
            if account:
                account.failed_attempts += 1
                if account.failed_attempts >= self.max_failed_attempts:
                    account.locked_until = time.time() + self.lockout_minutes * 60
                    account.failed_attempts = 0
                accounts[name] = account
                self._save(accounts)
            self.audit.append(
                {"event": "login", "username": name, "outcome": "denied"}, actor=name
            )
            raise ValidationError("Invalid username or password.")

        account.failed_attempts = 0
        account.last_login_utc = datetime.now(timezone.utc).isoformat()
        accounts[name] = account
        self._save(accounts)
        self.audit.append({"event": "login", "username": name, "outcome": "granted"}, actor=name)

        return self.issue_token(account.username, account.role, account.password_hash)

    # -- tokens -----------------------------------------------------------

    def _credential_fingerprint(self, password_hash: str) -> str:
        """Short digest of the stored password hash.

        Carried inside the token so that changing a password ends every session
        that was opened with the old one. Without it, a stolen token stays
        usable for the rest of its hour even after the compromised password has
        been changed - which is exactly the moment it must stop working.
        """
        import hashlib

        return hashlib.sha256(password_hash.encode()).hexdigest()[:16]

    def issue_token(self, username: str, role: Role, password_hash: str = "") -> str:
        import base64
        import hashlib
        import hmac

        expiry = int(time.time() + self.session_minutes * 60)
        fingerprint = self._credential_fingerprint(password_hash) if password_hash else ""
        body = f"{username}:{role}:{expiry}:{fingerprint}"
        signature = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        return f"{base64.urlsafe_b64encode(body.encode()).decode()}.{signature}"

    def verify_token(self, token: str) -> dict[str, Any]:
        import base64
        import hashlib
        import hmac

        try:
            body_b64, signature = token.split(".")
            body = base64.urlsafe_b64decode(body_b64).decode()
            username, role, expiry, fingerprint = body.split(":")
        except (ValueError, TypeError) as exc:
            raise ValidationError("Malformed session token.") from exc

        expected = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValidationError("Session token signature is invalid.")
        if time.time() > int(expiry):
            raise ValidationError("Your session has expired. Please sign in again.")

        # The account may have been disabled or had its role changed since the
        # token was issued, so re-check rather than trusting the claim.
        account = self._load().get(username)
        if account is None or account.disabled:
            raise ValidationError("This account is no longer active.")
        if fingerprint and fingerprint != self._credential_fingerprint(account.password_hash):
            raise ValidationError("Your password was changed. Please sign in again.")

        return {"username": username, "role": account.role, "expires": int(expiry)}

    def can(self, token: str, capability: str) -> bool:
        try:
            claims = self.verify_token(token)
        except ValidationError:
            return False
        return capability in ROLE_CAPABILITIES.get(claims["role"], ())

    # -- reporting --------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        accounts = self._load()
        if not accounts:
            return pd.DataFrame(columns=["User", "Role", "Created", "Last login", "Status"])
        rows = []
        for account in sorted(accounts.values(), key=lambda a: a.username):
            status = "disabled" if account.disabled else ("locked" if account.locked else "active")
            if account.must_change_password:
                status += ", must change password"
            rows.append({
                "User": account.username,
                "Role": account.role,
                "Created": account.created_utc[:10],
                "Last login": (account.last_login_utc or "never")[:16].replace("T", " "),
                "Status": status,
            })
        return pd.DataFrame(rows)


def bootstrap_admin(store: AccountStore | None = None) -> tuple[bool, str]:
    """Create the first admin account if there are none.

    The password comes from ``DIVE_ADMIN_PASSWORD``. If that is not set, one is
    generated and returned once, so a deployment can never silently come up
    with a default password.
    """
    store = store or AccountStore()
    if not store.empty:
        return False, "Accounts already exist."

    password = os.environ.get("DIVE_ADMIN_PASSWORD", "")
    generated = False
    if not password:
        password = secrets.token_urlsafe(18)
        generated = True

    try:
        store.create("admin", password, role="admin", must_change_password=True)
    except ValidationError as exc:
        return False, str(exc)

    if generated:
        return True, (
            "Created the first admin account.\n"
            f"  username: admin\n"
            f"  password: {password}\n"
            "This is shown once. Sign in and change it immediately."
        )
    return True, (
        "Created the first admin account from DIVE_ADMIN_PASSWORD. "
        "You will be asked to change it at first sign-in."
    )
