"""Security controls: audit, encryption at rest, access control, input validation.

A statement worth making plainly, because the alternative is a false sense of
safety: **no software is unbreakable, and this module does not make Dive
unhackable.** What it does is raise the cost of an attack and make a successful
one visible afterwards. Controls implemented here:

* a tamper-evident audit log - each entry carries the hash of the previous one,
  so a deleted or edited entry breaks the chain and ``verify()`` finds it
* authenticated encryption at rest for stored cohort data and model artefacts
* password hashing with scrypt, per-user salts, and lockout on repeated failure
* signed, expiring session tokens with constant-time comparison
* upload validation: size caps, extension and magic-byte checks, path-traversal
  rejection, and neutralisation of spreadsheet formula injection
* secret handling that keeps keys out of source, out of logs and out of repr()

What remains the deployer's responsibility, and cannot be solved in this file:
TLS termination and certificate management, network isolation, OS and
dependency patching, backup encryption and restore testing, key custody and
rotation, and an independent penetration test before any real patient data is
loaded. Clinical deployment additionally carries legal obligations (UK GDPR /
DPA 2018, a DPIA, and - if Dive's output is used to guide care rather than
research - possible medical device regulation) that are governance questions,
not code.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

__all__ = [
    "AuditLog",
    "SecretRef",
    "DataVault",
    "ENCRYPTED_SUFFIX",
    "PasswordHasher",
    "SessionManager",
    "UploadValidator",
    "ValidationError",
    "Role",
    "requires_role",
]


class ValidationError(Exception):
    """Raised when input fails a security check. Never silently corrected."""


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------

class AuditLog:
    """Append-only, hash-chained audit log.

    Each record stores the SHA-256 of the previous record, so removing or
    editing any entry invalidates every hash after it. That does not prevent
    an attacker with write access from rewriting the whole file, but it does
    mean they cannot quietly remove one line - which is the common case.

    For stronger guarantees the file should be shipped to append-only storage
    (a WORM bucket, or a syslog collector on a separate host).
    """

    GENESIS = "0" * 64

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return self.GENESIS
        last = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        if last is None:
            return self.GENESIS
        try:
            return json.loads(last)["entry_sha256"]
        except (json.JSONDecodeError, KeyError):
            return self.GENESIS

    @staticmethod
    def _digest(record: dict) -> str:
        return hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode()
        ).hexdigest()

    def append(self, event: dict, actor: str = "system") -> dict:
        """Write one audit record. Values are redacted before storage."""
        from .privacy import redact

        safe = json.loads(redact(json.dumps(event, default=str)))
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "prev_sha256": self._last_hash(),
            **safe,
        }
        record["entry_sha256"] = self._digest(record)

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def verify(self) -> tuple[bool, list[str]]:
        """Re-walk the chain. Returns ``(intact, problems)``."""
        if not self.path.exists():
            return True, []

        problems: list[str] = []
        previous = self.GENESIS
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    problems.append(f"line {number}: not valid JSON")
                    return False, problems

                claimed = record.pop("entry_sha256", None)
                if record.get("prev_sha256") != previous:
                    problems.append(
                        f"line {number}: chain broken - expected previous hash "
                        f"{previous[:12]}..., found {str(record.get('prev_sha256'))[:12]}..."
                    )
                if claimed != self._digest(record):
                    problems.append(f"line {number}: content does not match its hash")
                previous = claimed or self.GENESIS
        return not problems, problems

    def tail(self, n: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        out = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------

@dataclass
class SecretRef:
    """A secret held by reference, never printed.

    ``repr`` and ``str`` are overridden so a secret cannot reach a log or a
    traceback by accident - the single most common way API keys leak.
    """

    env_var: str
    _value: str | None = field(default=None, repr=False)

    def resolve(self) -> str | None:
        if self._value is not None:
            return self._value
        return os.environ.get(self.env_var)

    def present(self) -> bool:
        return bool(self.resolve())

    def fingerprint(self) -> str:
        """A stable, non-reversible identifier, safe to log."""
        value = self.resolve()
        if not value:
            return "absent"
        return hashlib.sha256(value.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        return f"SecretRef({self.env_var}=<{'set' if self.present() else 'unset'}>)"

    __str__ = __repr__


# --------------------------------------------------------------------------
# Encryption at rest
# --------------------------------------------------------------------------

#: Appended to a filename whose contents are encrypted. Kept visible in the
#: name so that a file on disk is never ambiguous about what it holds.
ENCRYPTED_SUFFIX = ".enc"


class DataVault:
    """Authenticated encryption for files at rest.

    Uses AES-GCM via ``cryptography`` when available, falling back to
    HMAC-authenticated ChaCha-style XOR only if it is not - and the fallback
    refuses to run unless explicitly permitted, because rolling your own
    cipher is a bad idea and it should be a deliberate choice, not a silent
    degradation.

    The key comes from ``DIVE_ENCRYPTION_KEY`` (base64, 32 bytes). It is never
    written to disk by this class.
    """

    KEY_ENV = "DIVE_ENCRYPTION_KEY"

    def __init__(self, key: bytes | None = None, allow_insecure_fallback: bool = False):
        self.allow_insecure_fallback = allow_insecure_fallback
        self._key = key or self._key_from_env()

    @classmethod
    def generate_key(cls) -> str:
        """A fresh base64 key. Store it in a secret manager, not in the repo."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

    def _key_from_env(self) -> bytes | None:
        raw = os.environ.get(self.KEY_ENV)
        if not raw:
            return None
        try:
            key = base64.urlsafe_b64decode(raw)
        except Exception as exc:
            raise ValidationError(f"{self.KEY_ENV} is not valid base64: {exc}") from exc
        if len(key) != 32:
            raise ValidationError(f"{self.KEY_ENV} must decode to 32 bytes, got {len(key)}")
        return key

    @property
    def available(self) -> bool:
        return self._key is not None

    def _require_key(self) -> bytes:
        if self._key is None:
            raise ValidationError(
                f"No encryption key. Set {self.KEY_ENV} to a base64 32-byte key "
                "(DataVault.generate_key() produces one)."
            )
        return self._key

    def encrypt(self, plaintext: bytes, associated_data: bytes = b"dive") -> bytes:
        key = self._require_key()
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = secrets.token_bytes(12)
            return b"AGCM" + nonce + AESGCM(key).encrypt(nonce, plaintext, associated_data)
        except BaseException:
            # A broken `cryptography` build (missing _cffi_backend, a mismatched
            # native wheel) raises well below ImportError - including as a Rust
            # panic - so catch broadly here. The alternative is that encryption
            # at rest silently fails to load on a deployment that thought it
            # had it.
            if not self.allow_insecure_fallback:
                raise ValidationError(
                    "The `cryptography` package is required for encryption at rest and is "
                    "not usable here. Install it (pip install cryptography cffi) rather "
                    "than relying on a hand-rolled cipher."
                )
            nonce = secrets.token_bytes(16)
            stream = self._keystream(key, nonce, len(plaintext))
            body = bytes(a ^ b for a, b in zip(plaintext, stream))
            tag = hmac.new(key, nonce + body + associated_data, hashlib.sha256).digest()
            return b"FALL" + nonce + tag + body

    def decrypt(self, blob: bytes, associated_data: bytes = b"dive") -> bytes:
        key = self._require_key()
        if blob.startswith(b"AGCM"):
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            except BaseException as exc:
                raise ValidationError(
                    "This data was encrypted with AES-GCM but `cryptography` is not "
                    f"usable here ({exc}). Install it to decrypt."
                ) from exc
            nonce, body = blob[4:16], blob[16:]
            try:
                return AESGCM(key).decrypt(nonce, body, associated_data)
            except Exception as exc:
                # InvalidTag carries no message at all. A file that has been
                # altered, truncated or encrypted under a different key should
                # say so, not surface an empty exception.
                raise ValidationError(
                    "Ciphertext failed authentication; refusing to decrypt. The file "
                    "has been altered or was encrypted with a different key."
                ) from exc
        if blob.startswith(b"FALL"):
            nonce, tag, body = blob[4:20], blob[20:52], blob[52:]
            expected = hmac.new(key, nonce + body + associated_data, hashlib.sha256).digest()
            if not hmac.compare_digest(tag, expected):
                raise ValidationError("Ciphertext failed authentication; refusing to decrypt.")
            stream = self._keystream(key, nonce, len(body))
            return bytes(a ^ b for a, b in zip(body, stream))
        raise ValidationError("Unrecognised ciphertext header.")

    @staticmethod
    def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            out.extend(hashlib.sha256(key + nonce + counter.to_bytes(8, "big")).digest())
            counter += 1
        return bytes(out[:length])

    def encrypt_file(self, source: Path, target: Path | None = None) -> Path:
        source = Path(source)
        target = Path(target) if target else source.with_suffix(source.suffix + ".enc")
        target.write_bytes(self.encrypt(source.read_bytes()))
        return target

    def decrypt_file(self, source: Path, target: Path) -> Path:
        Path(target).write_bytes(self.decrypt(Path(source).read_bytes()))
        return Path(target)

    # -- the pair the application actually uses ---------------------------

    def write_file(self, payload: bytes, target: Path) -> Path:
        """Write ``payload``, encrypted if a key is configured.

        Returns the path actually written - ``target`` with ``.enc`` appended
        when the payload was encrypted - so the caller records where the data
        really is rather than where it asked for it to go. With no key set the
        file is written in the clear, which is the honest outcome: pretending
        otherwise is worse than being unencrypted.
        """
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        if not self.available:
            target.write_bytes(payload)
            written = target
        else:
            written = target.with_name(target.name + ENCRYPTED_SUFFIX)
            written.write_bytes(self.encrypt(payload))

        try:
            os.chmod(written, 0o600)
        except OSError:
            pass
        return written

    def read_file(self, path: Path | str) -> bytes:
        """Read a file written by :meth:`write_file`, decrypting if needed."""
        path = Path(path)
        raw = path.read_bytes()
        if path.name.endswith(ENCRYPTED_SUFFIX):
            return self.decrypt(raw)
        return raw

    @staticmethod
    def plain_name(path: Path | str) -> str:
        """The filename without the encryption suffix, for extension checks."""
        name = Path(path).name
        return name[: -len(ENCRYPTED_SUFFIX)] if name.endswith(ENCRYPTED_SUFFIX) else name


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------

Role = Literal["viewer", "analyst", "admin"]

ROLE_RANK: dict[str, int] = {"viewer": 0, "analyst": 1, "admin": 2}

#: What each role may do. Deliberately coarse - fine-grained permissions that
#: nobody understands are worse than a few that everybody does.
ROLE_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "viewer": ("view_results", "view_aggregates"),
    "analyst": ("view_results", "view_aggregates", "view_individual", "run_models",
                "ingest_data", "register_metrics"),
    "admin": ("view_results", "view_aggregates", "view_individual", "run_models",
              "ingest_data", "register_metrics", "manage_users", "promote_models",
              "change_privacy_policy", "export_data"),
}


class PasswordHasher:
    """scrypt password hashing with per-password salt.

    scrypt is memory-hard, which is what makes offline cracking of a stolen
    hash database expensive. Parameters follow current OWASP guidance; raise
    ``n`` as hardware improves.
    """

    def __init__(self, n: int = 2 ** 15, r: int = 8, p: int = 1, dklen: int = 32):
        self.n, self.r, self.p, self.dklen = n, r, p, dklen

    @staticmethod
    def _maxmem(n: int, r: int, p: int) -> int:
        """scrypt needs 128*n*r bytes; OpenSSL caps at 32 MB unless told otherwise.

        With the OWASP-recommended n=2**15, r=8 that is 33.5 MB, so the default
        cap rejects the very parameters we want. Compute the requirement and
        allow headroom rather than weakening the parameters to fit the cap.
        """
        return int(128 * n * r * p * 1.5) + (1 << 20)

    def hash(self, password: str) -> str:
        if len(password) < 12:
            raise ValidationError("Password must be at least 12 characters.")
        salt = secrets.token_bytes(16)
        derived = hashlib.scrypt(
            password.encode(), salt=salt, n=self.n, r=self.r, p=self.p, dklen=self.dklen,
            maxmem=self._maxmem(self.n, self.r, self.p),
        )
        return f"scrypt${self.n}${self.r}${self.p}${base64.b64encode(salt).decode()}${base64.b64encode(derived).decode()}"

    def verify(self, password: str, stored: str) -> bool:
        try:
            scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
            if scheme != "scrypt":
                return False
            derived = hashlib.scrypt(
                password.encode(), salt=base64.b64decode(salt_b64),
                n=int(n), r=int(r), p=int(p), dklen=len(base64.b64decode(hash_b64)),
                maxmem=self._maxmem(int(n), int(r), int(p)),
            )
        except (ValueError, TypeError):
            return False
        # Constant-time: a timing difference here leaks how much of the hash matched.
        return hmac.compare_digest(derived, base64.b64decode(hash_b64))


@dataclass
class _Account:
    username: str
    password_hash: str
    role: Role = "viewer"
    failed_attempts: int = 0
    locked_until: float = 0.0


class SessionManager:
    """Signed, expiring session tokens and account lockout.

    Tokens are HMAC-signed over (username, role, expiry) with a server secret,
    so a token cannot be forged or have its role edited client-side.
    """

    def __init__(
        self,
        secret: bytes | None = None,
        session_minutes: int = 60,
        max_failed_attempts: int = 5,
        lockout_minutes: int = 15,
    ):
        self._secret = secret or os.environ.get("DIVE_SESSION_SECRET", "").encode() or secrets.token_bytes(32)
        self.session_minutes = session_minutes
        self.max_failed_attempts = max_failed_attempts
        self.lockout_minutes = lockout_minutes
        self.hasher = PasswordHasher()
        self._accounts: dict[str, _Account] = {}

    # -- accounts ---------------------------------------------------------

    def create_user(self, username: str, password: str, role: Role = "viewer") -> None:
        username = username.strip().lower()
        if not re.fullmatch(r"[a-z0-9._-]{3,64}", username):
            raise ValidationError(
                "Username must be 3-64 characters of lowercase letters, digits, dot, "
                "underscore or hyphen."
            )
        if role not in ROLE_RANK:
            raise ValidationError(f"Unknown role '{role}'.")
        if username in self._accounts:
            raise ValidationError(f"User '{username}' already exists.")
        self._accounts[username] = _Account(username, self.hasher.hash(password), role)

    def authenticate(self, username: str, password: str) -> str:
        """Return a session token, or raise. Failures are deliberately uniform."""
        username = username.strip().lower()
        account = self._accounts.get(username)
        now = time.time()

        if account and account.locked_until > now:
            wait = int((account.locked_until - now) / 60) + 1
            raise ValidationError(f"Account locked. Try again in {wait} minute(s).")

        # Always run a hash even for an unknown user, so response time does not
        # reveal whether the username exists.
        reference = account.password_hash if account else self.hasher.hash("x" * 16)
        valid = self.hasher.verify(password, reference) and account is not None

        if not valid:
            if account:
                account.failed_attempts += 1
                if account.failed_attempts >= self.max_failed_attempts:
                    account.locked_until = now + self.lockout_minutes * 60
                    account.failed_attempts = 0
            raise ValidationError("Invalid username or password.")

        account.failed_attempts = 0
        return self.issue_token(account.username, account.role)

    # -- tokens -----------------------------------------------------------

    def issue_token(self, username: str, role: Role) -> str:
        expiry = int(time.time() + self.session_minutes * 60)
        body = f"{username}:{role}:{expiry}"
        signature = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        return f"{base64.urlsafe_b64encode(body.encode()).decode()}.{signature}"

    def verify_token(self, token: str) -> dict[str, Any]:
        try:
            body_b64, signature = token.split(".")
            body = base64.urlsafe_b64decode(body_b64).decode()
            username, role, expiry = body.split(":")
        except (ValueError, TypeError) as exc:
            raise ValidationError("Malformed session token.") from exc

        expected = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValidationError("Session token signature is invalid.")
        if time.time() > int(expiry):
            raise ValidationError("Session has expired.")
        return {"username": username, "role": role, "expires": int(expiry)}

    def can(self, token: str, capability: str) -> bool:
        try:
            claims = self.verify_token(token)
        except ValidationError:
            return False
        return capability in ROLE_CAPABILITIES.get(claims["role"], ())


def requires_role(minimum: Role):
    """Decorator gating a function on the caller's role.

    The wrapped function must accept a ``token`` keyword.
    """
    def decorator(func):
        def wrapper(*args, session: SessionManager, token: str, **kwargs):
            claims = session.verify_token(token)
            if ROLE_RANK.get(claims["role"], -1) < ROLE_RANK[minimum]:
                raise ValidationError(
                    f"'{claims['role']}' is not permitted to do this; "
                    f"'{minimum}' or higher is required."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------

class UploadValidator:
    """Validates files before anything opens them.

    Covers the three things that actually go wrong with clinical spreadsheet
    uploads: a file far larger than any real cohort (resource exhaustion), a
    file whose extension lies about its contents, and a cell beginning with
    ``=`` that executes when someone later opens the export in Excel.
    """

    #: extension -> accepted leading magic bytes
    MAGIC: dict[str, tuple[bytes, ...]] = {
        ".xlsx": (b"PK\x03\x04",),
        ".xlsm": (b"PK\x03\x04",),
        ".xls": (b"\xd0\xcf\x11\xe0",),
        ".csv": (),
        ".tsv": (),
        ".json": (),
        ".parquet": (b"PAR1",),
    }

    #: Leading characters Excel and Sheets treat as the start of a formula.
    FORMULA_PREFIXES = ("=", "+", "-", "@", "\t=", "\r=")

    def __init__(self, max_bytes: int = 200 * 1024 * 1024, allowed_suffixes: Iterable[str] | None = None):
        self.max_bytes = max_bytes
        self.allowed_suffixes = set(allowed_suffixes or self.MAGIC.keys())

    def validate_filename(self, filename: str) -> str:
        """Return a safe basename, or raise.

        Rejects path traversal outright rather than sanitising it - a filename
        containing ``../`` is not a typo, and quietly stripping it hides an
        attempt.
        """
        name = unicodedata.normalize("NFKC", str(filename)).strip()
        if not name:
            raise ValidationError("Filename is empty.")
        if "\x00" in name:
            raise ValidationError("Filename contains a null byte.")
        if "/" in name or "\\" in name or ".." in name:
            raise ValidationError(f"Filename '{filename}' contains a path component.")
        if name.startswith("."):
            raise ValidationError("Filename may not start with a dot.")
        if len(name) > 255:
            raise ValidationError("Filename is too long.")

        suffix = Path(name).suffix.lower()
        if suffix not in self.allowed_suffixes:
            raise ValidationError(
                f"File type '{suffix}' is not accepted. Allowed: "
                f"{', '.join(sorted(self.allowed_suffixes))}."
            )
        return name

    def validate_bytes(self, data: bytes, filename: str) -> str:
        name = self.validate_filename(filename)
        if len(data) > self.max_bytes:
            raise ValidationError(
                f"File is {len(data) / 1e6:.1f} MB; the limit is "
                f"{self.max_bytes / 1e6:.0f} MB."
            )
        if not data:
            raise ValidationError("File is empty.")

        expected = self.MAGIC.get(Path(name).suffix.lower(), ())
        if expected and not any(data.startswith(m) for m in expected):
            raise ValidationError(
                f"File contents do not match the '{Path(name).suffix}' extension. "
                "The file may be corrupt or mislabelled."
            )
        return name

    @classmethod
    def neutralise_formula(cls, value: Any) -> Any:
        """Prefix a leading formula character so spreadsheets treat it as text."""
        if not isinstance(value, str):
            return value
        stripped = value.lstrip("\t\r\n ")
        if stripped.startswith(cls.FORMULA_PREFIXES):
            return "'" + value
        return value

    @classmethod
    def sanitise_for_export(cls, frame: Any) -> Any:
        """Neutralise formula injection across every text cell of a DataFrame."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            return frame
        out = frame.copy()
        for column in out.columns:
            if out[column].dtype == object:
                out[column] = out[column].map(cls.neutralise_formula)
        return out
