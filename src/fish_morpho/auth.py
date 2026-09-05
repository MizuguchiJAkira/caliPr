"""Local passphrase gate for the operations that can change training data.

What this is for
----------------
The labeler already binds to 127.0.0.1, so nothing here is defending against the
network. The thing worth guarding is narrower and more likely: during a demo, or
on a shared machine, somebody clicks Auto-label and then Save, and a set of model
predictions enters ``sidecars/`` as though a human had placed them. Train on
that and the model learns its own mistakes while the error curve improves,
because the labels are moving toward the predictions.

So this gates the automation, not the application. Hand labelling never needs a
passphrase.

Why hashed and not encrypted
----------------------------
A passphrase you can decrypt is a passphrase an attacker can decrypt, because the
key has to live somewhere on the same disk. What is stored is a **scrypt hash**:
a one-way function with a random salt and a deliberately expensive work factor.
Verifying means hashing the attempt and comparing; nothing can turn the stored
value back into the passphrase. Forgetting it means resetting, not recovering,
which is the correct trade.

Where it lives
--------------
``~/.calipr/auth.json`` — the user's home directory, **outside the repository
tree**. Not a gitignored file in the repo: gitignore is a rule someone can
override with ``git add -f``, or lose in a merge. A file git cannot see at all is
a stronger guarantee than a file git has been asked to ignore.

Limits, stated plainly
----------------------
Anyone who can run code as this user can read the sidecars and edit them
directly. This raises the cost of an accident, not of an attack. It is a guard
rail, not a security boundary, and should not be described as one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

#: scrypt work factors. n=2**15 costs ~40ms and 32 MiB per attempt here: nothing
#: when a human types a passphrase once, and enough memory-hardness to make a
#: dictionary run against the stored hash expensive rather than instant.
_N, _R, _P, _DKLEN = 2 ** 15, 8, 1, 32

#: scrypt needs 128*n*r bytes — 32 MiB at these factors — and OpenSSL refuses
#: anything over 32 MiB unless told otherwise, so it must be raised explicitly
#: or the derivation fails outright rather than running slowly.
_MAXMEM = 128 * _N * _R * _P * 2

#: How long an unlock lasts. A demo runs for an hour; a working day does not need
#: to re-authenticate, but an unattended laptop should not stay unlocked forever.
SESSION_SECONDS = 8 * 3600

#: Consecutive failures before the gate stops accepting attempts for a while.
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300


def auth_path() -> Path:
    """Where the credential lives. ``CALIPR_AUTH`` overrides, for tests."""
    env = os.environ.get("CALIPR_AUTH")
    if env:
        return Path(env)
    return Path.home() / ".calipr" / "auth.json"


def _derive(passphrase: str, salt: bytes) -> bytes:
    return hashlib.scrypt(passphrase.encode("utf-8"), salt=salt,
                          n=_N, r=_R, p=_P, dklen=_DKLEN, maxmem=_MAXMEM)


def set_passphrase(passphrase: str, path: Path | None = None) -> Path:
    """Store a new passphrase hash, replacing any existing one."""
    if len(passphrase) < 8:
        raise ValueError("passphrase must be at least 8 characters")
    path = path or auth_path()
    salt = secrets.token_bytes(16)
    doc = {
        "version": 1,
        "kdf": "scrypt",
        "n": _N, "r": _R, "p": _P, "dklen": _DKLEN,
        "salt": salt.hex(),
        "hash": _derive(passphrase, salt).hex(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    # Owner-only. Other accounts on the machine have no reason to read even a
    # hash, and a readable hash is a hash somebody can attack offline.
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def is_configured(path: Path | None = None) -> bool:
    return (path or auth_path()).is_file()


def verify(passphrase: str, path: Path | None = None) -> bool:
    """True if ``passphrase`` matches the stored hash."""
    path = path or auth_path()
    try:
        doc = json.loads(path.read_text())
        salt = bytes.fromhex(doc["salt"])
        want = bytes.fromhex(doc["hash"])
    except Exception:
        return False
    n, r, pp = int(doc.get("n", _N)), int(doc.get("r", _R)), int(doc.get("p", _P))
    got = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt,
                         n=n, r=r, p=pp, dklen=int(doc.get("dklen", _DKLEN)),
                         maxmem=128 * n * r * pp * 2)
    # Constant-time: a plain == leaks how much of the hash matched via timing.
    return hmac.compare_digest(got, want)


@dataclass
class Session:
    """In-memory unlock state. Deliberately not persisted anywhere."""

    token: str = ""
    expires: float = 0.0
    failures: int = 0
    locked_until: float = 0.0

    def unlocked(self, token: str | None) -> bool:
        return bool(self.token) and token == self.token and time.time() < self.expires

    def attempt(self, passphrase: str, path: Path | None = None) -> tuple[bool, str]:
        """(ok, message). Backs off after repeated failures."""
        now = time.time()
        if now < self.locked_until:
            return False, (f"too many attempts — locked for "
                           f"{int(self.locked_until - now)}s")
        if not verify(passphrase, path):
            self.failures += 1
            if self.failures >= MAX_ATTEMPTS:
                self.locked_until = now + LOCKOUT_SECONDS
                self.failures = 0
                return False, f"too many attempts — locked for {LOCKOUT_SECONDS}s"
            return False, (f"wrong passphrase "
                           f"({MAX_ATTEMPTS - self.failures} left)")
        self.token = secrets.token_urlsafe(32)
        self.expires = now + SESSION_SECONDS
        self.failures = 0
        return True, "unlocked"

    def lock(self) -> None:
        self.token, self.expires = "", 0.0
