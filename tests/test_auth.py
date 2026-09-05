"""Tests for the passphrase gate on automated landmarking.

The property that matters most is the first one: the passphrase must not be
recoverable from what is written to disk. Everything else here is behaviour
around that.
"""

from __future__ import annotations

import json
import time

import pytest

from fish_morpho import auth

PASS = "brook-trout-2026"


@pytest.fixture
def cred(tmp_path):
    path = tmp_path / "auth.json"
    auth.set_passphrase(PASS, path)
    return path


# --------------------------------------------------------------------------
# the stored credential
# --------------------------------------------------------------------------

def test_passphrase_is_not_recoverable_from_the_file(cred):
    """The whole point: what lands on disk must not contain the secret."""
    raw = cred.read_text()
    assert PASS not in raw
    assert PASS.encode().hex() not in raw
    doc = json.loads(raw)
    assert doc["kdf"] == "scrypt"
    # A hash with no salt is a hash a rainbow table answers.
    assert len(bytes.fromhex(doc["salt"])) == 16
    assert len(bytes.fromhex(doc["hash"])) == 32


def test_same_passphrase_hashes_differently_each_time(tmp_path):
    """Distinct salts, so two people choosing the same passphrase are not
    visibly identical to anyone reading both files."""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    auth.set_passphrase(PASS, a)
    auth.set_passphrase(PASS, b)
    da, db = json.loads(a.read_text()), json.loads(b.read_text())
    assert da["salt"] != db["salt"]
    assert da["hash"] != db["hash"]
    assert auth.verify(PASS, a) and auth.verify(PASS, b)


def test_file_is_owner_only(cred):
    assert cred.stat().st_mode & 0o077 == 0, "group/other must not read the hash"


def test_short_passphrase_refused(tmp_path):
    with pytest.raises(ValueError):
        auth.set_passphrase("short", tmp_path / "a.json")


def test_verify(cred):
    assert auth.verify(PASS, cred)
    assert not auth.verify(PASS + "x", cred)
    assert not auth.verify("", cred)


def test_verify_on_missing_or_corrupt_file_is_false(tmp_path):
    assert not auth.verify(PASS, tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert not auth.verify(PASS, bad)


def test_is_configured(tmp_path, cred):
    assert auth.is_configured(cred)
    assert not auth.is_configured(tmp_path / "absent.json")


def test_auth_path_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CALIPR_AUTH", str(tmp_path / "x.json"))
    assert auth.auth_path() == tmp_path / "x.json"
    monkeypatch.delenv("CALIPR_AUTH")
    # Default lives outside the repository, so git cannot see it at all.
    assert auth.auth_path().name == "auth.json"
    assert ".calipr" in str(auth.auth_path())


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------

def test_session_starts_locked(cred):
    s = auth.Session()
    assert not s.unlocked(None)
    assert not s.unlocked("")


def test_unlock_then_token_works(cred):
    s = auth.Session()
    ok, _ = s.attempt(PASS, cred)
    assert ok
    assert s.unlocked(s.token)
    assert not s.unlocked("some-other-token")
    assert not s.unlocked(None)


def test_wrong_passphrase_does_not_unlock(cred):
    s = auth.Session()
    ok, msg = s.attempt("wrong", cred)
    assert not ok
    assert not s.unlocked(s.token)
    assert "wrong passphrase" in msg


def test_lock_revokes_the_token(cred):
    s = auth.Session()
    s.attempt(PASS, cred)
    tok = s.token
    s.lock()
    assert not s.unlocked(tok)


def test_expired_session_is_locked(cred, monkeypatch):
    s = auth.Session()
    s.attempt(PASS, cred)
    tok = s.token
    monkeypatch.setattr(time, "time", lambda: s.expires + 1)
    assert not s.unlocked(tok), "a stale token must not keep working"


def test_repeated_failures_lock_out(cred):
    s = auth.Session()
    for _ in range(auth.MAX_ATTEMPTS - 1):
        assert not s.attempt("wrong", cred)[0]
    ok, msg = s.attempt("wrong", cred)
    assert not ok and "too many attempts" in msg
    # ...and the correct passphrase is refused while the lockout stands, so the
    # backoff cannot be stepped around by guessing right on the next try.
    ok, msg = s.attempt(PASS, cred)
    assert not ok and "locked" in msg


def test_successful_unlock_clears_the_failure_count(cred):
    s = auth.Session()
    s.attempt("wrong", cred)
    s.attempt("wrong", cred)
    assert s.attempt(PASS, cred)[0]
    assert s.failures == 0
