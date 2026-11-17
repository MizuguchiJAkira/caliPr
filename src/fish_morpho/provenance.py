"""A record beside every export of the command that reproduces it.

The labeler's Export buttons run the same command-line exporters a script would
(``scripts/export_measurements.py``, ``scripts/export_tps.py``), so every
workbook and landmark file has an exact command behind it. This writes that
command next to the output, with the version of caliPr and of Python that ran
it, so the result can be regenerated without the labeler.
"""

from __future__ import annotations

import datetime as dt
import platform
import shlex
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def version() -> str:
    try:
        from importlib.metadata import version as dist_version
        return dist_version("fish_morpho")
    except Exception:
        import re
        m = re.search(r'^version\s*=\s*"([^"]+)"',
                      (_REPO / "pyproject.toml").read_text(), re.M) \
            if (_REPO / "pyproject.toml").is_file() else None
        return m.group(1) if m else "unknown"


def _shown(arg: str) -> str:
    """An argument as it should be re-typed: a path inside the repository as a
    relative one, so the record does not carry this machine's folders."""
    try:
        p = Path(arg)
        if p.is_absolute():
            return str(p.resolve().relative_to(_REPO))
    except (ValueError, OSError):
        pass
    return arg


def write_rerun(output: Path, script: str, argv: list[str]) -> Path:
    """Write ``<output>.rerun.txt`` (or ``RERUN.txt`` inside an output folder)."""
    target = output / "RERUN.txt" if output.is_dir() else output.with_name(output.name + ".rerun.txt")
    commit = _git("rev-parse", "HEAD") or "unknown (not a git checkout)"
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    lines = [
        "# The command that produced this export. Run it from the repository's top folder.",
        "python " + " ".join(shlex.quote(_shown(a)) for a in [script, *argv]),
        "",
        f"caliPr {version()}, commit {commit}"
        + (" with uncommitted changes to tracked files" if dirty else ""),
        f"Python {platform.python_version()} on {platform.system()} {platform.machine()}",
        f"Run {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
    ]
    target.write_text("\n".join(lines) + "\n")
    return target


def record(output: Path, argv: list[str] | None = None) -> Path:
    """``write_rerun`` for the running script, with its own command line."""
    script = Path(sys.argv[0])
    try:
        script_shown = str(script.resolve().relative_to(_REPO))
    except ValueError:
        script_shown = script.name
    return write_rerun(output, script_shown, list(sys.argv[1:] if argv is None else argv))
