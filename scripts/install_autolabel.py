#!/usr/bin/env python3
"""Install Auto-label: the environment it runs in, then the trained models.

    python scripts/install_autolabel.py

Hand labelling and export need nothing beyond the labeler. Auto-label also needs
PyTorch and DeepLabCut (about 1.7 GB) and the trained models with Segment Anything
(about 570 MB). This installs both:

1. ``.venv-train`` (in the repository, or under ``CALIPR_MODELS`` when that is
   set), holding exactly ``autolabel-requirements.txt``. It is built beside its
   final name and renamed into place only once everything installed, so an
   interrupted install is never mistaken for a working one.
2. The models, by ``fetch_model.py``.

The labeler's Auto-label button offers to run this when the environment is
missing, and shows its output line by line as progress. Standard library only.

Inside the Mac app the environment is a copy of the app's own Python rather than
a venv. A venv records the path of the Python it was made from, and the app's path
changes whenever it is moved -- macOS even runs a freshly downloaded app from a
temporary copy -- which would leave the venv pointing at nothing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from fish_morpho import components, paths  # noqa: E402

REQUIREMENTS = _ROOT / "scripts" / "autolabel-requirements.txt"


def say(msg: str) -> None:
    print(msg, flush=True)


def copy_this_python(dest: Path) -> Path:
    """A self-contained copy of the running interpreter, without its packages."""
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib = Path(sys.base_prefix) / "lib" / ver
    (dest / "bin").mkdir(parents=True)
    shutil.copy2(Path(sys.executable).resolve(), dest / "bin" / ver)
    for alias in ("python3", "python"):
        (dest / "bin" / alias).symlink_to(ver)
    shutil.copytree(stdlib, dest / "lib" / ver, symlinks=True,
                    ignore=lambda d, names: names if Path(d).name == "site-packages" else
                    [n for n in names if n in ("__pycache__", "test", "idlelib")])
    (dest / "lib" / ver / "site-packages").mkdir(exist_ok=True)
    return dest / "bin" / "python"


def run(cmd: list, **kw) -> None:
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            say(line)
    if proc.wait() != 0:
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd[:4])} … "
                         f"(exit {proc.returncode})")


def build_environment(target: Path, standalone: bool) -> None:
    part = target.with_name(target.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    say("Setting up Python for Auto-label")
    if standalone:
        py = copy_this_python(part)
        run([py, "-m", "ensurepip", "--default-pip"])
    else:
        run([sys.executable, "-m", "venv", part])
        py = part / "bin" / "python"
    say("Downloading PyTorch and DeepLabCut (about 1.7 GB; this takes a while)")
    # The environment's own pip, never one found on PATH.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    run([py, "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
         "--progress-bar", "off", "-r", REQUIREMENTS], env=env)
    part.rename(target)
    say("Auto-label environment installed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="install_autolabel")
    ap.add_argument("--standalone", action="store_true",
                    default=bool(os.environ.get("CALIPR_APP")),
                    help="copy this Python instead of making a venv (the Mac app does)")
    args = ap.parse_args(argv)

    if not components.released("autolabel"):
        say("Auto-label is not available to download yet.")
        return 2
    target = paths.TRAINING_PYTHON.parent.parent
    target.parent.mkdir(parents=True, exist_ok=True)
    if paths.TRAINING_PYTHON.is_file():
        say(f"Auto-label environment already installed at {target}")
    else:
        build_environment(target, args.standalone)

    import fetch_model
    say("Downloading the trained models")
    return fetch_model.main(["--root", str(paths.MODELS)])


if __name__ == "__main__":
    raise SystemExit(main())
