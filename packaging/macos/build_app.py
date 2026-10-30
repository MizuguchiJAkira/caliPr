#!/usr/bin/env python3
"""Build caliPr.app, a Mac app that needs no Python, Terminal or install steps.

    python packaging/macos/build_app.py            # -> build/caliPr.app, dist/caliPr-<v>-mac.zip

Needs ``uv`` (https://docs.astral.sh/uv/) and the Xcode command line tools
(``xcode-select --install``), on an Apple Silicon Mac; the app it builds runs on
Apple Silicon Macs with macOS 12 or later.

What goes in:

- ``Contents/MacOS/caliPr``, a small compiled launcher (``launcher.c``), and
  ``Contents/MacOS/python3``, a standalone CPython (python-build-standalone, as
  uv installs it) with its standard library in ``Contents/lib``. Nothing on the
  user's machine is used or changed.
- The labeler's dependencies and pywebview (``requirements.txt``).
- ``scripts/``, ``src/`` and ``docs/`` from this repository, under
  ``Contents/Resources/app``. No photographs, sidecars, models or results: the
  app starts with no studies, and Auto-label installs its models on request.

The app is signed ad hoc, not by an Apple Developer ID, so the first time it is
opened macOS refuses; System Settings > Privacy & Security > "Open Anyway" lets
it run from then on. A notarised build would need a paid developer account.
"""

from __future__ import annotations

import argparse
import plistlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = "3.12"
BUNDLE_ID = "io.github.calipr"
# Not needed by anything the app runs, and a fifth of the standard library.
STDLIB_SKIP = {"test", "idlelib", "tkinter", "turtledemo", "__phello__", "lib2to3",
               "site-packages", "config-3.12-darwin", "EXTERNALLY-MANAGED"}
CODE_SKIP = {"__pycache__", ".DS_Store", ".pytest_cache"}


def sh(*cmd, **kw) -> str:
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, text=True,
                          stdout=subprocess.PIPE, **kw).stdout.strip()


def version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)
    return m.group(1) if m else "0"


def standalone_python() -> Path:
    exe = Path(sh("uv", "python", "find", "--managed-python", PY)).resolve()
    if not (exe.parent.parent / "lib" / f"python{PY}" / "os.py").is_file():
        raise SystemExit(f"{exe} is not a standalone Python; run `uv python install {PY}`")
    return exe


def build(out: Path) -> Path:
    app = out / "caliPr.app"
    shutil.rmtree(app, ignore_errors=True)
    contents = app / "Contents"
    macos, res = contents / "MacOS", contents / "Resources"
    macos.mkdir(parents=True)
    res.mkdir()

    print("Python")
    src = standalone_python()
    shutil.copy2(src, macos / "python3")
    lib = contents / "lib" / f"python{PY}"
    shutil.copytree(src.parent.parent / "lib" / f"python{PY}", lib, symlinks=True,
                    ignore=lambda d, names: [n for n in names if n in STDLIB_SKIP
                                             or n == "__pycache__"])
    (lib / "site-packages").mkdir()

    print("Packages")
    sh("uv", "pip", "install", "--python", macos / "python3", "--break-system-packages",
       "--quiet", "-r", HERE / "requirements.txt")
    site = lib / "site-packages"
    for junk in [site / "PyObjCTest", *site.glob("numpy/**/tests")]:
        shutil.rmtree(junk, ignore_errors=True)

    print("caliPr")
    code = res / "app"
    for part in ("scripts", "src", "docs"):
        shutil.copytree(REPO / part, code / part,
                        ignore=lambda d, names: [n for n in names if n in CODE_SKIP
                                                 or n.endswith(".egg-info")])
    (code / "packaging" / "macos").mkdir(parents=True)
    shutil.copy2(HERE / "calipr_app.py", code / "packaging" / "macos" / "calipr_app.py")
    for f in ("LICENSE", "README.md", "pyproject.toml"):
        shutil.copy2(REPO / f, code / f)

    print("Launcher and icon")
    sh("clang", "-O2", "-Wall", "-arch", "arm64", "-mmacosx-version-min=12.0",
       "-o", macos / "caliPr", HERE / "launcher.c")
    iconset = out / "caliPr.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir()
    for px in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            name = f"icon_{px}x{px}{'@2x' if scale == 2 else ''}.png"
            sh("sips", "-z", px * scale, px * scale, HERE / "caliPr.png",
               "--out", iconset / name)
    sh("iconutil", "-c", "icns", iconset, "-o", res / "caliPr.icns")
    shutil.rmtree(iconset)

    v = version()
    plistlib.dump({
        "CFBundleName": "caliPr",
        "CFBundleDisplayName": "caliPr",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": "caliPr",
        "CFBundleIconFile": "caliPr",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": v,
        "CFBundleVersion": v,
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.education",
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
        "NSHumanReadableCopyright": "MIT License",
    }, open(contents / "Info.plist", "wb"))

    print("Compiling")
    # By the app's own Python, whose bytecode it is. Unchecked-hash bytecode is
    # used whatever the files' dates, which a zip does not keep exactly; the app
    # never writes bytecode of its own (launcher.c).
    sh(macos / "python3", "-m", "compileall", "-q", "-j", "0",
       "--invalidation-mode", "unchecked-hash", lib, code)

    print("Signing (ad hoc)")
    for f in sorted(contents.rglob("*")):
        if f.is_file() and not f.is_symlink() and f.suffix in (".so", ".dylib"):
            subprocess.run(["codesign", "--force", "--sign", "-", str(f)],
                           check=True, capture_output=True)
    sh("codesign", "--force", "--sign", "-", macos / "python3")
    sh("codesign", "--force", "--deep", "--sign", "-", app)
    sh("codesign", "--verify", "--deep", "--strict", app)
    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_app")
    ap.add_argument("--out", type=Path, default=REPO / "build")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args(argv)
    if sys.platform != "darwin":
        raise SystemExit("the Mac app is built on a Mac")
    app = build(args.out)
    size = sh("du", "-sh", app).split()[0]
    print(f"\nbuilt {app} ({size})")
    if not args.no_zip:
        dist = REPO / "dist"
        dist.mkdir(exist_ok=True)
        z = dist / f"caliPr-{version()}-mac.zip"
        z.unlink(missing_ok=True)
        # ditto keeps the signature and permissions a plain zip would drop.
        sh("ditto", "-c", "-k", "--keepParent", app, z)
        print(f"zipped  {z} ({z.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
