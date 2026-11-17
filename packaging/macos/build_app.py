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
  ``Contents/Resources/app``, taken from a commit (``--ref``, default HEAD) with
  ``git archive`` rather than copied from the working folder, so uncommitted or
  untracked files never ship. No photographs, sidecars, models or results: the
  app starts with no studies, and Auto-label installs its models on request.
- ``Contents/Resources/THIRD_PARTY_NOTICES/``: the licence files of every
  bundled package and of the bundled Python, with an index.

The OpenCV wheels for macOS bundle FFmpeg configured with ``--enable-gpl``
(libav* and libsw* under GPLv3, plus x264 and x265), and the import of cv2 loads
them, so they cannot simply be left out. caliPr reads and writes only still
images, so the fix is an OpenCV wheel built without FFmpeg, pinned in
``requirements.txt``::

    ENABLE_HEADLESS=1 CMAKE_ARGS="-DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF" \\
        pip wheel --no-binary :all: --no-deps opencv-python-headless==5.0.0.93
    delocate-wheel -w wheels opencv_python_headless-*.whl

The build stops if it finds GPL FFmpeg libraries in OpenCV; ``--allow-gpl``
builds anyway for local testing, and such a build must not be published.

The app is signed ad hoc, not by an Apple Developer ID, so the first time it is
opened macOS refuses; System Settings > Privacy & Security > "Open Anyway" lets
it run from then on. A notarised build would need a paid developer account.
"""

from __future__ import annotations

import argparse
import io
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from email.parser import Parser
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


#: Libraries the standalone CPython links statically; their licences are listed in
#: python-build-standalone's distribution (PYTHON.json, "licenses").
PYTHON_STATIC = ("OpenSSL (Apache-2.0)", "libffi (MIT)", "mpdecimal (BSD-2-Clause)",
                 "expat (MIT)", "xz/liblzma (0BSD)", "zlib (Zlib)", "bzip2 (bzip2-1.0.6)",
                 "SQLite (public domain)", "ncurses (X11)", "readline-compatible libedit (BSD)")
#: FFmpeg as the OpenCV wheels bundle it (Homebrew's, configured --enable-gpl
#: --enable-version3): every FFmpeg library is then GPLv3, plus the GPL codecs.
GPL_DYLIBS = ("libavcodec", "libavformat", "libavutil", "libavfilter", "libavdevice",
              "libswscale", "libswresample", "libpostproc",
              "libx264", "libx265", "libvidstab", "librubberband")


def source_tree(ref: str, dest: Path) -> Path:
    """The app's code exactly as committed at ``ref``, unpacked into ``dest``."""
    data = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", ref,
                           "scripts", "src", "docs", "LICENSE", "README.md", "pyproject.toml",
                           "packaging/macos/calipr_app.py"],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        try:
            t.extractall(dest, filter="data")
        except TypeError:                  # Python before 3.11.4: no extraction filters
            t.extractall(dest)
    return dest


def write_notices(site: Path, pylib: Path, dest: Path) -> list[str]:
    """Copy every bundled component's licence files into ``dest``; return GPL dylibs found."""
    dest.mkdir(parents=True, exist_ok=True)
    index = ["Third-party software bundled in caliPr.app", "",
             "caliPr itself is MIT-licensed (LICENSE in Resources/app). The components",
             "below keep their own licences; their licence files are in this folder.", ""]
    for info in sorted(site.glob("*.dist-info")):
        meta = Parser().parsestr((info / "METADATA").read_text(errors="replace")) \
            if (info / "METADATA").is_file() else {}
        name = meta.get("Name") or info.name.split("-")[0]
        licence = meta.get("License-Expression") or meta.get("License") or ""
        licence = licence.splitlines()[0][:80] if licence else "see licence files"
        files = [f for f in info.rglob("*") if f.is_file() and re.match(
            r"(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", f.name, re.I)]
        target = dest / name
        for f in files:
            target.mkdir(exist_ok=True)
            shutil.copy2(f, target / f"{f.parent.name}-{f.name}" if f.parent != info else target / f.name)
        index.append(f"- {name} {meta.get('Version', '')}: {licence}"
                     + ("" if files else " (no licence file shipped by the package)"))
    for extra in sorted(site.glob("cv2/LICENSE*")):
        (dest / "opencv-python").mkdir(exist_ok=True)
        shutil.copy2(extra, dest / "opencv-python" / extra.name)
    if (pylib / "LICENSE.txt").is_file():
        (dest / "Python").mkdir(exist_ok=True)
        shutil.copy2(pylib / "LICENSE.txt", dest / "Python" / "LICENSE.txt")
    index += ["", f"- Python {PY} (PSF licence, Python/LICENSE.txt), built by python-build-standalone,",
              "  which links statically: " + ", ".join(PYTHON_STATIC) + "."]
    for pkg in ("cv2", "PIL", "numpy"):
        libs = sorted(p.name for p in site.glob(f"{pkg}/.dylibs/*"))
        if libs:
            index += ["", f"- Libraries bundled inside the {pkg} wheel (their licences are"
                      " listed in that package's licence files): " + ", ".join(libs) + "."]
    gpl = sorted({p.name for p in site.glob("cv2/.dylibs/*")
                  if any(p.name.startswith(g) for g in GPL_DYLIBS)})
    if gpl:
        index += ["", "- OpenCV's wheel bundles FFmpeg built with GPL components: " + ", ".join(gpl)
                  + ". They are covered by the GNU GPL v3; see opencv-python/LICENSE-3RD-PARTY.txt."]
    (dest / "INDEX.txt").write_text("\n".join(index) + "\n")
    return gpl


def build(out: Path, ref: str = "HEAD", allow_gpl: bool = False) -> Path:
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

    print(f"caliPr, as committed at {ref}")
    code = res / "app"
    with tempfile.TemporaryDirectory() as tmp:
        tree = source_tree(ref, Path(tmp))
        for part in ("scripts", "src", "docs"):
            shutil.copytree(tree / part, code / part,
                            ignore=lambda d, names: [n for n in names if n in CODE_SKIP
                                                     or n.endswith(".egg-info")])
        (code / "packaging" / "macos").mkdir(parents=True)
        shutil.copy2(tree / "packaging" / "macos" / "calipr_app.py",
                     code / "packaging" / "macos" / "calipr_app.py")
        for f in ("LICENSE", "README.md", "pyproject.toml"):
            shutil.copy2(tree / f, code / f)

    print("Third-party notices")
    gpl = write_notices(site, lib, res / "THIRD_PARTY_NOTICES")
    if gpl and not allow_gpl:
        raise SystemExit(
            "OpenCV's wheel bundles FFmpeg built under the GPL: " + ", ".join(gpl)
            + ".\ncaliPr uses no video, so build OpenCV without FFmpeg and pin that wheel in"
            " requirements.txt (see the module docstring). --allow-gpl builds anyway, for"
            " local testing only.")
    if gpl:
        print("  ! GPL-licensed libraries bundled with OpenCV: " + ", ".join(gpl))
        print("    Local testing only: do not publish this build.")

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
        "NSHumanReadableCopyright": "caliPr: MIT License. Bundled components keep their own "
                                    "licences (Resources/THIRD_PARTY_NOTICES).",
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
    ap.add_argument("--ref", default="HEAD",
                    help="the commit or tag whose code goes into the app (default HEAD)")
    ap.add_argument("--allow-gpl", action="store_true",
                    help="build even if OpenCV bundles GPL FFmpeg (local testing only)")
    args = ap.parse_args(argv)
    if sys.platform != "darwin":
        raise SystemExit("the Mac app is built on a Mac")
    app = build(args.out, args.ref, args.allow_gpl)
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
