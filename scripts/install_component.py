#!/usr/bin/env python3
"""Install an optional part of caliPr that is a set of data rather than a program.

    python scripts/install_component.py batch
    python scripts/install_component.py batch --from ~/Downloads/shiners-reference.zip

Batch identification compares the fish of a photograph with a reference set:
identified specimens, hand-landmarked, with the credits of the photographs they
were landmarked on. The set arrives as one study folder -- schema.json marking it
``"reference_set": true``, its sidecars, its photo credits, and an empty
``lateral/`` so the labeler lists it as a study -- and is unpacked into the data
folder beside the studies already there.

``scripts/components.json`` says where it is published and its SHA-256, so a
truncated or altered download is refused rather than used. A folder of the same
name already in the data folder is never replaced. Batch identification also needs
Auto-label (``install_autolabel.py``); the two can be installed in either order, and
the feature works once both are in place.

Standard library only, so it runs in the labeler's own environment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from fish_morpho import components, paths  # noqa: E402
from fetch_model import _download, _sha256  # noqa: E402

#: The parts this installs. Auto-label is a program and has its own installer.
DATA_PARTS = ("batch",)


def unpack(archive: Path, folder: str, data: Path) -> Path:
    """The study folder ``folder`` from ``archive``, placed in ``data``.

    Every member must lie inside that one folder: an archive reaching anywhere else
    (an absolute path, a ``..``) is refused whole, before anything is written.
    """
    dest = data / folder
    if dest.exists():
        raise SystemExit(f"{dest} already exists — left as it is. Remove it to install again.")
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            parts = PurePosixPath(name).parts
            if not parts or parts[0] != folder or ".." in parts or PurePosixPath(name).is_absolute():
                raise SystemExit(f"REFUSED — {archive.name} holds {name!r}, outside {folder}/. "
                                 "Nothing installed.")
        with tempfile.TemporaryDirectory(dir=data) as tmp:
            z.extractall(tmp)
            schema = Path(tmp) / folder / "schema.json"
            try:
                ok = json.loads(schema.read_text()).get("reference_set") is True
            except (OSError, ValueError):
                ok = False
            if not ok:
                raise SystemExit(f"REFUSED — {archive.name} is not a reference set "
                                 "(its schema.json does not say so). Nothing installed.")
            (Path(tmp) / folder / "lateral").mkdir(exist_ok=True)
            shutil.move(str(Path(tmp) / folder), str(dest))
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="install_component", description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=DATA_PARTS)
    ap.add_argument("--data", type=Path, default=paths.WORK / "data",
                    help="The data folder the studies are in.")
    ap.add_argument("--from", dest="source", type=Path,
                    help="An archive already on this machine, instead of downloading it.")
    ap.add_argument("--manifest", type=Path, default=components.MANIFEST, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    entry = components.manifest(args.manifest).get(args.part) or {}
    title = entry.get("title", args.part)
    if not entry.get("folder"):
        print(f"{title} is not available to download yet.", flush=True)
        return 2
    if not args.source and not (entry.get("released") and entry.get("url")):
        print(f"{title} is not available to download yet.", flush=True)
        return 2
    args.data.mkdir(parents=True, exist_ok=True)
    if (args.data / entry["folder"]).exists():
        print(f"{title}: {entry['folder']} is already in {args.data} — left as it is.", flush=True)
        return 0

    with tempfile.TemporaryDirectory(dir=args.data) as tmp:
        archive = Path(tmp) / (entry.get("file") or "part.zip")
        if args.source:
            shutil.copyfile(args.source, archive)
        else:
            print(f"{title}: downloading {entry['file']} ({(entry.get('bytes') or 0) / 1e6:.0f} MB)",
                  flush=True)
            _download(entry["url"], archive, entry.get("bytes") or 0)
        if entry.get("sha256"):
            digest = _sha256(archive)
            if digest != entry["sha256"]:
                print(f"{title}: REFUSED — the download does not match its recorded checksum "
                      f"(got {digest[:12]}…, expected {entry['sha256'][:12]}…). Nothing installed.",
                      flush=True)
                return 1
        dest = unpack(archive, entry["folder"], args.data)
    print(f"{title}: installed {dest}", flush=True)
    if not paths.TRAINING_PYTHON.is_file():
        print("It needs Auto-label too, which is not installed yet.", flush=True)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
