#!/usr/bin/env python3
"""Package a study as the reference set batch identification downloads.

    python scripts/package_component.py batch --study data/shiners --release components-2026-11 \
        --licence "CC BY-NC 4.0"

Writes ``dist/components/<name>.zip`` -- the study's schema.json (marked as a
reference set), its hand-landmarked sidecars, its photo credits
(``photo_manifest.csv``) and README, the characters read from its photographs
(``photo_characters.json``; run ``scripts/photo_characters.py`` first), a
LICENCE.md stating the licence the set is released under, and an empty
``lateral/`` -- and records the file, its address under the release, its size,
its SHA-256 and its licence in
``scripts/components.json``. No photographs are packaged: the landmarks are what
the identification compares, and the photographs remain the collections' to publish.

The entry is left unreleased. Publishing is a separate decision: attach the zip to
the release, then set ``"released": true``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho import components  # noqa: E402

#: What travels with the landmarks: credits, how to read them, and the characters
#: read from the photographs (which do not travel).
EXTRAS = ("photo_manifest.csv", "README.md", "photo_characters.json")
#: schema.json entries that steer one person's labelling, not anyone's comparison.
LOCAL_ONLY = ("label_first", "hand_only", "hand_only_why", "batch_of", "reference_study")


def release_base() -> str:
    """The repository's release download address, from where the models are published."""
    models = json.loads((_ROOT / "scripts" / "models.json").read_text())
    url = next(iter(models["models"].values()))["url"]
    return url.split("/releases/download/")[0] + "/releases/download"


def licence_note(licence: str, credits: bool) -> str:
    """LICENCE.md for the set: what it is released under, and where the photographs are credited."""
    out = (f"# Licence\n\nThe landmarks and photo characters in this reference set are released under "
           f"{licence}. The photographs they were read from are not included.")
    if credits:
        out += " Each photograph's source, creator and licence is listed in photo_manifest.csv."
    return out + "\n"


def package(study: Path, name: str, out: Path, licence: str) -> Path:
    schema = json.loads((study / "schema.json").read_text())
    for k in LOCAL_ONLY:
        schema.pop(k, None)
    schema["reference_set"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}/schema.json", json.dumps(schema, indent=1) + "\n")
        z.writestr(f"{name}/lateral/", "")
        for f in sorted((study / "sidecars").glob("*.json")):
            d = json.loads(f.read_text())
            if not ((d.get("lateral") or {}).get("keypoints")):
                continue                       # a note with no landmarks compares nothing
            z.write(f, f"{name}/sidecars/{f.name}")
            kept += 1
        for extra in EXTRAS:
            if (study / extra).is_file():
                z.write(study / extra, f"{name}/{extra}")
        z.writestr(f"{name}/LICENCE.md", licence_note(licence, (study / "photo_manifest.csv").is_file()))
    print(f"{out}: {kept} landmarked specimens")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="package_component", description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=("batch",))
    ap.add_argument("--study", type=Path, required=True, help="The study to package.")
    ap.add_argument("--name", help="The folder it unpacks to (the study's name if not given).")
    ap.add_argument("--release", required=True, help="The GitHub release tag it will be attached to.")
    ap.add_argument("--licence", required=True,
                    help='What the set is released under, e.g. "CC BY-NC 4.0" when any of its '
                         "photographs are licensed for non-commercial use.")
    ap.add_argument("--out", type=Path, default=_ROOT / "dist" / "components")
    args = ap.parse_args(argv)

    name = args.name or args.study.name
    zpath = package(args.study, name, args.out / f"{name}-reference.zip", args.licence)
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    man = components.manifest()
    entry = man.setdefault(args.part, {})
    entry.update({"file": zpath.name, "url": f"{release_base()}/{args.release}/{zpath.name}",
                  "sha256": digest, "bytes": zpath.stat().st_size, "folder": name,
                  "licence": args.licence, "released": False})
    components.MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
    print(f"recorded in {components.MANIFEST.relative_to(_ROOT)} (not released)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
