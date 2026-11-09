#!/usr/bin/env python3
"""Read the identification characters from a study's photographs.

    python scripts/photo_characters.py --study data/shiners

For every hand-labelled fish with its photograph, ``fish_morpho.photo_characters``
-- how dark the caudal spot, the snout and the side stripe are, and how upright the
scale margins are -- read at the places its landmarks set, into
``<study>/photo_characters.json``, with a fingerprint of those landmarks. A fish
whose landmarks have not changed since is not read again; one whose have is, and
one no longer labelled or photographed is dropped. Batch identification compares
a fish with the reference species on these beside shape, and a reference set
downloaded without its photographs carries the file instead.

The labeler runs this itself before scoring a batch photograph, when any sidecar
is newer than the file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import cv2  # noqa: E402

from fish_morpho import photo_characters  # noqa: E402
from fish_morpho.species_match import CHARACTERS_FILE  # noqa: E402

PHOTO_TYPES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def refresh(study: Path) -> dict:
    out_path = study / CHARACTERS_FILE
    try:
        had = json.loads(out_path.read_text()).get("fish") or {}
    except (OSError, ValueError):
        had = {}
    photos = {p.stem: p for p in (study / "lateral").iterdir()
              if p.suffix.lower() in PHOTO_TYPES} if (study / "lateral").is_dir() else {}
    fish, read, kept = {}, 0, 0
    for f in sorted((study / "sidecars").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        fid = d.get("fish_id") or f.stem
        kp = {k: tuple(v[:2]) for k, v in ((d.get("lateral") or {}).get("keypoints") or {}).items() if v}
        photo = photos.get(fid) or photos.get(f"{fid}_L")
        if not kp or photo is None:
            continue
        key = photo_characters.landmarks_key(kp)
        if (had.get(fid) or {}).get("landmarks") == key:
            fish[fid] = had[fid]
            kept += 1
            continue
        img = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        if img is None:
            continue
        fish[fid] = {"landmarks": key, "characters": photo_characters.characters(img, kp)}
        read += 1
    doc = {"about": "Characters read from each fish's photograph where its landmarks put them "
                    "(fish_morpho.photo_characters); 'landmarks' fingerprints the landmarks used.",
           "fish": fish}
    with tempfile.NamedTemporaryFile("w", dir=study, suffix=".tmp", delete=False) as tmp:
        json.dump(doc, tmp, indent=1)
    os.replace(tmp.name, out_path)
    return {"read": read, "kept": kept, "fish": len(fish)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="photo_characters", description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", type=Path, required=True)
    args = ap.parse_args(argv)
    r = refresh(args.study)
    print(f"{args.study / CHARACTERS_FILE}: {r['fish']} fish ({r['read']} read, {r['kept']} unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
