#!/usr/bin/env python3
"""Give a scale to every head-on view that has mouth corners but no ruler.

    python scripts/scale_frontals.py --dataset cornell --dry-run
    python scripts/scale_frontals.py --dataset cornell

Mouth width is measured in the mirror, which sits at a different distance from
the camera than the fish, so it needs its own scale -- and a borrowed one will
not do: on the brook trout rig the mirror was moved between sessions, reading
about 19.6 px/mm for ASN_25-35 and 22.6 elsewhere.

The mirror carries a small vertical ruler. The tick detector scans for a
horizontal one, so the head-on view is turned a quarter turn before detection.
Checked against the 43 frontal views calibrated by hand: found on 42, with a
median disagreement of 2.7% -- and following the mirror between sessions.

It reads a consistent ~2.6% lower than those hand clicks. Which of the two is
nearer the truth cannot be settled from these photographs, but mixing them in
one column would leave a method offset between fish -- and the split is uneven
by strain (HRN 5 hand-scaled to 34 detected, TXD 22 to 27), so the offset would
read as a strain difference. So the detector is put on the hand clicks' basis by
their median ratio over the fish that have both, and every value this writes
says so in its notes.

Only fish with mouth corners and no frontal calibration are touched. A hand
calibration is never replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import preprocess_cornell as pc  # noqa: E402

from fish_morpho.ruler_calibration import detect_tick_scale  # noqa: E402

#: A detected mirror scale outside this is refused. The hand calibrations span
#: 17.9-23.6 px/mm; the imperial ticks the detector must not lock onto would
#: read 1.59x coarser, far outside it.
PLAUSIBLE = (15.0, 27.0)


def mirror_scale(image_path: Path) -> float | None:
    """px/mm of the mirror's ruler, or None if it cannot be found."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    box = pc.view_frames(img)["frontal"]
    if box is None:
        return None
    x0, y0, x1, y1 = box
    turned = cv2.rotate(img[y0:y1, x0:x1], cv2.ROTATE_90_CLOCKWISE)
    try:
        ppm = detect_tick_scale(turned, x_frac=(0.02, 0.98)).px_per_mm
    except Exception:
        return None
    return ppm if PLAUSIBLE[0] <= ppm <= PLAUSIBLE[1] else None


def hand_scale(cal: dict) -> float | None:
    if cal.get("mode") == "manual" and cal.get("known_mm"):
        return math.dist(cal["point_a"], cal["point_b"]) / float(cal["known_mm"])
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scale_frontals")
    ap.add_argument("--dataset", default="cornell")
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    study = args.data_root / args.dataset

    docs = {}
    for p in sorted((study / "sidecars").glob("*.json")):
        d = json.loads(p.read_text())
        if len(((d.get("frontal") or {}).get("keypoints")) or {}) >= 2:
            docs[p] = d

    # The basis: detector against hand, on every fish that has both.
    ratios = []
    for p, d in docs.items():
        h = hand_scale((d["frontal"].get("calibration")) or {})
        if h is None:
            continue
        m = mirror_scale(study / "lateral" / f"{d['fish_id']}_L.JPEG")
        if m:
            ratios.append(h / m)
    if len(ratios) < 5:
        print("too few hand-calibrated frontals to put the detector on their basis")
        return 1
    k = st.median(ratios)
    print(f"detector vs {len(ratios)} hand calibrations: hand/detected median {k:.4f} "
          f"(detector reads {100 * (1 - 1 / k):.1f}% low)")

    placed = failed = 0
    for p, d in docs.items():
        fr = d["frontal"]
        if (fr.get("calibration") or {}).get("mode") not in (None, "none"):
            continue
        m = mirror_scale(study / "lateral" / f"{d['fish_id']}_L.JPEG")
        name = d["fish_id"].split("_", 2)[-1]
        if m is None:
            failed += 1
            print(f"  {name:9s} mirror ruler not found — place the frontal ruler by hand")
            continue
        ppm = m * k
        placed += 1
        if args.dry_run:
            continue
        fr["calibration"] = {
            "mode": "ticks", "px_per_mm": round(ppm, 4), "placed_by": "ruler-ticks",
            "notes": (f"mirror ruler detected on the head-on view turned a quarter turn "
                      f"({m:.2f} px/mm), x{k:.4f} to put it on the basis of the "
                      f"{len(ratios)} hand-clicked frontal calibrations")}
        p.write_text(json.dumps(d, indent=2))
    verb = "would scale" if args.dry_run else "scaled"
    print(f"\n{verb} {placed} head-on views · {failed} where the mirror ruler was not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
