"""Draw each specimen's landmarks and outlines onto its photograph.

The numbers are checkable by anyone; the placements are not, unless someone can
see them. These renders are what makes an annotation reviewable — by a second
labeller, by a supervisor signing off on a dataset, or by whoever reads the paper
and wants to know what "standard length" meant in practice.

Downscaled and cropped to the annotation, so a folder of them can be flipped
through rather than opened one at a time.

Usage::

    python scripts/render_overlays.py --dataset alewife
    python scripts/render_overlays.py --dataset cornell --out /tmp/check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

POLY_COLOR = (255, 170, 60)     # BGR, blue — outlines
KP_COLOR = (80, 90, 255)        # BGR, red  — keypoints
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPEG", ".JPG")


def find_image(images: Path, fish_id: str) -> Path | None:
    for suf in IMAGE_SUFFIXES:
        for cand in (images / f"{fish_id}{suf}", images / f"{fish_id}_L{suf}"):
            if cand.is_file():
                return cand
    return None


def render(sidecar: dict, image: Path, width: int = 1400,
           label: bool = True) -> np.ndarray | None:
    lat = sidecar.get("lateral") or {}
    polys = lat.get("polygons") or {}
    kps = lat.get("keypoints") or {}
    if not polys and not kps:
        return None
    im = cv2.imread(str(image))
    if im is None:
        return None
    H, W = im.shape[:2]

    pts = [p for v in polys.values() for p in v] + list(kps.values())
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = int(0.06 * max(max(xs) - min(xs), max(ys) - min(ys))) + 30
    x0, y0 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
    x1, y1 = min(W, int(max(xs)) + pad), min(H, int(max(ys)) + pad)

    lw = max(2, int(W / 1400))
    for verts in polys.values():
        if len(verts) >= 3:
            cv2.polylines(im, [np.array(verts, np.int32)], True, POLY_COLOR,
                          lw, cv2.LINE_AA)
    r = max(3, int(W / 900))
    for p in kps.values():
        cv2.circle(im, (int(p[0]), int(p[1])), r, KP_COLOR, -1, cv2.LINE_AA)
        cv2.circle(im, (int(p[0]), int(p[1])), r, (20, 20, 20), 1, cv2.LINE_AA)

    crop = im[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    k = width / crop.shape[1]
    out = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)

    if label:
        bar = np.full((34, out.shape[1], 3), 22, np.uint8)
        txt = (f"{sidecar.get('fish_id', image.stem)}   "
               f"{len(kps)} landmarks, {len(polys)} outlines")
        cv2.putText(bar, txt, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (225, 225, 225), 1, cv2.LINE_AA)
        out = np.vstack([bar, out])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="render_overlays")
    ap.add_argument("--dataset")
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    ap.add_argument("--images", type=Path, default=None)
    ap.add_argument("--sidecars", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--width", type=int, default=1400)
    args = ap.parse_args(argv)

    if args.images and args.sidecars:
        images, sidecars, name = args.images, args.sidecars, args.dataset or "custom"
    else:
        if not args.dataset:
            ap.error("--dataset is required (or pass --images and --sidecars)")
        base = args.data_root / args.dataset
        images, sidecars, name = base / "lateral", base / "sidecars", args.dataset

    out = args.out or (_ROOT / "results" / name / "overlays")
    out.mkdir(parents=True, exist_ok=True)

    made = skipped = 0
    for path in sorted(sidecars.glob("*.json")):
        sc = json.loads(path.read_text())
        img = find_image(images, sc.get("fish_id", path.stem))
        if img is None:
            skipped += 1
            continue
        canvas = render(sc, img, width=args.width)
        if canvas is None:
            skipped += 1
            continue
        cv2.imwrite(str(out / f"{path.stem}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        made += 1

    print(f"wrote {made} overlay(s) to {out}")
    if skipped:
        print(f"  skipped {skipped} (no image, or nothing annotated yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
