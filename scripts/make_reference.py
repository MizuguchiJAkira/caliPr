"""Generate the labeling-UI reference example from a labeled specimen.

The labeler shows an annotated "example fish" beside the canvas so annotators
can see where each landmark belongs. This builds those assets from a real
hand-labeled sidecar, so the reference reflects the lab's actual labeling
convention rather than an approximation.

Writes into ``scripts/labeling_ui/``:
  reference_base.jpg   downscaled crop, no annotations (used for the zoom view)
  reference_annot.jpg  same crop with polygons + keypoints drawn (overview)
  reference.json       landmark coordinates in that crop's pixel space

Usage::

    python scripts/make_reference.py --specimen Salvelinus_fontinalis_HRN_5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = _ROOT / "scripts" / "labeling_ui"

POLY_COLOR = (255, 170, 50)   # BGR — matches the UI's polygon blue
KP_COLOR = (60, 80, 255)      # BGR — matches the UI's keypoint red


def build(specimen: str, sidecars: Path, images: Path, pad: int = 140,
          target_w: int = 1540) -> None:
    sc_path = sidecars / f"{specimen}.json"
    sidecar = json.loads(sc_path.read_text())
    lateral = sidecar.get("lateral") or {}
    polygons = lateral.get("polygons") or {}
    keypoints = lateral.get("keypoints") or {}
    if not polygons or not keypoints:
        raise SystemExit(f"{specimen}: sidecar has no lateral polygons/keypoints")

    img_path = images / f"{specimen}_L.JPEG"
    im = cv2.imread(str(img_path))
    if im is None:
        raise SystemExit(f"Could not read {img_path}")
    H, W = im.shape[:2]

    # Crop tightly around everything the annotator needs to see.
    pts = [p for verts in polygons.values() for p in verts] + list(keypoints.values())
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = max(0, int(min(xs)) - pad)
    x1 = min(W, int(max(xs)) + pad)
    y0 = max(0, int(min(ys)) - pad)
    y1 = min(H, int(max(ys)) + pad)

    scale = target_w / (x1 - x0)
    def T(p):
        return [round((p[0] - x0) * scale, 1), round((p[1] - y0) * scale, 1)]

    base = cv2.resize(im[y0:y1, x0:x1], None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)
    bh, bw = base.shape[:2]
    UI_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(UI_DIR / "reference_base.jpg"), base,
                [cv2.IMWRITE_JPEG_QUALITY, 86])

    annot = base.copy()
    for verts in polygons.values():
        cv2.polylines(annot, [np.array([T(v) for v in verts], np.int32)],
                      True, POLY_COLOR, 2, cv2.LINE_AA)
    for xy in keypoints.values():
        x, y = (int(v) for v in T(xy))
        cv2.circle(annot, (x, y), 5, KP_COLOR, -1, cv2.LINE_AA)
        cv2.circle(annot, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(UI_DIR / "reference_annot.jpg"), annot,
                [cv2.IMWRITE_JPEG_QUALITY, 86])

    (UI_DIR / "reference.json").write_text(json.dumps({
        "w": bw, "h": bh,
        "specimen": specimen,
        "keypoints": {k: T(v) for k, v in keypoints.items()},
        "polygons": {k: [T(v) for v in verts] for k, verts in polygons.items()},
        "note": f"Hand-labeled reference: {specimen}",
    }, indent=1))

    print(f"reference built from {specimen}: {bw}x{bh}, "
          f"{len(keypoints)} keypoints, {len(polygons)} polygons")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_reference")
    ap.add_argument("--specimen", required=True,
                    help="Sidecar stem, e.g. Salvelinus_fontinalis_HRN_5")
    ap.add_argument("--sidecars", type=Path,
                    default=_ROOT / "data" / "cornell" / "sidecars")
    ap.add_argument("--images", type=Path,
                    default=_ROOT / "data" / "cornell" / "lateral")
    args = ap.parse_args(argv)
    build(args.specimen, args.sidecars, args.images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
