"""Generate the labeling-UI reference example from a labeled specimen.

The labeler shows an annotated "example fish" beside the canvas so annotators
can see where each landmark belongs. This builds those assets from a real
hand-labeled sidecar, so the reference reflects the lab's actual labeling
convention rather than an approximation.

Writes into ``scripts/labeling_ui/``:
  reference_base.jpg   crop with no annotations, used for the ZOOM view. Its
                       resolution sets how sharp that zoom can be: the panel is
                       ~430 px wide and a dorsal fin is ~625 px in the source, so
                       anything under ~0.7x native has to upscale and turns to
                       mush exactly where the annotator is looking hardest.
  reference_annot.jpg  same crop with polygons + keypoints drawn (overview)
  reference.json       landmark coordinates in that crop's pixel space

Usage::

    python scripts/make_reference.py --specimen Salvelinus_fontinalis_HRN_5

Rebuild this whenever the labeling convention changes. A reference built from a
sidecar that has since been re-labelled teaches the OLD convention, which is
worse than having no reference at all -- it looks authoritative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = _ROOT / "scripts" / "labeling_ui"

POLY_COLOR = (255, 170, 50)   # BGR — matches the UI's polygon blue
KP_COLOR = (60, 80, 255)      # BGR — matches the UI's keypoint red


def build(specimen: str, sidecars: Path, images: Path, pad: int = 140,
          target_w: int = 3600, out_dir: Path = UI_DIR, image_path: Path | None = None,
          skip_keypoints=(), skip_polygons=()) -> dict:
    """Write the three reference files for ``specimen`` into ``out_dir``.

    ``image_path`` names the photograph when it is not ``<specimen>_L.JPEG`` in
    ``images`` -- a study's own example can be any labelled fish. Landmarks and
    outlines the study does not collect (``skip_*``) are left off, so the example
    shows only what the annotator is asked for. Raises ValueError if the fish has
    no landmarks to show.
    """
    sc_path = sidecars / f"{specimen}.json"
    sidecar = json.loads(sc_path.read_text())
    lateral = sidecar.get("lateral") or {}
    polygons = {k: v for k, v in (lateral.get("polygons") or {}).items()
                if v and len(v) >= 3 and k not in set(skip_polygons)}
    keypoints = {k: v for k, v in (lateral.get("keypoints") or {}).items()
                 if v and k not in set(skip_keypoints)}
    if not keypoints:
        raise ValueError(f"{specimen} has no landmarks saved on its side view")

    img_path = image_path or images / f"{specimen}_L.JPEG"
    im = cv2.imread(str(img_path))
    if im is None:
        raise ValueError(f"could not read {img_path}")
    H, W = im.shape[:2]

    # Crop tightly around everything the annotator needs to see.
    pts = [p for verts in polygons.values() for p in verts] + list(keypoints.values())
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = max(0, int(min(xs)) - pad)
    x1 = min(W, int(max(xs)) + pad)
    y0 = max(0, int(min(ys)) - pad)
    y1 = min(H, int(max(ys)) + pad)

    # Never enlarged past the photograph's own resolution: that adds no detail.
    scale = min(target_w, x1 - x0) / (x1 - x0)
    def T(p):
        return [round((p[0] - x0) * scale, 1), round((p[1] - y0) * scale, 1)]

    base = cv2.resize(im[y0:y1, x0:x1], None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)
    bh, bw = base.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "reference_base.jpg"), base,
                [cv2.IMWRITE_JPEG_QUALITY, 86])

    annot = base.copy()
    for verts in polygons.values():
        cv2.polylines(annot, [np.array([T(v) for v in verts], np.int32)],
                      True, POLY_COLOR, 2, cv2.LINE_AA)
    for xy in keypoints.values():
        x, y = (int(v) for v in T(xy))
        cv2.circle(annot, (x, y), 5, KP_COLOR, -1, cv2.LINE_AA)
        cv2.circle(annot, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "reference_annot.jpg"), annot,
                [cv2.IMWRITE_JPEG_QUALITY, 86])

    ref = {
        "w": bw, "h": bh,
        "specimen": specimen,
        "keypoints": {k: T(v) for k, v in keypoints.items()},
        "polygons": {k: [T(v) for v in verts] for k, verts in polygons.items()},
        "note": f"Hand-labeled reference: {specimen}",
    }
    (out_dir / "reference.json").write_text(json.dumps(ref, indent=1))

    print(f"reference built from {specimen}: {bw}x{bh}, "
          f"{len(keypoints)} keypoints, {len(polygons)} polygons")
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_reference")
    ap.add_argument("--specimen", required=True,
                    help="Sidecar stem, e.g. Salvelinus_fontinalis_HRN_5")
    ap.add_argument("--sidecars", type=Path,
                    default=_ROOT / "data" / "cornell" / "sidecars")
    ap.add_argument("--images", type=Path,
                    default=_ROOT / "data" / "cornell" / "lateral")
    ap.add_argument("--image", type=Path, default=None,
                    help="The photograph itself, when it is not <specimen>_L.JPEG.")
    ap.add_argument("--out", type=Path, default=UI_DIR,
                    help="Where to write the three files (default: the labeler's own).")
    ap.add_argument("--skip-keypoints", default="",
                    help="Comma-separated landmarks the study does not collect.")
    ap.add_argument("--skip-polygons", default="",
                    help="Comma-separated outlines the study does not collect.")
    args = ap.parse_args(argv)
    split = lambda v: tuple(x for x in v.split(",") if x)
    try:
        build(args.specimen, args.sidecars, args.images, out_dir=args.out,
              image_path=args.image, skip_keypoints=split(args.skip_keypoints),
              skip_polygons=split(args.skip_polygons))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
