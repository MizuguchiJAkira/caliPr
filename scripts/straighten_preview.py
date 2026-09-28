"""Draw a fish as its midline straightens it, to check before measuring.

MorFishJ keeps the straightened image it measured on (``<photo>_straightened.jpg``)
so the straightening can be looked at afterwards. caliPr measures the landmarks
in the straightened frame without making that image, so this makes it on request:
the photograph resampled across the midline, as ImageJ's Straighten does, with the
landmarks and outlines drawn where the measurements put them.

    python scripts/straighten_preview.py --image photo.jpg --labels labels.json --out s.jpg

``labels.json`` is a sidecar's lateral block, or any JSON with ``midline`` and
optionally ``keypoints`` and ``polygons``. The labeler runs this for its
Straightened view, on the midline as it stands, saved or not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho import straighten  # noqa: E402


def render(image_path: Path, block: dict, max_width: int = 2400):
    import cv2
    import numpy as np

    nodes = straighten.midline_of(block)
    if not nodes:
        raise ValueError("draw a midline of two points or more first")
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read {image_path.name}")
    kps = {n: p for n, p in (block.get("keypoints") or {}).items() if p}
    polys = {n: v for n, v in (block.get("polygons") or {}).items() if v and len(v) >= 2}
    snout = next((tuple(kps[n]) for n in straighten.SNOUT_LANDMARKS if n in kps), None)
    ml = straighten.Midline.fit(nodes).oriented(snout)
    every = list(kps.values()) + [q for v in polys.values() for q in v]
    out, place = straighten.straighten_image(img, ml, every, max_width=max_width)
    h, w = out.shape[:2]
    lw = max(1, round(w / 900))
    # the midline, now straight
    (x0, y0), (x1, _) = place([ml.curve[0], ml.curve[-1]])
    cv2.line(out, (int(x0), int(y0)), (int(x1), int(y0)), (214, 85, 138), lw, cv2.LINE_AA)
    for name, verts in polys.items():
        pts = np.array(place(verts), dtype=np.int32)
        cv2.polylines(out, [pts], True, (58, 162, 76), lw, cv2.LINE_AA)
    names = list(kps)
    folded = {names[i] for i in ml.folded(list(kps.values()))} if kps else set()
    for (name, p), (x, y) in zip(kps.items(), place(list(kps.values()))):
        colour = (40, 170, 250) if name in folded else (63, 50, 201)
        cv2.circle(out, (int(x), int(y)), 3 * lw + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 3 * lw, colour, -1, cv2.LINE_AA)
    info = {"length_px": round(ml.length, 1), "turn_deg": round(ml.turn_deg, 1),
            "folded": sorted(folded), "width": w, "height": h}
    return out, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="straighten_preview")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True,
                    help="JSON with midline, keypoints, polygons (a sidecar's lateral block)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--info", type=Path, default=None, help="also write what was done, as JSON")
    ap.add_argument("--max-width", type=int, default=2400)
    args = ap.parse_args(argv)
    import cv2
    block = json.loads(args.labels.read_text())
    block = block.get("lateral", block)
    try:
        out, info = render(args.image, block, args.max_width)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    cv2.imwrite(str(args.out), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if args.info:
        args.info.write_text(json.dumps(info))
    print(f"wrote {args.out} ({info['width']}x{info['height']}), midline "
          f"{info['length_px']:.0f} px turning {info['turn_deg']:.0f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
