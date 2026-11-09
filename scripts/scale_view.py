#!/usr/bin/env python3
"""The labeler's fish-scale tools, run apart from the server.

    scale_view.py enhance --image P --out O.jpg [--mode local] [--strength 4]
                          [--channel luminance] [--scale-size PX]
    scale_view.py count --image P --path '[[x, y], ...]' [--sl PX] [--axis '[[x, y], [x, y]]']

``enhance`` writes the photograph with its scale margins brought up, grey, at most
``--max-side`` px on its long side (the labeler draws it over the photograph's own
rectangle, so a point placed on it is placed on the photograph). ``count`` prints
``fish_scales.count_along`` as JSON. A subprocess, as the labeler's other image
operations are: a 6000 px photograph decoded and filtered is a few hundred MB the
server does not keep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import cv2  # noqa: E402

from fish_morpho import fish_scales  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scale_view", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enhance")
    e.add_argument("--image", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--mode", choices=fish_scales.MODES, default="local")
    e.add_argument("--strength", type=float, default=4.0)
    e.add_argument("--channel", choices=fish_scales.CHANNELS, default="luminance")
    e.add_argument("--scale-size", type=float, default=0.0, help="One scale, px on the photograph.")
    e.add_argument("--max-side", type=int, default=4000)
    c = sub.add_parser("count")
    c.add_argument("--image", type=Path, required=True)
    c.add_argument("--path", required=True, help="JSON [[x, y], ...] in photograph pixels.")
    c.add_argument("--sl", type=float, default=0.0, help="Standard length, px, to bound the spacing.")
    c.add_argument("--axis", default="", help="JSON [snout, end of the vertebral column], to place the row on the body.")
    args = ap.parse_args(argv)

    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 1
    if args.cmd == "count":
        path = json.loads(args.path)
        axis = json.loads(args.axis) if args.axis else None
        print(json.dumps(fish_scales.count_along(img, path, sl_px=args.sl or None, axis=axis)))
        return 0
    f = min(1.0, args.max_side / max(img.shape[:2]))
    if f < 1:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    out = fish_scales.enhance(img, mode=args.mode, strength=args.strength,
                              scale_size_px=(args.scale_size * f) or None, channel=args.channel)
    cv2.imwrite(str(args.out), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
