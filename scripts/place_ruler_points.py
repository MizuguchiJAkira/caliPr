#!/usr/bin/env python3
"""Put the two ruler points on every fish that has no scale yet.

    python scripts/place_ruler_points.py --dataset cornell --dry-run
    python scripts/place_ruler_points.py --dataset cornell

The same thing the labeler's button does, over a whole study. The ruler detector
locates every millimetre tick; the two ends of that run become ``ruler_point_a``
and ``ruler_point_b``, and the span between them is an exact whole number of
millimetres, so the scale is derived from two points that can be seen and dragged
rather than from a number nothing downstream can check.

Unlike the landmark scripts here, **this one writes to sidecars.** A calibration
is not a suggestion to review -- it is either right or it is visibly wrong on the
ruler -- so it is written, marked ``placed_by: ruler-ticks`` so it never reads as
hand-placed, and restricted three ways:

* only fish whose calibration is absent or ``none``. An existing scale, hand or
  tick, is never touched.
* only where the detector's own millimetres land on the ruler's own ticks. Below
  that, the scale is wrong and writing it would be worse than leaving none --
  those fish are named so the two points can be placed by hand.
* the ends of the tick run, not a short span. This rig's millimetres are a
  median 5% wider at one end than the other, so a long span is the best single
  number available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

#: Fraction of a ruler's millimetre positions that must land within a quarter of
#: a millimetre of a real tick before the detected scale is believed. Measured
#: over 18 of these photographs: a working ruler sits at 40-100%, and the two
#: that are wrong sit at 1%.
MIN_ON_TICK = 0.30


def short(fid: str) -> str:
    """The part of a fish id a person uses: the strain and number."""
    m = re.search(r"([A-Z]{2,4}_\d+)$", fid)
    return m.group(1) if m else fid[:12]


def get(base: str, path: str, timeout: int = 180) -> dict:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"error": f"HTTP {e.code}"}
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e)}


def points_from(ticks: dict) -> tuple[list[int], list[int], int] | None:
    """The two ends of the tick run, and the millimetres between them."""
    tk = ticks.get("ticks") or []
    span = ticks.get("span_mm")
    if len(tk) < 2 or not span:
        return None
    # x + offset is where the real tick is; x alone is where this scale put it.
    a = [int(round(tk[0][0] + tk[0][2])), int(round(tk[0][1]))]
    b = [int(round(tk[-1][0] + tk[-1][2])), int(round(tk[-1][1]))]
    return a, b, int(span)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="place_ruler_points")
    ap.add_argument("--dataset", default="cornell")
    ap.add_argument("--base", default="http://127.0.0.1:8765",
                    help="a running labeler; it owns the ruler detector")
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    study = args.data_root / args.dataset
    specs = get(args.base, f"/api/specimens?dataset={args.dataset}")
    if isinstance(specs, dict) and not specs.get("specimens"):
        print(f"could not read specimens: {specs.get('error', specs)}", file=sys.stderr)
        return 1
    specs = specs["specimens"] if isinstance(specs, dict) else specs

    todo = []
    for s in specs:
        sc = study / "sidecars" / f"{s['id']}.json"
        if not sc.is_file() or not s.get("lateral"):
            continue
        doc = json.loads(sc.read_text())
        mode = ((doc.get("lateral") or {}).get("calibration") or {}).get("mode")
        if mode in (None, "none"):
            todo.append((s["id"], s["lateral"], sc))

    print(f"{len(specs)} fish · {len(todo)} with no scale\n")
    placed = refused = failed = 0
    t0 = time.time()
    for i, (fid, image, sc) in enumerate(todo, 1):
        r = get(args.base,
                f"/api/autocal/{urllib.parse.quote(image)}?dataset={args.dataset}")
        T = r.get("ticks")
        name = short(fid)
        if not T:
            failed += 1
            print(f"  {name:9s} no ruler found — place both points by hand")
            continue
        n, on = len(T.get("ticks") or []), T.get("within_quarter_mm", 0)
        if not n or on < MIN_ON_TICK * n:
            refused += 1
            print(f"  {name:9s} REFUSED — only {on} of {n} millimetres land on a "
                  f"tick; this scale is wrong, place both points by hand")
            continue
        got = points_from(T)
        if got is None:
            failed += 1
            print(f"  {name:9s} ticks found but no usable span")
            continue
        a, b, span = got
        ppm = math.dist(a, b) / span
        placed += 1
        if args.dry_run:
            print(f"  {name:9s} would place {span} mm apart → {ppm:.2f} px/mm")
            continue
        doc = json.loads(sc.read_text())
        lat = doc.setdefault("lateral", {})
        lat["calibration"] = {"mode": "manual", "point_a": a, "point_b": b,
                              "known_mm": span, "placed_by": "ruler-ticks"}
        sc.write_text(json.dumps(doc, indent=2))
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  placed {placed}  refused {refused}  "
                  f"failed {failed}  ({time.time() - t0:.0f}s)", flush=True)

    verb = "would place" if args.dry_run else "placed"
    print(f"\n{verb} both ruler points on {placed} fish · {refused} refused as "
          f"untrustworthy · {failed} with no ruler found")
    if refused or failed:
        print("those need the two points placed by hand, which the labeler shows "
              "as NO SCALE until they are")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
