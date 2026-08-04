"""Turn DLC's evaluation output into a per-landmark verdict in millimetres.

DLC reports pixel error in the *downscaled* training frames. That number can't
be judged on its own: 5 px sounds small, but at scale 0.25 on a 25 px/mm rig it
is 0.8 mm of real fish — comparable to the manual pipeline's entire agreement
budget with calipers (~1.3%).

So this converts every landmark's error into specimen millimetres and grades it
against the manual pipeline, which is the standard the model has to meet:

  good      < 0.5 mm   comfortably inside manual noise
  usable    < 1.0 mm   about the manual pipeline's own agreement
  marginal  < 2.0 mm   would widen error on traits anchored to it
  poor      >= 2.0 mm  not usable for measurement yet

Usage::

    python scripts/dlc_report.py --project dlc_project/jcalipr-jcalipr-<date> \\
        --split dlc/split.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent


def median_px_per_mm(sidecars: Path) -> float:
    vals = []
    for p in sidecars.glob("*.json"):
        cal = (json.loads(p.read_text()).get("lateral") or {}).get("calibration")
        if not cal:
            continue
        if cal.get("mode") == "ticks":
            vals.append(float(cal["px_per_mm"]))
        else:
            a, b = cal["point_a"], cal["point_b"]
            km = float(cal.get("known_mm") or 0)
            if km > 0:
                vals.append(math.hypot(b[0] - a[0], b[1] - a[1]) / km)
    vals.sort()
    return vals[len(vals) // 2] if vals else float("nan")


def grade(mm: float) -> str:
    if mm < 0.5:
        return "good"
    if mm < 1.0:
        return "usable"
    if mm < 2.0:
        return "marginal"
    return "poor"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dlc_report")
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=_ROOT / "dlc/split.json")
    ap.add_argument("--sidecars", type=Path, default=_ROOT / "data/cornell/sidecars")
    args = ap.parse_args(argv)

    split = json.loads(args.split.read_text())
    scale = float(split.get("scale", 1.0))
    ppm_full = median_px_per_mm(args.sidecars)
    ppm_frame = ppm_full * scale            # px/mm in the downscaled frames

    csvs = sorted(args.project.rglob("*-results.csv")) + \
           sorted(args.project.rglob("CombinedEvaluation-results.csv"))
    per_kp = [p for p in args.project.rglob("*.csv") if "PerKeypoint" in p.name or "per_keypoint" in p.name.lower()]
    src = per_kp[0] if per_kp else (csvs[0] if csvs else None)
    if src is None:
        print("No DLC evaluation CSV found. Did evaluate_network run?")
        return 1

    print(f"source: {src.relative_to(args.project)}")
    print(f"scale: {scale}   rig: {ppm_full:.2f} px/mm full-res "
          f"-> {ppm_frame:.2f} px/mm in training frames\n")

    df = pd.read_csv(src)
    # Per-keypoint files carry one row per bodypart with a test error column.
    cols = {c.lower(): c for c in df.columns}
    bp_col = next((cols[c] for c in cols if "bodypart" in c or "keypoint" in c), None)
    err_col = next((cols[c] for c in cols
                    if "test" in c and "error" in c and "p-cut" not in c), None)
    if bp_col is None or err_col is None:
        print("Columns:", list(df.columns))
        print("\nCould not identify bodypart/test-error columns; raw table:")
        print(df.to_string(index=False))
        return 0

    rows = []
    for _, r in df.iterrows():
        try:
            px = float(r[err_col])
        except (TypeError, ValueError):
            continue
        rows.append((str(r[bp_col]), px, px / ppm_frame))
    rows.sort(key=lambda t: t[2])

    print(f"{'landmark':32} {'test px':>8} {'mm':>7}  verdict")
    for name, px, mm in rows:
        print(f"{name:32} {px:8.2f} {mm:7.3f}  {grade(mm)}")
    if rows:
        mms = [m for _, _, m in rows]
        print(f"\n{'MEDIAN':32} {'':8} {sorted(mms)[len(mms)//2]:7.3f}")
        print(f"{'WORST':32} {'':8} {max(mms):7.3f}  ({rows[-1][0]})")
    print(f"\nheld-out specimens: {len(split['test'])}")
    print("benchmark: manual pipeline agrees with calipers to ~1.3% "
          "(~1.5 mm on a 120 mm fish)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
