"""Per-landmark keypoint error in specimen millimetres, with confidence gating.

DLC's summary CSV reports a *mean* pixel error per landmark. On a 9-specimen
held-out set that is actively misleading: one bad prediction moves the mean by
more than the other eight combined. ``pectoral_ray_tip`` looked like a 12.26 mm
catastrophe by that measure and is 0.76 mm by median — the difference was a
single fish.

So this reads the raw prediction HDF5 (which carries per-point ``likelihood``)
rather than the summary, and reports:

* median and mean, so a skewed distribution is visible rather than hidden;
* error in **millimetres**, since that is what decides usability — the manual
  pipeline agrees with calipers to ~1.4 mm, and a landmark is only useful if it
  beats that;
* what happens once low-confidence predictions are **rejected**. The model
  already signals its own failures (the bad ``pectoral_ray_tip`` came with
  likelihood 0.21 against 0.62–0.89 elsewhere); throwing that signal away and
  reporting a confident wrong number is exactly what the pipeline's
  data-compromise handling exists to prevent. A rejected point becomes a
  missing landmark, and the traits that depend on it surface as NaN with a
  reason attached.

Usage::

    python scripts/dlc_report.py --project dlc_project/jcalipr-<...> \\
        --pcutoff 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


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
    ap.add_argument("--pcutoff", type=float, default=0.5,
                    help="Absolute likelihood floor (see --relative).")
    ap.add_argument("--relative", type=float, default=0.5, metavar="FRAC",
                    help="Reject a prediction whose likelihood is below FRAC x that "
                         "landmark's own median. Likelihood is NOT calibrated across "
                         "landmarks — caudal_base sits at 0.24-0.45 while being "
                         "accurate to 0.97 mm — so an absolute floor discards good "
                         "predictions wholesale. Set to 0 to use --pcutoff instead.")
    ap.add_argument("--px-per-mm", type=float, default=25.0,
                    help="Full-resolution rig scale; combined with split.json's scale.")
    ap.add_argument("--per-specimen", metavar="LANDMARK",
                    help="Break one landmark down by specimen.")
    args = ap.parse_args(argv)

    import pandas as pd

    split = json.loads(args.split.read_text())
    test = set(split["test"])
    ppm = args.px_per_mm * float(split.get("scale", 1.0))

    gt_path = next((args.project / "labeled-data").rglob("CollectedData_*.h5"))
    pred_path = next(args.project.rglob("*snapshot_best-*.h5"), None)
    if pred_path is None:
        pred_path = next(args.project.rglob("*snapshot-*.h5"))
    gt = pd.read_hdf(gt_path)
    pred = pd.read_hdf(pred_path)

    sg = gt.columns[0][0]
    sp = pred.columns[0][0]
    # single-animal projects still nest an "individuals" level in predictions
    indiv = pred.columns[0][1] if pred.columns.nlevels == 4 else None

    def pcol(part, coord):
        return (sp, indiv, part, coord) if indiv else (sp, part, coord)

    parts = list(dict.fromkeys(c[1] for c in gt.columns))
    per_part: dict[str, list[tuple[str, float, float]]] = {}
    for idx in gt.index:
        fish = str(idx[-1]).replace(".png", "")
        if fish not in test or idx not in pred.index:
            continue
        for part in parts:
            gx = gt.loc[idx, (sg, part, "x")]
            gy = gt.loc[idx, (sg, part, "y")]
            if pd.isna(gx):
                continue                      # landmark genuinely absent
            px = pred.loc[idx, pcol(part, "x")]
            py = pred.loc[idx, pcol(part, "y")]
            lk = float(pred.loc[idx, pcol(part, "likelihood")])
            d = math.hypot(float(px) - float(gx), float(py) - float(gy)) / ppm
            per_part.setdefault(part, []).append((fish, d, lk))

    # Per-landmark rejection thresholds, since likelihood scales differ by landmark.
    thresh: dict[str, float] = {}
    for part, rows in per_part.items():
        med_lk = st.median([r[2] for r in rows])
        thresh[part] = (args.relative * med_lk) if args.relative > 0 else args.pcutoff

    if not per_part:
        print("No overlap between held-out split and predictions.")
        return 1

    if args.per_specimen:
        rows = sorted(per_part.get(args.per_specimen, []), key=lambda r: -r[1])
        print(f"{args.per_specimen} — per held-out specimen\n")
        print(f"{'specimen':12} {'err mm':>8} {'conf':>6}")
        for f, d, lk in rows:
            mark = "  <-- rejected" if lk < thresh[args.per_specimen] else ""
            print(f"  {f.replace('Salvelinus_fontinalis_',''):10} {d:8.2f} {lk:6.2f}{mark}")
        return 0

    print(f"held-out: {len(test)} specimens   scale {split.get('scale')} "
          f"({ppm:.2f} px/mm)   gate: {args.relative}x per-landmark median\n")
    print(f"{'landmark':32} {'median':>7} {'mean':>7} {'kept':>6} {'med|kept':>9}  verdict")
    med_all, med_kept = [], []
    for part in parts:
        rows = per_part.get(part)
        if not rows:
            continue
        d = [r[1] for r in rows]
        keep = [r[1] for r in rows if r[2] >= thresh[part]]
        m = st.median(d)
        mk = st.median(keep) if keep else float("nan")
        med_all.append(m)
        if keep:
            med_kept.append(mk)
        kept = f"{len(keep)}/{len(rows)}"
        print(f"{part:32} {m:7.2f} {st.mean(d):7.2f} {kept:>6} {mk:9.2f}  {grade(mk if keep else m)}")

    print(f"\n{'OVERALL median of medians':32} {st.median(med_all):7.2f} "
          f"{'':7} {'':6} {st.median(med_kept):9.2f}")
    dropped = sum(1 for p2,rows in per_part.items() for r in rows if r[2] < thresh[p2])
    total = sum(len(r) for r in per_part.values())
    print(f"\nrejected {dropped}/{total} predictions below {args.pcutoff} "
          f"({dropped/total*100:.1f}%) — these become missing landmarks, not wrong numbers")
    print("benchmark: the manual pipeline agrees with calipers to ~1.4 mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
