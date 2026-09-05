"""What the model gets wrong on new fish, measured from the corrections you make.

    python scripts/correction_report.py --dataset data/cornell

Every sidecar saved after an Auto-label records, per landmark, whether the human
**corrected** it (moved it, and how far), **accepted** it (pressed A to confirm),
or left it **unreviewed**. Aggregated over specimens that is a running error
estimate on exactly the fish the model has never seen — which is a better signal
than a nine-specimen holdout frozen in August, and it costs nothing extra because
it falls out of labelling you were doing anyway.

Read the three states differently
---------------------------------
**Corrections are the strong evidence.** The human saw a wrong point and moved
it. The distance is a real error measurement, and the corrected coordinate is a
genuine label.

**Acceptances are weak evidence, and the weakness has a direction.** Showing
somebody a point and asking whether it is right is not the same as asking them
where the point goes. Pre-annotation anchors people: a plausible-looking marker
gets waved through more often than an empty image gets mislabelled, so
acceptances are biased *toward agreeing with the model*. Train on them as if they
were fresh labels and the model's systematic errors get confirmed by a human who
was primed by those very errors, while the error curve improves. They are worth
recording and worth much less than corrections.

**Unreviewed points are not evidence.** They mean nobody looked.

So this reports the three separately and never pools them. What it is for is
deciding where to spend labelling effort and which landmarks still need work —
not for quietly manufacturing a training set.

A caveat on the correction distances
------------------------------------
They are biased *upward* as an error estimate. A human moves a point when the
error is big enough to be worth the click; sub-pixel errors never get corrected
and never enter the numbers. Read the median correction as "how wrong it is when
it is visibly wrong", not as the model's mean error.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def load(sidecars: Path):
    """Sidecars that carry assist provenance, newest schema only."""
    out = []
    for f in sorted(sidecars.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        a = (d.get("metadata") or {}).get("assist")
        if a:
            out.append((d.get("fish_id", f.stem), a))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="correction_report")
    ap.add_argument("--dataset", type=Path, default=_ROOT / "data/cornell")
    ap.add_argument("--sidecars", type=Path, default=None)
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="Report distances in mm as well as pixels.")
    ap.add_argument("--json", type=Path, default=None,
                    help="Also write the aggregate as JSON.")
    args = ap.parse_args(argv)

    sidecars = args.sidecars or (args.dataset / "sidecars")
    rows = load(sidecars)
    if not rows:
        print(f"No assisted sidecars in {sidecars}.")
        print("Save a specimen after using Auto-label and this will have "
              "something to report.")
        return 0

    corrected: dict[str, list[float]] = defaultdict(list)
    accepted: dict[str, int] = defaultdict(int)
    unreviewed: dict[str, int] = defaultdict(int)
    conf_when_corrected: dict[str, list[float]] = defaultdict(list)
    per_fish = []

    for fid, a in rows:
        c = a.get("corrected") or {}
        acc = a.get("accepted") or []
        unr = a.get("unreviewed") or []
        for name, rec in c.items():
            corrected[name].append(float(rec.get("px", 0.0)))
            if rec.get("conf") is not None:
                conf_when_corrected[name].append(float(rec["conf"]))
        for name in acc:
            accepted[name] += 1
        for name in unr:
            unreviewed[name] += 1
        per_fish.append((fid, len(c), len(acc), len(unr)))

    names = sorted(set(corrected) | set(accepted) | set(unreviewed))
    scale = args.px_per_mm
    unit = "mm" if scale else "px"

    print(f"{len(rows)} assisted specimen(s) in {sidecars}\n")
    print(f"{'landmark':32} {'corr':>5} {'acc':>4} {'unrev':>5} "
          f"{'median':>8} {'worst':>8}   correction rate")
    print("-" * 96)

    summary = {}
    for n in names:
        d = corrected[n]
        nc, na, nu = len(d), accepted[n], unreviewed[n]
        seen = nc + na                      # points a human actually looked at
        rate = (nc / seen) if seen else float("nan")
        med = st.median(d) if d else float("nan")
        worst = max(d) if d else float("nan")
        if scale:
            med, worst = med / scale, worst / scale
        bar = "" if math.isnan(rate) else ("!" * min(10, int(rate * 10 + 0.5)))
        print(f"  {n:30} {nc:>5} {na:>4} {nu:>5} "
              f"{med:>8.1f} {worst:>8.1f}   "
              f"{'' if math.isnan(rate) else f'{rate*100:>3.0f}%'} {bar}")
        summary[n] = {"corrected": nc, "accepted": na, "unreviewed": nu,
                      "median": None if math.isnan(med) else round(med, 2),
                      "worst": None if math.isnan(worst) else round(worst, 2),
                      "correction_rate": None if math.isnan(rate) else round(rate, 3)}

    tot_c = sum(len(v) for v in corrected.values())
    tot_a = sum(accepted.values())
    tot_u = sum(unreviewed.values())
    seen = tot_c + tot_a
    print("-" * 96)
    print(f"  {'TOTAL':30} {tot_c:>5} {tot_a:>4} {tot_u:>5}")
    if seen:
        print(f"\n  {tot_c}/{seen} reviewed points needed moving "
              f"({tot_c / seen * 100:.0f}%).")

    ranked = sorted(((n, v) for n, v in summary.items() if v["correction_rate"] is not None),
                    key=lambda kv: (-kv[1]["correction_rate"], -(kv[1]["median"] or 0)))
    if ranked:
        print("\nMost often wrong — where more training data would pay:")
        for n, v in ranked[:5]:
            print(f"  {n:30} corrected {v['correction_rate']*100:.0f}% of the time, "
                  f"median {v['median']} {unit}")

    flagged = {n: st.median(c) for n, c in conf_when_corrected.items() if c}
    if flagged:
        lo = sum(1 for cs in conf_when_corrected.values() for c in cs if c < 0.6)
        print(f"\n  {lo}/{tot_c} corrections were on points the model had already "
              f"flagged below 0.6.")
        print("  A high share here means the confidence gate is doing its job; a low "
              "one\n  means the model is confidently wrong, which is the worse failure.")

    if tot_u:
        print(f"\n  NOTE: {tot_u} predicted point(s) were never reviewed. They are "
              f"not evidence\n  of anything and are excluded from the rates above.")

    print("\nCorrections are the trustworthy half. Acceptances are a human agreeing "
          "with a\nsuggestion they were shown, which is a weaker claim than an "
          "independent label —\nsee this script's docstring before training on them.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"specimens": len(rows), "unit": unit, "landmarks": summary,
             "per_fish": [{"fish_id": f, "corrected": c, "accepted": a,
                           "unreviewed": u} for f, c, a, u in per_fish]}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
