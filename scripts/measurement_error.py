"""Measurement error of every trait, from a study's blind re-label rounds.

    python scripts/measurement_error.py --dataset cornell
    python scripts/measurement_error.py --dataset cornell --new-round 20 --operator JC

The first prints ICC(1) with its 95% interval, %ME and ICC(3,1) per trait, for
each set of rounds that re-labelled the same fish (see fish_morpho.repeatability).
The second sets up a blind round of 20 labelled fish, drawn across the study's
groups, as the study folder ``cornell.relabel-<k>`` beside it: open that in the
labeler and label every fish from scratch. The measurements workbook carries the
same table on its Measurement error sheet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho import repeatability  # noqa: E402
from fish_morpho.landmark_config import TRAITS  # noqa: E402


def as_json(designs) -> dict:
    label = {t.code: t.label for t in TRAITS}
    out = []
    for d in designs:
        traits = []
        for code, te in d.traits.items():
            if te.icc1 is None and te.n < 3:
                continue
            traits.append({"code": code, "label": label.get(code, code), "n": te.n, "k": te.k,
                           "icc1": te.icc1, "icc1_lo": te.icc1_lo, "icc1_hi": te.icc1_hi,
                           "pct_me": te.pct_me, "icc31": te.icc31, "bias": te.bias,
                           "mean_abs_diff": te.mean_abs_diff, "tem": te.tem,
                           "verdict": te.verdict})
        out.append({"rounds": [r.get("round") for r in d.rounds], "fish": len(d.fish),
                    "sessions": d.sessions, "traits": traits})
    return {"designs": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="measurement_error")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    ap.add_argument("--json", type=Path, default=None, help="Write the result here as JSON.")
    ap.add_argument("--new-round", type=int, default=None, metavar="N",
                    help="Set up a blind re-label round of N labelled fish instead.")
    ap.add_argument("--operator", default="", help="Who will re-label the round.")
    ap.add_argument("--same-as", type=int, default=None,
                    help="Re-label the fish of this earlier round again.")
    args = ap.parse_args(argv)
    # Each fish's data notes are the measurements export's to report, not this.
    import logging
    logging.basicConfig(level=logging.ERROR)
    study = args.data_root / args.dataset
    if not (study / "lateral").is_dir():
        ap.error(f"{study} is not a study (no lateral/)")

    if args.new_round is not None or args.same_as is not None:
        try:
            r = repeatability.create_round(study, args.new_round or 0, args.operator,
                                           same_as=args.same_as)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"round {r['round']}: {r['n']} fish in {Path(r['folder']).name}/ -- open it in "
              f"the labeler and label every fish from scratch")
        return 0

    designs = repeatability.analyse(study)
    if args.json:
        args.json.write_text(json.dumps(as_json(designs)))
    if not designs:
        print("No re-label round with three or more fish done yet.")
        return 0
    for d in designs:
        print(f"\n{len(d.fish)} fish, {len(d.sessions)} sessions:")
        for i, s in enumerate(d.sessions, 1):
            print(f"  {i}. {s}")
        print(f"  {'trait':<6} {'ICC(1)':>7} {'95% CI':>15} {'%ME':>6} {'ICC(3,1)':>9}  verdict")
        for code, te in d.traits.items():
            if te.icc1 is None:
                continue
            print(f"  {code:<6} {te.icc1:7.3f} [{te.icc1_lo:6.3f},{te.icc1_hi:6.3f}] "
                  f"{te.pct_me:6.1f} {te.icc31 if te.icc31 is not None else float('nan'):9.3f}  "
                  f"{te.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
