#!/usr/bin/env python3
"""Measure a dataset's landmark plausibility bands from its own hand labels.

Writes ``plausibility.json`` beside the sidecars it was fitted on. Predictions are
then checked against it: a landmark outside the range every labelled fish occupies
is reported as impossible rather than placed.

    python scripts/fit_plausibility.py --dataset cornell

Only hand-labelled sidecars are used -- fitting on the model's own output would
launder its mistakes into the definition of what is possible. Re-run it after a
labelling session; more fish means tighter, better-founded bands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fish_morpho import plausibility  # noqa: E402


def hand_labelled(sidecar_dir: Path):
    """The ``lateral`` block of every sidecar a person labelled."""
    for path in sorted(sidecar_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        meta = doc.get("metadata") or {}
        if meta.get("source", "") == "predicted":
            continue
        lat = doc.get("lateral")
        if not lat:
            continue
        # The bands define what the model is allowed to output, so they cannot be
        # fitted on what the model output. Drop points nobody reviewed, and an
        # outline the model drew and nobody edited -- the outline-shape guard in
        # particular would otherwise learn SAM's failure modes as normal.
        assist = meta.get("assist") or {}
        stale = set(assist.get("unreviewed") or []) | set(meta.get("unreviewed_predictions") or [])
        model_polys = set(assist.get("polygons_from_model") or [])
        lat = dict(lat)
        if stale:
            lat["keypoints"] = {k: v for k, v in (lat.get("keypoints") or {}).items()
                                if k not in stale}
        if model_polys:
            lat["polygons"] = {k: v for k, v in (lat.get("polygons") or {}).items()
                               if k not in model_polys}
        yield lat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fit_plausibility")
    ap.add_argument("--dataset", default="cornell")
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--margin", type=float, default=plausibility.DEFAULT_MARGIN,
                    help="slack each side of the observed range, as a multiple "
                         "of that range")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = args.data_root / args.dataset
    sidecars = root / "sidecars"
    if not sidecars.is_dir():
        print(f"no sidecars at {sidecars}", file=sys.stderr)
        return 1

    bands = plausibility.fit(hand_labelled(sidecars), margin=args.margin)
    lm = bands["landmarks"]
    fitted = {n: e for n, e in lm.items() if e.get("axial")}

    print(f"{bands['fish']} labelled fish carried a traced {plausibility.BODY} outline\n")
    print(f"{'landmark':32}{'n':>4}{'band (fraction of body)':>28}")
    for name, entry in sorted(lm.items(), key=lambda kv: kv[1].get("axial", [9])[0]):
        if entry.get("axial"):
            lo, hi = entry["axial"]
            obs = entry["axial_observed"]
            print(f"  {name:30}{entry['n']:4}   {lo:.3f} – {hi:.3f}"
                  f"   (seen {obs[0]:.3f} – {obs[1]:.3f})")
        else:
            print(f"  {name:30}{entry['n']:4}   — too few to measure")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    out = root / "plausibility.json"
    out.write_text(json.dumps(bands, indent=2) + "\n")
    print(f"\nwrote {out}  ({len(fitted)} landmarks checkable)")
    for line in plausibility.describe(bands):
        print(f"  · {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
