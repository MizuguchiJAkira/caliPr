#!/usr/bin/env python3
"""Standard length from the photographs against the hatchery's caliper length.

An independent check of calibration and landmarking: the same fish were measured
with calipers at the hatchery before they were photographed. For every brook
trout whose sheet row can be matched to its photograph, this measures standard
length from the sidecar exactly as the exporters do and compares it with the
caliper value.

    python scripts/caliper_check.py
    python scripts/caliper_check.py --csv results/caliper_check.csv

The photographs are not needed: the sidecars hold the landmarks and calibration.

Matching photographs to sheet rows
----------------------------------
The sheet numbers specimens per strain; the photographs are named per strain in
the order they were taken. The two agree except where the photo series has
reshoots or a swap, found by comparing the sheet's photo numbers with the photo
series:

* HRN: photographs 28 and 38 are each other's specimen.
* TXD: photograph 14 is a second photograph of specimen 13, and photograph 49 a
  second photograph of specimen 48, so both are left out; photographs 15-48 are
  specimens 14-47, and photograph 50 is specimen 48.
* ASN: photograph n is specimen n.

The sheet covers ASN, HRN and TXD; the other strains' caliper lengths are not in
the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho.pipeline import (  # noqa: E402
    SpecimenInput, _excluded_structures, process_specimen)

SHEET = _ROOT / "data" / "cornell_raw" / "jonah_photos_info_v2.csv"
STUDY = _ROOT / "data" / "cornell"
STRAINS = ("ASN", "HRN", "TXD")


def sheet_specimen(strain: str, photo: int) -> int | None:
    """The sheet's specimen number for a strain's photograph number, or None."""
    if strain == "HRN":
        return {28: 38, 38: 28}.get(photo, photo)
    if strain == "TXD":
        if photo in (14, 49):
            return None                     # reshoots of 13 and 48
        if 15 <= photo <= 48:
            return photo - 1
        if photo == 50:
            return 48
        return photo
    return photo


def caliper_lengths(sheet: Path) -> dict[tuple[str, int], float]:
    out = {}
    with sheet.open(newline="") as f:
        for row in csv.DictReader(f):
            m = re.match(r"ST-([A-Z]+)\b", row.get("field_number") or "")
            sl = (row.get("sl(mm)") or "").strip()
            spec = (row.get("specimen") or "").strip()
            if m and sl and spec.isdigit():
                out[(m.group(1), int(spec))] = float(sl)
    return out


def photo_lengths(study: Path) -> dict[str, float]:
    excluded = _excluded_structures(study)
    out = {}
    for path in sorted((study / "sidecars").glob("*.json")):
        sc = json.loads(path.read_text())
        spec = SpecimenInput(fish_id=sc.get("fish_id", path.stem),
                             image_path=study / "lateral" / path.stem,
                             sidecar_path=path, sidecar=sc)
        try:
            rec = process_specimen(spec, excluded=excluded)
        except (ValueError, FileNotFoundError):
            continue
        v = rec.measurements.get("SL")
        if v is not None and v.unit == "mm" and not math.isnan(v.value) and not v.missing_landmarks:
            out[spec.fish_id] = v.value
    return out


def compare(study: Path = STUDY, sheet: Path = SHEET) -> list[dict]:
    calipers = caliper_lengths(sheet)
    rows = []
    for fid, photo_sl in photo_lengths(study).items():
        m = re.search(r"_([A-Z]+)_(\d+)$", fid)
        if not m or m.group(1) not in STRAINS:
            continue
        strain, photo = m.group(1), int(m.group(2))
        spec = sheet_specimen(strain, photo)
        caliper = calipers.get((strain, spec)) if spec is not None else None
        if caliper is None:
            continue
        rows.append({"fish_id": fid, "strain": strain, "sheet_specimen": spec,
                     "photo_sl_mm": round(photo_sl, 2), "caliper_sl_mm": caliper,
                     "diff_pct": round(100 * (photo_sl - caliper) / caliper, 3)})
    return rows


def summary(rows: list[dict]) -> dict:
    d = [r["diff_pct"] for r in rows]
    a = [abs(x) for x in d]
    return {"n": len(d), "median_abs_pct": statistics.median(a), "mean_abs_pct": statistics.fmean(a),
            "mean_bias_pct": statistics.fmean(d), "over_5_pct": sum(x > 5 for x in a)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="caliper_check", description=__doc__.split("\n")[0])
    ap.add_argument("--study", type=Path, default=STUDY)
    ap.add_argument("--sheet", type=Path, default=SHEET)
    ap.add_argument("--csv", type=Path, help="also write one row per fish here")
    args = ap.parse_args(argv)
    if not args.sheet.is_file():
        print(f"no caliper sheet at {args.sheet}")
        return 1
    rows = compare(args.study, args.sheet)
    if not rows:
        print("no photographed fish matched a caliper row")
        return 1
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    s = summary(rows)
    print(f"standard length, photograph vs caliper: {s['n']} fish; "
          f"median |difference| {s['median_abs_pct']:.2f}%, mean {s['mean_abs_pct']:.2f}%, "
          f"mean bias {s['mean_bias_pct']:+.2f}%, {s['over_5_pct']} fish beyond 5%")
    for strain in STRAINS:
        sub = [r for r in rows if r["strain"] == strain]
        if sub:
            t = summary(sub)
            print(f"  {strain}: {t['n']} fish, median |difference| {t['median_abs_pct']:.2f}%, "
                  f"bias {t['mean_bias_pct']:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
