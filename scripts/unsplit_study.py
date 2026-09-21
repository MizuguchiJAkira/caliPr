#!/usr/bin/env python3
"""Stop storing a lateral and a frontal crop: keep one photograph per specimen.

The two crops existed to give the models a consistent frame. They can have one
without being stored: the crop is made in memory at prediction time, at the same
seam and with the same margins, and on 173 runs over these photographs it gives
the model the same pixels -- identical predictions, median 0 px apart. What the
stored crops added was a class of failures of their own: a mirror boundary found
in the wrong place cut 10 heads out of the frontal view, and re-cutting a
photograph after labelling stranded HRN_4's landmarks 450 px into the ruler.

A fixed fraction of the frame cannot replace them, which is worth writing down
because it is the obvious thing to try. Across these 131 photographs the mirror
seam runs from 0 to 0.35 of the width while the leftmost snout sits at 0.197, so
there is no fraction that is past every mirror and before every fish: cutting at
0.25 hands the lateral model a second, mirrored head on some fish and cuts the
real one off 29 others. The seam has to be found per photograph, which
``preprocess_cornell.view_frames`` does.

This migrates a study to one photograph per fish:

    python scripts/unsplit_study.py --stage     # build and check, touching nothing
    python scripts/unsplit_study.py --apply     # swap it in, keeping the old crops

**Staging** writes the normalised originals and migrated sidecars under
``<dataset>/.migration/`` and verifies each one. Nothing live is touched, so it can
run while someone is labelling.

**Applying** moves the old crops aside (never deletes them), puts the photographs
in their place, and writes the migrated sidecars.

The coordinates move because the frame does. A lateral crop runs from its start
column to the right edge, so every lateral coordinate shifts right by that start
column -- exact arithmetic, and checked here against the photograph itself before
anything is written. Frontal coordinates do not move at all: that crop starts at
the frame's own corner. A uniform shift changes no distance, area or angle, so no
measured trait changes; what changes is that each coordinate again names the pixel
it was clicked on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import preprocess_cornell as pc  # noqa: E402

from fish_morpho import image_identity as ident  # noqa: E402

#: How well a slice of the photograph must match the stored crop to accept the
#: offset: mean absolute difference in grey levels on a downscaled comparison.
#: JPEG re-encoding alone gives ~1; a different crop gives tens.
MATCH_TOLERANCE = 6.0
SEARCH_PX = 40


def normalised(raw: Path) -> np.ndarray:
    return pc.normalize_orientation(raw)


def find_offset(full: np.ndarray, crop: np.ndarray, guess: int) -> tuple[int | None, float]:
    """The column of ``full`` where ``crop`` begins, confirmed against the pixels.

    The arithmetic says ``guess``; this proves it, and searches a little either side
    so a photograph whose crop was cut differently is caught rather than assumed.
    """
    fh, fw = full.shape[:2]
    ch, cw = crop.shape[:2]
    if ch != fh:
        return None, float("inf")
    small_crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (cw // 8, fh // 8))
    best, best_score = None, float("inf")
    for dx in range(guess - SEARCH_PX, guess + SEARCH_PX + 1, 4):
        if dx < 0 or dx + cw > fw:
            continue
        slice_ = full[:, dx:dx + cw]
        small = cv2.resize(cv2.cvtColor(slice_, cv2.COLOR_BGR2GRAY), (cw // 8, fh // 8))
        score = float(np.abs(small.astype(np.float32) - small_crop.astype(np.float32)).mean())
        if score < best_score:
            best, best_score = dx, score
    return (best, best_score) if best_score <= MATCH_TOLERANCE else (None, best_score)


def raw_for(dataset: Path, raw_dir: Path, fid: str) -> Path | None:
    """The original photograph behind a fish, whatever the spelling of its name."""
    want = fid.lower().replace("fontinalis", "fontinalus")
    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem.strip().lower()
        if stem in (want, f"{want}_l"):
            return p
    return None


def stage(dataset: Path, raw_dir: Path) -> int:
    out = dataset / ".migration"
    photos, sidecars = out / "photos", out / "sidecars"
    for d in (photos, sidecars):
        d.mkdir(parents=True, exist_ok=True)
    report = []
    problems = 0
    for lat_path in sorted((dataset / "lateral").glob("*_L.JPEG")):
        fid = lat_path.name[:-7]
        raw = raw_for(dataset, raw_dir, fid)
        row = {"fish_id": fid}
        if raw is None:
            row["problem"] = "no original photograph"
            report.append(row)
            problems += 1
            continue
        full = normalised(raw)
        crop = cv2.imread(str(lat_path))
        guess = full.shape[1] - crop.shape[1]
        dx, score = find_offset(full, crop, guess)
        row.update(full=[int(full.shape[1]), int(full.shape[0])],
                   crop=[int(crop.shape[1]), int(crop.shape[0])],
                   guess=int(guess), dx=None if dx is None else int(dx),
                   match=round(score, 2))
        if dx is None:
            row["problem"] = (f"the stored crop is not a slice of this photograph "
                              f"(best match {score:.1f} grey levels)")
            report.append(row)
            problems += 1
            continue

        # the frontal crop starts at the frame's own corner, so it needs no shift
        fro = dataset / "frontal" / f"{fid}_F.JPEG"
        if fro.is_file():
            f = cv2.imread(str(fro))
            row["frontal_matches_corner"] = bool(
                f.shape[0] == full.shape[0]
                and find_offset(full, f, 0)[0] == 0)
            if not row["frontal_matches_corner"]:
                row["problem"] = "the frontal crop does not start at the frame's corner"
                problems += 1

        cv2.imwrite(str(photos / f"{fid}_L.JPEG"), full, [cv2.IMWRITE_JPEG_QUALITY, 95])

        sc = dataset / "sidecars" / f"{fid}.json"
        if sc.is_file():
            doc = json.loads(sc.read_text())
            moved = ident.shift_view(doc, "lateral", dx) if dx else 0
            meta = doc.setdefault("metadata", {})
            if dx:
                meta.setdefault("coordinate_history", []).append({
                    "at": dt.datetime.now().isoformat(timespec="seconds"), "view": "lateral",
                    "dx": dx, "points": moved,
                    "reason": "one photograph per fish: the lateral crop's start column"})
            meta.pop("images", None)      # re-recorded against the photograph on apply
            row["labels_moved"] = moved
            (sidecars / f"{fid}.json").write_text(json.dumps(doc, indent=2))
        report.append(row)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    ok = [r for r in report if "problem" not in r]
    print(f"staged {len(ok)} of {len(report)} fish in {out}")
    shifts = sorted({r["dx"] for r in ok})
    print(f"  lateral shift: {min(shifts)}-{max(shifts)} px, "
          f"worst pixel match {max(r['match'] for r in ok):.1f} grey levels")
    for r in report:
        if "problem" in r:
            print(f"  PROBLEM {r['fish_id'][22:]}: {r['problem']}")
    return 1 if problems else 0


def apply(dataset: Path) -> int:
    out = dataset / ".migration"
    photos, sidecars = out / "photos", out / "sidecars"
    if not (out / "report.json").is_file():
        print("nothing staged — run --stage first")
        return 1
    report = json.loads((out / "report.json").read_text())
    problems = [r for r in report if "problem" in r]
    if problems:
        print(f"{len(problems)} fish did not stage cleanly; fix those first:")
        for r in problems[:10]:
            print(f"  {r['fish_id']}: {r['problem']}")
        return 1

    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    kept = dataset / f".crops-before-{stamp}"
    kept.mkdir(parents=True)
    for name in ("lateral", "frontal"):
        if (dataset / name).is_dir():
            shutil.move(str(dataset / name), str(kept / name))
    shutil.copytree(str(dataset / "sidecars"), str(kept / "sidecars"))   # labels as they were
    shutil.move(str(photos), str(dataset / "lateral"))
    for p in sidecars.glob("*.json"):
        shutil.copy2(p, dataset / "sidecars" / p.name)

    # predictions were made in the crops' coordinates and mean nothing here
    for cache in ("sidecars_auto",):
        if (dataset / cache).is_dir():
            shutil.move(str(dataset / cache), str(kept / cache))

    # every coordinate now belongs to the photograph it names
    manifest = {}
    for sc in sorted((dataset / "sidecars").glob("*.json")):
        doc = json.loads(sc.read_text())
        fid = doc.get("fish_id", sc.stem)
        img = next(iter(sorted((dataset / "lateral").glob(f"{fid}_L.*"))), None)
        if img is None:
            continue
        fp = ident.fingerprint(img)
        entry = {}
        for view in ("lateral", "frontal"):
            if ident.has_labels(doc, view):
                entry[view] = fp                      # one photograph, both views
        if entry:
            manifest[fid] = entry
        doc.setdefault("metadata", {})["images"] = entry
        sc.write_text(json.dumps(doc, indent=2))
    (dataset / ident.MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    prof = dataset / "schema.json"
    doc = json.loads(prof.read_text()) if prof.is_file() else {}
    doc["single_photo"] = True
    prof.write_text(json.dumps(doc, indent=2) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    print(f"applied: {len(manifest)} fish now have one photograph each")
    print(f"the old crops and the predictions made against them are in {kept}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="unsplit_study")
    ap.add_argument("--dataset", type=Path, default=_ROOT / "data" / "cornell")
    ap.add_argument("--raw", type=Path, default=_ROOT / "data" / "cornell_raw" / "jonah")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    return stage(args.dataset, args.raw) if args.stage else apply(args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
