#!/usr/bin/env python3
"""Find labels that no longer sit on their fish, and move them back exactly.

A lateral crop runs from its start column to the right edge of the photograph, so
re-cropping moves every pixel of the fish and leaves saved coordinates behind.
HRN_4 was labelled on a crop starting at x=1656 and re-cropped to start at
x=1206; its landmarks, outline and ruler clicks then sat 450 px into the ruler.
``preprocess_cornell.py`` now refuses to do that. This repairs what it already did,
and checks everything else.

    python scripts/realign_labels.py --audit
    python scripts/realign_labels.py --fish HRN_4 --dx 450
    python scripts/realign_labels.py --record-fingerprints

**Audit** measures, for every fish with a traced body outline, how much darker the
fish is just inside the outline than the foam just outside it, at the saved
position and at every horizontal shift within +-900 px. Labels on the fish score
far higher where they are than anywhere else; displaced labels score low where
they are and high at the shift that puts them back. Fish with landmarks but no
outline are not scored this way: a four-point eye box also scores high on the
snout's edge against the foam, which flagged HRN_10 at -158 px when its points
were exactly right. Those rely on their recorded image fingerprint instead.

**Repair** moves a fish's lateral coordinates -- landmarks, outline, and the two
ruler clicks -- by one shift. When the image's size was recorded at labelling the
shift is exact arithmetic (a crop that is N px wider started N px further left).
Labels older than fingerprints have no record, so the shift must be given with
``--dx``; the image evidence is then used only to confirm it, and a shift that
does not put the outline on the fish is refused. Every repair keeps a backup and
appends to ``metadata.coordinate_history``.

A uniform shift changes no distance, area or angle, so no measured trait moves.
What it restores is each coordinate naming the right pixel: what the labeler
draws, and what the landmark model is trained on.
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

from fish_morpho import image_identity as ident  # noqa: E402

BODY = "body_plus_caudal"
DOWNSCALE = 4
SEARCH_PX = 900
#: Contrast between foam outside and fish inside an outline that sits on the fish.
#: Measured across the 52 traced trout: aligned outlines score 20.6-37.6 (median
#: 27.9); HRN_4, displaced, scored 5.4 where it sat and 26.3 once shifted back.
#: The threshold sits between the two groups.
ON_FISH = 12.0
#: How far an aligned outline may be from the best-scoring shift. Across the same
#: 52 the largest was 19 px, from tracing jitter; the displacement this exists for
#: was 450.
ALIGNED_WITHIN = 40


def lateral_image(dataset: Path, fid: str) -> Path | None:
    for p in sorted((dataset / "lateral").glob(f"{fid}_L.*")):
        return p
    return None


def _prepared(path: Path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    h, w = g.shape
    small = cv2.resize(g, (w // DOWNSCALE, h // DOWNSCALE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)


def outline_contrast(small, polygon, dx: float = 0.0) -> float:
    """Foam just outside the outline minus fish just inside it (grey levels)."""
    h, w = small.shape
    pts = np.round((np.asarray(polygon, float) + [dx, 0]) / DOWNSCALE).astype(np.int32)
    if pts[:, 0].max() < 0 or pts[:, 0].min() >= w:
        return float("-inf")
    inside = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inside, [pts], 1)
    k = np.ones((9, 9), np.uint8)
    ring_out = cv2.dilate(inside, k) - inside
    ring_in = inside - cv2.erode(inside, k)
    o, i = small[ring_out > 0], small[ring_in > 0]
    if len(o) < 50 or len(i) < 50:
        return float("-inf")
    return float(o.mean() - i.mean())


def best_shift(small, polygon) -> tuple[int, float, float]:
    """(shift that best puts the outline on the fish, contrast there, contrast now)."""
    coarse = range(-SEARCH_PX, SEARCH_PX + 1, 2 * DOWNSCALE)
    scores = {dx: outline_contrast(small, polygon, dx) for dx in coarse}
    top = max(scores, key=scores.get)
    fine = {dx: outline_contrast(small, polygon, dx)
            for dx in range(top - 2 * DOWNSCALE, top + 2 * DOWNSCALE + 1)}
    top = max(fine, key=fine.get)
    return top, fine[top], outline_contrast(small, polygon, 0)


def audit(dataset: Path) -> list[dict]:
    manifest = ident.load_manifest(dataset)
    rows = []
    for sc in sorted((dataset / "sidecars").glob("*.json")):
        try:
            doc = json.loads(sc.read_text())
        except Exception as exc:
            rows.append({"fish": sc.stem, "result": f"unreadable: {exc}"})
            continue
        fid = doc.get("fish_id", sc.stem)
        if not ident.has_labels(doc, "lateral"):
            continue
        img = lateral_image(dataset, fid)
        cur = ident.fingerprint(img) if img else None
        rec = ident.recorded_for(doc, manifest, fid, "lateral")
        status = ident.compare(rec, cur)
        row = {"fish": fid, "fingerprint": status}
        if status == ident.SIZE_CHANGED:
            row["exact_dx"] = cur["width"] - rec["width"]
        poly = ((doc.get("lateral") or {}).get("polygons") or {}).get(BODY)
        if poly and img:
            small = _prepared(img)
            dx, best, now = best_shift(small, poly)
            row.update(proposed_dx=dx, contrast_now=round(now, 1), contrast_best=round(best, 1))
            row["result"] = ("aligned" if abs(dx) <= ALIGNED_WITHIN and now >= ON_FISH
                             else "DISPLACED")
        else:
            row["result"] = ("DISPLACED" if status == ident.SIZE_CHANGED
                             else "no outline — checked by fingerprint only")
        rows.append(row)
    return rows


def repair(dataset: Path, fid: str, dx: int | None, dry_run: bool) -> int:
    fid = _resolve(dataset, fid)
    sc = dataset / "sidecars" / f"{fid}.json"
    doc = json.loads(sc.read_text())
    img = lateral_image(dataset, fid)
    if img is None or not ident.has_labels(doc, "lateral"):
        print(f"{fid}: no lateral image or no lateral labels — nothing to repair")
        return 1
    cur = ident.fingerprint(img)
    rec = ident.recorded_for(doc, ident.load_manifest(dataset), fid, "lateral")
    if dx is None:
        if ident.compare(rec, cur) == ident.SIZE_CHANGED:
            dx = cur["width"] - rec["width"]
            print(f"{fid}: recorded width {rec['width']}, now {cur['width']} -> exact shift {dx:+d} px")
        else:
            poly = ((doc.get("lateral") or {}).get("polygons") or {}).get(BODY)
            hint = ""
            if poly:
                est, _, _ = best_shift(_prepared(img), poly)
                hint = f" The image evidence suggests about {est:+d} px."
            print(f"{fid}: no recorded image size, so the shift cannot be worked out exactly. "
                  f"Establish it (e.g. from the raw photograph's crop) and pass --dx.{hint}")
            return 2

    poly = ((doc.get("lateral") or {}).get("polygons") or {}).get(BODY)
    if poly:
        small = _prepared(img)
        before, after = outline_contrast(small, poly, 0), outline_contrast(small, poly, dx)
        print(f"{fid}: outline contrast {before:.1f} where it is, {after:.1f} after {dx:+d} px")
        if after < ON_FISH or after <= before:
            print(f"{fid}: REFUSED — that shift does not put the outline on the fish. Nothing written.")
            return 3
    else:
        print(f"{fid}: no traced outline to confirm the shift against; applying {dx:+d} px as given")

    if dry_run:
        print(f"{fid}: dry run — nothing written")
        return 0
    backup = sc.with_name(f"{fid}.pre-realign-{dt.datetime.now():%Y%m%dT%H%M%S}.json.bak")
    shutil.copy2(sc, backup)
    moved = ident.shift_view(doc, "lateral", dx)
    meta = doc.setdefault("metadata", {})
    meta.setdefault("coordinate_history", []).append({
        "at": dt.datetime.now().isoformat(timespec="seconds"), "view": "lateral",
        "dx": dx, "points": moved, "reason": "realigned to the image on disk"})
    meta.setdefault("images", {})["lateral"] = cur      # these coordinates now fit this image
    sc.write_text(json.dumps(doc, indent=2))
    _record(dataset, fid, doc)
    print(f"{fid}: moved {moved} lateral coordinates by {dx:+d} px (backup {backup.name})")
    return 0


def _record(dataset: Path, fid: str, doc: dict) -> None:
    manifest = ident.load_manifest(dataset)
    entry = manifest.setdefault(fid, {})
    for view in ("lateral", "frontal"):
        if not ident.has_labels(doc, view):
            continue
        folder, suffix = (("lateral", "_L") if view == "lateral" else ("frontal", "_F"))
        img = next(iter(sorted((dataset / folder).glob(f"{fid}{suffix}.*"))), None)
        fp = ident.fingerprint(img) if img else None
        if fp:
            entry[view] = fp
    (dataset / ident.MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def record_fingerprints(dataset: Path) -> int:
    """Record the image under every set of labels that audits clean.

    Refuses outright if any fish audits as displaced: recording the current image
    for labels that do not fit it would make the mismatch look legitimate forever.
    """
    rows = audit(dataset)
    bad = [r for r in rows if r.get("result") == "DISPLACED"]
    if bad:
        print("Not recording: repair these first —")
        for r in bad:
            print(f"   {r['fish']}  proposed {r.get('exact_dx', r.get('proposed_dx'))!s} px")
        return 1
    n = 0
    for sc in sorted((dataset / "sidecars").glob("*.json")):
        doc = json.loads(sc.read_text())
        if ident.has_labels(doc, "lateral") or ident.has_labels(doc, "frontal"):
            _record(dataset, doc.get("fish_id", sc.stem), doc)
            n += 1
    print(f"recorded image fingerprints for {n} labelled fish in {dataset / ident.MANIFEST}")
    return 0


def _resolve(dataset: Path, fid: str) -> str:
    if (dataset / "sidecars" / f"{fid}.json").is_file():
        return fid
    hits = [p.stem for p in (dataset / "sidecars").glob(f"*_{fid}.json")]
    if len(hits) != 1:
        raise SystemExit(f"no single sidecar matches {fid!r}: {hits or 'none'}")
    return hits[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="realign_labels")
    ap.add_argument("--dataset", default="cornell")
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="check every labelled fish")
    mode.add_argument("--fish", help="repair one fish (full id, or e.g. HRN_4)")
    mode.add_argument("--record-fingerprints", action="store_true",
                      help="record the image under every clean set of labels")
    ap.add_argument("--dx", type=int, help="the shift, in px, for labels with no recorded size")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    dataset = args.data_root / args.dataset

    if args.audit:
        rows = audit(dataset)
        bad = [r for r in rows if r.get("result") == "DISPLACED"]
        print(f"{len(rows)} fish with lateral labels: {len(rows) - len(bad)} fine, {len(bad)} displaced\n")
        for r in rows:
            if r.get("result") != "aligned":
                extra = (f"  shift {r.get('exact_dx', r.get('proposed_dx')):+d} px"
                         if r.get("result") == "DISPLACED" else "")
                print(f"  {r['fish']:34} {r['result']}{extra}")
        return 1 if bad else 0
    if args.record_fingerprints:
        return record_fingerprints(dataset)
    return repair(dataset, args.fish, args.dx, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
