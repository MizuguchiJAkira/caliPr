#!/usr/bin/env python3
"""Crops of each hand-traced fin, for training a fin outliner.

Segment Anything, untrained, missed fin area by 8-35% (median, per fin) against dense
re-tracings, because alcohol-preserved fins have almost no contrast against the
foam (docs/what-we-tried.md). A model trained on the lab's own tracings is the next
thing to try. This writes what it trains on.

    python scripts/build_fin_dataset.py            # -> fin_seg/
    python scripts/build_fin_dataset.py --framed-by predicted.json --out fin_seg_pred

With ``--framed-by``, each crop is framed from *predicted* landmarks (a JSON of
``{fish_id: {"keypoints": {...}}}``) while the mask is still the hand tracing, so a
fin outliner can be measured the way it would be used: on crops placed by the
keypoint model. A fin whose predicted base or tip is missing -- not found, or
dropped as anatomically impossible -- gets no crop and is recorded as such.

Each example is one fin on one fish:

* a crop centred on the midpoint of the fin's base and tip landmarks, 0.34 x 0.17
  of standard length. Framed from landmarks the keypoint model predicts, so the
  same crop can be made when predicting; sized from standard length because every
  traced fin fits within 0.14 SL of that midpoint, where base-to-tip distance does
  not (the dorsal's tip is the apex of a long low fin, 56-271 px from its base).
* the hand-traced outline as a mask, and the base and tip in crop coordinates, so
  the model is told which of the fins in the crop to outline.

Only outlines of at least ``FIN_POLYGON_TARGET_VERTICES`` vertices are used: sparse
ones miss area by an unknown sign and would teach it that error. Fish whose labels
no longer fit their image are skipped, as for the keypoint model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho import image_identity as ident  # noqa: E402
from fish_morpho.landmark_config import FIN_POLYGON_TARGET_VERTICES  # noqa: E402

#: The landmarks a fin's crop is framed from: its base, then its tip.
FIN_LANDMARKS = {
    "pectoral": ("pectoral_insertion_upper", "pectoral_ray_tip"),
    "dorsal": ("dorsal_base_center", "dorsal_tip"),
    "pelvic": ("pelvic_base_center", "pelvic_tip"),
    "anal": ("anal_base_center", "anal_tip"),
}
FINS = tuple(FIN_LANDMARKS)
#: Crop half-width and half-height as fractions of standard length. The largest
#: traced fin reaches 0.137 SL across and 0.064 SL up or down from its midpoint.
HALF_W_SL, HALF_H_SL = 0.17, 0.085
OUT_W, OUT_H = 448, 224


def crop_box(base, tip, sl: float) -> tuple[float, float, float, float]:
    cx, cy = (base[0] + tip[0]) / 2, (base[1] + tip[1]) / 2
    return cx - HALF_W_SL * sl, cy - HALF_H_SL * sl, cx + HALF_W_SL * sl, cy + HALF_H_SL * sl


def cut(image: np.ndarray, box) -> tuple[np.ndarray, float, float]:
    """The box resampled to OUT_W x OUT_H, padding outside the photograph with its edge.

    Returns the crop and the scale from image px to crop px along x and y.
    """
    x0, y0, x1, y1 = box
    sx, sy = OUT_W / (x1 - x0), OUT_H / (y1 - y0)
    M = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], np.float64)
    out = cv2.warpAffine(image, M, (OUT_W, OUT_H), flags=cv2.INTER_AREA,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, sx, sy


def to_crop(points, box, sx, sy) -> np.ndarray:
    p = np.asarray(points, float)
    return np.stack([(p[:, 0] - box[0]) * sx, (p[:, 1] - box[1]) * sy], axis=1)


def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def px_per_mm(block: dict) -> float | None:
    cal = block.get("calibration") or {}
    if cal.get("mode") == "ticks":
        return float(cal["px_per_mm"])
    if cal.get("mode") == "manual" and cal.get("known_mm"):
        return math.dist(cal["point_a"], cal["point_b"]) / float(cal["known_mm"])
    return None


def build(dataset: Path, out: Path, framed_by: dict | None = None) -> list[dict]:
    manifest = ident.load_manifest(dataset)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sc in sorted((dataset / "sidecars").glob("*.json")):
        doc = json.loads(sc.read_text())
        fid = doc.get("fish_id", sc.stem)
        lat = doc.get("lateral") or {}
        polys, kps = lat.get("polygons") or {}, lat.get("keypoints") or {}
        wanted = [f for f in FINS if len(polys.get(f) or []) >= FIN_POLYGON_TARGET_VERTICES
                  and all(k in kps for k in FIN_LANDMARKS[f])]
        if not wanted or "premaxilla_tip" not in kps or "caudal_base" not in kps:
            continue
        # Where the crop is framed from: this fish's own labels, or a prediction.
        frame = kps
        if framed_by is not None:
            if fid not in framed_by:
                continue
            frame = framed_by[fid].get("keypoints") or {}
        img_path = next(iter(sorted((dataset / "lateral").glob(f"{fid}_L.*"))), None)
        if img_path is None:
            continue
        rec = ident.recorded_for(doc, manifest, fid, "lateral")
        if ident.compare(rec, ident.fingerprint(img_path)) == ident.SIZE_CHANGED:
            print(f"  SKIP {fid}: labels do not fit the image on disk")
            continue
        image = cv2.imread(str(img_path))
        for fin in wanted:
            need = (*FIN_LANDMARKS[fin], "premaxilla_tip", "caudal_base")
            absent = [k for k in need if k not in frame]
            if absent:
                rows.append({"name": f"{fid}__{fin}", "fish_id": fid, "fin": fin, "no_crop": True,
                             "missing": absent, "area_px": round(polygon_area(
                                 np.asarray(polys[fin], float)), 1)})
                continue
            sl = math.dist(frame["premaxilla_tip"], frame["caudal_base"])
            base, tip = (frame[k] for k in FIN_LANDMARKS[fin])
            box = crop_box(base, tip, sl)
            crop, sx, sy = cut(image, box)
            poly = to_crop(polys[fin], box, sx, sy)
            mask = np.zeros((OUT_H, OUT_W), np.uint8)
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
            bt = to_crop([base, tip], box, sx, sy)
            name = f"{fid}__{fin}"
            np.savez_compressed(out / f"{name}.npz", image=crop[:, :, ::-1], mask=mask,
                                base_tip=bt.astype(np.float32))
            rows.append({
                "name": name, "fish_id": fid, "fin": fin, "box": [round(v, 1) for v in box],
                "scale": [sx, sy], "sl_px": round(sl, 1), "px_per_mm": px_per_mm(lat),
                "vertices": len(polys[fin]),
                "area_px": round(polygon_area(np.asarray(polys[fin], float)), 1),
                "outside_crop": bool((poly < 0).any() or (poly[:, 0] > OUT_W).any()
                                     or (poly[:, 1] > OUT_H).any()),
            })
    (out / "index.json").write_text(json.dumps(rows, indent=1))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_fin_dataset")
    ap.add_argument("--dataset", type=Path, default=_ROOT / "data" / "cornell")
    ap.add_argument("--out", type=Path, default=_ROOT / "fin_seg")
    ap.add_argument("--framed-by", type=Path, default=None,
                    help="frame crops from these predicted landmarks instead of the labels")
    args = ap.parse_args(argv)
    framed = json.loads(args.framed_by.read_text()) if args.framed_by else None
    rows = build(args.dataset, args.out, framed)
    fish = {r["fish_id"] for r in rows}
    made = [r for r in rows if not r.get("no_crop")]
    print(f"{len(made)} fin crops from {len(fish)} fish -> {args.out}")
    for fin in FINS:
        n = sum(r["fin"] == fin for r in made)
        clipped = sum(r["fin"] == fin and r["outside_crop"] for r in made)
        none = sum(r["fin"] == fin for r in rows if r.get("no_crop"))
        print(f"  {fin:9} {n:3}" + (f"  ({clipped} reach outside the crop)" if clipped else "")
              + (f"  ({none} with no predicted base/tip)" if none else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
