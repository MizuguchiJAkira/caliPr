"""Test whether SAM can replace hand-traced polygons, prompted by keypoints.

The planned auto stack is DeepLabCut for keypoints + SAM for the 5 polygons,
with SAM prompted by the predicted keypoints. SAM is zero-shot, so this needs
no training data — and the 35 hand-traced specimens are a ready-made benchmark.

What is measured
----------------
Mask IoU is the obvious metric but not the decisive one: the pipeline never
consumes a mask, it consumes an **area in mm²** (Bs, CFs, PFs, DFs, PlFs, AFs).
A mask can score a mediocre IoU and still give the right area, or score well and
be biased. So this reports both, and treats the **area error** as the verdict.

Prompting
---------
Each polygon is prompted with the anatomical keypoints that bound it — the same
points DLC will predict — so this measures the real auto-mode configuration,
not an idealised one. ``body_plus_caudal`` gets several points along the body
axis because it is large and elongated.

Usage::

    python scripts/eval_sam_polygons.py --limit 10
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

from fish_morpho.measurement_engine import shoelace_area  # noqa: E402

# Which keypoints prompt which polygon (all positive points).
PROMPTS: dict[str, list[str]] = {
    "pectoral": ["pectoral_insertion_upper", "pectoral_ray_tip"],
    "dorsal": ["dorsal_base_center", "dorsal_tip"],
    "pelvic": ["pelvic_base_center", "pelvic_tip"],
    "anal": ["anal_base_center", "anal_tip"],
    "body_plus_caudal": [
        "premaxilla_tip", "operculum_posterior", "pectoral_insertion_upper",
        "peduncle_narrowest_dorsal", "peduncle_narrowest_ventral", "caudal_base",
    ],
}


def px_per_mm(sidecar: dict) -> float | None:
    cal = (sidecar.get("lateral") or {}).get("calibration")
    if not cal:
        return None
    if cal.get("mode") == "ticks":
        return float(cal["px_per_mm"])
    a, b = cal["point_a"], cal["point_b"]
    known = float(cal.get("known_mm") or 0)
    return math.hypot(b[0] - a[0], b[1] - a[1]) / known if known > 0 else None


def mask_from_polygon(poly, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval_sam_polygons")
    ap.add_argument("--sidecars", type=Path, default=_ROOT / "data/cornell/sidecars")
    ap.add_argument("--images", type=Path, default=_ROOT / "data/cornell/lateral")
    ap.add_argument("--model", default="facebook/sam-vit-base")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", type=Path, help="Write an overlay image per specimen here.")
    args = ap.parse_args(argv)

    import torch
    from transformers import SamModel, SamProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {args.model} on {device} …")
    model = SamModel.from_pretrained(args.model).to(device).eval()
    processor = SamProcessor.from_pretrained(args.model)

    files = sorted(args.sidecars.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    results: dict[str, list[tuple[float, float]]] = {}   # poly -> [(iou, area_err_pct)]
    for path in files:
        sc = json.loads(path.read_text())
        fid = sc["fish_id"]
        img_path = args.images / f"{fid}_L.JPEG"
        if not img_path.is_file():
            continue
        lat = sc.get("lateral") or {}
        kps, polys = lat.get("keypoints") or {}, lat.get("polygons") or {}
        ppm_full = px_per_mm(sc)
        if not ppm_full:
            continue

        bgr = cv2.imread(str(img_path))
        small = cv2.resize(bgr, None, fx=args.scale, fy=args.scale,
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        h, w = small.shape[:2]
        ppm = ppm_full * args.scale

        overlay = small.copy() if args.dump else None
        for name, prompt_kps in PROMPTS.items():
            gt = polys.get(name)
            pts = [kps[k] for k in prompt_kps if k in kps]
            if not gt or len(pts) < 2:
                continue
            pts_s = [[p[0] * args.scale, p[1] * args.scale] for p in pts]

            # A bare point prompt makes SAM return object-level granularity —
            # for a fin it segments the whole fish. Fins therefore get a box
            # prompt built from their own two landmarks (padded), which pins
            # SAM to that region; the body keeps points, where object-level is
            # exactly what we want.
            if name == "body_plus_caudal":
                inputs = processor(rgb, input_points=[[pts_s]], return_tensors="pt")
            else:
                xs = [p[0] for p in pts_s]; ys = [p[1] for p in pts_s]
                pad = 0.35 * math.hypot(max(xs) - min(xs), max(ys) - min(ys)) + 6
                box = [max(0, min(xs) - pad), max(0, min(ys) - pad),
                       min(w, max(xs) + pad), min(h, max(ys) + pad)]
                inputs = processor(rgb, input_boxes=[[box]],
                                   input_points=[[pts_s]], return_tensors="pt")
            # the processor emits float64 point coords; MPS has no float64
            inputs = {k: (v.to(torch.float32) if getattr(v, "dtype", None) == torch.float64 else v)
                      for k, v in inputs.items()}
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, multimask_output=True)
            masks = processor.image_processor.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu())[0][0]
            scores = out.iou_scores.cpu()[0][0]
            mask = masks[int(scores.argmax())].numpy().astype(np.uint8)

            gt_poly = [[p[0] * args.scale, p[1] * args.scale] for p in gt]
            gt_mask = mask_from_polygon(gt_poly, (h, w))
            inter = int((mask & gt_mask).sum())
            union = int((mask | gt_mask).sum())
            iou = inter / union if union else 0.0

            sam_mm2 = float(mask.sum()) / (ppm ** 2)
            gt_mm2 = shoelace_area([tuple(p) for p in gt_poly]) / (ppm ** 2)
            err = (sam_mm2 - gt_mm2) / gt_mm2 * 100 if gt_mm2 else float("nan")
            results.setdefault(name, []).append((iou, err))

            if overlay is not None:
                cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cs, -1, (60, 220, 60), 2)
                cv2.polylines(overlay, [np.array(gt_poly, np.int32)], True, (255, 170, 50), 2)
        if overlay is not None:
            args.dump.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.dump / f"{fid}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {fid.replace('Salvelinus_fontinalis_','')}", flush=True)

    print(f"\n{'polygon':20} {'n':>3} {'median IoU':>11} {'median |area err|':>18} {'bias':>8}")
    for name in PROMPTS:
        rs = results.get(name)
        if not rs:
            continue
        ious = sorted(r[0] for r in rs)
        errs = [r[1] for r in rs]
        med_iou = ious[len(ious) // 2]
        abs_err = sorted(abs(e) for e in errs)
        med_abs = abs_err[len(abs_err) // 2]
        bias = sum(errs) / len(errs)
        print(f"{name:20} {len(rs):3d} {med_iou:11.3f} {med_abs:17.1f}% {bias:+7.1f}%")
    print("\nverdict basis: the pipeline consumes AREA, not masks — "
          "hand tracing agrees with calipers to ~1.3%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
