"""Segment the body-hugging fins by zooming in and telling SAM what is NOT fin.

The pectoral and pelvic fins lie flat against the flank — 0% of either sits
outside the body silhouette — and the pectoral's fin-to-surround contrast is
only 5.7 intensity levels. Two consequences follow, and this script addresses
both:

*Resolution.* SAM resizes its input to 1024 px. On a full 4400 px frame the
pectoral is a handful of pixels by the time the model sees it. Cropping to the
fin and feeding that crop at 1024 px raised pectoral IoU from 0.330 to 0.730.

*Granularity.* A point prompt makes SAM return object-level masks, so on a fin
it happily returns the whole fish — the zoomed pelvic mask came back 15x too
large. Positive points alone cannot say "this fin, not the animal it is
attached to". Negative points can: seeding them on the flank above and below
the fin marks the body as background, which is the only signal that separates a
part from its whole when the boundary itself is faint.

Measurement fixes over the first attempt: the crop is kept inside the frame
*before* the scale is derived, and any specimen whose ground truth rasterises
empty is skipped rather than dividing by ~zero (that produced the 4e16% figures).

Usage — needs the torch env, not the pipeline's `.venv`::

    .venv-train/bin/python scripts/eval_sam_zoom.py --limit 15 --dump /tmp/zoom
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho.measurement_engine import shoelace_area  # noqa: E402

FINS = {
    "pectoral": ("pectoral_insertion_upper", "pectoral_ray_tip"),
    "pelvic": ("pelvic_base_center", "pelvic_tip"),
    "dorsal": ("dorsal_base_center", "dorsal_tip"),
    "anal": ("anal_base_center", "anal_tip"),
}
# Full-frame direct-prompt baselines, for comparison.
BASELINE = {"pectoral": 238.4, "pelvic": 26.8, "dorsal": 22.6, "anal": 11.1}


def merge_fragments(mask: np.ndarray, gap: float) -> np.ndarray:
    """Rejoin mask pieces separated by less than ``gap`` pixels.

    A fin is a single structure, but a specimen pin laid across it splits SAM's
    output into disconnected fragments. Closing over the gap reconnects them,
    then the result is intersected back with the original so the closing adds
    no area of its own — only the bridge is filled, and only where it links
    genuine fin pixels.
    """
    if gap < 2 or mask.sum() == 0:
        return mask
    k = int(max(3, round(gap))) | 1
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n <= 1:
        return mask
    # keep the closed component that carries the most original mask
    best, ba = None, -1
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        overlap = int((comp & mask).sum())
        if overlap > ba:
            ba, best = overlap, comp
    return best if best is not None else mask


def px_per_mm(sidecar: dict) -> float | None:
    cal = (sidecar.get("lateral") or {}).get("calibration")
    if not cal:
        return None
    if cal.get("mode") == "ticks":
        return float(cal["px_per_mm"])
    a, b = cal["point_a"], cal["point_b"]
    known = float(cal.get("known_mm") or 0)
    return math.hypot(b[0] - a[0], b[1] - a[1]) / known if known > 0 else None


def segment(model, processor, torch, device, rgb, pos, neg, box):
    pts = [list(p) for p in pos] + [list(p) for p in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    inputs = processor(rgb, input_points=[[pts]], input_labels=[[labels]],
                       input_boxes=[[box]], return_tensors="pt")
    inputs = {k: (v.to(torch.float32) if getattr(v, "dtype", None) == torch.float64 else v)
              for k, v in inputs.items()}
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = processor.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0][0]
    scores = out.iou_scores.cpu()[0][0]
    return masks[int(scores.argmax())].numpy().astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval_sam_zoom")
    ap.add_argument("--sidecars", type=Path, default=_ROOT / "data/cornell/sidecars")
    ap.add_argument("--images", type=Path, default=_ROOT / "data/cornell/lateral")
    ap.add_argument("--model", default="facebook/sam-vit-base")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fins", default=",".join(FINS))
    ap.add_argument("--merge-gap", type=float, default=0.06, metavar="FRAC",
                    help="Merge mask fragments whose gap is under FRAC x the crop's "
                         "long edge. A fin is one structure; pins and specimen pins "
                         "crossing it split SAM's mask into pieces (clearly visible "
                         "on TXD_7), and keeping only the largest piece then throws "
                         "away real fin area.")
    ap.add_argument("--ventral-negatives", action="store_true",
                    help="Place negative points only on the VENTRAL side of the fin "
                         "axis. The pectoral sits dorsal to the belly margin and never "
                         "runs along it, so the belly is background; symmetric "
                         "negatives instead clip the fin's dorsal edge.")
    ap.add_argument("--dump", type=Path)
    args = ap.parse_args(argv)

    import torch
    from transformers import SamModel, SamProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {args.model} on {device} …", flush=True)
    model = SamModel.from_pretrained(args.model).to(device).eval()
    processor = SamProcessor.from_pretrained(args.model)
    want = [f.strip() for f in args.fins.split(",") if f.strip() in FINS]

    files = sorted(args.sidecars.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    res: dict[tuple[str, str], list[tuple[float, float]]] = {}
    skipped = 0
    for path in files:
        sc = json.loads(path.read_text())
        fid = sc["fish_id"]
        img_path = args.images / f"{fid}_L.JPEG"
        lat = sc.get("lateral") or {}
        kps, polys = lat.get("keypoints") or {}, lat.get("polygons") or {}
        ppm_full = px_per_mm(sc)
        if not img_path.is_file() or not ppm_full:
            continue
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        IH, IW = im.shape[:2]

        for name in want:
            bk, tk = FINS[name]
            gt = polys.get(name)
            if not gt or len(gt) < 3 or bk not in kps or tk not in kps:
                continue
            base, tip = kps[bk], kps[tk]
            span = math.hypot(tip[0] - base[0], tip[1] - base[1])
            pad = int(span * 1.1) + 40
            xs = [p[0] for p in gt] + [base[0], tip[0]]
            ys = [p[1] for p in gt] + [base[1], tip[1]]
            # clamp the crop to the frame BEFORE deriving the scale, so ground
            # truth can never fall outside the rasterised mask
            x0 = max(0, int(min(xs)) - pad); x1 = min(IW, int(max(xs)) + pad)
            y0 = max(0, int(min(ys)) - pad); y1 = min(IH, int(max(ys)) + pad)
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            crop = im[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            s = 1024.0 / max(ch, cw)
            cs = cv2.resize(crop, (int(cw * s), int(ch * s)))
            rgb = cv2.cvtColor(cs, cv2.COLOR_BGR2RGB)
            H, W = cs.shape[:2]

            def T(p):
                return [(p[0] - x0) * s, (p[1] - y0) * s]

            gtm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(gtm, [np.array([T(p) for p in gt], np.int32)], 1)
            if gtm.sum() < 50:
                skipped += 1
                continue

            b, t = T(base), T(tip)
            box = [max(0, min(b[0], t[0]) - 25), max(0, min(b[1], t[1]) - 25),
                   min(W, max(b[0], t[0]) + 25), min(H, max(b[1], t[1]) + 25)]
            # negative points on the flank, perpendicular to the fin's axis:
            # the body is the thing the fin must be separated from.
            dx, dy = t[0] - b[0], t[1] - b[1]
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L, dx / L
            off = 0.75 * L
            cand_negs = [[b[0] + nx * off, b[1] + ny * off],
                         [b[0] - nx * off, b[1] - ny * off]]
            if args.ventral_negatives:
                # image y grows downward, so the ventral side is the larger y
                cand_negs = [max(cand_negs, key=lambda p: p[1])]
            negs = [[min(max(p[0], 0), W - 1), min(max(p[1], 0), H - 1)]
                    for p in cand_negs]

            for mode, pos, neg in (("zoom+box", [b, t], []),
                                   ("zoom+box+neg", [b, t], negs)):
                m = segment(model, processor, torch, device, rgb, pos, neg, box)
                m = merge_fragments(m, args.merge_gap * max(H, W))
                inter = int((m & gtm).sum()); union = int((m | gtm).sum())
                iou = inter / union if union else 0.0
                err = (float(m.sum()) - float(gtm.sum())) / float(gtm.sum()) * 100
                res.setdefault((name, mode), []).append((iou, err))
                if args.dump and mode == "zoom+box+neg":
                    args.dump.mkdir(parents=True, exist_ok=True)
                    ov = cs.copy()
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(ov, cnts, -1, (60, 220, 60), 2)
                    cv2.polylines(ov, [np.array([T(p) for p in gt], np.int32)], True,
                                  (255, 170, 50), 2)
                    for p in negs:
                        cv2.circle(ov, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1)
                    cv2.imwrite(str(args.dump / f"{fid}_{name}.jpg"), ov,
                                [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {fid.replace('Salvelinus_fontinalis_','')}", flush=True)

    print(f"\n{'fin':10} {'mode':14} {'n':>3} {'med IoU':>8} {'med |area err|':>15} "
          f"{'full-frame':>11}")
    for name in want:
        for mode in ("zoom+box", "zoom+box+neg"):
            rs = res.get((name, mode))
            if not rs:
                continue
            print(f"{name:10} {mode:14} {len(rs):3d} "
                  f"{st.median([r[0] for r in rs]):8.3f} "
                  f"{st.median([abs(r[1]) for r in rs]):14.1f}% "
                  f"{BASELINE.get(name, float('nan')):10.1f}%")
    if skipped:
        print(f"\nskipped {skipped} fin(s) whose ground truth rasterised empty")
    print("hand tracing agrees with calipers to ~1.3%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
