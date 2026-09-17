#!/usr/bin/env python3
"""Train a fin outliner on the lab's own tracings, and measure it on fish it never saw.

Reads the crops written by ``build_fin_dataset.py``. One model serves all four fins:
it is given the crop and a map marking the fin's base and tip landmarks, and returns
the fin's mask. Sharing it across fins quadruples what it learns from -- about twenty
tracings per fin exist.

    .venv-train/bin/python scripts/train_fin_segmenter.py --folds 5      # cross-validate
    .venv-train/bin/python scripts/train_fin_segmenter.py --final        # train on all

**Cross-validation** splits by fish, so no fin of a held-out fish is seen in
training, and reports the error in fin *area* -- what the traits use -- per fin,
against the hand tracing. The comparison that matters is Segment Anything, untrained,
at a median 8% (anal) to 35% (dorsal) against dense re-tracings.

The base and tip it is given during training are jittered, because when it is used
they come from the keypoint model, not a person.

Encoder: ResNet-50 (GroupNorm, ImageNet), the backbone DeepLabCut already uses, so
nothing is downloaded. The stem and first stage stay frozen: with ~80 examples,
retraining them would learn the photographs rather than fins.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")     # the encoder weights are already cached

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
FINS = ("pectoral", "dorsal", "pelvic", "anal")
ENCODER = "resnet50_gn.a1h_in1k"
#: How far the base and tip are moved in training, in crop px (448 px ~ 0.34 SL).
#: The keypoint model's held-out error on fin landmarks is ~1 mm, roughly 10 px here.
PROMPT_JITTER = 10.0
PROMPT_SIGMA = 7.0


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class FinSegmenter(nn.Module):
    """ResNet features, a small top-down decoder, one mask logit per pixel."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(ENCODER, features_only=True, pretrained=pretrained,
                                         in_chans=4, out_indices=(1, 2, 3, 4))
        chans = self.encoder.feature_info.channels()
        self.lateral = nn.ModuleList(nn.Conv2d(c, 96, 1) for c in chans)
        self.smooth = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.ReLU(),
                                    nn.Conv2d(96, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU())
        self.head = nn.Conv2d(64, 1, 1)
        for name, p in self.encoder.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1")):
                p.requires_grad = False

    def forward(self, x):
        feats = self.encoder(x)
        y = self.lateral[-1](feats[-1])
        for f, lat in zip(reversed(feats[:-1]), reversed(self.lateral[:-1])):
            y = F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False) + lat(f)
        y = self.head(self.smooth(y))
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load(data: Path) -> list[dict]:
    rows = json.loads((data / "index.json").read_text())
    for r in rows:
        z = np.load(data / f"{r['name']}.npz")
        r["image"], r["mask"], r["base_tip"] = z["image"], z["mask"], z["base_tip"]
    return rows


def prompt_map(base_tip: np.ndarray, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros((h, w), np.float32)
    for x, y in base_tip:
        out = np.maximum(out, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * PROMPT_SIGMA ** 2)))
    return out


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def tensorise(image: np.ndarray, base_tip: np.ndarray) -> torch.Tensor:
    h, w = image.shape[:2]
    rgb = (image.astype(np.float32) / 255.0 - MEAN) / STD
    x = np.concatenate([rgb, prompt_map(base_tip, h, w)[..., None] * 2 - 1], axis=2)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def augment(r: dict, rng: random.Random):
    """A plausible other photograph of the same fin: turned, scaled, lit differently,
    with the landmarks placed a little off. Never mirrored -- the fish face left."""
    img, mask, bt = r["image"], r["mask"], r["base_tip"].copy()
    h, w = img.shape[:2]
    angle, scale = rng.uniform(-12, 12), rng.uniform(0.88, 1.12)
    tx, ty = rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.08, 0.08) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    bt = (np.c_[bt, np.ones(2)] @ M.T).astype(np.float32)
    bt += np.array([[rng.gauss(0, PROMPT_JITTER / 2) for _ in range(2)] for _ in range(2)], np.float32)
    img = img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
    grey = img.mean(axis=2, keepdims=True)
    img = grey + (img - grey) * rng.uniform(0.7, 1.3)
    return np.clip(img, 0, 255).astype(np.uint8), mask, bt


def batches(rows, size, rng, train: bool):
    order = list(range(len(rows)))
    if train:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        xs, ys = [], []
        for j in order[i:i + size]:
            r = rows[j]
            img, mask, bt = augment(r, rng) if train else (r["image"], r["mask"], r["base_tip"])
            xs.append(tensorise(img, bt))
            ys.append(torch.from_numpy(mask.astype(np.float32))[None])
        yield torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# training and measurement
# ---------------------------------------------------------------------------

def dice_bce(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


def train(rows, epochs: int, device: str, seed: int, log=print) -> FinSegmenter:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = FinSegmenter().to(device)
    enc = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    dec = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": 1e-4}, {"params": dec, "lr": 5e-4}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in batches(rows, 4, rng, train=True):
            x, y = x.to(device), y.to(device)
            loss = dice_bce(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(x), n + len(x)
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"    epoch {ep:3d}  loss {total / n:.3f}")
    return model


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(np.uint8)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == keep).astype(np.uint8)


@torch.no_grad()
def predict_masks(model, rows, device: str) -> list[np.ndarray]:
    model.eval()
    out = []
    for r in rows:
        x = tensorise(r["image"], r["base_tip"])[None].to(device)
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        out.append(largest_component(prob > 0.5))
    return out


def measure(rows, masks) -> list[dict]:
    res = []
    for r, m in zip(rows, masks):
        sx, sy = r["scale"]
        area_pred = float(m.sum()) / (sx * sy)                 # back to image px
        # Compare like with like: the hand polygon rasterised the same way.
        truth = r["mask"].astype(bool)
        area_true_raster = float(truth.sum()) / (sx * sy)
        inter = float((truth & m.astype(bool)).sum())
        union = float((truth | m.astype(bool)).sum())
        res.append({"name": r["name"], "fish_id": r["fish_id"], "fin": r["fin"],
                    "area_err_pct": round(100 * (area_pred - area_true_raster) / area_true_raster, 1),
                    "iou": round(inter / union, 3) if union else 0.0})
    return res


def summarise(results: list[dict]) -> dict:
    out = {}
    for fin in FINS:
        e = [r["area_err_pct"] for r in results if r["fin"] == fin]
        iou = [r["iou"] for r in results if r["fin"] == fin]
        if not e:
            continue
        out[fin] = {"n": len(e), "median_abs_err_pct": round(float(np.median(np.abs(e))), 1),
                    "mean_err_pct": round(float(np.mean(e)), 1),
                    "worst_abs_err_pct": round(float(np.max(np.abs(e))), 1),
                    "median_iou": round(float(np.median(iou)), 3)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="train_fin_segmenter")
    ap.add_argument("--data", type=Path, default=_ROOT / "fin_seg")
    ap.add_argument("--out", type=Path, default=_ROOT / "fin_seg_runs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=None, help="run one fold (to time it)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--final", action="store_true", help="train on every fish and save the model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rows = load(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    fish = sorted({r["fish_id"] for r in rows})
    print(f"{len(rows)} fin crops from {len(fish)} fish, device {args.device}")

    if args.final:
        t0 = time.time()
        model = train(rows, args.epochs, args.device, args.seed)
        path = args.out / "fin_segmenter.pt"
        torch.save({"state_dict": model.state_dict(), "encoder": ENCODER, "fins": FINS,
                    "crop": {"half_w_sl": 0.17, "half_h_sl": 0.085, "w": 448, "h": 224},
                    "trained_on": fish, "epochs": args.epochs}, path)
        print(f"saved {path} ({time.time() - t0:.0f}s)")
        return 0

    rng = random.Random(args.seed)
    shuffled = fish[:]
    rng.shuffle(shuffled)
    folds = [shuffled[i::args.folds] for i in range(args.folds)]
    results = []
    for k, held in enumerate(folds):
        if args.only_fold is not None and k != args.only_fold:
            continue
        t0 = time.time()
        tr = [r for r in rows if r["fish_id"] not in held]
        te = [r for r in rows if r["fish_id"] in held]
        print(f"fold {k}: train {len(tr)} crops, test {len(te)} crops from {len(held)} fish")
        model = train(tr, args.epochs, args.device, args.seed + k)
        masks = predict_masks(model, te, args.device)
        fold_res = measure(te, masks)
        # Keep what it drew, so a bad number can be looked at rather than argued about.
        pred_dir = args.out / "predicted"
        pred_dir.mkdir(exist_ok=True)
        for r, m in zip(te, masks):
            cv2.imwrite(str(pred_dir / f"{r['name']}.png"), m * 255)
        for r in fold_res:
            r["fold"] = k
        results += fold_res
        print(f"  fold {k} done in {time.time() - t0:.0f}s: "
              + ", ".join(f"{r['fin']} {r['area_err_pct']:+.0f}%" for r in fold_res))
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()
    summary = summarise(results)
    (args.out / "cv_results.json").write_text(json.dumps({"summary": summary, "results": results,
                                                          "epochs": args.epochs, "folds": args.folds},
                                                         indent=1))
    print("\nheld-out fin area error (vs hand tracing):")
    for fin, s in summary.items():
        print(f"  {fin:9} n={s['n']:2}  median |err| {s['median_abs_err_pct']:5.1f}%  "
              f"mean {s['mean_err_pct']:+5.1f}%  worst {s['worst_abs_err_pct']:5.1f}%  IoU {s['median_iou']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
