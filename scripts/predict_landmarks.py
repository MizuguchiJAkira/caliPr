"""Place landmarks on unlabelled photographs with the trained keypoint model.

    python scripts/predict_landmarks.py --dataset data/cornell --overlays

Writes one sidecar per photograph into ``<dataset>/sidecars_auto/`` — deliberately
*not* the directory holding hand labels — plus, with ``--overlays``, an annotated
JPEG per fish showing where the model put each point and how sure it was.

Three properties this has to have, in order of how bad it is to get them wrong:

**A prediction must never be mistaken for a hand label.** Every sidecar written
here carries ``metadata.source = "predicted"``. ``build_dlc_dataset.py`` skips
those, because a model trained on its own output learns its own mistakes and the
error curve looks like progress while it happens. The separate output directory
is a second line of defence, not the main one.

**Predictions must not overwrite hand labels.** They go to a different directory
and even there an existing file is left alone unless ``--overwrite`` is passed.
Fifty-five hand-labelled specimens took hours; nothing automatic should be able
to spend them.

**The model's uncertainty has to survive into the output.** Each keypoint records
its likelihood, and anything under ``--min-confidence`` is written to a
``low_confidence`` list rather than being silently dropped or silently trusted.
On a specimen whose fins are folded flat — which alcohol preservation does often
— the fin landmarks are the ones that go wrong, and they are also the ones the
model marks down. That correspondence is the useful part: it means a human can be
pointed at the three points worth checking instead of all twenty-three.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

#: Below this likelihood a point is reported as low-confidence. 0.6 is where the
#: separation sat on the held-out set; it is a reporting threshold, not a claim
#: about calibration, and likelihoods are not comparable between landmarks.
DEFAULT_MIN_CONFIDENCE = 0.6


def find_project(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    hits = sorted((_ROOT / "dlc_project").glob("*/config.yaml"))
    if not hits:
        raise SystemExit("no DLC project under dlc_project/ — pass --project")
    return hits[-1].parent


def find_config_and_snapshot(project: Path, snapshot: Path | None):
    cfgs = sorted(project.glob(
        "dlc-models-pytorch/iteration-*/*/train/pytorch_config.yaml"))
    if not cfgs:
        raise SystemExit(f"no pytorch_config.yaml under {project}")
    cfg = cfgs[-1]
    if snapshot is not None:
        if not snapshot.is_file():
            raise SystemExit(f"snapshot not found: {snapshot}")
        return cfg, snapshot
    # "best" is chosen on the validation metric; prefer it over the last epoch,
    # which is only the point training happened to stop at.
    best = sorted(cfg.parent.glob("snapshot-best-*.pt"))
    last = sorted(cfg.parent.glob("snapshot-*.pt"))
    if best:
        return cfg, best[-1]
    if last:
        return cfg, last[-1]
    raise SystemExit(f"no snapshot .pt under {cfg.parent}")


def training_scale(project: Path, override: float | None) -> float:
    """The scale the model was trained at, which its outputs are in.

    Getting this wrong scales every coordinate uniformly and the landmarks still
    look plausibly fish-shaped, just in the wrong place — so it is read from the
    dataset that produced the model rather than assumed.
    """
    if override is not None:
        return override
    for cand in (_ROOT / "dlc" / "split.json", project / "split.json"):
        if cand.is_file():
            try:
                s = json.loads(cand.read_text()).get("scale")
                if s:
                    return float(s)
            except Exception:
                pass
    print("  WARNING: no split.json found; assuming the 0.25 default. "
          "Pass --scale if the model was trained at another resolution.")
    return 0.25


def collect_images(args) -> tuple[list[Path], Path | None]:
    if args.images:
        p = args.images
        if p.is_file():
            return [p], None
        return sorted(q for q in p.iterdir()
                      if q.suffix.lower() in (".jpg", ".jpeg", ".png")), None
    base = args.dataset
    lateral = base / "lateral"
    if not lateral.is_dir():
        raise SystemExit(f"{lateral} does not exist")
    imgs = sorted(q for q in lateral.iterdir()
                  if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return imgs, base


def stem_of(path: Path) -> str:
    s = path.stem
    return s[:-2] if s.endswith("_L") else s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="predict_landmarks")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path,
                     help="Dataset directory holding lateral/.")
    src.add_argument("--images", type=Path,
                     help="A single photograph, or a directory of them.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write sidecars. Defaults to "
                         "<dataset>/sidecars_auto — never the hand-label dir.")
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    ap.add_argument("--skip-labelled", type=Path, default=None,
                    help="A hand-label directory; specimens already in it are "
                         "skipped, so a batch run does not spend time on fish "
                         "somebody has already done properly.")
    ap.add_argument("--overlays", action="store_true",
                    help="Also write an annotated JPEG per specimen.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    import cv2
    import numpy as np
    import ruamel.yaml
    from deeplabcut.pose_estimation_pytorch import apis

    project = find_project(args.project)
    cfg, snapshot = find_config_and_snapshot(project, args.snapshot)
    scale = training_scale(project, args.scale)

    yaml = ruamel.yaml.YAML()
    with open(cfg) as fh:
        conf = yaml.load(fh)
    names = list(conf["metadata"]["bodyparts"])

    images, base = collect_images(args)
    if args.skip_labelled and args.skip_labelled.is_dir():
        done = {p.stem for p in args.skip_labelled.glob("*.json")}
        before = len(images)
        images = [p for p in images if stem_of(p) not in done]
        if before != len(images):
            print(f"  skipping {before - len(images)} already hand-labelled")
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit("no images to predict on")

    out_dir = args.out or ((base / "sidecars_auto") if base
                           else _ROOT / "results" / "auto" / "sidecars")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model    {snapshot.name}  (trained at scale {scale})")
    print(f"images   {len(images)}")
    print(f"sidecars {out_dir}")

    # The model sees images at the scale it was trained on, so feed it that and
    # map the coordinates back afterwards.
    tmp = Path(tempfile.mkdtemp(prefix="calipr_predict_"))
    try:
        keep: dict[str, Path] = {}
        for p in images:
            im = cv2.imread(str(p))
            if im is None:
                print(f"  SKIP {p.name}: unreadable")
                continue
            small = cv2.resize(im, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            dst = tmp / f"{p.stem}.png"
            cv2.imwrite(str(dst), small)
            keep[dst.name] = p

        res = apis.analyze_image_folder(
            model_cfg=str(cfg), images=str(tmp), snapshot_path=str(snapshot),
            device=args.device, progress_bar=True)

        written = skipped = 0
        conf_all: list[float] = []
        overlay_dir = out_dir.parent / "overlays_auto"
        if args.overlays:
            overlay_dir.mkdir(parents=True, exist_ok=True)

        for key, pred in sorted(res.items()):
            src_img = keep.get(Path(key).name)
            if src_img is None:
                continue
            arr = np.asarray(pred["bodyparts"]).reshape(-1, 3)
            fid = stem_of(src_img)
            dest = out_dir / f"{fid}.json"
            if dest.exists() and not args.overwrite:
                print(f"  SKIP {fid}: sidecar exists (use --overwrite)")
                skipped += 1
                continue

            kps, confs, low = {}, {}, []
            for name, (x, y, c) in zip(names, arr):
                kps[name] = [round(float(x) / scale, 1), round(float(y) / scale, 1)]
                confs[name] = round(float(c), 3)
                conf_all.append(float(c))
                if c < args.min_confidence:
                    low.append(name)

            sidecar = {
                "fish_id": fid,
                "metadata": {
                    # Load-bearing: build_dlc_dataset.py refuses to train on this,
                    # and nothing should mistake it for a hand label.
                    "source": "predicted",
                    "model": snapshot.name,
                    "image": src_img.name,
                    "keypoint_confidence": confs,
                    "low_confidence": sorted(low),
                },
                "lateral": {
                    "keypoints": kps,
                    "calibration": {"mode": "none",
                                    "notes": "predicted; no scale reference read"},
                },
            }
            dest.write_text(json.dumps(sidecar, indent=2))
            written += 1

            if args.overlays:
                vis = cv2.imread(str(src_img))
                for name, (x, y) in kps.items():
                    ok = confs[name] >= args.min_confidence
                    col = (0, 200, 0) if ok else (0, 165, 255)
                    cv2.circle(vis, (int(x), int(y)), 16, col, -1)
                    label = f"{name} {confs[name]:.2f}"
                    for c2, th in (((255, 255, 255), 5), ((0, 0, 0), 2)):
                        cv2.putText(vis, label, (int(x) + 22, int(y) + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, c2, th)
                cv2.imwrite(str(overlay_dir / f"{fid}.jpg"),
                            cv2.resize(vis, None, fx=0.35, fy=0.35),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nwrote {written} sidecar(s)" + (f", skipped {skipped}" if skipped else ""))
    if args.overlays and written:
        print(f"overlays {overlay_dir}")
    if conf_all:
        arr = sorted(conf_all)
        med = arr[len(arr) // 2]
        lo = sum(1 for c in arr if c < args.min_confidence)
        print(f"confidence: median {med:.2f}, "
              f"{lo}/{len(arr)} below {args.min_confidence}")
        print("Low-confidence points are where this model is usually wrong — "
              "check those, not all of them.")
    print("\nThese are PREDICTIONS. They are excluded from training and must be "
          "corrected by hand before they are treated as data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
