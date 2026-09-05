"""Convert hand-labeled sidecars into a DeepLabCut project.

Builds a DLC 3.x project from ``data/cornell/sidecars/`` + ``data/cornell/lateral/``:
creates the project skeleton, copies the labeled images into a single video
folder, and writes ``CollectedData_<scorer>.h5/.csv`` in DLC's MultiIndex format.

Two details that matter for correctness:

*Missing keypoints stay missing.* A clipped snout means ``premaxilla_tip`` does
not exist in that frame. Writing a placeholder (0, or the frame edge) would
teach the model to predict the frame edge. DLC treats NaN as "unlabeled" and
skips it in the loss, so absent landmarks are written as NaN.

*The split is stratified by strain.* ASN/HRN/TXD differ in appearance, and an
unstratified random split can leave a strain out of training entirely, which
makes the held-out error meaningless.

Images are downscaled by ``--scale`` (default 0.25): the crops are ~4400x4000,
far larger than any keypoint backbone consumes, and DLC would otherwise spend
the whole run resizing. Coordinates are scaled to match.

Usage::

    python scripts/build_dlc_dataset.py --out dlc/ --test-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho.landmark_config import KEYPOINTS, View  # noqa: E402

LATERAL_KP = [k.name for k in KEYPOINTS if k.view == View.LATERAL]
SCORER = "jcalipr"
VIDEO = "cornell_lateral"


def find_image(images: Path, fid: str, recorded: str | None = None):
    """Locate a specimen's photograph without assuming one naming convention.

    The trout rig names its crops ``<fish_id>_L.JPEG``; the alewife series and
    anything a contributor photographs do not. Hardcoding the trout pattern
    silently skipped every specimen from every other dataset, which looks
    identical to "nothing is labelled yet".
    """
    if recorded:
        p = images / recorded
        if p.is_file():
            return p
    lookup = {p.name.lower(): p for p in images.iterdir() if p.is_file()}
    for stem in (fid, f"{fid}_L"):
        for ext in (".jpeg", ".jpg", ".png", ".tif", ".tiff"):
            hit = lookup.get(f"{stem}{ext}".lower())
            if hit is not None:
                return hit
    return None


def load_specimens(sidecars: Path, images: Path, group: str = ""):
    out = []
    for path in sorted(sidecars.glob("*.json")):
        data = json.loads(path.read_text())
        fid = data["fish_id"]
        img = find_image(images, fid, (data.get("metadata") or {}).get("image"))
        if img is None:
            print(f"  SKIP {fid}: no image in {images}")
            continue
        kps = (data.get("lateral") or {}).get("keypoints") or {}
        if not kps:
            print(f"  SKIP {fid}: no lateral keypoints")
            continue
        m = re.search(r"_([A-Z]{2,4})_\d+$", fid)
        strain = m.group(1) if m else "UNK"
        out.append({
            "fish_id": fid,
            "image": img,
            "keypoints": kps,
            # Pooling two datasets must stratify on the dataset too, or a split
            # can put every alewife in test and report a trout model's error.
            "strain": f"{group}:{strain}" if group else strain,
            "compromised": bool((data.get("metadata") or {}).get("exclude_traits")),
        })
    return out


def stratified_split(specs, test_frac, seed):
    """Hold out ``test_frac`` of each strain, so every strain is in both sets."""
    rng = random.Random(seed)
    train, test = [], []
    by_strain: dict[str, list] = {}
    for s in specs:
        by_strain.setdefault(s["strain"], []).append(s)
    if len(by_strain) == 1 and "UNK" in by_strain:
        # Nothing matched the strain pattern, so this is a plain random split.
        # Say so: a split that only looks stratified is worse than one that
        # admits it is not.
        print("  NOTE: no strain/group parsed from any fish_id — "
              "splitting at random, so held-out error is not group-balanced")
    for strain in sorted(by_strain):
        group = sorted(by_strain[strain], key=lambda s: s["fish_id"])
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_frac))
        test += group[:n_test]
        train += group[n_test:]
    return train, test


def build(out_dir: Path, sources, scale: float,
          test_frac: float, seed: int) -> None:
    specs = []
    seen: dict[str, str] = {}
    for sidecars, images, group in sources:
        if len(sources) > 1:
            print(f"  {group}: {sidecars}")
        for spec in load_specimens(sidecars, images, group if len(sources) > 1 else ""):
            # Two datasets can hold the same fish_id; one would overwrite the
            # other's frame and silently drop a specimen from training.
            if spec["fish_id"] in seen:
                print(f"  SKIP {spec['fish_id']}: id already taken by "
                      f"{seen[spec['fish_id']]}")
                continue
            seen[spec["fish_id"]] = group
            specs.append(spec)
    if not specs:
        raise SystemExit("No labeled specimens found.")
    train, test = stratified_split(specs, test_frac, seed)

    labeled = out_dir / "labeled-data" / VIDEO
    labeled.mkdir(parents=True, exist_ok=True)

    rows, index = [], []
    for spec in sorted(specs, key=lambda s: s["fish_id"]):
        im = cv2.imread(str(spec["image"]))
        if im is None:
            print(f"  SKIP {spec['fish_id']}: unreadable image")
            continue
        small = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        name = f"{spec['fish_id']}.png"
        cv2.imwrite(str(labeled / name), small)

        row = []
        for kp in LATERAL_KP:
            p = spec["keypoints"].get(kp)
            # absent landmark -> NaN, never a placeholder (see module docstring)
            row += [np.nan, np.nan] if p is None else [p[0] * scale, p[1] * scale]
        rows.append(row)
        index.append(("labeled-data", VIDEO, name))

    columns = pd.MultiIndex.from_product(
        [[SCORER], LATERAL_KP, ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(rows, columns=columns,
                      index=pd.MultiIndex.from_tuples(index))
    df.to_hdf(labeled / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")
    df.to_csv(labeled / f"CollectedData_{SCORER}.csv")

    split = {
        "train": sorted(s["fish_id"] for s in train),
        "test": sorted(s["fish_id"] for s in test),
        "scale": scale,
        "seed": seed,
    }
    (out_dir / "split.json").write_text(json.dumps(split, indent=2))

    n_nan = int(df.isna().sum().sum() // 2)
    print(f"\nwrote {len(rows)} frames to {labeled}")
    print(f"  keypoint columns : {len(LATERAL_KP)}")
    print(f"  missing landmarks: {n_nan} (kept as NaN)")
    print(f"  train/test       : {len(train)}/{len(test)}")
    for strain in sorted({s['strain'] for s in specs}):
        tr = sum(1 for s in train if s["strain"] == strain)
        te = sum(1 for s in test if s["strain"] == strain)
        print(f"    {strain}: {tr} train / {te} test")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_dlc_dataset")
    ap.add_argument("--out", type=Path, default=_ROOT / "dlc")
    ap.add_argument("--sidecars", type=Path, default=None)
    ap.add_argument("--images", type=Path, default=None)
    ap.add_argument("--dataset", action="append", default=[], metavar="DIR",
                    help="A dataset directory holding sidecars/ and lateral/. "
                         "Repeatable: pooling every fish photographed the same "
                         "way gives the shared anatomy more to learn from, and "
                         "the split stratifies on dataset so one cannot end up "
                         "entirely in test.")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)
    if args.dataset:
        sources = [(Path(d) / "sidecars", Path(d) / "lateral", Path(d).name)
                   for d in args.dataset]
    elif args.sidecars and args.images:
        sources = [(args.sidecars, args.images, args.sidecars.parent.name)]
    else:
        sources = [(_ROOT / "data/cornell/sidecars",
                    _ROOT / "data/cornell/lateral", "cornell")]
    for sc, im, _ in sources:
        if not sc.is_dir() or not im.is_dir():
            raise SystemExit(f"missing {sc} or {im}")
    build(args.out, sources, args.scale, args.test_frac, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
