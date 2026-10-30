"""Build one training set from every labelled study, on the shared landmark list.

    python scripts/build_union_dataset.py --dataset data/cornell --dataset data/alewife \\
        --dataset data/notropis --dataset data/CUMV_63033 --dataset data/CUMV_24306

Two things differ from ``build_dlc_dataset.py``, which builds one scheme's model:

*Landmarks from different schemes train one model.* Each study's points are put
under the union names (``fish_morpho.landmark_union``), so a trout's eye and an
alewife's orbit teach the same output. A landmark a study did not collect is NaN,
which DeepLabCut leaves out of the loss.

*Each fish is shown twice, at the two scales prediction uses.* The old model saw
every photograph shrunk by one fixed factor, so a fish photographed from farther
away (the TEM series, 18 px/mm against 25-29) was a size it had never seen, and its
points went to pins and cards. Prediction now runs in two passes, and training
matches both:

- ``<fish>__whole``: the whole photograph, longest side 1024 px -- the first pass,
  which finds the fish among everything else in the frame. Its brightness and
  contrast are varied, since photographs from different rigs differ in both.
- ``<fish>__fish``: the fish cut out and scaled to a standard length -- the second
  pass, which places the points (``landmark_union.fish_frame``).

Both images of a fish go to the same side of the split, which is stratified by
study and strain as before.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import build_dlc_dataset as bdd  # noqa: E402
import preprocess_cornell as pc  # noqa: E402
from fish_morpho import landmark_union as lu  # noqa: E402

SCORER = "jcalipr"
VIDEO = "fish_union"


def photometric(im: np.ndarray, rng: random.Random) -> np.ndarray:
    """The same photograph a little darker or lighter, flatter or harsher."""
    gamma = rng.uniform(0.65, 1.5)
    contrast = rng.uniform(0.75, 1.3)
    shift = rng.uniform(-18, 18)
    x = (im.astype(np.float32) / 255.0) ** gamma * 255.0
    x = (x - 128.0) * contrast + 128.0 + shift
    return np.clip(x, 0, 255).astype(np.uint8)


def frames_for(spec: dict, rng: random.Random):
    """Yield (suffix, image, union keypoints in that image's pixels) for one fish."""
    im = cv2.imread(str(spec["image"]))
    if im is None:
        return
    ox = oy = 0
    if spec.get("crop"):
        x0, y0, x1, y1 = spec["crop"]
        im, ox, oy = im[y0:y1, x0:x1], x0, y0
    kps = {n: (float(p[0]) - ox, float(p[1]) - oy) for n, p in lu.to_union(spec["keypoints"]).items()}
    if len(kps) < 3:
        return
    h, w = im.shape[:2]

    s = lu.WHOLE_SIDE / max(h, w)
    whole = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    yield "whole", photometric(whole, rng), {n: (x * s, y * s) for n, (x, y) in kps.items()}

    frame = lu.fish_frame(list(kps.values()))
    if frame is None:
        return
    crop, to_crop = lu.cut_frame(im, frame)
    yield "fish", crop, {n: to_crop(x, y) for n, (x, y) in kps.items()}


def keep_split(usable, previous: dict, test_frac: float, seed: int):
    """The previous split for the fish it had, and a stratified one for the rest.

    Adding fish must not move fish already held out into training, or the models
    before and after can no longer be scored on the same fish. The new fish of each
    group are split so the group's held-out share stays near ``test_frac``.
    """
    was_test = set(previous.get("fish_test") or ())
    was = was_test | set(previous.get("fish_train") or ())
    old = [s for s in usable if s["fish_id"] in was]
    new = [s for s in usable if s["fish_id"] not in was]
    train = [s for s in old if s["fish_id"] not in was_test]
    test = [s for s in old if s["fish_id"] in was_test]
    rng = random.Random(seed)
    by_group: dict[str, list] = {}
    for s in new:
        by_group.setdefault(s["strain"], []).append(s)
    for g in sorted(by_group):
        group = sorted(by_group[g], key=lambda s: s["fish_id"])
        rng.shuffle(group)
        total = len(group) + sum(1 for s in old if s["strain"] == g)
        held = sum(1 for s in test if s["strain"] == g)
        n_test = max(0, min(len(group), round(total * test_frac) - held))
        test += group[:n_test]
        train += group[n_test:]
    print(f"  kept the split of {len(old)} fish; split {len(new)} new ones")
    return train, test


def build(out_dir: Path, sources, test_frac: float, seed: int,
          previous: dict | None = None, exclude=()) -> None:
    bdd.use_view("lateral")
    specs, seen = [], {}
    for sidecars, images, group in sources:
        print(f"  {group}: {sidecars}")
        for spec in bdd.load_specimens(sidecars, images, group):
            if spec["fish_id"] in exclude:
                print(f"  SKIP {spec['fish_id']}: excluded")
                continue
            if spec["fish_id"] in seen:
                print(f"  SKIP {spec['fish_id']}: id already taken by {seen[spec['fish_id']]}")
                continue
            seen[spec["fish_id"]] = group
            specs.append(spec)
    # Frame each single-photograph study the way prediction will.
    usable = []
    for spec in specs:
        spec["crop"] = None
        if spec.get("single_photo"):
            im = cv2.imread(str(spec["image"]))
            box = pc.view_frames(im)["lateral"] if im is not None else None
            # No seam found: the whole photograph, mirror and all, as prediction
            # will then get it -- which also shows the model that the head in the
            # mirror carries none of the side view's points.
            spec["crop"] = tuple(box) if box is not None else None
        usable.append(spec)
    if previous:
        train, test = keep_split(usable, previous, test_frac, seed)
    else:
        train, test = bdd.stratified_split(usable, test_frac, seed)
    side = {s["fish_id"]: "train" for s in train} | {s["fish_id"]: "test" for s in test}

    labeled = out_dir / "labeled-data" / VIDEO
    if labeled.exists():
        for old in labeled.iterdir():
            old.unlink()
    labeled.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows, index, frames = [], [], {"train": [], "test": []}
    for spec in sorted(usable, key=lambda s: s["fish_id"]):
        for suffix, img, pts in frames_for(spec, rng):
            name = f"{spec['fish_id']}__{suffix}"
            cv2.imwrite(str(labeled / f"{name}.png"), img)
            row = []
            for kp in lu.UNION:
                p = pts.get(kp)
                row += [np.nan, np.nan] if p is None else [p[0], p[1]]
            rows.append(row)
            index.append(("labeled-data", VIDEO, f"{name}.png"))
            frames[side[spec["fish_id"]]].append(name)

    columns = pd.MultiIndex.from_product([[SCORER], list(lu.UNION), ["x", "y"]],
                                         names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(rows, columns=columns, index=pd.MultiIndex.from_tuples(index))
    df.to_hdf(labeled / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")
    df.to_csv(labeled / f"CollectedData_{SCORER}.csv")
    split = {
        # train_dlc maps these names onto the annotation rows
        "train": sorted(frames["train"]), "test": sorted(frames["test"]),
        "fish_train": sorted(s["fish_id"] for s in train),
        "fish_test": sorted(s["fish_id"] for s in test),
        "groups": {s["fish_id"]: s["strain"] for s in usable},
        "seed": seed, "view": "union", "keypoints": list(lu.UNION),
        "framing": lu.FRAMING,
        # The median fish of the training labels, which prediction fits to its
        # first pass to frame the whole fish (landmark_union.fit_template).
        "template": lu.template(lu.to_union(s["keypoints"]) for s in train),
    }
    (out_dir / "split.json").write_text(json.dumps(split, indent=2))
    print(f"\nwrote {len(rows)} frames ({len(usable)} fish) to {labeled}")
    print(f"  landmarks: {len(lu.UNION)}; labelled points: {int(df.notna().sum().sum() // 2)}")
    for g in sorted({s["strain"] for s in usable}):
        tr = sum(1 for s in train if s["strain"] == g)
        te = sum(1 for s in test if s["strain"] == g)
        print(f"    {g}: {tr} train / {te} test")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_union_dataset")
    ap.add_argument("--dataset", action="append", required=True, metavar="DIR",
                    help="A study folder with sidecars/ and lateral/. Repeatable.")
    ap.add_argument("--out", type=Path, default=_ROOT / "dlc_union")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--keep-split", type=Path, metavar="SPLIT_JSON",
                    help="keep this earlier split for the fish it had; split only new fish")
    ap.add_argument("--exclude", action="append", default=[], metavar="FISH_ID",
                    help="Leave this fish out of both sides, e.g. one with a label known "
                         "to be wrong. Repeatable.")
    args = ap.parse_args(argv)
    sources = [(Path(d) / "sidecars", Path(d) / "lateral", Path(d).name) for d in args.dataset]
    for sc, im, _ in sources:
        if not sc.is_dir() or not im.is_dir():
            raise SystemExit(f"missing {sc} or {im}")
    previous = json.loads(args.keep_split.read_text()) if args.keep_split else None
    build(args.out, sources, args.test_frac, args.seed, previous, set(args.exclude))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
