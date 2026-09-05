"""Create the DLC project, train, and evaluate on the held-out specimens.

Run after ``build_dlc_dataset.py``. Steps, each idempotent:

1. ``create_new_project`` (no videos — our frames are stills, not extracted
   from video) and rewrite ``config.yaml`` so ``scorer`` and ``bodyparts``
   match what the converter wrote.
2. Move the converted ``labeled-data`` into the project.
3. ``create_training_dataset`` using the *stratified* indices from
   ``split.json`` rather than DLC's random split, so every strain appears in
   both train and test.
4. ``train_network`` on the MPS (Apple GPU) device when available.
5. ``evaluate_network(per_keypoint_evaluation=True)`` — the pooled error hides
   which landmarks fail, and that is the thing that decides usability.

Usage::

    python scripts/train_dlc.py --built dlc --project-dir dlc_project --epochs 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import ruamel.yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho.landmark_config import KEYPOINTS, View  # noqa: E402

LATERAL_KP = [k.name for k in KEYPOINTS if k.view == View.LATERAL]
SCORER = "jcalipr"
PROJECT = "jcalipr"
VIDEO = "cornell_lateral"


def read_yaml(p: Path):
    y = ruamel.yaml.YAML()
    with p.open() as f:
        return y.load(f), y


def write_yaml(p: Path, data, y):
    with p.open("w") as f:
        y.dump(data, f)


def _make_stub_video(built: Path, project_dir: Path) -> Path:
    """Write a 2-frame mp4 named after the labeled-data folder."""
    import cv2

    frames = sorted((built / "labeled-data" / VIDEO).glob("*.png"))
    if not frames:
        raise SystemExit(f"No frames in {built/'labeled-data'/VIDEO}")
    img = cv2.imread(str(frames[0]))
    h, w = img.shape[:2]
    out = project_dir / f"{VIDEO}.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (w, h))
    for _ in range(2):
        vw.write(img)
    vw.release()
    return out


def setup_project(built: Path, project_dir: Path) -> Path:
    import deeplabcut

    project_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(project_dir.glob(f"{PROJECT}-{SCORER}-*"))
    if existing:
        cfg_path = existing[0] / "config.yaml"
        print(f"  reusing project {existing[0].name}")
    else:
        # DLC rolls the whole project back if given no videos, and it keys
        # labeled-data/<name> off the video name — so synthesize a tiny stub
        # video named after our frame folder. It is never read for training;
        # our frames are stills, not extracted from footage.
        stub = _make_stub_video(built, project_dir)
        created = deeplabcut.create_new_project(
            PROJECT, SCORER, [str(stub)], working_directory=str(project_dir),
            copy_videos=False,
        )
        if not isinstance(created, (str, Path)) or str(created) == "nothingcreated":
            raise SystemExit("create_new_project failed to produce a config.yaml")
        cfg_path = Path(created)
        print(f"  created {cfg_path.parent.name}")

    cfg, y = read_yaml(cfg_path)
    cfg["scorer"] = SCORER
    cfg["bodyparts"] = list(LATERAL_KP)
    cfg["TrainingFraction"] = [0.8]
    cfg["skeleton"] = []
    cfg["video_sets"] = {}
    write_yaml(cfg_path, cfg, y)

    # place converted frames + annotations inside the project
    src = built / "labeled-data" / VIDEO
    dst = cfg_path.parent / "labeled-data" / VIDEO
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"  labeled-data -> {dst.relative_to(cfg_path.parent)}"
          f" ({len(list(dst.glob('*.png')))} frames)")
    return cfg_path


def split_indices(cfg_path: Path, split: dict):
    """Map fish ids in split.json to row positions in DLC's CollectedData."""
    h5 = next((cfg_path.parent / "labeled-data").rglob("CollectedData_*.h5"))
    df = pd.read_hdf(h5)
    names = [(i[-1] if isinstance(i, tuple) else i) for i in df.index]
    fish = [str(n).replace(".png", "") for n in names]
    train = [i for i, f in enumerate(fish) if f in set(split["train"])]
    test = [i for i, f in enumerate(fish) if f in set(split["test"])]
    return train, test


def patch_training_config(cfg_path: Path, resume: Path | None,
                          lr: float | None) -> None:
    """Write warm-start settings into the generated pytorch_config.yaml.

    ``create_training_dataset`` writes this file, so it can only be edited
    between that call and ``train_network``. Two keys matter:

    ``resume_training_from``
        Loads a snapshot before training starts. A full run from ImageNet costs
        ~37 min here; picking up a fitted model to absorb ten new specimens
        should not.

    ``load_scheduler_state_dict: false``
        Without it the scheduler state comes back from the snapshot too, so a
        resumed run silently keeps the old learning-rate schedule and any
        override below is discarded.
    """
    yaml = ruamel.yaml.YAML()
    # Without this ruamel wraps a long snapshot path onto a continuation line.
    # It round-trips correctly, but a config a human cannot read at a glance is
    # a config nobody checks.
    yaml.width = 4096
    hits = sorted(cfg_path.parent.glob(
        "dlc-models-pytorch/iteration-*/*/train/pytorch_config.yaml"))
    if not hits:
        raise SystemExit("no pytorch_config.yaml — did create_training_dataset run?")
    target = hits[-1]
    with open(target) as fh:
        conf = yaml.load(fh)

    if resume is not None:
        snap = resume.resolve()
        if not snap.is_file():
            raise SystemExit(f"snapshot not found: {snap}")
        conf["resume_training_from"] = str(snap)
        conf.setdefault("runner", {})["load_scheduler_state_dict"] = False
        print(f"  warm start from {snap.name}")
    if lr is not None:
        conf["runner"]["optimizer"]["params"]["lr"] = lr
        print(f"  learning rate {lr}")

    with open(target, "w") as fh:
        yaml.dump(conf, fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="train_dlc")
    ap.add_argument("--built", type=Path, default=_ROOT / "dlc")
    ap.add_argument("--project-dir", type=Path, default=_ROOT / "dlc_project")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--net", default="resnet_50")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--resume", type=Path, default=None, metavar="SNAPSHOT.pt",
                    help="Warm-start from an existing snapshot instead of "
                         "ImageNet. Adding a handful of specimens then costs a "
                         "short run rather than a full one. The keypoint set "
                         "must match the snapshot's.")
    ap.add_argument("--lr", type=float, default=None,
                    help="Override the optimizer learning rate. Worth lowering "
                         "on a warm start: the default schedule assumes it is "
                         "starting from ImageNet, not from a fitted model.")
    args = ap.parse_args(argv)

    import deeplabcut
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[1/5] project setup (device={device})")
    cfg_path = setup_project(args.built, args.project_dir)

    split = json.loads((args.built / "split.json").read_text())
    train_idx, test_idx = split_indices(cfg_path, split)
    print(f"[2/5] stratified split: {len(train_idx)} train / {len(test_idx)} test")
    if not train_idx or not test_idx:
        raise SystemExit("Split did not map onto the annotation index.")

    print("[3/5] create_training_dataset")
    deeplabcut.create_training_dataset(
        str(cfg_path), num_shuffles=1,
        trainIndices=[train_idx], testIndices=[test_idx],
        net_type=args.net, userfeedback=False,
    )

    if args.resume or args.lr is not None:
        patch_training_config(cfg_path, args.resume, args.lr)

    if not args.skip_train:
        print(f"[4/5] train_network ({args.epochs} epochs"
              + (f", resuming from {args.resume.name}" if args.resume else "")
              + ")")
        deeplabcut.train_network(
            str(cfg_path), shuffle=1, device=device,
            epochs=args.epochs, batch_size=args.batch_size,
            save_epochs=max(10, args.epochs // 5),
        )

    print("[5/5] evaluate_network (per-keypoint)")
    deeplabcut.evaluate_network(
        str(cfg_path), Shuffles=(1,), plotting=False,
        per_keypoint_evaluation=True, device=device,
    )
    print(f"\nproject: {cfg_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
