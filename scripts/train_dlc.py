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

#: Frame folder and default project directory per view; the frontal model never
#: shares a directory with the lateral one.
VIEW_VIDEO = {"lateral": "cornell_lateral", "frontal": "cornell_frontal", "union": "fish_union"}
VIEW_PROJECT = {"lateral": "dlc_project", "frontal": "dlc_project_frontal",
                "union": "dlc_project_union"}
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
    # A reused project is rewritten below -- its landmark list replaced and its
    # labelled frames deleted. If it was built for other landmarks, that would
    # silently destroy a different model's project. Refuse instead.
    if existing and cfg.get("bodyparts") and list(cfg["bodyparts"]) != list(LATERAL_KP):
        raise SystemExit(
            f"refusing to reuse {existing[0]}: it was set up for "
            f"{len(cfg['bodyparts'])} landmarks ({', '.join(list(cfg['bodyparts'])[:3])}...), "
            f"not these {len(LATERAL_KP)}. Use a separate --project-dir.")
    cfg["scorer"] = SCORER
    cfg["bodyparts"] = list(LATERAL_KP)
    # Overwritten below once the split is known. DLC registers a shuffle under
    # the fraction it actually got and looks it up by TrainingFraction[0], so a
    # hardcoded value only works while the split happens to match it.
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
    # The project carries its own record of scale and landmarks, so prediction
    # never has to infer them from a dataset directory that may since have been
    # rebuilt for a different model.
    shutil.copy2(built / "split.json", cfg_path.parent / "split.json")
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

    The old schedule is not carried over: a snapshot keeps its scheduler's state,
    which, restored, replaces the new run's learning-rate drops with the old run's
    (long past, so never reached). DeepLabCut 3.0.1 reads a
    ``runner.load_scheduler_state_dict`` switch but its config validation rejects
    the key, so instead the run starts from a copy of the snapshot with the
    scheduler and optimizer state left out -- the weights and the epoch count only.
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
        import torch

        state = torch.load(str(snap), map_location="cpu", weights_only=False)
        weights = {k: state[k] for k in ("model", "metadata") if k in state}
        start = target.parent / f"warmstart-{snap.name}"
        torch.save(weights, str(start))
        conf["resume_training_from"] = str(start)
        print(f"  warm start from {snap.name} (weights and epoch; its learning-rate schedule left behind)")
    if lr is not None:
        conf["runner"]["optimizer"]["params"]["lr"] = lr
        print(f"  learning rate {lr}")

    with open(target, "w") as fh:
        yaml.dump(conf, fh)


#: Augmentation for the cross-species model (``--profile union``). Its frames are
#: already scaled to a standard fish length or show the whole photograph, so the
#: scaling range covers the gap between those and the rest: one rig photographing
#: from farther away than another. Scaling x and y independently (albumentations
#: does when ``keep_ratio`` is off, its default) stretches a fish deeper or longer
#: than any in training, which is what a new body shape is to the model. A lateral
#: view has no left/right landmark pairs, so a mirrored fish needs no relabelling.
UNION_AUGMENT = {
    "affine": {"p": 0.9, "rotation": 25, "scaling": [0.6, 1.4], "translation": 40},
    "hflip": 0.5,
    "crop_sampling": {"width": 448, "height": 448, "max_shift": 0.1, "method": "hybrid"},
    "hist_eq": True,
    "motion_blur": True,
    "covering": True,
    "grayscale": True,
    "gaussian_noise": 12.75,
}


def patch_profile(cfg_path: Path, profile: str, epochs: int, start_epoch: int = 0,
                  lr: float | None = None) -> None:
    """Write a training profile into the generated pytorch_config.yaml.

    Also moves the learning-rate drops to 60% and 85% of the run: the generated
    schedule drops at fixed epochs (90, 120) that suit only a 200-epoch run.

    A warm start counts on from the snapshot's epoch (DeepLabCut trains epochs
    ``start_epoch + 1`` to ``start_epoch + epochs``), and its scheduler changes the
    rate only on the exact epoch of a drop -- so drops placed at 60% and 85% of the
    new run alone are never reached, and the run keeps its starting rate to the
    end. They are placed after ``start_epoch`` instead, and with ``lr`` given the
    rates they drop to keep a full run's proportions (a fifth, then a fiftieth).
    """
    yaml = ruamel.yaml.YAML()
    yaml.width = 4096
    target = sorted(cfg_path.parent.glob(
        "dlc-models-pytorch/iteration-*/*/train/pytorch_config.yaml"))[-1]
    with open(target) as fh:
        conf = yaml.load(fh)
    if profile == "union":
        conf["data"]["train"].update(UNION_AUGMENT)
    sched = conf["runner"].get("scheduler") or {}
    if sched.get("type") == "LRListScheduler":
        sched["params"]["milestones"] = [start_epoch + int(epochs * 0.6),
                                         start_epoch + int(epochs * 0.85)]
        if lr is not None:
            sched["params"]["lr_list"] = [[lr / 5], [lr / 50]]
    conf["runner"]["snapshots"]["save_epochs"] = max(10, epochs // 5)
    with open(target, "w") as fh:
        yaml.dump(conf, fh)
    print(f"  profile {profile}: lr drops at {sched.get('params', {}).get('milestones')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="train_dlc")
    ap.add_argument("--built", type=Path, default=_ROOT / "dlc")
    ap.add_argument("--project-dir", type=Path, default=None,
                    help="defaults to dlc_project/ for lateral, dlc_project_frontal/ "
                         "for frontal -- the view is read from the built dataset")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--net", default="resnet_50")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--resume", type=Path, default=None, metavar="SNAPSHOT.pt",
                    help="Warm-start from an existing snapshot instead of "
                         "ImageNet. Adding a handful of specimens then costs a "
                         "short run rather than a full one. The keypoint set "
                         "must match the snapshot's.")
    ap.add_argument("--profile", choices=["default", "union"], default="default",
                    help="'union': the augmentation for the cross-species model")
    ap.add_argument("--lr", type=float, default=None,
                    help="Override the optimizer learning rate. Worth lowering "
                         "on a warm start: the default schedule assumes it is "
                         "starting from ImageNet, not from a fitted model.")
    args = ap.parse_args(argv)

    # The built dataset says which landmarks it holds, so the trainer cannot be
    # pointed at one view and train it as the other.
    global LATERAL_KP, VIDEO
    split = json.loads((args.built / "split.json").read_text())
    view = split.get("view", "lateral")
    if split.get("keypoints"):
        LATERAL_KP = list(split["keypoints"])
    VIDEO = VIEW_VIDEO[view]
    project_dir = args.project_dir or (_ROOT / VIEW_PROJECT[view])
    print(f"  view: {view} ({len(LATERAL_KP)} landmarks) -> {project_dir}")

    import deeplabcut
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[1/5] project setup (device={device})")
    cfg_path = setup_project(args.built, project_dir)

    train_idx, test_idx = split_indices(cfg_path, split)
    print(f"[2/5] stratified split: {len(train_idx)} train / {len(test_idx)} test")

    # create_training_dataset registers the shuffle under the fraction it is
    # handed, rounded to two places, and train_network looks it up by
    # TrainingFraction[0]. They have to agree or the shuffle is unfindable:
    # "Couldn't find any shuffles with trainingsetindex=0, shuffle=1".
    frac = round(len(train_idx) / max(len(train_idx) + len(test_idx), 1), 2)
    cfg_now, y_now = read_yaml(cfg_path)
    if cfg_now.get("TrainingFraction") != [frac]:
        cfg_now["TrainingFraction"] = [frac]
        write_yaml(cfg_path, cfg_now, y_now)
        print(f"  TrainingFraction set to {frac} to match the split")
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
    start_epoch = 0
    if args.resume:
        snap = torch.load(str(args.resume), map_location="cpu", weights_only=False)
        start_epoch = int((snap.get("metadata") or {}).get("epoch") or 0)
        del snap
        print(f"  the snapshot was saved at epoch {start_epoch}; training continues from there")
    patch_profile(cfg_path, args.profile, args.epochs, start_epoch, args.lr)

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
