#!/usr/bin/env python3
"""Package a trained landmark model so someone else can run Auto-label.

The trained projects (``dlc_project/``, ``dlc_project_frontal/``) are not in the
repository: a project holds every snapshot from training and the labelled frames,
and one snapshot alone is ~98 MB. This writes, per view, a zip holding only what
inference needs -- the snapshot the labeler would load, its two configs, the scale
it was trained at, and a short model card -- and records each zip's checksum in
``scripts/models.json``, which ``fetch_model.py`` downloads against.

    python scripts/package_model.py --release models-2026-09-17

Then attach ``dist/models/*.zip`` to a GitHub release with that tag and commit
``scripts/models.json``.

Absolute paths in the configs name this machine's folders. They are replaced with
a placeholder here and set to wherever the model is unpacked on install.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import predict_landmarks as pl  # noqa: E402
from fetch_model import set_paths  # noqa: E402

REPO = "MizuguchiJAkira/caliPr"
MANIFEST = _ROOT / "scripts" / "models.json"
PLACEHOLDER = "SET-ON-INSTALL"
_PATH_KEYS = ("project_path", "pose_config_path")


def bodyparts(config_yaml: str) -> list[str]:
    names, inside = [], False
    for line in config_yaml.split("\n"):
        if re.match(r"^bodyparts:\s*$", line):
            inside = True
            continue
        if inside:
            m = re.match(r"^-\s+(\S+)\s*$", line)
            if not m:
                break
            names.append(m.group(1))
    return names


def trained_scale(project: Path, parts: list[str]) -> tuple[float, dict]:
    """The scale this model was trained at, from the dataset that produced it.

    Refuses rather than guessing: a wrong scale moves every predicted point.
    """
    for cand in [project / "split.json", *sorted(_ROOT.glob("dlc*/split.json"))]:
        if not cand.is_file():
            continue
        split = json.loads(cand.read_text())
        if not split.get("scale"):
            continue
        if cand.parent == project or list(split.get("keypoints") or []) == parts:
            return float(split["scale"]), split
        # dlc/split.json predates datasets recording their landmarks; it was
        # always the lateral one.
        if (split.get("keypoints") is None and cand.parent.name == "dlc"
                and project.parent.name == pl.VIEW_PROJECT["lateral"]):
            return float(split["scale"]), split
    raise SystemExit(f"{project.name}: no training dataset records this model's landmarks "
                     f"and scale; cannot package it without knowing the scale")


def card(view: str, project: Path, snapshot: Path, scale: float, parts: list[str]) -> str:
    return f"""# caliPr {view} landmark model

- Project: `{project.name}`
- Snapshot: `{snapshot.name}` (DeepLabCut 3.0.1, PyTorch, ResNet-50)
- Input scale: {scale} of the full-resolution photograph
- Landmarks ({len(parts)}): {", ".join(parts)}

Trained on hand-labelled photographs of brook trout (*Salvelinus fontinalis*) from
the Cornell photo rig: fish facing left on foam, a ruler along the top and a mirror
showing the head-on view. It is not a general fish model; on photographs taken
another way it places points with low confidence or wrongly.

Installed by `python scripts/fetch_model.py`. Accuracy and limits:
https://github.com/{REPO}#accuracy

Packaged {dt.date.today().isoformat()}. MIT licence, as the repository.
"""


def package(view: str, out_dir: Path, release: str) -> dict:
    project = pl.find_project(None, view)
    cfg, snapshot = pl.find_config_and_snapshot(project, None)
    config_text = (project / "config.yaml").read_text()
    parts = bodyparts(config_text)
    if not parts:
        raise SystemExit(f"{project}: no bodyparts in config.yaml")
    scale, split = trained_scale(project, parts)

    rel_train = cfg.parent.relative_to(project)
    trained = "-".join(project.name.split("-")[-3:])     # jcalipr-jcalipr-2026-08-05
    name = f"calipr-trout-{view}-{trained}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    root = project.name
    blank = {k: PLACEHOLDER for k in _PATH_KEYS}
    with zipfile.ZipFile(target, "w") as z:
        def text(arc: str, body: str):
            z.writestr(zipfile.ZipInfo(f"{root}/{arc}", date_time=(2026, 1, 1, 0, 0, 0)),
                       body, compress_type=zipfile.ZIP_DEFLATED)
        text("config.yaml", set_paths(config_text, {"project_path": PLACEHOLDER}))
        text(f"{rel_train}/pytorch_config.yaml", set_paths(cfg.read_text(), blank))
        # Just what the loader needs from the split; which fish trained is not.
        text("split.json", json.dumps({"view": view, "scale": scale, "keypoints": parts},
                                      indent=2))
        text("model.json", json.dumps({"snapshot": snapshot.name,
                                       "why": "the snapshot the labeler loads for this view"},
                                      indent=2))
        text("MODEL.md", card(view, project, snapshot, scale, parts))
        z.write(snapshot, f"{root}/{rel_train}/{snapshot.name}", compress_type=zipfile.ZIP_STORED)

    data = target.read_bytes()
    entry = {
        "file": name,
        "url": f"https://github.com/{REPO}/releases/download/{release}/{name}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "folder": pl.VIEW_PROJECT[view],
        "project": project.name,
        "snapshot": snapshot.name,
        "scale": scale,
        "landmarks": len(parts),
    }
    print(f"{view}: {target}  {len(data) / 1e6:.1f} MB  {snapshot.name} at {scale}, "
          f"{len(parts)} landmarks")
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="package_model")
    ap.add_argument("--release", required=True, help="the GitHub release tag the zips go on")
    ap.add_argument("--views", nargs="+", default=["lateral", "frontal"],
                    choices=["lateral", "frontal"])
    ap.add_argument("--out", type=Path, default=_ROOT / "dist" / "models")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args(argv)
    models = {view: package(view, args.out, args.release) for view in args.views}
    args.manifest.write_text(json.dumps({"release": args.release, "models": models},
                                        indent=2) + "\n")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
