#!/usr/bin/env python3
"""Package a trained landmark model so someone else can run Auto-label.

The trained projects (``dlc_project/``, ``dlc_project_frontal/``) are not in the
repository: a project holds every snapshot from training and the labelled frames,
and one snapshot alone is ~98 MB. This writes, per view, a zip holding only what
inference needs -- the snapshot the labeler would load, its two configs, the scale
it was trained at, and a short model card -- and records each zip's checksum in
``scripts/models.json``, which ``fetch_model.py`` downloads against.

    python scripts/package_model.py --release models-2026-09-17

Then attach the zips this run printed, by name (not ``dist/models/*.zip``, which
may hold older packages), to a GitHub release with that tag and commit
``scripts/models.json``.

Absolute paths in the configs name this machine's folders. ``project_path`` and
``pose_config_path`` are replaced with a placeholder here and set to wherever the
model is unpacked on install; any other value holding an absolute path (such as
``resume_training_from``) is cleared, and a package that still mentions this
machine's home folder is refused.
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
#: Upstream components a packaged model carries, credited on its model card.
UPSTREAM = ("Trained with DeepLabCut 3.0.1 (Mathis et al. 2018; LGPL-3.0). The ResNet-50 "
            "backbone was initialised from timm's resnet50_gn.a1h_in1k weights (Apache-2.0), "
            "pretrained on ImageNet-1k.")


def strip_local_paths(text: str) -> str:
    """Clear YAML values that are absolute paths on this machine.

    Line-based, like ``set_paths``. A value written on the key's line or folded
    onto the following, more-indented lines is replaced by ``null``.
    """
    lines = text.split("\n")
    out, i = [], 0
    looks_local = re.compile(r"""^['"]?(/Users/|/home/|/private/|/Volumes/|[A-Za-z]:[\\/])""")
    while i < len(lines):
        m = re.match(r"^(\s*)([A-Za-z_][\w.-]*):(.*)$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, key, rest = m.groups()
        value = rest.strip()
        j = i + 1
        if not value:              # a long value folded onto the next lines
            k = j
            while k < len(lines) and lines[k].strip() and \
                    len(lines[k]) - len(lines[k].lstrip()) > len(indent) and \
                    not re.match(r"^\s*[A-Za-z_][\w.-]*:", lines[k]):
                k += 1
            folded = " ".join(x.strip() for x in lines[j:k])
            if folded and looks_local.match(folded):
                out.append(f"{indent}{key}: null")
                i = k
                continue
        elif looks_local.match(value):
            out.append(f"{indent}{key}: null")
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def mentions_this_machine(text: str) -> bool:
    home = str(Path.home())
    return home in text or bool(re.search(r"/Users/[^/\s'\"]+/", text))


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


def terms(licence: str, credits: bool) -> str:
    """The card's closing lines: the weights' licence, and where the training data is credited."""
    out = f"Licence for these weights: {licence}."
    if credits:
        out += (" The photographs and landmarks they were trained on, and their sources"
                " and licences, are listed in CREDITS.md.")
    return out


def card(view: str, project: Path, snapshot: Path, scale: float, parts: list[str],
         licence: str, credits: bool = False) -> str:
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

{UPSTREAM}

Packaged {dt.date.today().isoformat()}. {terms(licence, credits)}
"""


def union_card(project: Path, snapshot: Path, parts: list[str], split: dict,
               licence: str, credits: bool = False) -> str:
    groups = sorted({g.split(":")[0] for g in (split.get("groups") or {}).values()})
    return f"""# caliPr side-view landmark model (cross-species)

- Project: `{project.name}`
- Snapshot: `{snapshot.name}` (DeepLabCut 3.0.1, PyTorch, ResNet-50)
- Landmarks ({len(parts)}): {", ".join(parts)}
- Trained on: {len(split.get("fish_train") or [])} hand-labelled fish from {", ".join(groups)}

Predicts in two passes: the whole photograph to find the fish, then the fish framed
at a standard length to place the points, under caliPr's or BGNN's landmark names
as the study uses. Trained mostly on brook trout and alewife; other body shapes
need labelled examples to fine-tune on.

Installed by `python scripts/fetch_model.py`. Accuracy and limits:
https://github.com/{REPO}#automated-landmarking

{UPSTREAM}

Packaged {dt.date.today().isoformat()}. {terms(licence, credits)}
"""


def package(view: str, out_dir: Path, release: str, licence: str,
            credits: Path | None = None) -> dict:
    project = pl.find_project(None, view)
    cfg, snapshot = pl.find_config_and_snapshot(project, None)
    config_text = (project / "config.yaml").read_text()
    parts = bodyparts(config_text)
    if not parts:
        raise SystemExit(f"{project}: no bodyparts in config.yaml")
    union = project.parent.name == pl.UNION_PROJECT
    if union and credits is None:
        # Trained on other collections' photographs and other people's landmarks:
        # it does not ship without saying whose.
        raise SystemExit("the cross-species model needs --credits: a file naming the "
                         "photographs and landmarks it was trained on and their licences")
    if union:
        # Framed per fish, not shrunk by one factor: what the loader needs is the
        # framing and the template, not a scale.
        split = json.loads((project / "split.json").read_text())
        scale = None
    else:
        scale, split = trained_scale(project, parts)

    rel_train = cfg.parent.relative_to(project)
    trained = "-".join(project.name.split("-")[-3:])     # jcalipr-jcalipr-2026-08-05
    name = f"calipr-{'fish' if union else 'trout'}-{view}-{trained}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    root = project.name
    blank = {k: PLACEHOLDER for k in _PATH_KEYS}
    with zipfile.ZipFile(target, "w") as z:
        def text(arc: str, body: str):
            if mentions_this_machine(body):
                raise SystemExit(f"{arc} would publish a path on this machine; not packaged")
            z.writestr(zipfile.ZipInfo(f"{root}/{arc}", date_time=(2026, 1, 1, 0, 0, 0)),
                       body, compress_type=zipfile.ZIP_DEFLATED)
        text("config.yaml", strip_local_paths(set_paths(config_text, {"project_path": PLACEHOLDER})))
        text(f"{rel_train}/pytorch_config.yaml", strip_local_paths(set_paths(cfg.read_text(), blank)))
        # Just what the loader needs from the split; which fish trained is not.
        if union:
            text("split.json", json.dumps({k: split[k] for k in
                                           ("view", "keypoints", "framing", "template")},
                                          indent=2))
        else:
            text("split.json", json.dumps({"view": view, "scale": scale, "keypoints": parts},
                                          indent=2))
        text("model.json", json.dumps({"snapshot": snapshot.name,
                                       "why": "the snapshot the labeler loads for this view"},
                                      indent=2))
        has_credits = credits is not None
        text("MODEL.md", union_card(project, snapshot, parts, split, licence, has_credits)
             if union else card(view, project, snapshot, scale, parts, licence, has_credits))
        if has_credits:
            text("CREDITS.md", credits.read_text())
        z.write(snapshot, f"{root}/{rel_train}/{snapshot.name}", compress_type=zipfile.ZIP_STORED)

    data = target.read_bytes()
    entry = {
        "file": name,
        "url": f"https://github.com/{REPO}/releases/download/{release}/{name}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "folder": pl.UNION_PROJECT if union else pl.VIEW_PROJECT[view],
        "project": project.name,
        "snapshot": snapshot.name,
        "scale": scale,
        "landmarks": len(parts),
        "licence": licence,
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
    ap.add_argument("--licence", required=True,
                    help='the licence of the weights, as it goes on the model card, '
                         'e.g. "CC BY 4.0" or "CC BY-NC 4.0"')
    ap.add_argument("--credits", type=Path,
                    help="a Markdown file crediting the training photographs and landmarks; "
                         "shipped as CREDITS.md (required for the cross-species model)")
    args = ap.parse_args(argv)
    # Models not packaged this time keep their entries, and their URLs into the
    # release they were published in: repackaging one view must not unpublish
    # the others.
    previous = (json.loads(args.manifest.read_text()).get("models") or {}
                if args.manifest.is_file() else {})
    models = dict(previous)
    models.update({view: package(view, args.out, args.release, args.licence, args.credits)
                   for view in args.views})
    args.manifest.write_text(json.dumps({"release": args.release, "models": models},
                                        indent=2) + "\n")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
