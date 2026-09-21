#!/usr/bin/env python3
"""Download the trained landmark models, so Auto-label works on this machine.

The models are not in the repository (one is ~98 MB). They are attached to a
GitHub release, and ``scripts/models.json`` records where each one is and its
SHA-256, so a truncated or altered download is refused rather than loaded.

    python scripts/fetch_model.py                 # lateral and frontal
    python scripts/fetch_model.py --views frontal
    python scripts/fetch_model.py --from ~/Downloads   # zips already on disk

The landmark models unpack to ``dlc_project/`` or ``dlc_project_frontal/``, and
the fin outliner is a single file that lands at ``fin_seg_runs/``. Whichever it
is, anything already installed under that name is never replaced: it may be one
trained here.

Auto-label also needs the training stack, installed separately from the labeler:

    python3.11 -m venv .venv-train
    .venv-train/bin/pip install "deeplabcut==3.0.1" transformers

If that environment exists, this also downloads Segment Anything's weights
(~375 MB, from Hugging Face), which the labeler uses to find the fish's outline.
Otherwise the first Auto-label would stall for a minute while it downloads them.

Standard library only, so it runs in the labeler's own environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = _ROOT / "scripts" / "models.json"
SAM = "facebook/sam-vit-base"


def set_paths(text: str, values: dict[str, str]) -> str:
    """Set ``project_path``/``pose_config_path`` in DeepLabCut's YAML, as text.

    Line-based, because this environment has no YAML library. Handles a value on
    the key's own line and a long one folded onto the next, more-indented line,
    which is how DeepLabCut writes a long path.
    """
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"^(\s*)(project_path|pose_config_path):(.*)$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, key, rest = m.groups()
        i += 1
        if not rest.strip():
            while i < len(lines) and lines[i].strip() and \
                    len(lines[i]) - len(lines[i].lstrip()) > len(indent):
                i += 1
        value = values.get(key)
        out.append(f"{indent}{key}:" if value is None
                   else f"{indent}{key}: '" + value.replace("'", "''") + "'")
    return "\n".join(out)


def _download(url: str, dest: Path, size: int) -> None:
    if "://" not in url:                       # a local file, for testing a package
        shutil.copyfile(url, dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "caliPr-fetch-model"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as fh:
        got, shown = 0, -1
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            got += len(chunk)
            pct = int(100 * got / size) // 10 * 10 if size else 0
            if pct != shown:
                print(f"    {pct:3d}%  {got / 1e6:6.1f} of {size / 1e6:.1f} MB", flush=True)
                shown = pct


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def install_file(view: str, entry: dict, root: Path) -> bool:
    """A model that is one file, not a project: download, check, put it in place.

    The fin outliner is a single ``.pt``. It has no project directory and no
    configs naming the machine it was trained on, so none of the unpacking below
    applies to it -- which is why it could not be published at all until this
    existed.
    """
    dest = root / entry["path"]
    if dest.exists():
        print(f"{view}: already installed at {dest.relative_to(root)} — left as it is")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"{view}: downloading {entry['file']} ({entry['bytes'] / 1e6:.0f} MB)")
    try:
        _download(entry["url"], part, entry["bytes"])
        digest = _sha256(part)
        if digest != entry["sha256"]:
            print(f"{view}: REFUSED — the download does not match its recorded checksum "
                  f"(got {digest[:12]}…, expected {entry['sha256'][:12]}…). Nothing installed.")
            return False
        part.replace(dest)
        print(f"{view}: installed {dest.relative_to(root)}")
        return True
    finally:
        part.unlink(missing_ok=True)


def install(view: str, entry: dict, root: Path) -> bool:
    # Two shapes of model live in the manifest: a DeepLabCut project, which
    # arrives as a zip and needs its configs repointed at this machine, and a
    # single file, which needs none of that.
    if entry.get("path"):
        return install_file(view, entry, root)
    folder = root / entry["folder"]
    dest = folder / entry["project"]
    if dest.exists():
        print(f"{view}: already installed at {dest.relative_to(root)} — left as it is")
        return True
    folder.mkdir(parents=True, exist_ok=True)
    part = folder / f".{entry['file']}.part"
    unpack = folder / f".unpack-{entry['project']}"
    print(f"{view}: downloading {entry['file']} ({entry['bytes'] / 1e6:.0f} MB)")
    try:
        _download(entry["url"], part, entry["bytes"])
        digest = _sha256(part)
        if digest != entry["sha256"]:
            print(f"{view}: REFUSED — the download does not match its recorded checksum "
                  f"(got {digest[:12]}…, expected {entry['sha256'][:12]}…). Nothing installed.")
            return False
        shutil.rmtree(unpack, ignore_errors=True)
        unpack.mkdir()
        with zipfile.ZipFile(part) as z:
            for name in z.namelist():
                target = (unpack / name).resolve()
                if not name.startswith(f"{entry['project']}/") or \
                        unpack.resolve() not in target.parents:
                    print(f"{view}: REFUSED — the archive holds a path outside the model "
                          f"folder ({name}). Nothing installed.")
                    return False
            z.extractall(unpack)
        project = unpack / entry["project"]
        # The configs name the folders the model was trained in; point them here.
        final = dest.resolve()
        (project / "config.yaml").write_text(
            set_paths((project / "config.yaml").read_text(), {"project_path": str(final)}))
        for cfg in project.glob("dlc-models-pytorch/iteration-*/*/train/pytorch_config.yaml"):
            cfg.write_text(set_paths(cfg.read_text(), {
                "project_path": str(final),
                "pose_config_path": str(final / cfg.relative_to(project))}))
        project.rename(dest)
        print(f"{view}: installed {entry['snapshot']} at {dest.relative_to(root)}")
        return True
    finally:
        part.unlink(missing_ok=True)
        shutil.rmtree(unpack, ignore_errors=True)


def fetch_sam(root: Path) -> None:
    py = root / ".venv-train" / "bin" / "python"
    if not py.is_file():
        print("\nAuto-label also needs the training stack, which is not installed yet:\n"
              "    python3.11 -m venv .venv-train\n"
              '    .venv-train/bin/pip install "deeplabcut==3.0.1" transformers\n'
              "then run this again to fetch the outline model too.")
        return
    print(f"\nfetching Segment Anything ({SAM}, ~375 MB, skipped if already cached)")
    code = ("from transformers import SamModel, SamProcessor; "
            f"SamProcessor.from_pretrained('{SAM}'); SamModel.from_pretrained('{SAM}')")
    r = subprocess.run([str(py), "-c", code])
    print("Segment Anything ready" if r.returncode == 0 else
          "Segment Anything could not be fetched; the first Auto-label will try again")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_model")
    ap.add_argument("--views", nargs="+",
                    help="which models to fetch (default: every published one)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--root", type=Path, default=_ROOT, help="where to install (the repository)")
    ap.add_argument("--from", dest="source", type=Path,
                    help="a folder holding the zips already (offline, or before a release "
                         "is published); still checked against the recorded checksums")
    ap.add_argument("--no-sam", action="store_true", help="skip the Segment Anything weights")
    args = ap.parse_args(argv)
    if not args.manifest.is_file():
        print(f"no model has been published yet ({args.manifest.name} is missing)")
        return 1
    manifest = json.loads(args.manifest.read_text())
    models = manifest.get("models") or {}
    views = args.views or list(models)
    missing = [v for v in views if v not in models]
    if missing:
        print(f"no published model for: {', '.join(missing)}")
        return 1
    if args.source:
        models = {v: dict(m, url=str(args.source.expanduser() / m["file"]))
                  for v, m in models.items()}
        absent = [m["file"] for v, m in models.items() if v in views and not Path(m["url"]).is_file()]
        if absent:
            print(f"not in {args.source}: {', '.join(absent)}")
            return 1
    ok = all([install(v, models[v], args.root) for v in views])
    if ok and not args.no_sam:
        fetch_sam(args.root)
    if ok:
        print("\nDone. Start the labeler (or click Auto-label again if it is running).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
