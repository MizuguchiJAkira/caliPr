"""Turn a standalone-labeler export into sidecars, verifying it first.

The standalone labeler exports in two shapes:

``calipr_bundle_<key>.zip``
    ``labels.json`` plus every photograph, byte for byte as the labeller opened
    them. Use this when the labeller supplied their own specimens — it is the
    only form that lets the training set be rebuilt here, since coordinates
    without their pixels train nothing.

``calipr_labels_<key>.json``
    Coordinates only. Small enough to email, and correct only for someone who
    already holds the identical photographs.

Why the verification matters
----------------------------
A landmark set is valid for exactly the pixels it was drawn on. The likely
accident is not a wrong click but a **resized photograph** — a phone download, a
Preview re-export, a mail client shrinking an attachment. That scales every
coordinate by a constant factor, and the result still lands inside the frame and
still looks like a plausible fish, so nothing downstream can notice. Training on
it teaches the model a systematically displaced anatomy.

So the export records each photograph's byte count and a content hash, and this
script recomputes both from the copy here. A hash match is proof the two files
are the same bytes; a size or dimension match alone is not, because a re-encode
at the same dimensions changes every pixel while changing neither.

Also checked, because each is silent in its own way:

* **Landmark names must be in the schema.** A name this repo does not know is
  dropped by every downstream consumer without comment.
* **Coordinates must be inside the image.**
* **An extracted photograph must not collide with a different one already here.**
  Two contributors both labelling ``IMG_0042`` would otherwise overwrite each
  other. Use ``--prefix`` to namespace one contributor's files.
* **Existing sidecars are never overwritten** without ``--overwrite``.

Usage::

    python scripts/import_standalone_labels.py --labels calipr_bundle_alewife.zip \\
        --images data/alewife/lateral --out data/alewife/sidecars \\
        --annotator "R. Chen" --prefix rchen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho.landmark_config import KEYPOINTS, View  # noqa: E402

KNOWN = {k.name for k in KEYPOINTS if k.view == View.LATERAL}


def fnv1a(data: bytes) -> str:
    """Mirror of the labeler's fallback hash, for browsers without SubtleCrypto.

    ``crypto.subtle`` is absent outside a secure context, and a double-clicked
    file:// page is not always one. The labeler falls back to this; a weaker
    hash still catches a resize or a re-encode, which is the whole job.
    """
    h1, h2 = 0x811C9DC5, 0x01000193
    for i, b in enumerate(data):
        h1 = ((h1 ^ b) * 16777619) & 0xFFFFFFFF
        if (i & 2047) == 0:
            h2 = ((h2 ^ h1) * 16777619) & 0xFFFFFFFF
    return f"fnv1a:{h1:08x}{h2:08x}"


def digest(data: bytes, style: str) -> str:
    return fnv1a(data) if style.startswith("fnv1a:") else hashlib.sha256(data).hexdigest()


def image_size(path: Path):
    """(width, height) from the header, or None if it cannot be read."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def read_export(src: Path):
    """(labels dict, {name: bytes} for a bundle or {} for labels-only)."""
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            broken = z.testzip()
            if broken is not None:
                raise SystemExit(f"{src}: corrupt zip entry {broken}")
            names = z.namelist()
            if "labels.json" not in names:
                raise SystemExit(f"{src}: no labels.json in the bundle")
            doc = json.loads(z.read("labels.json"))
            imgs = {n[len("images/"):]: z.read(n)
                    for n in names if n.startswith("images/") and not n.endswith("/")}
        return doc, imgs
    return json.loads(src.read_text()), {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="import_standalone_labels")
    ap.add_argument("--labels", type=Path, required=True,
                    help="The .zip bundle or .json labels file from the labeler.")
    ap.add_argument("--images", type=Path, required=True,
                    help="Dataset image directory. Bundled photos are written "
                         "here; for a labels-only import they must already be.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotator", default="",
                    help="Recorded in each sidecar. Worth setting: "
                         "between-annotator differences are a real effect and you "
                         "cannot check for them after the fact without this.")
    ap.add_argument("--prefix", default="",
                    help="Prepended to every filename and fish_id from this "
                         "export, so two contributors' IMG_0042 stay distinct.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace sidecars that already exist.")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="Import even where the photograph here is not the one "
                         "labelled. Only for a mismatch you have explained.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    doc, bundled = read_export(args.labels)
    if doc.get("format") != "calipr-landmarks/1":
        print(f"Unexpected format {doc.get('format')!r}; expected calipr-landmarks/1")
        return 2

    specimens = doc.get("specimens") or {}
    args.images.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pre = f"{args.prefix}_" if args.prefix else ""

    written = skipped = extracted = unverified = 0
    problems: list[str] = []

    for filename, block in sorted(specimens.items()):
        local_name = pre + Path(filename).name
        stem = Path(local_name).stem
        img = args.images / local_name
        fp = block.get("file") or {}

        # --- the photograph -------------------------------------------------
        if filename in bundled:
            payload = bundled[filename]
            if img.is_file():
                here = img.read_bytes()
                if here != payload:
                    problems.append(
                        f"{stem}: a DIFFERENT {local_name} is already in "
                        f"{args.images} — refusing to overwrite it "
                        f"(use --prefix to namespace this contributor)")
                    skipped += 1
                    continue
            elif not args.dry_run:
                img.write_bytes(payload)
                extracted += 1
            else:
                extracted += 1
        elif not img.is_file():
            problems.append(
                f"{stem}: no image {local_name} in {args.images}, and the export "
                f"carries none — ask for the bundle export, not labels-only")
            skipped += 1
            continue

        # --- is it the photograph that was labelled? -------------------------
        if filename in bundled:
            pass                      # byte-identical by construction
        elif fp.get("sha256"):
            raw = img.read_bytes()
            if len(raw) != fp.get("bytes", len(raw)):
                problems.append(
                    f"{stem}: {len(raw)} bytes here, {fp['bytes']} when labelled "
                    f"— not the same file")
                if not args.allow_mismatch:
                    skipped += 1
                    continue
            elif digest(raw, fp["sha256"]) != fp["sha256"]:
                problems.append(
                    f"{stem}: same size but different content hash — the "
                    f"photograph was re-encoded after labelling")
                if not args.allow_mismatch:
                    skipped += 1
                    continue
        else:
            unverified += 1

        w, h = block.get("width"), block.get("height")
        actual = image_size(img)
        if actual and w and h and (actual[0] != w or actual[1] != h):
            problems.append(
                f"{stem}: labelled on {w}x{h}, the copy here is "
                f"{actual[0]}x{actual[1]} — every coordinate is off by "
                f"{actual[0] / w:.3g}x, do NOT import without rescaling")
            if not args.allow_mismatch:
                skipped += 1
                continue

        # --- the landmarks ----------------------------------------------------
        kps = block.get("keypoints") or {}
        bad = [n for n in kps if n not in KNOWN]
        if bad:
            problems.append(f"{stem}: unknown landmark(s) {', '.join(sorted(bad))}")
            skipped += 1
            continue
        if w and h:
            oob = [n for n, p in kps.items()
                   if not (0 <= p[0] <= w and 0 <= p[1] <= h)]
            if oob:
                problems.append(f"{stem}: {len(oob)} landmark(s) outside the image "
                                f"({', '.join(sorted(oob)[:3])})")
                skipped += 1
                continue

        dest = args.out / f"{stem}.json"
        if dest.exists() and not args.overwrite:
            problems.append(f"{stem}: sidecar already exists, left alone "
                            f"(use --overwrite)")
            skipped += 1
            continue

        meta = {"source": "standalone-labeler", "image": local_name}
        if args.annotator:
            meta["annotator"] = args.annotator
        if fp.get("sha256"):
            meta["image_sha256"] = fp["sha256"]
        sidecar = {
            "fish_id": stem,
            "metadata": meta,
            "lateral": {
                "keypoints": {n: [int(p[0]), int(p[1])] for n, p in kps.items()},
                # The standalone collects no ruler, so the series is scale-free by
                # construction. Recording that explicitly keeps the pipeline from
                # rejecting the specimen for a missing calibration.
                "calibration": {"mode": "none",
                                "notes": "standalone labeler: no scale reference"},
            },
        }
        if not args.dry_run:
            dest.write_text(json.dumps(sidecar, indent=2))
        written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {written} sidecar(s) to {args.out}")
    if extracted:
        print(f"  {'would extract' if args.dry_run else 'extracted'} "
              f"{extracted} photograph(s) to {args.images}")
    if unverified:
        print(f"  {unverified} specimen(s) carried no fingerprint (labeler older "
              f"than this check) — content not verified")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
    if skipped:
        print(f"\nskipped {skipped}")
    return 1 if skipped and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
