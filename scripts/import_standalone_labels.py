"""Turn a classmate's standalone-labeler export into sidecars this repo can use.

The standalone labeler writes one JSON holding every specimen they labelled. This
converts it into the per-specimen sidecar files the measurement engine, the DLC
dataset builder and the TPS exporter all read, so work done on a laptop with no
Python installed lands in the pipeline unchanged.

Checks worth having, because a mislabelled batch is expensive to discover late:

* **Coordinates must be inside the image.** The export records each image's own
  width and height, so a point outside them means the labeller's file did not
  match the photograph named in it.
* **Landmark names must be in the schema.** A name this repo does not know would
  be silently dropped by every downstream consumer.
* **The image must exist here.** A sidecar without its photograph cannot be
  measured or trained on.
* **Existing sidecars are never overwritten** without ``--overwrite``, so a second
  import cannot quietly discard your own labelling.

Usage::

    python scripts/import_standalone_labels.py --labels calipr_labels_alewife.json \\
        --images data/alewife/lateral --out data/alewife/sidecars \\
        --annotator "R. Chen"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho.landmark_config import KEYPOINTS, View  # noqa: E402

KNOWN = {k.name for k in KEYPOINTS if k.view == View.LATERAL}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="import_standalone_labels")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotator", default="",
                    help="Recorded in each sidecar's metadata. Worth setting: "
                         "between-annotator differences are a real effect and you "
                         "cannot check for them after the fact without this.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    doc = json.loads(args.labels.read_text())
    if doc.get("format") != "calipr-landmarks/1":
        print(f"Unexpected format {doc.get('format')!r}; expected calipr-landmarks/1")
        return 2

    specimens = doc.get("specimens") or {}
    args.out.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    problems: list[str] = []

    for filename, block in sorted(specimens.items()):
        stem = Path(filename).stem
        img = args.images / filename
        if not img.is_file():
            problems.append(f"{stem}: no image {filename} in {args.images}")
            skipped += 1
            continue

        kps = block.get("keypoints") or {}
        w, h = block.get("width"), block.get("height")
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
                                f"— wrong photograph? ({', '.join(sorted(oob)[:3])})")
                skipped += 1
                continue

        dest = args.out / f"{stem}.json"
        if dest.exists() and not args.overwrite:
            problems.append(f"{stem}: sidecar already exists, left alone "
                            f"(use --overwrite)")
            skipped += 1
            continue

        meta = {"source": "standalone-labeler"}
        if args.annotator:
            meta["annotator"] = args.annotator
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
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
    if skipped:
        print(f"\nskipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
