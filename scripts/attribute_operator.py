"""Credit the landmarks saved before operator IDs existed to the person who placed them.

    python scripts/attribute_operator.py --dataset cornell --operator JC          # report
    python scripts/attribute_operator.py --dataset cornell --operator JC --write  # do it

Labels saved before the labeler recorded who placed each point say nothing about
it, and a repeatability study reads them as "operator not recorded". When one
person placed them all, this says so: every landmark, outline, ruler point and
midline without an operator gets ``--operator``. A model's point that nobody
reviewed (``metadata.assist.unreviewed``) gets "model" instead, since nobody
placed it. Points that already carry an operator are left alone.

A report by default; ``--write`` changes the files, after copying them to
data/.labels-backup-<time>/. Do not run it for a study several people labelled.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def attribute(doc: dict, operator: str) -> int:
    """Fill in the missing operators of one sidecar, in place; how many were set."""
    meta = doc.get("metadata") or {}
    set_n = 0
    for view in ("lateral", "frontal"):
        block = doc.get(view)
        if not block:
            continue
        assist = meta.get("assist" if view == "lateral" else f"assist_{view}") or {}
        unreviewed = set(assist.get("unreviewed") or [])
        names = list((block.get("keypoints") or {})) + list((block.get("polygons") or {}))
        cal = block.get("calibration") or {}
        if cal.get("point_a") and cal.get("point_b"):
            names += ["ruler_point_a", "ruler_point_b"]
        if block.get("midline"):
            names.append("midline")
        ops = block.setdefault("operators", {})
        for n in names:
            if not ops.get(n):
                ops[n] = "model" if n in unreviewed else operator
                set_n += 1
        if not ops:
            del block["operators"]
    return set_n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="attribute_operator")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--operator", required=True, help="Initials or a short ID, e.g. JC.")
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    ap.add_argument("--write", action="store_true", help="Change the files (after a backup).")
    args = ap.parse_args(argv)
    op = args.operator.strip()
    if not op or op == "model":
        ap.error("give the operator's initials or ID")
    sidecars = args.data_root / args.dataset / "sidecars"
    if not sidecars.is_dir():
        ap.error(f"{sidecars} does not exist")
    changes = {}
    for f in sorted(sidecars.glob("*.json")):
        doc = json.loads(f.read_text())
        n = attribute(doc, op)
        if n:
            changes[f] = (doc, n)
    total = sum(n for _, n in changes.values())
    print(f"{total} landmark(s) on {len(changes)} fish have no operator"
          + (f" and would be credited to {op} (unreviewed model points to 'model')"
             if not args.write else ""))
    if not args.write or not changes:
        if changes and not args.write:
            print("Nothing written. Add --write to do it.")
        return 0
    backup = args.data_root / f".labels-backup-{datetime.datetime.now():%Y%m%dT%H%M%S}"
    backup.mkdir(parents=True)
    for f, (doc, _) in changes.items():
        shutil.copy2(f, backup / f.name)
        f.write_text(json.dumps(doc, indent=2))
    print(f"Written. The files as they were are in {backup.relative_to(_ROOT) if _ROOT in backup.parents else backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
