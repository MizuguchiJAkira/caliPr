"""Clear fin base/tip keypoints from saved sidecars so they can be re-placed.

Re-placing a landmark is not the same as nudging the old one. A stale point that
is 2 mm off still *looks* placed -- it renders green, it satisfies the retrace
mode's "both keypoints present" check, and the eye anchors to it rather than to
the anatomy. Removing it first forces a fresh judgement.

The fin tips are the highest-value thing to redo: ``dorsal_tip`` is the keypoint
model's worst landmark (2.55 mm median held-out error) and ``pelvic_tip`` carries
a 7.95 mm tail, so these labels cap what any model trained on them can learn.

Nothing else in the sidecar is touched -- polygons, other keypoints, calibration,
and the data-compromise record are all preserved, and the file is rewritten with
the same ``indent=2`` the labeler uses so the diff shows only removed keys.

Refuses to run without ``--apply``, and refuses to run on a dirty tree unless
``--force`` is given, because the sidecars are tracked and git is the undo:

    git checkout -- data/cornell/sidecars

Usage::

    python scripts/wipe_fin_keypoints.py                    # dry run, tips
    python scripts/wipe_fin_keypoints.py --apply
    python scripts/wipe_fin_keypoints.py --which both --apply
    python scripts/wipe_fin_keypoints.py --fins dorsal,pelvic --apply
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho.landmark_config import (  # noqa: E402
    FIN_KEYPOINTS,
    TRAITS,
)


def targets(which: str, fins: list[str]) -> set[str]:
    out: set[str] = set()
    for fin in fins:
        base, tip = FIN_KEYPOINTS[fin]
        if which in ("tips", "both"):
            out.add(tip)
        if which in ("bases", "both"):
            out.add(base)
    return out


def affected_traits(names: set[str]) -> list[str]:
    return [t.code for t in TRAITS if set(t.required_keypoints) & names]


def tree_is_dirty(paths: Path) -> bool:
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", str(paths)],
                           cwd=_ROOT, capture_output=True, text=True, timeout=20)
        return bool(r.stdout.strip())
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wipe_fin_keypoints")
    ap.add_argument("--sidecars", type=Path, default=_ROOT / "data/cornell/sidecars")
    ap.add_argument("--which", choices=("tips", "bases", "both"), default="tips")
    ap.add_argument("--fins", default=",".join(FIN_KEYPOINTS),
                    help="Comma-separated fins to clear (default: all four).")
    ap.add_argument("--specimens", default="",
                    help="Comma-separated substrings; default is every sidecar.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write. Without it this only reports.")
    ap.add_argument("--force", action="store_true",
                    help="Proceed even if the sidecars have uncommitted changes.")
    args = ap.parse_args(argv)

    fins = [f.strip() for f in args.fins.split(",") if f.strip()]
    bad = [f for f in fins if f not in FIN_KEYPOINTS]
    if bad:
        ap.error(f"unknown fin(s): {', '.join(bad)}")

    names = targets(args.which, fins)
    files = sorted(args.sidecars.glob("*.json"))
    if args.specimens:
        want = [s.strip() for s in args.specimens.split(",") if s.strip()]
        files = [f for f in files if any(w in f.stem for w in want)]
    if not files:
        print("No sidecars matched.")
        return 1

    print(f"clearing {args.which} for {', '.join(fins)}")
    print(f"keypoints: {', '.join(sorted(names))}")
    print(f"traits that will read NaN until re-placed: "
          f"{', '.join(affected_traits(names)) or '(none)'}\n")

    if args.apply and not args.force and tree_is_dirty(args.sidecars):
        print("REFUSING: the sidecars have uncommitted changes, so git cannot "
              "undo this.\nCommit or stash them first, or pass --force.")
        return 2

    total = 0
    touched: list[tuple[str, list[str]]] = []
    for path in files:
        data = json.loads(path.read_text())
        block = data.get("lateral") or {}
        kps = block.get("keypoints") or {}
        gone = [n for n in sorted(names) if n in kps]
        if not gone:
            continue
        total += len(gone)
        touched.append((path.stem, gone))
        if args.apply:
            for n in gone:
                del kps[n]
            path.write_text(json.dumps(data, indent=2))

    for stem, gone in touched:
        short = stem.replace("Salvelinus_fontinalis_", "")
        print(f"  {short:10} -{len(gone)}  {', '.join(g.replace('_tip','').replace('_base_center','') for g in gone)}")

    verb = "removed" if args.apply else "would remove"
    print(f"\n{verb} {total} keypoints across {len(touched)} specimens")
    if args.apply:
        print("undo with:  git checkout -- data/cornell/sidecars")
    else:
        print("dry run — nothing written. Add --apply to do it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
