"""Clear fin keypoints and/or fin outlines from saved sidecars, to be re-done.

Re-placing a landmark is not the same as nudging the old one. A stale point that
is 2 mm off still *looks* placed -- it renders green, it satisfies the retrace
mode's "both keypoints present" check, and the eye anchors to it rather than to
the anatomy. Removing it first forces a fresh judgement.

The fin tips are the highest-value thing to redo: ``dorsal_tip`` is the keypoint
model's worst landmark (2.55 mm median held-out error) and ``pelvic_tip`` carries
a 7.95 mm tail, so these labels cap what any model trained on them can learn.

``--which traces`` clears a fin's outline polygon. That is a far bigger loss than
a keypoint -- 7 to 16 clicks per fin per specimen -- and it is only worth doing
when the existing outline is *actively wrong*, not merely sparse. A sparse outline
whose vertices sit on the margin is a free head start: adding points to it is both
faster and no less accurate than starting over.

Nothing outside the named parts is touched -- the body outline, other keypoints,
calibration, and the data-compromise record are all preserved, and the file is
rewritten with the same ``indent=2`` the labeler uses so the diff shows only
removed keys.

Refuses to run without ``--apply``, and refuses to run on a dirty tree unless
``--force`` is given, because the sidecars are tracked and git is the undo:

    git checkout -- data/cornell/sidecars

Usage::

    python scripts/wipe_fin_keypoints.py                    # dry run, tips
    python scripts/wipe_fin_keypoints.py --apply
    python scripts/wipe_fin_keypoints.py --which tips,bases --apply
    python scripts/wipe_fin_keypoints.py --which traces --fins dorsal,anal --apply
    python scripts/wipe_fin_keypoints.py --which all --apply
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


PARTS = ("tips", "bases", "traces")


def targets(which: set[str], fins: list[str]) -> tuple[set[str], set[str]]:
    """Return (keypoint names, polygon names) to clear."""
    kps: set[str] = set()
    polys: set[str] = set()
    for fin in fins:
        base, tip = FIN_KEYPOINTS[fin]
        if "tips" in which:
            kps.add(tip)
        if "bases" in which:
            kps.add(base)
        if "traces" in which:
            polys.add(fin)
    return kps, polys


def affected_traits(kps: set[str], polys: set[str]) -> list[str]:
    return [t.code for t in TRAITS
            if (set(t.required_keypoints) & kps) or (set(t.required_polygons) & polys)]


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
    ap.add_argument("--which", default="tips",
                    help="Comma-separated from tips,bases,traces (or 'all'). "
                         "'traces' clears the fin OUTLINE polygon, which is a much "
                         "bigger loss than a keypoint -- it is 7-16 clicks of work "
                         "and it is only worth wiping if the existing outline is "
                         "actively wrong rather than merely sparse. A sparse trace "
                         "whose vertices sit on the margin is a free head start: "
                         "adding points to it is faster and no less accurate.")
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

    which = set(PARTS) if args.which.strip() == "all" else {
        w.strip() for w in args.which.split(",") if w.strip()}
    unknown = which - set(PARTS)
    if unknown:
        ap.error(f"--which: unknown part(s) {', '.join(sorted(unknown))}; "
                 f"pick from {', '.join(PARTS)} or 'all'")
    names, polys = targets(which, fins)
    files = sorted(args.sidecars.glob("*.json"))
    if args.specimens:
        want = [s.strip() for s in args.specimens.split(",") if s.strip()]
        files = [f for f in files if any(w in f.stem for w in want)]
    if not files:
        print("No sidecars matched.")
        return 1

    print(f"clearing {'+'.join(sorted(which))} for {', '.join(fins)}")
    if names:
        print(f"  keypoints: {', '.join(sorted(names))}")
    if polys:
        print(f"  outlines : {', '.join(sorted(polys))}")
    print(f"traits that will read NaN until re-done: "
          f"{', '.join(affected_traits(names, polys)) or '(none)'}\n")

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
        pg = block.get("polygons") or {}
        gone = [n for n in sorted(names) if n in kps]
        gone_polys = [n for n in sorted(polys) if n in pg]
        if not gone and not gone_polys:
            continue
        total += len(gone) + len(gone_polys)
        touched.append((path.stem, gone + [f"{n} outline({len(pg[n])})"
                                           for n in gone_polys]))
        if args.apply:
            for n in gone:
                del kps[n]
            for n in gone_polys:
                del pg[n]
            path.write_text(json.dumps(data, indent=2))

    for stem, gone in touched:
        short = stem.replace("Salvelinus_fontinalis_", "")
        print(f"  {short:10} -{len(gone):<2} {', '.join(g.replace('_tip','').replace('_base_center','') for g in gone)}")

    verb = "removed" if args.apply else "would remove"
    print(f"\n{verb} {total} annotations across {len(touched)} specimens")
    if args.apply:
        print("undo with:  git checkout -- data/cornell/sidecars")
    else:
        print("dry run — nothing written. Add --apply to do it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
