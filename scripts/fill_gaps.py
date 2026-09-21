#!/usr/bin/env python3
"""Predict the landmarks a labelled fish is still missing, and only those.

    python scripts/fill_gaps.py --dataset cornell --dry-run
    python scripts/fill_gaps.py --dataset cornell

This writes **no labels**. It fills the prediction cache, and the labeler puts
those points into the gaps when the fish is opened -- leaving every landmark a
person placed exactly as it is. They become data only when Save is pressed, as
for any other prediction.

Two things make a gap unreachable, and both are worth knowing before running it.

**The model does not know every landmark the study collects.** On the brook trout
set it knows 19 of 23; the four it does not -- the dorsal and anal base endpoints
-- are also the four most often missing, 330 of 666 gaps. Nothing here can touch
them.

**A gap is sometimes the finding.** A landmark can be missing because the
labeller looked and the structure was not there: "dorsal nonexistent, prediction
model labeled adipose fin" is a real note from this dataset, describing exactly
what filling such a gap does. So a fish whose own note says a structure is absent
is skipped for that structure's landmarks, and every skip is printed. The matching
is a keyword rule over free text -- deliberately kept here, in a script whose
output names what it skipped, rather than inside the labeler where it would be a
silent judgement about somebody's prose.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho import schemes  # noqa: E402

#: What the lateral model was trained to place. The study collects more than
#: this, and the difference is not a gap anything here can fill: on the brook
#: trout set the four it does not know are the dorsal and anal base endpoints,
#: which are also the four most often missing.
def model_landmarks(root: Path) -> set[str]:
    """Landmark names from the trained project's own config, or empty if absent."""
    for cfg in sorted(root.glob("dlc_project/*/config.yaml")):
        m = re.search(r"bodyparts:\n((?:\s*-\s*\S+\n)+)", cfg.read_text())
        if m:
            return {ln.strip("- \n") for ln in m.group(1).splitlines()}
    return set()


def short(fid: str) -> str:
    """The part of a fish id a person uses: the strain and number."""
    m = re.search(r"([A-Z]{2,4}_\d+)$", fid)
    return m.group(1) if m else fid[:12]

#: Words a labeller used for "this structure is not there to label". Matched
#: against the fish's own data note, and only alongside the structure's name.
ABSENT = re.compile(
    r"not (really )?visible|invisible|inviisble|nonexistent|destroyed|hardly visible"
    r"|folded|damaged|worn down|poorly formed|missing|clipped",
    re.I)

#: Structures a note can name, and the prefix their landmarks share.
PARTS = ("dorsal", "anal", "pelvic", "pectoral", "caudal", "peduncle")


def blocked_parts(note: str) -> set[str]:
    """Which structures this fish's note says are not there to label."""
    if not note or not ABSENT.search(note):
        return set()
    return {w for w in PARTS if re.search(w, note, re.I)}


def get(base: str, path: str, timeout: int = 600) -> dict:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}"}
    except Exception as e:                                   # noqa: BLE001
        return {"ok": False, "error": str(e)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fill_gaps")
    ap.add_argument("--dataset", default="cornell")
    ap.add_argument("--base", default="http://127.0.0.1:8765",
                    help="a running labeler; it owns the model and the cache")
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be predicted and skipped, and stop")
    args = ap.parse_args(argv)

    study = args.data_root / args.dataset
    order, _ = schemes.study_landmarks(study)
    prof = {}
    if (study / "schema.json").is_file():
        prof = json.loads((study / "schema.json").read_text())
    want = [n for n in order if n not in set(prof.get("exclude_keypoints") or [])]

    specs = get(args.base, f"/api/specimens?dataset={args.dataset}")
    if isinstance(specs, dict) and not specs.get("specimens"):
        print(f"could not read specimens: {specs.get('error', specs)}", file=sys.stderr)
        return 1
    specs = specs["specimens"] if isinstance(specs, dict) else specs
    known = model_landmarks(_ROOT)

    todo, skipped, nothing = [], [], 0
    for s in specs:
        if not s.get("labeled"):
            continue                       # "Auto-label all unlabelled" is for those
        sc = study / "sidecars" / f"{s['id']}.json"
        if not sc.is_file():
            continue
        doc = json.loads(sc.read_text())
        placed = (doc.get("lateral") or {}).get("keypoints") or {}
        note = ((doc.get("metadata") or {}).get("data_note") or "")
        gaps = [n for n in want if n not in placed]
        if not gaps:
            nothing += 1
            continue
        blocked = blocked_parts(note)
        held = sorted(g for g in gaps if any(w in g for w in blocked))
        if held:
            skipped.append((s["id"], held, note))
            if len(held) == len(gaps):
                continue                   # nothing left worth predicting
        todo.append((s["id"], [g for g in gaps if g not in held]))

    print(f"{len(specs)} fish · {nothing} with no gaps · {len(todo)} to predict "
          f"· {len(skipped)} carrying a note that holds a structure back")
    if skipped:
        print("\nheld back — the fish's own note says the structure is not there:")
        for fid, held, note in skipped:
            print(f"  {short(fid):9s} {', '.join(held)[:56]:58s} “{note[:60]}”")
    gaps_all = sum(len(g) for _, g in todo)
    reach = (sum(sum(1 for n in g if n in known) for _, g in todo)
             if known else gaps_all)
    print(f"\n{gaps_all} gaps on those fish; the model has a landmark for {reach} of them"
          + ("" if known else " (could not read the model's landmark list)"))
    if args.dry_run:
        print(f"--dry-run: nothing predicted")
        return 0

    print()
    ok = failed = 0
    t0 = time.time()
    for i, (fid, gaps) in enumerate(todo, 1):
        r = get(args.base, f"/api/predict/{fid}?dataset={args.dataset}")
        if r.get("ok"):
            ok += 1
        else:
            failed += 1
            print(f"  {short(fid)}: {r.get('error', '')[:80]}")
        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  ok {ok}  failed {failed}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"\ncached a prediction for {ok} fish. Nothing was written to any sidecar: "
          f"open a fish and its gaps are filled, with your own points untouched.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
