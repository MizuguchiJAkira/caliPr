#!/usr/bin/env python3
"""How far a landmark model's points fall from the hand-placed ones, on held-out fish.

    .venv-train/bin/python scripts/eval_landmarks.py --split dlc_union/split.json \\
        --dataset data/cornell --dataset data/alewife \\
        --model before=dlc_project --model after=dlc_project_union \\
        --exclude-trained dlc/split.json

Each model is run exactly as Auto-label runs it -- the labeler's own worker, on the
whole photograph (or the study's crop of it), with the outline and the anatomy
checks -- on the fish in the split's held-out list. A fish any of the models was
trained on (``--exclude-trained``) is left out for all of them, so every model is
scored on the same fish and none on fish it has seen.

Only points a person placed or moved are scored. A point a model placed and a
person accepted unmoved (``metadata.assist.accepted``) is that model's own answer:
scored against, it makes the model that placed it look perfect and measures
nothing about the others.

Reported per group (study or strain), for the landmarks both the hand labels and
the model have:

- median error as a percentage of the fish's length (snout to caudal base, from
  the hand labels), which compares across species and photograph scales;
- median error in mm, where the photograph has a scale;
- the share of points more than 5% of the fish's length out -- not a near miss
  but a point on the wrong part of the fish, or off it;
- the share of points outside the hand-traced outline of the fish (where traced),
  allowing 2% of its length: points on the background, a pin or the ruler;
- how many the model's own checks dropped as impossible.

Writes every point's result to ``--out`` as JSON as well.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import build_dlc_dataset as bdd  # noqa: E402
import preprocess_cornell as pc  # noqa: E402
from export_tps import px_per_mm  # noqa: E402
from fish_morpho import landmark_union as lu  # noqa: E402
from fish_morpho import paths, schemes  # noqa: E402

FAR = 0.05            # of fish length: a point on the wrong part of the fish
OUTLINE_SLACK = 0.02  # of fish length: how far outside the traced outline still counts
LENGTH_ENDS = (("premaxilla_tip", "lower_jaw_tip"), ("caudal_base",))


def fish_length(kps: dict) -> float | None:
    snout = next((kps[n] for n in LENGTH_ENDS[0] if n in kps), None)
    tail = next((kps[n] for n in LENGTH_ENDS[1] if n in kps), None)
    if snout is None or tail is None:
        xs = [p[0] for p in kps.values()]
        return (max(xs) - min(xs)) if len(xs) >= 2 else None
    return math.dist(snout, tail)


def outside(point, polygon) -> float:
    """Distance from a point to a polygon it lies outside, 0 if inside."""
    x, y = point
    inside = False
    n = len(polygon)
    best = math.inf
    for i in range(n):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy or 1)))
        best = min(best, math.hypot(x - x1 - t * dx, y - y1 - t * dy))
    return 0.0 if inside else best


class Worker:
    """The labeler's prediction worker, on one model."""

    def __init__(self, python: Path, project: Path, device: str = "mps"):
        self.proc = subprocess.Popen(
            [str(python), str(_ROOT / "scripts" / "predict_worker.py"), "--project", str(project),
             "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=str(_ROOT))
        hello = json.loads(self.proc.stdout.readline())
        if not hello.get("ready"):
            raise SystemExit(f"{project}: {hello.get('error')}")
        self.info = hello

    def predict(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=60)


def resolve_project(path: Path) -> Path:
    """A DeepLabCut project, given it or the folder it was trained into."""
    if (path / "config.yaml").is_file():
        return path
    found = sorted(path.glob("*/config.yaml"))
    if not found:
        raise SystemExit(f"no DeepLabCut project in {path}")
    return found[-1].parent


def held_out(split_path: Path, sources, exclude: list[Path]) -> list[dict]:
    split = json.loads(split_path.read_text())
    test = set(split.get("fish_test") or split.get("test") or [])
    trained = set()
    for p in exclude:
        s = json.loads(p.read_text())
        trained |= set(s.get("fish_train") or s.get("train") or [])
    bdd.use_view("lateral")
    specs = []
    for sidecars, images, group in sources:
        for spec in bdd.load_specimens(sidecars, images, group):
            if spec["fish_id"] in test and spec["fish_id"] not in trained:
                data = json.loads((sidecars / f"{spec['fish_id']}.json").read_text())
                spec["study"] = sidecars.parent
                spec["px_per_mm"] = px_per_mm(data)
                spec["outline"] = ((data.get("lateral") or {}).get("polygons") or {}).get(
                    "body_plus_caudal")
                spec["group"] = split.get("groups", {}).get(spec["fish_id"], group)
                assist = (data.get("metadata") or {}).get("assist") or {}
                spec["model_points"] = (set(assist.get("accepted") or ())
                                        | set(assist.get("unreviewed") or ()))
                specs.append(spec)
    return specs


def crop_for(spec: dict):
    """The rectangle the labeler predicts in, for a one-photograph-per-fish study."""
    if not spec.get("single_photo"):
        return None
    import cv2
    im = cv2.imread(str(spec["image"]))
    box = pc.view_frames(im)["lateral"] if im is not None else None
    return list(box) if box is not None else None


def score(specs, worker: Worker) -> list[dict]:
    rows = []
    for i, spec in enumerate(specs, 1):
        wanted, _ = schemes.study_landmarks(spec["study"])
        crop = crop_for(spec)
        r = worker.predict({"image": str(spec["image"]), "crop": crop, "polygons": True,
                            "landmarks": list(wanted)})
        if not r.get("ok"):
            print(f"  {spec['fish_id']}: {r.get('error')}")
            rows.append({"fish": spec["fish_id"], "group": spec["group"], "failed": True})
            continue
        truth = lu.to_union({n: p for n, p in spec["keypoints"].items()
                             if n not in spec["model_points"]})
        length = fish_length(lu.to_union(spec["keypoints"]))
        pred = lu.to_union(r.get("keypoints") or {})
        for name, p in pred.items():
            row = {"fish": spec["fish_id"], "group": spec["group"], "landmark": name,
                   "pred": p}
            if spec.get("outline") and length:
                row["off_outline"] = outside(p, spec["outline"]) > OUTLINE_SLACK * length
            if name in truth and length:
                d = math.dist(p, truth[name])
                row["err_pct"] = 100 * d / length
                if spec["px_per_mm"]:
                    row["err_mm"] = d / spec["px_per_mm"]
            rows.append(row)
        dropped = r.get("implausible") or {}
        rows.append({"fish": spec["fish_id"], "group": spec["group"],
                     "dropped": len(dropped)})
        print(f"  [{i}/{len(specs)}] {spec['fish_id']}: {len(pred)} points, "
              f"{len(dropped)} dropped", flush=True)
    return rows


def summary(rows: list[dict]) -> dict:
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
        by["ALL"].append(r)
    for g, rs in by.items():
        pct = [r["err_pct"] for r in rs if "err_pct" in r]
        mm = [r["err_mm"] for r in rs if "err_mm" in r]
        off = [r["off_outline"] for r in rs if "off_outline" in r]
        out[g] = {
            "fish": len({r["fish"] for r in rs}),
            "points": len(pct),
            "median_pct": statistics.median(pct) if pct else None,
            "median_mm": statistics.median(mm) if mm else None,
            "far_share": sum(e > 100 * FAR for e in pct) / len(pct) if pct else None,
            "off_outline_share": sum(off) / len(off) if off else None,
            "dropped": sum(r.get("dropped", 0) for r in rs),
            "failed": sum(1 for r in rs if r.get("failed")),
        }
    return out


def table(results: dict[str, dict]) -> str:
    groups = sorted({g for s in results.values() for g in s}, key=lambda g: (g == "ALL", g))
    fmt = lambda v, f: "—" if v is None else f.format(v)   # noqa: E731
    lines = [f"{'group':14} {'model':8} {'fish':>4} {'pts':>5} {'med %L':>7} {'med mm':>7} "
             f"{'>5%L':>6} {'off':>6} {'drop':>5}"]
    for g in groups:
        for label, s in results.items():
            if g not in s:
                continue
            x = s[g]
            lines.append(f"{g:14} {label:8} {x['fish']:4d} {x['points']:5d} "
                         f"{fmt(x['median_pct'], '{:6.2f}%'):>7} {fmt(x['median_mm'], '{:6.2f}'):>7} "
                         f"{fmt(x['far_share'] and 100 * x['far_share'], '{:5.1f}%'):>6} "
                         f"{fmt(x['off_outline_share'] and 100 * x['off_outline_share'], '{:5.1f}%'):>6} "
                         f"{x['dropped']:5d}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval_landmarks")
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--dataset", action="append", required=True, metavar="DIR")
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PROJECT",
                    help="a model to score, by its project folder. Repeatable.")
    ap.add_argument("--exclude-trained", action="append", type=Path, default=[],
                    metavar="SPLIT", help="leave out every fish this split trained on")
    ap.add_argument("--python", type=Path, default=paths.TRAINING_PYTHON)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N fish")
    ap.add_argument("--out", type=Path, default=paths.WORK / "results" / "landmark_eval.json")
    args = ap.parse_args(argv)

    sources = [(Path(d) / "sidecars", Path(d) / "lateral", Path(d).name) for d in args.dataset]
    models = [(label, resolve_project(Path(p))) for label, p in
              (m.split("=", 1) for m in args.model)]
    exclude = list(args.exclude_trained) + [p / "split.json" for _, p in models
                                            if (p / "split.json").is_file()]
    exclude = [p for p in exclude if p.resolve() != args.split.resolve()]
    specs = held_out(args.split, sources, exclude)
    if args.limit:
        specs = specs[::max(1, len(specs) // args.limit)][:args.limit]
    print(f"{len(specs)} held-out fish, none trained on by any model scored\n")

    results, every = {}, {}
    for label, project in models:
        print(f"{label}: {project}")
        w = Worker(args.python, project, args.device)
        try:
            rows = score(specs, w)
        finally:
            w.close()
        every[label] = {"snapshot": w.info.get("model"), "rows": rows}
        results[label] = summary(rows)
    print()
    print(table(results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": results, "models": every}, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
