#!/usr/bin/env python3
"""Body depth and pectoral fin size from automatic outlines, for every fish alike.

    .venv-train/bin/python scripts/auto_body_traits.py --dataset data/cornell

A study that stopped tracing the body outline and the fins still has what is
needed to measure them: the landmarks it does place, and the photograph. This
outlines both automatically, from the fish's own (hand-placed or reviewed)
landmarks, and measures:

- **MBd**, maximum body depth: the greatest depth of the body square to the line
  from the snout tip to the caudal base, between the back of the gill cover and
  the caudal base. The outline is Segment Anything's, prompted with the fish's
  landmarks and smoothed off the fins it wraps (``predict_worker.smooth_body``);
  where the pectoral fin hangs below the belly, the belly is bridged straight
  across the fin's span. Square to that line rather than vertical in the image,
  so a fish lying at a slant is measured as if it lay level.
- **PFb**, body depth at the pectoral fin: the same, at the pectoral insertion.
- **PFl**/**PFs**, **PlFl**/**PlFs**, **AFs**: the pectoral and pelvic fins'
  length (base to the outline's farthest point) and area, and the anal fin's area,
  from the trained fin outliner. The outliner is framed from a fin's base and tip, and every
  fish is framed the same way: the tip where it typically lies relative to the base
  in the study's own hand-labelled fish. A study that placed tips on some groups
  and not others would otherwise compare hand-framed fins with model-framed ones.
  For the pectoral, framed this way the outline is within 1.2% (median) of the one
  framed from the hand-placed tip, in length and in area; framed from the
  trout-only model's tips instead, four TEM fish came out with "fins" a third to
  over half their body length. The pectoral and pelvic bases are the hand-placed
  insertion and pelvic base. The anal fin is framed from the midpoint of its two
  base ends as the pinned side-view model places them, on every fish alike
  (``<out>/model_landmarks.json``), Cornell having stopped placing the anal base.
  That is good enough to find the fin -- the area is within 1.8% of the hand
  tracing, median -- but not to measure from: the model is unsure of the point on
  every trout (confidence under 0.3) and a fifth of its placements are over 5 mm
  out, so anal fin height (AFh) is not given. A fin whose length (pectoral, pelvic)
  or area (anal) falls outside what the hand-measured fins span is left blank.

The same quantities are measured on the hand-traced outlines and hand-placed tips
where a fish has them, as a check. A body depth outside 0.17-0.34 of standard
length is an outline that took in something else (a ruler, a card); the outline is
tried again cropped close, and left blank if it fails again.

Writes ``<out>/auto_body_traits.csv`` (one row per fish, millimetres) and each
fish's body outline to ``<out>/outlines/<fish>.json``. Nothing is written to the
study's sidecars, and no fin outline or model-placed fin point is kept: fins are
for hand-tracing, and an automatic one on disk is one step from training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import predict_worker as pw  # noqa: E402
from export_tps import px_per_mm  # noqa: E402
from fish_morpho import paths  # noqa: E402

DEPTH_RANGE = (0.17, 0.34)       # of SL: brook trout hand outlines run 0.20-0.29
#: name -> (length code, area code, base landmark, tip landmark, base from). The
#: base is the hand-placed one, or -- for a fin whose base the study did not place
#: on every group -- the midpoint of the fin's two base ends as the model places
#: them (``model_fin_bases``). A hand-based fin is checked by its length, a
#: model-based one by its area, which does not depend on where the base is.
FINS = {
    "pectoral": ("PFl", "PFs", "pectoral_insertion_upper", "pectoral_ray_tip", "hand"),
    "pelvic": ("PlFl", "PlFs", "pelvic_base_center", "pelvic_tip", "hand"),
    "dorsal": ("DFh", "DFs", "dorsal_base_center", "dorsal_tip", "model"),
    "anal": (None, "AFs", "anal_base_center", "anal_tip", "model"),
}


def axis_frame(kps):
    """Coordinates along (u) and across (v) the snout -> caudal base line."""
    a, b = np.array(kps["premaxilla_tip"], float), np.array(kps["caudal_base"], float)
    L = float(np.linalg.norm(b - a))
    ux = (b - a) / L
    uy = np.array([-ux[1], ux[0]])
    return (lambda p: (float(np.dot(np.asarray(p, float) - a, ux)),
                       float(np.dot(np.asarray(p, float) - a, uy)))), L


def crossings(poly_uv, u):
    vs = []
    for i in range(len(poly_uv)):
        (u1, v1), (u2, v2) = poly_uv[i], poly_uv[(i + 1) % len(poly_uv)]
        if (u1 - u) * (u2 - u) <= 0 and u1 != u2:
            vs.append(v1 + (u - u1) * (v2 - v1) / (u2 - u1))
    return (min(vs), max(vs)) if len(vs) >= 2 else (math.nan, math.nan)


def depth_profile(poly, kps, fin=None, start=None):
    """Depth square to the axis at 300 places from ``start`` (default: the back of
    the gill cover) to the caudal base, the belly bridged under the pectoral fin."""
    to, L = axis_frame(kps)
    puv = [to(p) for p in poly]
    u0 = to(kps["operculum_posterior"])[0] if start is None else start
    us = np.linspace(u0, L, 300)
    edges = np.array([crossings(puv, u) for u in us])
    top, bot = edges[:, 0], edges[:, 1].copy()
    if fin:
        fu = [to(p)[0] for p in fin]
        under = (us >= min(fu)) & (us <= max(fu))
        clear = ~under & ~np.isnan(bot)
        if under.any() and clear.sum() >= 2:
            bot[under] = np.interp(us[under], us[clear], bot[clear])
    return us, bot - top, L, to


def body_depths(poly, kps, fin=None):
    """(MBd, where along SL it is, PFb) in pixels."""
    us, dep, L, to = depth_profile(poly, kps, fin)
    if np.all(np.isnan(dep)):
        return math.nan, math.nan, math.nan
    i = int(np.nanargmax(dep))
    pfb = math.nan
    if "pectoral_insertion_upper" in kps:
        u = to(kps["pectoral_insertion_upper"])[0]
        ups, udep, _, _ = depth_profile(poly, kps, fin, start=min(u, us[0]) - 1)
        pfb = float(np.interp(u, ups, udep))
    return float(dep[i]), float(us[i] / L), pfb


def area(p) -> float:
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
                   for i in range(len(p)))) / 2


def body_outline(photo, kps, margin_v: float):
    """Segment Anything's outline of the fish, prompted with its own landmarks."""
    to, L = axis_frame(kps)
    xs = [p[0] for p in kps.values()]
    ys = [p[1] for p in kps.values()]
    x0 = int(max(0, min(xs) - 0.06 * L)); x1 = int(min(photo.shape[1], max(xs) + 0.2 * L))
    y0 = int(max(0, min(ys) - margin_v * L)); y1 = int(min(photo.shape[0], max(ys) + margin_v * L))
    crop = photo[y0:y1, x0:x1]
    s = min(1.0, 1500.0 / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    a, b = np.array(kps["premaxilla_tip"], float), np.array(kps["caudal_base"], float)
    prompts = [kps["premaxilla_tip"], kps["operculum_posterior"], list(a + 0.5 * (b - a)),
               list(a + 0.75 * (b - a)), kps["caudal_base"]]
    for n in ("pectoral_insertion_upper",):
        if n in kps:
            prompts.append(kps[n])
    if "dorsal_base_anterior" in kps and "pelvic_base_center" in kps:
        prompts.append(list((np.array(kps["dorsal_base_anterior"]) + np.array(kps["pelvic_base_center"])) / 2))
    pts = [[(p[0] - x0) * s, (p[1] - y0) * s] for p in prompts]
    poly = pw._Sam.body_outline(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), pts, DEVICE, s)
    return pw.smooth_body([[p[0] + x0, p[1] + y0] for p in poly], kps)


def typical_tip(study: Path, base: str, tip: str):
    """Where a fin's tip lies relative to its base, in SL along and across the body
    (median over the study's fish with both hand-placed), and the range of the fin's
    length in SL (1st to 99th percentile), widened by a fifth either way."""
    import statistics
    offs, lens = [], []
    for f in (study / "sidecars").glob("*.json"):
        k = (json.loads(f.read_text()).get("lateral") or {}).get("keypoints") or {}
        if all(n in k for n in ("premaxilla_tip", "caudal_base", base, tip)):
            to, L = axis_frame(k)
            a, b = to(k[base]), to(k[tip])
            offs.append(((b[0] - a[0]) / L, (b[1] - a[1]) / L))
            lens.append(math.dist(k[base], k[tip]) / L)
    if len(offs) < 10:
        raise SystemExit(f"fewer than ten fish with a hand-placed {base} and {tip}")
    lens.sort()
    lo, hi = lens[int(0.01 * (len(lens) - 1))], lens[int(0.99 * (len(lens) - 1))]
    return ((statistics.median(o[0] for o in offs), statistics.median(o[1] for o in offs)),
            (0.8 * lo, 1.2 * hi), len(offs))


#: The smallest square-root area (in SL) taken as a fin rather than a failed
#: outline. A fin folded flat shows little of itself -- smaller than any of the
#: spread fins that were hand-traced -- and is still the fin; outlines that found
#: nothing come out near 0.005.
MIN_FIN_ROOT_AREA = 0.010


def hand_area_range(study: Path, fin: str):
    """The range of a fin's square-root area in SL: from ``MIN_FIN_ROOT_AREA`` (a
    folded fin is small, not wrong) to the largest hand tracing (99th percentile)
    widened by a fifth."""
    vals = []
    for f in (study / "sidecars").glob("*.json"):
        lat = json.loads(f.read_text()).get("lateral") or {}
        k, poly = lat.get("keypoints") or {}, (lat.get("polygons") or {}).get(fin)
        if poly and len(poly) >= 3 and "premaxilla_tip" in k and "caudal_base" in k:
            vals.append(math.sqrt(area(poly)) / math.dist(k["premaxilla_tip"], k["caudal_base"]))
    if len(vals) < 10:
        raise SystemExit(f"fewer than ten hand-traced {fin} fins to check the outlines against")
    vals.sort()
    return MIN_FIN_ROOT_AREA, 1.2 * vals[int(0.99 * (len(vals) - 1))]


def at_offset(kps, origin, offset):
    """``origin`` moved by ``offset`` (in SL, along and across the body)."""
    a, b = np.array(kps["premaxilla_tip"], float), np.array(kps["caudal_base"], float)
    L = float(np.linalg.norm(b - a)); ux = (b - a) / L; uy = np.array([-ux[1], ux[0]])
    return list(np.array(origin, float) + L * (offset[0] * ux + offset[1] * uy))


def model_fin_bases(study: Path, out: Path) -> dict:
    """Each model-based fin's base centre on every fish, from the pinned side-view
    model: the midpoint of the fin's two base ends. Run once, kept in
    model_landmarks.json, and run again only if a fin is asked for that it lacks."""
    import subprocess
    fins = [f for f, spec in FINS.items() if spec[4] == "model"]
    wanted = [f"{f}_base_{end}" for f in fins for end in ("anterior", "posterior")]
    cache = out / "model_landmarks.json"
    if cache.is_file():
        found = json.loads(cache.read_text())
        if set(found.get("landmarks") or ()) >= set(wanted):
            return found
    import preprocess_cornell as pc
    single = json.loads((study / "schema.json").read_text()).get("single_photo") \
        if (study / "schema.json").is_file() else False
    proc = subprocess.Popen([str(paths.TRAINING_PYTHON), str(_ROOT / "scripts" / "predict_worker.py"),
                             "--device", DEVICE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1, cwd=str(_ROOT))
    hello = json.loads(proc.stdout.readline())
    if not hello.get("union"):
        raise SystemExit("fin bases from the model need the cross-species model pinned")
    found = {"model": hello.get("model"), "landmarks": wanted, "fish": {}}
    sidecars = sorted((study / "sidecars").glob("*.json"))
    for k, f in enumerate(sidecars, 1):
        fid = json.loads(f.read_text())["fish_id"]
        img = find_image(study, fid)
        if img is None:
            continue
        crop = None
        if single:
            box = pc.view_frames(cv2.imread(str(img)))["lateral"]
            crop = list(box) if box else None
        proc.stdin.write(json.dumps({"image": str(img), "crop": crop, "polygons": False,
                                     "landmarks": wanted}) + "\n")
        proc.stdin.flush()
        r = json.loads(proc.stdout.readline())
        kp, cf = r.get("keypoints") or {}, r.get("confidence") or {}
        entry = {}
        for fin in fins:
            a, b = kp.get(f"{fin}_base_anterior"), kp.get(f"{fin}_base_posterior")
            if a and b:
                entry[f"{fin}_base_center"] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
                entry[f"{fin}_confidence"] = [cf.get(f"{fin}_base_anterior"), cf.get(f"{fin}_base_posterior")]
        found["fish"][fid] = entry
        if k % 25 == 0:
            print(f"  fin bases {k}/{len(sidecars)}", flush=True)
    proc.stdin.close()
    proc.wait(timeout=120)
    cache.write_text(json.dumps(found, indent=1))
    return found


def find_image(study: Path, fid: str):
    return next((p for p in (study / "lateral").iterdir()
                 if p.stem == fid or p.stem.startswith(fid + "_")), None)


def main(argv=None) -> int:
    global DEVICE
    ap = argparse.ArgumentParser(prog="auto_body_traits")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args(argv)
    DEVICE = args.device
    study = args.dataset
    out = args.out or (paths.WORK / "results" / study.name / "auto_body_traits")
    (out / "outlines").mkdir(parents=True, exist_ok=True)

    typical = {}
    for fin, (lcode, _, base, tip, source) in FINS.items():
        off, rng, n = typical_tip(study, base, tip)
        what = "length"
        if source == "model":                # checked by area: it does not depend on the base
            rng, what = hand_area_range(study, fin), "square-root area"
        typical[fin] = (off, rng)
        print(f"{fin}: tip placed {off[0]:.3f} SL back and {off[1]:.3f} SL across from the base "
              f"(median of {n} hand-placed); {what} kept within {rng[0]:.3f}-{rng[1]:.3f} SL")
    modelled = model_fin_bases(study, out)["fish"]
    rows = []
    sidecars = sorted((study / "sidecars").glob("*.json"))
    for k, f in enumerate(sidecars, 1):
        d = json.loads(f.read_text())
        fid = d["fish_id"]
        kps = (d.get("lateral") or {}).get("keypoints") or {}
        img = find_image(study, fid)
        ppm = px_per_mm(d)
        if img is None or not ppm or any(n not in kps for n in
                                         ("premaxilla_tip", "caudal_base", "operculum_posterior")):
            continue
        photo = cv2.imread(str(img))
        _, L = axis_frame(kps)
        row = {"fish_id": fid, "SL_mm": L / ppm}

        # fins, each framed from its base and the typical tip position
        fins, polys = {}, (d.get("lateral") or {}).get("polygons") or {}
        for fin, (lcode, acode, base_name, tip_name, source) in FINS.items():
            base = ((modelled.get(fid) or {}).get(base_name) if source == "model"
                    else kps.get(base_name))
            outline, why = None, ""
            if base:
                tip = at_offset(kps, base, typical[fin][0])
                outline = pw._Fin.outline(photo, dict(kps, **{base_name: base, tip_name: tip}), fin, DEVICE)
                if outline:
                    by_area = source == "model"
                    size = math.sqrt(area(outline)) if by_area else max(math.dist(base, q) for q in outline)
                    lo, hi = typical[fin][1]
                    if not lo <= size / L <= hi:
                        why = f"no usable outline ({'square-root area' if by_area else 'length'} {size / L:.3f} of SL)"
                        outline = None
                else:
                    why = "no fin found"
            else:
                why = f"no {base_name}"
            fins[fin] = outline
            if lcode:
                row[f"{lcode}_mm"] = max(math.dist(base, q) for q in outline) / ppm if outline else math.nan
            row[f"{acode}_mm2"] = area(outline) / ppm ** 2 if outline else math.nan
            row[f"{fin}_flag"] = why
            # the hand-measured versions, where there are any
            hb, ht, hp = kps.get(base_name), kps.get(tip_name), polys.get(fin)
            if lcode:
                row[f"{lcode}_hand_mm"] = math.dist(hb, ht) / ppm if hb and ht else math.nan
            row[f"{acode}_hand_mm2"] = area(hp) / ppm ** 2 if hp and len(hp) >= 3 else math.nan
        for fin, spec in FINS.items():
            if spec[4] != "model":
                continue
            mb, hb = (modelled.get(fid) or {}).get(spec[2]), kps.get(spec[2])
            if hb is None and f"{fin}_base_anterior" in kps and f"{fin}_base_posterior" in kps:
                a, b = kps[f"{fin}_base_anterior"], kps[f"{fin}_base_posterior"]
                hb = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
            row[f"{fin}_base_model_err_mm"] = math.dist(mb, hb) / ppm if mb and hb else math.nan
        fin = fins["pectoral"]

        # body
        flag = ""
        poly = body_outline(photo, kps, 0.35)
        mbd, at, pfb = body_depths(poly, kps, fin)
        if not DEPTH_RANGE[0] <= mbd / L <= DEPTH_RANGE[1]:
            poly = body_outline(photo, kps, 0.16)            # closer, without the ruler
            mbd, at, pfb = body_depths(poly, kps, fin)
            flag = "outline retried close"
            if not DEPTH_RANGE[0] <= mbd / L <= DEPTH_RANGE[1]:
                flag = f"no usable outline (depth {mbd / L:.2f} of SL)"
                mbd = at = pfb = math.nan
        row.update(MBd_mm=mbd / ppm, MBd_at_SL=at, PFb_mm=pfb / ppm, body_flag=flag)

        # the hand-traced body, where there is one
        hb = polys.get("body_plus_caudal")
        row["MBd_hand_mm"] = (body_depths(hb, kps)[0] if hb and len(hb) >= 8 else math.nan) / ppm
        rows.append(row)
        # Only the body outline is kept, to look at. The fin outlines are not: an
        # automatic fin outline left on disk is one step from becoming fin training
        # data, which is to be hand-traced.
        (out / "outlines" / f"{fid}.json").write_text(json.dumps({"body": poly}))
        if k % 25 == 0:
            print(f"  {k}/{len(sidecars)}", flush=True)

    with open(out / "auto_body_traits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({kk: ("" if isinstance(v, float) and math.isnan(v) else
                             round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()})
    # The model's fin base points served this run; they are not kept either.
    (out / "model_landmarks.json").unlink(missing_ok=True)
    print(f"wrote {len(rows)} fish to {out / 'auto_body_traits.csv'}")
    return 0


DEVICE = "mps"

if __name__ == "__main__":
    raise SystemExit(main())
