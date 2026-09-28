"""Export labelled specimens to TPS, the format geomorph and tpsDig read.

Why this exists
---------------
``geomorph::digitize2d()`` is a *digitizing* tool -- it opens each photograph and
asks you to click landmarks, then writes a .tps file. Everything downstream in
geomorph (``gpagen``, ``plotTangentSpace``, ``procD.lm``) starts from that .tps
and never cares how it was made. So a labeller that already records named
landmarks can write the .tps directly and skip ``digitize2d`` entirely, which
also skips whatever is wrong with it on a given machine.

Two format details are easy to get wrong and both silently corrupt the data:

*Y origin.* TPS coordinates are Cartesian, measured from the BOTTOM-left. Image
coordinates run from the TOP-left. Writing image y unflipped mirrors every
specimen vertically; Procrustes superimposition will happily align the mirrored
shapes, so nothing errors and the biology comes out upside down.

*Missing landmarks.* TPS has no NA. The convention geomorph understands is a
negative coordinate, read back with ``readland.tps(..., negNA = TRUE)``. Writing
a 0 instead would place a real landmark at the image corner and drag the whole
Procrustes fit.

Landmark names
--------------
TPS identifies landmarks by ORDER, not name -- row 7 is whatever the protocol says
row 7 is. That is fine for geomorph but useless for anything else, so this writes
a companion ``landmark_names.csv`` and an R snippet that attaches the names to the
array's dimnames. The order is taken from ``landmark_config`` and is therefore the
same order the measurement engine and the pose model use.

Usage::

    python scripts/export_tps.py --sidecars data/alewife/sidecars \\
        --images data/alewife/lateral --out results/alewife/tps
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho import schemes  # noqa: E402
from fish_morpho.landmark_config import (  # noqa: E402
    KEYPOINTS,
    View,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPEG", ".JPG")
LANDMARK_LABELS: dict[str, str] = {}

def scheme_of(profile_dir: Path | None) -> str | None:
    """The landmark scheme this study collects, if it is not caliPr's own."""
    if profile_dir is None:
        return None
    prof = profile_dir / "schema.json"
    if not prof.is_file():
        return None
    try:
        name = json.loads(prof.read_text()).get("scheme")
    except Exception:
        return None
    return name if schemes.get(name) else None


def landmark_order(profile_dir: Path | None) -> tuple[str, ...]:
    """Landmark order for this dataset: what the study collects, in export order.

    Row N must mean the same thing in every specimen, so the order comes from the
    study's schema rather than from whatever a given sidecar happens to contain.
    One source for every export -- see ``schemes.study_landmarks``.
    """
    return schemes.study_landmarks(profile_dir)[0]


def landmark_labels(profile_dir: Path | None) -> dict[str, str]:
    """What this study calls each landmark, where that differs from its name."""
    return schemes.study_landmarks(profile_dir)[1]


#: Coordinate written for a landmark the annotator did not place. Negative by
#: convention so ``readland.tps(negNA = TRUE)`` turns it into NA.
MISSING = -1.0


def find_image(images: Path, fish_id: str) -> Path | None:
    """The photograph, under the name it actually has on disk.

    Matched without regard to case, because the suffix a series uses varies, but
    returned as the directory spells it: ``Label`` in the exported table is how a
    row is joined to anything else, and on a case-insensitive filesystem testing
    ``images / f"{fish_id}_L.jpeg"`` happily opens ``..._L.JPEG`` and then reports
    the name that was asked for rather than the one that exists.
    """
    try:
        lookup = {p.name.lower(): p for p in images.iterdir() if p.is_file()}
    except OSError:
        return None
    for suf in IMAGE_SUFFIXES:
        for stem in (fish_id, f"{fish_id}_L"):
            hit = lookup.get(f"{stem}{suf}".lower())
            if hit is not None:
                return hit
    return None


def image_height(path: Path) -> int | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size[1]
    except Exception:
        return None


def px_per_mm(sidecar: dict) -> float | None:
    """Scale factor, or None when the series was shot without a usable reference."""
    cal = (sidecar.get("lateral") or {}).get("calibration") or {}
    mode = cal.get("mode", "manual")
    if mode == "none":
        return None
    if mode == "ticks":
        return float(cal["px_per_mm"])
    a, b = cal.get("point_a"), cal.get("point_b")
    known = float(cal.get("known_mm") or 0)
    if not a or not b or known <= 0:
        return None
    return math.hypot(b[0] - a[0], b[1] - a[1]) / known


PER_SPECIMEN_README = """\
Landmarks, one file per specimen
================================

Each CSV is one specimen, laid out as ImageJ writes a Multi-Measure result:

    (row), Label, X, Y
    1,     <photograph>, x, y
    ...
    N,     <photograph>, x, y

Row n is landmark n in every file; landmark_key.csv says which landmark that is.
X and Y are measured from the photograph's TOP-LEFT corner, Y DOWNWARD (ImageJ's
default), in {units} -- except a specimen with no scale, which is in pixels
(specimens.csv says which). A landmark not placed on a specimen is NA; its row is
kept, so row n is always landmark n.

specimens.csv   one row per file: the specimen, its strain, units, scale, and
                how many landmarks it lacks; who placed them, where recorded;
                then any Darwin Core records the study keeps (catalogNumber,
                scientificName, locality ...)
landmark_key.csv
read_specimens.R  reads the folder into a geomorph array and runs a Procrustes
                  superimposition and PCA on the complete specimens
"""

READ_SPECIMENS_R = """\
# Read a folder of per-specimen landmark files (ImageJ layout) into geomorph.
# Run from this folder:  setwd("<this folder>");  source("read_specimens.R", echo = TRUE)
library(geomorph)

key <- read.csv("landmark_key.csv")
sp  <- read.csv("specimens.csv")

# p landmarks x 2 coordinates x n specimens, in the order of specimens.csv
A <- simplify2array(lapply(sp$file, function(f)
  as.matrix(read.csv(f, check.names = FALSE)[, c("X", "Y")])))
dimnames(A) <- list(key$name, c("X", "Y"), sp$ID)
A[, "Y", ] <- -A[, "Y", ]     # ImageJ's Y runs down the photograph; flip it so plots are upright

# Procrustes needs every landmark on every specimen.
complete <- apply(A, 3, function(m) !anyNA(m))
cat(sum(!complete), "of", length(complete), "specimens lack a landmark and are left out\n")
A  <- A[, , complete]
sp <- sp[complete, ]

# Units must agree for centroid size to mean one thing; shape is unaffected either way.
if (length(unique(sp$units)) > 1) warning("specimens are in different units: ",
                                          paste(unique(sp$units), collapse = ", "))

gpa <- gpagen(A, print.progress = FALSE)
dimnames(gpa$coords) <- dimnames(A)      # gpagen drops the landmark names
pca <- gm.prcomp(gpa$coords)
print(summary(pca))
plot(pca, pch = 19, col = as.integer(factor(sp$group)))
legend("topright", legend = levels(factor(sp$group)), col = seq_along(levels(factor(sp$group))),
       pch = 19, bty = "n")
"""


def write_per_specimen(folder: Path, specimens, order, labels, metas, units: str,
                       ds_dir: Path, straightened=frozenset()) -> Path:
    """One CSV per specimen in ImageJ's Multi-Measure layout, with a key beside it.

    The layout a folder-per-series workflow in R expects: every file the same
    landmarks in the same rows, so reading them in order gives a p x 2 x n array
    ready for gpagen. A missing landmark keeps its row as NA -- dropping the row
    would shift every landmark after it onto the wrong number.
    """
    from fish_morpho import darwin_core, grouping
    folder.mkdir(parents=True, exist_ok=True)
    # Only what this writes is cleared, so an older export does not leave a
    # specimen behind that is no longer labelled.
    for old in folder.glob("*.csv"):
        old.unlink()
    per_mm = {"cm": 10.0, "mm": 1.0}[units]
    g_table = grouping.load_group_table(ds_dir)
    g_pattern = grouping.filename_pattern(ds_dir)
    dwc = darwin_core.load(ds_dir)
    terms = darwin_core.used_terms(dwc, [s[0] for s in specimens])
    with_ops = any((metas.get(s[0]) or {}).get("_operators") for s in specimens)
    rows = []
    for fid, kps, img, h, ppm in specimens:
        name = f"{fid}.csv"
        with (folder / name).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([" ", "Label", "X", "Y"])
            for i, n in enumerate(order, start=1):
                pt = kps.get(n)
                if pt is None:
                    w.writerow([i, img.name, "NA", "NA"])
                    continue
                x, y = (pt[0] / ppm / per_mm, pt[1] / ppm / per_mm) if ppm else (pt[0], pt[1])
                w.writerow([i, img.name, f"{float(x):.3f}", f"{float(y):.3f}"])
        meta = metas.get(fid) or {}
        rows.append([name, fid, grouping.resolve(fid, meta, g_table, g_pattern),
                     units if ppm else "px", f"{ppm:.4f}" if ppm else "",
                     sum(1 for n in order if n not in kps),
                     (meta.get("data_note") or "").replace("\n", " "),
                     *([meta.get("_operators", "")] if with_ops else []),
                     *(["yes" if fid in straightened else ""] if straightened else []),
                     *((dwc.get(fid) or {}).get(t, "") for t in terms)])
    with (folder / "specimens.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "ID", "group", "units", "px_per_mm", "landmarks_missing", "data_note",
                    *(["operators"] if with_ops else []),
                    *(["straightened"] if straightened else []), *terms])
        w.writerows(rows)
    with (folder / "landmark_key.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "name", "label"])
        for i, n in enumerate(order, start=1):
            w.writerow([i, n, labels.get(n, n)])
    (folder / "README.txt").write_text(PER_SPECIMEN_README.format(units=units))
    (folder / "read_specimens.R").write_text(READ_SPECIMENS_R)
    print(f"wrote {len(rows)} specimen files ({units}) to {folder}")
    return folder


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_tps")
    ap.add_argument("--sidecars", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="landmarks",
                    help="Base filename for the .tps (default: landmarks).")
    ap.add_argument("--schema-dir", type=Path, default=None,
                    help="Directory holding schema.json. Defaults to the image "
                         "folder's parent.")
    ap.add_argument("--force-scale", action="store_true",
                    help="Write SCALE even when only some specimens have one. Off "
                         "by default: it makes centroid size millimetres for some "
                         "specimens and pixels for others, in one column.")
    ap.add_argument("--per-specimen", type=Path, default=None,
                    help="Also write a folder with one CSV per specimen, laid out as "
                         "ImageJ writes one: landmark 1..N, Label, X, Y.")
    ap.add_argument("--units", choices=("cm", "mm"), default="cm",
                    help="Units of the per-specimen files (default cm, as ImageJ "
                         "is usually set). A specimen with no scale is in pixels.")
    ap.add_argument("--straighten", action="store_true",
                    help="Write a fish that has a midline straightened along it, as "
                         "MorFishJ straightens a bent specimen (see fish_morpho."
                         "straighten). A fish without one is written as clicked. "
                         "specimens.csv says which were straightened.")
    ap.add_argument("--require-complete", action="store_true",
                    help="Skip specimens missing any landmark instead of writing "
                         "negatives. geomorph can estimate missing landmarks, but "
                         "only if you would rather it did not.")
    args = ap.parse_args(argv)

    # the profile lives beside the image folder, e.g. data/alewife/schema.json
    profile_dir = args.schema_dir or args.images.parent
    order, labels = schemes.study_landmarks(profile_dir)
    globals()["LANDMARK_ORDER"] = order
    globals()["LANDMARK_LABELS"] = labels

    args.out.mkdir(parents=True, exist_ok=True)
    tps_path = args.out / f"{args.name}.tps"

    # Pass 1: collect, so the landmark set and the scale policy can be decided
    # from the whole series rather than per specimen.
    specimens: list[tuple[str, dict, Path, int, float | None]] = []
    metas: dict[str, dict] = {}
    straightened: set[str] = set()
    skipped = 0
    for path in sorted(args.sidecars.glob("*.json")):
        sc = json.loads(path.read_text())
        fid = sc.get("fish_id", path.stem)
        kps = ((sc.get("lateral") or {}).get("keypoints")) or {}
        if not kps:
            skipped += 1
            continue
        if args.straighten:
            from fish_morpho import straighten
            try:
                block, done = straighten.straighten_block(sc.get("lateral") or {})
            except ValueError as exc:
                print(f"  ! {fid}: midline unusable ({exc}) — written as clicked")
                block, done = {}, {}
            if done:
                kps = block["keypoints"]
                straightened.add(fid)

        img = find_image(args.images, fid)
        h = image_height(img) if img else None
        if h is None:
            print(f"  ! {fid}: no image found, cannot flip y — skipped")
            skipped += 1
            continue

        specimens.append((fid, kps, img, h, px_per_mm(sc)))
        metas[fid] = dict(sc.get("metadata") or {})
        from fish_morpho import operators
        if operators.summary(sc):
            metas[fid]["_operators"] = operators.summary(sc)

    if not specimens:
        print("No labelled specimens found.")
        return 1

    # A landmark nobody has placed cannot be exported as all-NA: geomorph's
    # estimate.missing() infers a missing point from the same point in other
    # specimens, so with none to learn from it fails with a subscript error that
    # says nothing about the cause. Drop those columns and say which.
    never = [n for n in order if not any(n in k for _, k, _, _, _ in specimens)]
    if never:
        order = tuple(n for n in order if n not in never)
        globals()["LANDMARK_ORDER"] = order

    # Mixing scaled and unscaled specimens in one file silently mixes units:
    # Procrustes removes scale so shape survives, but centroid size would be
    # millimetres for some specimens and pixels for others. Refuse the mixture.
    n_scaled = sum(1 for *_, ppm in specimens if ppm)
    mixed = 0 < n_scaled < len(specimens)
    use_scale = (n_scaled == len(specimens)) or (mixed and args.force_scale)

    rows: list[str] = []
    written = 0
    incomplete: list[tuple[str, int]] = []
    for fid, kps, img, h, ppm in specimens:
        missing = [n for n in order if n not in kps]
        if missing and args.require_complete:
            incomplete.append((fid, len(missing)))
            skipped += 1
            continue
        if missing:
            incomplete.append((fid, len(missing)))

        rows.append(f"LM={len(order)}")
        for name in order:
            pt = kps.get(name)
            if pt is None:
                rows.append(f"{MISSING} {MISSING}")
            else:
                rows.append(f"{float(pt[0]):.4f} {h - float(pt[1]):.4f}")  # y: top-left -> bottom-left
        rows.append(f"IMAGE={img.name}")
        rows.append(f"ID={fid}")
        if use_scale and ppm:
            rows.append(f"SCALE={1.0 / ppm:.8f}")   # TPS SCALE multiplies px -> mm
        rows.append("")
        written += 1

    tps_path.write_text("\n".join(rows))

    # The same coordinates in the shape ImageJ's Multi-Measure writes: one row per
    # landmark per specimen, the landmark named, the photograph repeated. Image
    # coordinates -- y DOWNWARD, as ImageJ gives them, not the Cartesian y of the
    # .tps beside it -- so a series digitised in ImageJ and one digitised here can
    # be pooled. Millimetres where the specimen has a scale, pixels where it does
    # not, stated per row.
    imagej_path = args.out / "landmarks_imagej.csv"
    with imagej_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["landmark", "Label", "X", "Y", "units"])
        for fid, kps, img, h, ppm in specimens:
            for name in order:
                pt = kps.get(name)
                label = LANDMARK_LABELS.get(name, name)
                if pt is None:
                    w.writerow([label, img.name, "", "", "mm" if ppm else "px"])
                    continue
                x, y = (pt[0] / ppm, pt[1] / ppm) if ppm else (pt[0], pt[1])
                w.writerow([label, img.name, f"{float(x):.3f}", f"{float(y):.3f}",
                            "mm" if ppm else "px"])

    # One row per specimen: what an analysis needs beside the coordinates. The
    # group, so a test does not have to parse it out of an ID with a regex; and
    # the specimen's own scale, so centroid size can be put in millimetres in R
    # even though the .tps is in pixels throughout. The .tps stays in pixels on
    # purpose -- half a series rescaled and half not makes centroid size mean two
    # different things -- and this is how to get millimetres anyway.
    # Then the study's Darwin Core records -- catalogue number, scientific name,
    # locality -- so a specimen in R can be traced to the museum's record.
    from fish_morpho import darwin_core, grouping
    ds_dir = args.schema_dir or args.sidecars.parent
    g_table = grouping.load_group_table(ds_dir)
    g_pattern = grouping.filename_pattern(ds_dir)
    dwc = darwin_core.load(ds_dir)
    terms = darwin_core.used_terms(dwc, [s[0] for s in specimens])
    spec_path = args.out / "specimens.csv"
    with spec_path.open("w", newline="") as f:
        w = csv.writer(f)
        with_ops = any((metas.get(s[0]) or {}).get("_operators") for s in specimens)
        w.writerow(["ID", "group", "px_per_mm", "landmarks_missing", "data_note",
                    *(["operators"] if with_ops else []),
                    *(["straightened"] if straightened else []), *terms])
        for fid, kps, img, h, ppm in specimens:
            meta = metas.get(fid) or {}
            w.writerow([fid, grouping.resolve(fid, meta, g_table, g_pattern),
                        f"{ppm:.4f}" if ppm else "",
                        sum(1 for n in order if n not in kps),
                        (meta.get("data_note") or "").replace("\n", " "),
                        *([meta.get("_operators", "")] if with_ops else []),
                        *(["yes" if fid in straightened else ""] if straightened else []),
                        *((dwc.get(fid) or {}).get(t, "") for t in terms)])

    names_path = args.out / "landmark_names.csv"
    with names_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "name", "label"])
        for i, n in enumerate(LANDMARK_ORDER, start=1):
            w.writerow([i, n, LANDMARK_LABELS.get(n, n)])

    if args.per_specimen:
        write_per_specimen(args.per_specimen, specimens, LANDMARK_ORDER, LANDMARK_LABELS,
                           metas, args.units, args.schema_dir or args.sidecars.parent,
                           straightened)
    if straightened:
        print(f"  {len(straightened)} specimen(s) straightened along their midline")

    # The study this came from, so the snippet does not title a brook trout
    # plot "Alewife" -- which it did, in live code sitting under a block of
    # commented-out lines where it read as commentary.
    study = (args.schema_dir or args.sidecars.parent).name
    r_path = args.out / "load_landmarks.R"
    # The analysis script is a real .R file in scripts/r/, copied here with the
    # study's name filled in. It used to be a Python f-string, where every R
    # brace had to be doubled and nothing could run it -- which is how a brook
    # trout export came to title its shape space "Alewife".
    template = (_ROOT / "scripts" / "r" / "load_landmarks.R").read_text()
    r_path.write_text(template.replace("@STUDY@", study))

    print(f"wrote {tps_path}  ({written} specimens, {len(LANDMARK_ORDER)} landmarks each)")
    print(f"      {names_path}")
    print(f"      {r_path}")
    if incomplete:
        print(f"\n{len(incomplete)} specimen(s) missing landmarks "
              f"(written as {MISSING}, read back as NA):")
        for fid, n in incomplete[:10]:
            print(f"  {fid}: {n} missing")
    if never:
        print(f"\ndropped {len(never)} landmark(s) that no specimen has yet: "
              f"{', '.join(never)}\n  (all-NA columns make estimate.missing() fail; "
              f"re-export once they are labelled)")
    if mixed and not args.force_scale:
        print(f"\n{n_scaled} of {len(specimens)} specimens carry a scale. SCALE was "
              f"written for NONE of them, so every coordinate is in PIXELS and "
              f"centroid size is comparable.\n  Procrustes removes scale, so shape "
              f"analysis is unaffected. Use --force-scale to override.")
    elif not use_scale:
        print("\nNo specimen has a scale reference; coordinates are PIXELS. "
              "Procrustes removes scale, so shape analysis is unaffected.")
    if skipped:
        print(f"\nskipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
