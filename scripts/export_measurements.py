"""Measure every labelled specimen in a dataset and write the Excel workbook.

A thin wrapper over ``fish_morpho.pipeline`` that knows the repository's layout,
so producing the spreadsheet is one command with a dataset name rather than three
paths that have to agree with each other.

    python scripts/export_measurements.py --dataset alewife

Writes ``results/<dataset>/measurements.xlsx`` with six sheets:

  About         what the file is, the commit it came from, the millimetre /
                pixel split, and the check counts — so the workbook can be read
                by someone who did not run it.
  Measurements  one row per specimen, one column per trait, plus a ``units``
                column — a series photographed without a usable scale reference
                measures in PIXELS, and mixing those with millimetres under one
                header is a mistake nothing downstream can catch.
  Ratios        every length divided by standard length and every area by SL
                squared, so the numbers are dimensionless and comparable between
                fish of different sizes. This is the sheet a between-population
                comparison wants, and the only meaningful one when there is no
                scale.
  Shape         Mosimann log-shape variables: each length over the geometric
                mean of all lengths, logged. The defensible size correction when
                the groups being compared may differ in body size.
  QC            calibration method and confidence per view, which landmarks were
                missing, and any recorded data compromise.
  Validation    the automated checks, most severe first. Read before analysing.

and, when the study has Darwin Core records (darwin_core.csv in its folder):

  Specimens          each specimen's institution, catalogue number, scientific
                     name, locality ... under their Darwin Core term names
  MeasurementOrFact  the Measurements sheet in Darwin Core's long form, one row
                     per value with its unit, who placed its landmarks, and the
                     definition it was measured by

and, when the study has a blind re-label round (see fish_morpho.repeatability):

  Measurement error  ICC and %ME per trait, from the fish labelled twice

Specimens that cannot be processed are named and skipped rather than aborting the
batch.

Choosing what goes in, for an analysis that cannot take blanks (a PCA in R drops
every row with an NA):

    --preview-json F     also write the Measurements sheet as JSON, with each
                         column's count of blanks -- what the labeller's export
                         preview shows before anything is saved
    --leave-out SL,Jl    leave these trait columns out of every sheet
    --complete-only      keep only the specimens with a value in every column
                         that remains

Both choices are recorded on the About sheet.

    --csv F              also write the Measurements sheet as CSV, for R: one row
                         per specimen, trait codes as the headers (SL, CPd, ...),
                         blanks as NA
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from fish_morpho import darwin_core, repeatability  # noqa: E402
from fish_morpho.landmark_config import traits_requiring  # noqa: E402
from fish_morpho.pipeline import run  # noqa: E402


def dropped_traits(dataset_dir: Path) -> tuple[str, ...]:
    """Traits the dataset's schema profile puts out of scope.

    A landmark the study never collects would otherwise leave a column of blanks,
    which reads as "measured and missing" rather than "never in scope".
    """
    f = dataset_dir / "schema.json"
    if not f.is_file():
        return ()
    try:
        prof = json.loads(f.read_text())
    except Exception:
        return ()
    return tuple(traits_requiring(prof.get("exclude_keypoints") or [],
                                  prof.get("exclude_polygons") or []))

log = logging.getLogger("export_measurements")


def write_csv(workbook: Path, target: Path) -> Path:
    """The Measurements sheet as a CSV that R reads without coaxing.

    Headers are the trait codes alone. The workbook's "SL — Standard Length"
    carries an em-dash that read.csv mangles under the wrong encoding, and a
    code is what a loadings plot has room to print. Blank traits are written NA,
    so R cannot read an empty cell any other way. UTF-8, no index column.
    """
    import csv
    from openpyxl import load_workbook
    ws = load_workbook(workbook, read_only=True)["Measurements"]
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    u = hdr.index("units")
    head = [*hdr[:u + 1], *(str(h).split(" — ")[0] for h in hdr[u + 1:])]
    with open(target, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in it:
            w.writerow([*("" if v is None else v for v in r[:u + 1]),
                        *("NA" if v in ("", None) else v for v in r[u + 1:])])
    return target


def preview(workbook: Path) -> dict:
    """The Measurements sheet as the export preview shows it.

    One entry per trait column with its count of blanks, and one per specimen
    with its values, blank as None. Read back from the written workbook rather
    than recomputed, so the preview is exactly what the file holds.
    """
    from openpyxl import load_workbook
    ws = load_workbook(workbook, read_only=True)["Measurements"]
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    u = hdr.index("units")
    cols = []
    for h in hdr[u + 1:]:
        code, _, label = str(h).partition(" — ")
        cols.append({"code": code, "label": label or code, "missing": 0})
    rows = []
    for r in it:
        vals = [None if v in ("", None) else v for v in r[u + 1:]]
        for c, v in zip(cols, vals):
            c["missing"] += v is None
        rows.append({"id": r[hdr.index("fish_id")],
                     "group": r[hdr.index("group")] if "group" in hdr else "",
                     "units": r[u], "values": vals})
    return {"columns": cols, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_measurements")
    ap.add_argument("--dataset", help="Folder name under --data-root, e.g. alewife.")
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    ap.add_argument("--images", type=Path, default=None,
                    help="Override the image directory.")
    ap.add_argument("--labels", type=Path, default=None,
                    help="Override the sidecar directory.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--leave-out", default="",
                    help="Comma-separated trait codes to leave out, e.g. DFa,PlFa.")
    ap.add_argument("--complete-only", action="store_true",
                    help="Keep only specimens with a value in every remaining column.")
    ap.add_argument("--preview-json", type=Path, default=None,
                    help="Also write the Measurements sheet here as JSON.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Also write the Measurements sheet here as CSV, for R.")
    ap.add_argument("--log-level", default="WARNING",
                    choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(message)s")

    if args.images and args.labels:
        images, labels = args.images, args.labels
        name = args.dataset or images.parent.name
    else:
        if not args.dataset:
            avail = sorted(d.name for d in args.data_root.iterdir()
                           if d.is_dir() and (d / "lateral").is_dir()) \
                    if args.data_root.is_dir() else []
            ap.error("--dataset is required (or pass --images and --labels). "
                     f"Available: {', '.join(avail) or 'none found'}")
        base = args.data_root / args.dataset
        images, labels, name = base / "lateral", base / "sidecars", args.dataset
        if not images.is_dir():
            ap.error(f"{images} does not exist")

    out = args.out or (_ROOT / "results" / name / "measurements.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    left_out = tuple(c.strip() for c in args.leave_out.split(",") if c.strip())

    # A study on another landmark scheme has no caliPr traits; its measurements
    # are its landmark coordinates. See scheme_export.
    from scheme_export import export as export_scheme, scheme_of
    if scheme_of(images.parent):
        try:
            s = export_scheme(images.parent, labels, out, csv_path=args.csv,
                              preview_json=args.preview_json, left_out=left_out,
                              complete_only=args.complete_only)
        except Exception as exc:
            log.error("%s", exc)
            return 1
        dwc = darwin_core.add_sheets(out, images.parent, measurements=False)
        print(f"wrote {out}")
        print(f"  {s['specimens']} specimens x {s['landmarks']} landmarks ({s['scheme']}): "
              f"coordinates, {s['mm']} in mm — this scheme has no caliPr traits")
        if dwc["with_record"]:
            print(f"  Darwin Core records for {dwc['with_record']} of {dwc['specimens']} "
                  f"specimens (Specimens sheet)")
        if args.csv:
            print(f"wrote {args.csv}")
        return 0

    try:
        written = run(images_dir=images, labels_dir=labels, output_path=out,
                      mode="manual", model_config=None,
                      drop_traits=dropped_traits(images.parent),
                      left_out=left_out, complete_only=args.complete_only)
    except Exception as exc:
        log.error("%s", exc)
        return 1
    dwc = darwin_core.add_sheets(written, images.parent, labels_dir=labels)
    me = repeatability.add_sheet(written, images.parent, labels)
    if args.preview_json:
        args.preview_json.write_text(json.dumps(preview(written)))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(written, args.csv)
        print(f"wrote {args.csv}")

    from openpyxl import load_workbook
    wb = load_workbook(written)
    ws = wb["Measurements"]
    hdr = [c.value for c in ws[1]]
    u = hdr.index("units")
    units = [r[u] for r in ws.iter_rows(min_row=2, values_only=True)]
    dropped = dropped_traits(images.parent)
    print(f"wrote {written}")
    if dropped:
        print(f"  omitted {len(dropped)} trait column(s) out of scope for this "
              f"study: {', '.join(sorted(dropped))}")
    if left_out:
        print(f"  left out at your request: {', '.join(left_out)}")
    if dwc["with_record"]:
        print(f"  Darwin Core records for {dwc['with_record']} of {dwc['specimens']} "
              f"specimens; {dwc['measurement_rows']} MeasurementOrFact rows")
    if me.get("designs"):
        print(f"  measurement error: {me['fish']} fish re-labelled blind, ICC and %ME "
              f"for {me['traits']} traits (Measurement error sheet)")
    print(f"  {ws.max_row - 1} specimens x {len(hdr)} columns, "
          f"sheets: {', '.join(wb.sheetnames)}")
    if units:
        mm, px = units.count("mm"), units.count("px")
        print(f"  units: {mm} in mm, {px} in pixels"
              + ("   <- MIXED; use the Ratios sheet to compare across rows"
                 if mm and px else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
