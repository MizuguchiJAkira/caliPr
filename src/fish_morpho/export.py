"""Excel export for a batch of measurement sets.

Six sheets, because a bare measurement table answers fewer questions than it
appears to: the numbers alone cannot say what units a row is in, how a group
comparison should be size-corrected, or whether anything about the batch looked
wrong on the way out. About, Ratios, Shape, QC and Validation each carry one of
those, so the workbook explains itself to someone who did not run it.

We use openpyxl directly (no pandas) to keep dependencies light.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.worksheet import Worksheet

from .landmark_config import Unit
from .measurement_engine import (
    MeasurementSet,
    measurement_column_order,
    measurement_labels,
)
from .ruler_calibration import CalibrationResult


DEFAULT_METADATA_COLUMNS: tuple[str, ...] = (
    "fish_id",
    # Second, not last: morphometrics is comparative, and the column a reader
    # sorts and filters by should sit beside the identifier rather than after
    # three fields they rarely use.
    "group",
    "locality",
    "collection_date",
    "image_filename",
)


@dataclass
class ExportRecord:
    """One specimen's data packaged for export."""

    measurements: MeasurementSet
    # Calibrations per view — rendered on the QC sheet for provenance.
    calibrations: dict[str, CalibrationResult]
    image_filename: str = ""
    # Raw annotation and frame size, carried so the validation pass can check
    # things the measurements alone cannot reveal: a mirrored specimen, or a
    # sidecar written against a different photograph.
    keypoints: dict[str, tuple[float, float]] = field(default_factory=dict)
    polygons: dict[str, list] = field(default_factory=dict)
    image_size: tuple[int, int] | None = None


def export_to_xlsx(
    records: Sequence[ExportRecord],
    output_path: str | Path,
    metadata_columns: Iterable[str] = DEFAULT_METADATA_COLUMNS,
    drop_traits: Iterable[str] = (),
    issues: Sequence = None,
    provenance: dict | None = None,
    landmark_labels: dict[str, str] | None = None,
) -> Path:
    """Write ``records`` to an xlsx workbook at ``output_path``.

    The workbook has five sheets:

    * ``About`` — what the file is, when and from which commit it came, the
      unit split, the check counts, and any traits held out of scope.
    * ``Measurements`` — metadata columns + one column per measurement,
      with numeric values in mm / mm^2.
    * ``Ratios`` — every length over standard length and every area over SL
      squared, so the values are dimensionless and comparable between fish of
      different sizes. The only sheet where a scale-free specimen and a
      calibrated one can honestly share a column.
    * ``Shape`` — Mosimann log-shape variables, each length divided by the
      geometric mean of one fixed set of body and head lengths and logged. The
      size correction to use when comparing groups that may differ in size; see
      ``_write_shape_sheet``.
    * ``QC`` — calibration method / confidence / notes per view, plus a
      ``missing_landmarks`` column summarizing any gaps.
    * ``Validation`` — the checks from :mod:`fish_morpho.validation`, most
      severe first. Written only when ``issues`` is passed.

    Returns the resolved path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    meas_sheet = wb.active
    assert meas_sheet is not None
    meas_sheet.title = "Measurements"
    drop = set(drop_traits)
    _write_measurements_sheet(meas_sheet, records, list(metadata_columns), drop)

    ratio_sheet = wb.create_sheet("Ratios")
    _write_ratios_sheet(ratio_sheet, records, list(metadata_columns), drop)

    shape_sheet = wb.create_sheet("Shape")
    _write_shape_sheet(shape_sheet, records, list(metadata_columns), drop)

    qc_sheet = wb.create_sheet("QC")
    _write_qc_sheet(qc_sheet, records)

    if issues is not None:
        _write_validation_sheet(wb.create_sheet("Validation"), issues)

    # Last so it lands rightmost, first so it is what opens: moved to index 0.
    about = wb.create_sheet("About")
    _write_about_sheet(about, records, drop, issues, provenance or {})
    wb.move_sheet("About", offset=-(len(wb.sheetnames) - 1))
    wb.active = 0

    wb.save(output_path)
    return output_path


def _unit_of(rec) -> str | None:
    """"mm", "px", or None when the specimen carries no calibration at all."""
    lat = rec.calibrations.get("lateral")
    return None if lat is None else ("px" if lat.method == "none" else "mm")


def _ordered(records):
    """Records sorted by group, then by fish_id.

    Grouping a spreadsheet is most of what makes it usable for a comparison, and
    a reader should not have to sort it themselves. Ungrouped specimens sort last
    rather than first, where an empty string would otherwise put them.
    """
    def key(rec):
        g = str(rec.measurements.metadata.get("group", "") or "")
        return (g == "", g, rec.measurements.fish_id)
    return sorted(records, key=key)


def _write_measurements_sheet(
    sheet: Worksheet,
    records: Sequence[ExportRecord],
    metadata_columns: list[str],
    drop_traits: set[str] = frozenset(),
) -> None:
    measurement_keys = [k for k in measurement_column_order() if k not in drop_traits]
    labels = measurement_labels()

    # A workbook can hold both scaled and scale-free specimens -- a series shot
    # without a usable ruler measures in PIXELS -- so the units of each row travel
    # with the row, in a column of their own.
    #
    # The header has to agree with them. Where every row shares one unit the
    # header states it; where they differ no single unit is true of the column,
    # so the header carries none and the row's own value is the only answer. What
    # it must never do is say "(mm)" over a column of pixels, which is what a
    # fixed header did.
    units_present = {_unit_of(r) for r in records} - {None}
    only = units_present.pop() if len(units_present) == 1 else None

    def head(key: str) -> str:
        label = labels[key]
        if only == "mm":
            return label
        return (label.replace("(mm^2)", "(px^2)").replace("(mm)", "(px)")
                if only == "px"
                else label.replace(" (mm^2)", "").replace(" (mm)", ""))

    header = [*metadata_columns, "units", *(head(k) for k in measurement_keys)]
    sheet.append(header)

    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="E6E6E6")
    for col_idx in range(1, len(header) + 1):
        cell = sheet.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for rec in _ordered(records):
        row: list[float | str] = []
        for col in metadata_columns:
            if col == "image_filename":
                row.append(rec.image_filename)
            elif col == "fish_id":
                row.append(rec.measurements.fish_id)
            else:
                row.append(rec.measurements.metadata.get(col, ""))
        # Three states, not two. A specimen with no lateral calibration at all is
        # not "in pixels" -- it has no lateral measurements to have units for, and
        # saying px would invent a claim about empty cells.
        lat = rec.calibrations.get("lateral")
        row.append("" if lat is None else ("px" if lat.method == "none" else "mm"))
        for key in measurement_keys:
            v = rec.measurements.values.get(key)
            if v is None or math.isnan(v.value):
                row.append("")
            else:
                row.append(round(v.value, 3))
        sheet.append(row)

    # Reasonable column widths.
    for col_idx in range(1, len(header) + 1):
        letter = sheet.cell(row=1, column=col_idx).column_letter
        sheet.column_dimensions[letter].width = max(
            14, min(40, len(str(header[col_idx - 1])) + 2)
        )


def _write_ratios_sheet(
    sheet: Worksheet,
    records: Sequence[ExportRecord],
    metadata_columns: list[str],
    drop_traits: set[str] = frozenset(),
) -> None:
    """Size-corrected traits: lengths / SL, areas / SL^2, angles unchanged.

    Comparing raw lengths between populations mostly compares how big the fish
    happened to be, so a shape comparison wants ratios; forming them here means
    everyone forms them the same way.

    Ratios also need no calibration. Every landmark on a planar specimen shares
    one magnification, so trait/SL is exact whatever that magnification is --
    which makes this the only sheet where a scale-free specimen and a calibrated
    one can honestly sit in the same column.
    """
    # SL is the denominator here, so its own column would read 1 on every row.
    keys = [k for k in measurement_column_order()
            if k not in drop_traits and k != "SL"]
    labels = measurement_labels()

    # Say what each column was actually divided by. The header used to read
    # "/SL" on every column, including the angles (which are not divided) and
    # the areas (which are divided by SL squared).
    from .landmark_config import TRAITS as _T
    unit_of = {t.code: t.unit for t in _T}

    def head(k):
        u = unit_of.get(k)
        if u == Unit.DEG:
            return f"{labels[k]}"
        if u == Unit.MM2:
            return f"{labels[k]} /SL²"
        return f"{labels[k]} /SL"

    header = [*metadata_columns, *(head(k) for k in keys)]
    sheet.append(header)
    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="E6E6E6")
    for col_idx in range(1, len(header) + 1):
        cell = sheet.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for rec in _ordered(records):
        sl_val = rec.measurements.values.get("SL")
        sl = sl_val.value if sl_val is not None else float("nan")
        row: list[float | str] = []
        for col in metadata_columns:
            if col == "image_filename":
                row.append(rec.image_filename)
            elif col == "fish_id":
                row.append(rec.measurements.fish_id)
            else:
                row.append(rec.measurements.metadata.get(col, ""))
        for key in keys:
            v = rec.measurements.values.get(key)
            if v is None or math.isnan(v.value) or not _cross_view_ok(rec, key):
                row.append("")
            elif v.unit == Unit.DEG:            # an angle is already scale-free
                row.append(round(v.value, 3))
            elif math.isnan(sl) or sl <= 0:
                row.append("")                  # no SL, no ratio
            elif v.unit == Unit.MM2:
                row.append(round(v.value / (sl ** 2), 6))
            else:
                row.append(round(v.value / sl, 6))
        sheet.append(row)

    for col_idx in range(1, len(header) + 1):
        letter = sheet.cell(row=1, column=col_idx).column_letter
        sheet.column_dimensions[letter].width = max(
            14, min(40, len(str(header[col_idx - 1])) + 2))


def _write_validation_sheet(sheet: Worksheet, issues: Sequence) -> None:
    """Every check that fired, most severe first.

    In the workbook rather than only in a terminal, because the workbook is what
    gets emailed, opened months later, and handed to someone who never ran the
    export. A caveat that lives in a console scrollback is a caveat nobody has.
    """
    sheet.append(["level", "check", "fish_id", "detail"])
    for i, c in enumerate(sheet[1], start=1):
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="E6E6E6")
    colour = {"error": "FFC7CE", "warning": "FFEB9C", "note": "EDEDED"}
    for iss in issues:
        sheet.append([iss.level, iss.check, iss.fish_id, iss.message])
        fill = PatternFill("solid", fgColor=colour.get(iss.level, "FFFFFF"))
        sheet.cell(row=sheet.max_row, column=1).fill = fill
    if not issues:
        sheet.append(["", "", "", "No checks fired."])
    for col, w in zip("ABCD", (10, 22, 34, 110)):
        sheet.column_dimensions[col].width = w
    sheet.freeze_panes = "A2"


def _write_about_sheet(sheet: Worksheet, records, drop_traits, issues,
                       provenance: dict) -> None:
    """What this file is, how it was made, and what not to do with it."""
    def unit_of(rec):
        lat = rec.calibrations.get("lateral")
        return None if lat is None else ("px" if lat.method == "none" else "mm")

    left_out = list(provenance.get("left_out") or [])
    incomplete = provenance.get("incomplete")
    out_of_scope = set(drop_traits) - set(left_out)
    mm = sum(1 for r in records if unit_of(r) == "mm")
    px = sum(1 for r in records if unit_of(r) == "px")
    counts = {"error": 0, "warning": 0, "note": 0}
    for i in (issues or []):
        counts[i.level] = counts.get(i.level, 0) + 1

    rows: list[tuple[str, str]] = [
        ("caliPr measurement export", ""),
        ("", ""),
        ("dataset", str(provenance.get("dataset", ""))),
        ("generated", provenance.get("generated", "")),
        ("caliPr commit", provenance.get("commit", "unknown")),
        ("specimens", str(len(records))),
        ("", ""),
        ("SHEETS", ""),
        ("Measurements", "One row per specimen, one column per trait. The 'units' "
                         "column says whether THAT ROW is millimetres or pixels."),
        ("Ratios", "Each length over standard length, each area over SL squared. "
                   "Dimensionless. Assumes shape does not change with size."),
        ("Shape", "Mosimann log-shape variables: each length over the geometric "
                  "mean of all lengths, logged. Use this to compare groups that "
                  "may differ in body size — it is the defensible size correction."),
        ("QC", "Calibration method and confidence per view, missing landmarks, "
               "and any recorded data compromise, per specimen."),
        ("Validation", "Automated checks. Read this before analysing."),
        ("", ""),
        ("UNITS", ""),
        ("millimetres", f"{mm} specimen(s)"),
        ("pixels (no scale reference)", f"{px} specimen(s)"),
        ("", "A specimen photographed without a usable scale measures in PIXELS. "
             "Never compare a raw length across rows of different units — use "
             "Shape or Ratios, which are unitless."
         if (mm and px) else ""),
        ("", ""),
        ("CHECKS", f"{counts['error']} error(s), {counts['warning']} warning(s), "
                   f"{counts['note']} note(s) — see the Validation sheet"),
        ("", ""),
        ("OUT OF SCOPE", ", ".join(sorted(out_of_scope)) if out_of_scope
                         else "no traits excluded"),
        ("", "Traits the study does not collect have no column at all, rather "
             "than a column of blanks."
         if out_of_scope else ""),
    ]
    straight = [r.measurements.fish_id for r in records
                if r.measurements.metadata.get("straightened")]
    if straight:
        rows += [
            ("", ""),
            ("STRAIGHTENED", f"{len(straight)} specimen(s) measured along a midline"),
            ("", "A bent or tilted fish, measured as MorFishJ measures ImageJ's "
                 "straightened image: each landmark placed by its distance along "
                 "the midline drawn on the photograph and its distance off it. "
                 "The photograph and the landmarks as clicked are unchanged. "
                 "The QC sheet says which specimens and how far each midline turns."),
        ]
    # What the person exporting chose to leave out of this particular file, kept
    # apart from the study's scope: a column dropped for its blanks was measured.
    if left_out or incomplete is not None:
        rows += [
            ("", ""),
            ("LEFT OUT AT EXPORT", ""),
            ("columns", ", ".join(left_out) if left_out else "none"),
        ]
        if incomplete is not None:
            rows.append(("specimens", (
                f"{len(incomplete)} with a blank in a kept column: "
                + ", ".join(incomplete)) if incomplete
                else "none — every specimen had a value in every kept column"))
            rows.append(("", "Complete rows only: every specimen here has a value in "
                             "every column, so an analysis that drops rows with NA "
                             "drops nothing further."))
    for k, v in rows:
        sheet.append([k, v])
    sheet["A1"].font = Font(bold=True, size=14)
    for r in range(1, sheet.max_row + 1):
        a = sheet.cell(row=r, column=1)
        if a.value in ("SHEETS", "UNITS", "CHECKS", "OUT OF SCOPE", "LEFT OUT AT EXPORT"):
            a.font = Font(bold=True)
        sheet.cell(row=r, column=2).alignment = Alignment(wrap_text=True,
                                                          vertical="top")
    sheet.column_dimensions["A"].width = 30
    sheet.column_dimensions["B"].width = 100


def _write_shape_sheet(
    sheet: Worksheet,
    records: Sequence[ExportRecord],
    metadata_columns: list[str],
    drop_traits: set[str] = frozenset(),
) -> None:
    """Mosimann log-shape variables: log(trait) - log(geometric mean of traits).

    The Ratios sheet divides by standard length, which is only a fair size
    correction if shape does not change with size. It usually does, and if one
    group is systematically smaller than the other -- landlocked forms often are --
    dividing by SL leaves size sitting inside the "shape" numbers and a population
    difference can be read where only a size difference exists.

    Dividing instead by the GEOMETRIC MEAN of length traits gives the standard
    isometric size correction. The variables are dimensionless, so a scale-free
    specimen and a calibrated one are directly comparable, and their log scale is
    what the usual multivariate tools (PCA, MANOVA) assume.

    **The size is taken over one fixed set of traits, the same for every fish.**
    It used to be taken over whichever traits each fish happened to have, which
    on the brook trout set meant 13 different sets: the 43 fish with a mouth width
    had it folded into their size and the other 88 did not, shifting every column
    between the two groups by a constant a PCA would read as shape. The set is the
    lateral lengths of body and head -- not fins, whose extent depends on how they
    dried, and not the frontal view, which has its own scale. A fish missing one
    of those gets no size and a blank row, rather than a size nobody else has.

    Every length is still reported against that size, fins and mouth width
    included. Lengths only: an area does not share units with a length, and an
    angle is already scale-free.
    """
    keys = [k for k in measurement_column_order()
            if k not in drop_traits and _trait_unit(k) == Unit.MM]
    size_keys = [k for k in keys if _is_size_trait(k)]
    labels = measurement_labels()

    header = [*metadata_columns,
              f"size (geom. mean of {', '.join(size_keys)})",
              *(f"log {labels[k].split(' — ')[0]}" for k in keys)]
    sheet.append(header)
    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="E6E6E6")
    for col_idx in range(1, len(header) + 1):
        c = sheet.cell(row=1, column=col_idx)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center")

    for rec in _ordered(records):
        vals = {}
        for k in keys:
            mv = rec.measurements.values.get(k)
            if (mv is not None and not math.isnan(mv.value) and mv.value > 0
                    and _cross_view_ok(rec, k)):
                vals[k] = mv.value
        row: list[float | str] = []
        for col in metadata_columns:
            if col == "image_filename":
                row.append(rec.image_filename)
            elif col == "fish_id":
                row.append(rec.measurements.fish_id)
            else:
                row.append(rec.measurements.metadata.get(col, ""))
        if len(size_keys) < 3 or any(k not in vals for k in size_keys):
            # Missing part of the size set: no size, rather than a size computed
            # over different traits from every other fish.
            sheet.append([*row, "", *([""] * len(keys))])
            continue
        gm = math.exp(sum(math.log(vals[k]) for k in size_keys) / len(size_keys))
        row.append(round(gm, 4))
        for k in keys:
            row.append(round(math.log(vals[k] / gm), 6) if k in vals else "")
        sheet.append(row)

    for col_idx in range(1, len(header) + 1):
        letter = sheet.cell(row=1, column=col_idx).column_letter
        sheet.column_dimensions[letter].width = max(
            14, min(40, len(str(header[col_idx - 1])) + 2))


def _trait_unit(code: str):
    from .landmark_config import TRAITS
    for t in TRAITS:
        if t.code == code:
            return t.unit
    return None


#: Landmarks whose position depends on how a fin was held when it dried.
_FIN_LANDMARKS = frozenset({
    "pectoral_insertion_upper", "pectoral_ray_tip", "dorsal_base_center",
    "dorsal_tip", "pelvic_base_center", "pelvic_tip", "anal_base_center",
    "anal_tip", "dorsal_base_anterior", "dorsal_base_posterior",
    "anal_base_anterior", "anal_base_posterior"})


def _trait_view(code: str):
    from .landmark_config import TRAITS
    for t in TRAITS:
        if t.code == code:
            return t.view
    return None


def _cross_view_ok(rec, code: str) -> bool:
    """Whether a trait from another view can be set against this fish's lateral
    size. Only if both are in millimetres: a fish with no lateral scale measures
    its lengths in pixels while its mouth width is in millimetres, and TXD_35's
    mouth width came out at 5% of typical from dividing one by the other."""
    from .landmark_config import View
    if _trait_view(code) != View.FRONTAL:
        return True
    return _unit_of(rec) == "mm"


def _is_size_trait(code: str) -> bool:
    """Whether a trait belongs in the size estimate: a lateral length of the
    body or head, and not of a fin. A fin's extent depends on the posture it
    dried in, which is exactly the variation a size estimate must not carry."""
    from .landmark_config import FIN_POLYGONS, TRAITS, View
    for t in TRAITS:
        if t.code == code:
            return (t.unit == Unit.MM and t.view == View.LATERAL
                    and not set(t.required_keypoints) & _FIN_LANDMARKS
                    and not set(t.required_polygons) & set(FIN_POLYGONS))
    return False


def _write_qc_sheet(sheet: Worksheet, records: Sequence[ExportRecord]) -> None:
    header = [
        "fish_id",
        "image_filename",
        "view",
        "calibration_method",
        "px_per_mm",
        "confidence",
        "calibration_notes",
        "missing_landmarks",
        "operators",
        "straightened",
        "data_note",
    ]
    sheet.append(header)
    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="E6E6E6")
    for col_idx in range(1, len(header) + 1):
        cell = sheet.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill

    for rec in _ordered(records):
        missing = sorted(
            {
                lm
                for mv in rec.measurements.values.values()
                for lm in mv.missing_landmarks
            }
        )
        missing_str = ", ".join(missing) if missing else ""
        data_note = str(rec.measurements.metadata.get("data_note", "") or "")
        # Only the side view is ever straightened.
        straight = str(rec.measurements.metadata.get("straightened", "") or "")
        who = str(rec.measurements.metadata.get("operators", "") or "")
        for view_name, calib in rec.calibrations.items():
            sheet.append(
                [
                    rec.measurements.fish_id,
                    rec.image_filename,
                    view_name,
                    calib.method,
                    round(calib.px_per_mm, 4),
                    round(calib.confidence, 3),
                    calib.notes,
                    missing_str,
                    who,
                    straight if view_name == "lateral" else "",
                    data_note,
                ]
            )

    for col_idx in range(1, len(header) + 1):
        letter = sheet.cell(row=1, column=col_idx).column_letter
        sheet.column_dimensions[letter].width = 18
