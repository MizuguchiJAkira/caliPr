"""Checks run before an export, so problems surface in the workbook not the paper.

Every check here exists because its failure mode is *silent*. None of these throw
an error, produce an obviously wrong number, or look different from good data once
they are in a spreadsheet — which is exactly why they need a pass of their own.

The checks, and what each one is protecting against:

``orientation``
    The measurement engine splits the body outline at the peduncle and calls the
    anterior side "body" and the posterior side "caudal", identifying them by
    min-x and max-x under a fish-faces-left convention. A mirrored specimen
    therefore swaps ``Bs`` and ``CFs`` — two plausible numbers in the wrong
    columns, with nothing to signal it.

``landmark_in_frame``
    A coordinate outside the image means the sidecar was written against a
    different photograph, usually after a re-crop. The geometry stays
    self-consistent, so nothing downstream can notice.

``duplicate_id``
    Two sidecars claiming one fish_id silently collapse to one row.

``mixed_units``
    A series with some specimens calibrated and some not puts millimetres and
    pixels in one column.

``calibration_outlier``
    A mistyped known_mm scales every trait for that specimen. Compared within the
    collection lot, because camera distance varies between lots in some series
    and a whole-batch median then flags healthy specimens.

``shape_outlier``
    A specimen whose size-corrected proportions sit far from its peers. Usually a
    misplaced landmark rather than an unusual fish, and far cheaper to catch here
    than in a scatter plot at the end.

``sparse_outline``
    A fin outline too coarse for its area to be trusted.

``incomplete``
    Landmarks never placed. Not an error — labelling in progress looks exactly
    like this — but the count belongs in front of whoever reads the sheet.
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass
from typing import Iterable, Sequence

from .landmark_config import (
    FIN_POLYGON_TARGET_VERTICES,
    FIN_POLYGONS,
    Unit,
)

#: Robust z-score past which a size-corrected trait is called an outlier. 3.5 on
#: a median/MAD scale is the conventional threshold and is not tuned to this data.
OUTLIER_Z = 3.5

#: Fraction by which a specimen's scale may differ from its lot's median before
#: it is flagged. Wide, because it is hunting a 5x typo, not a 5% wobble.
CALIB_TOLERANCE = 0.25


@dataclass(frozen=True)
class Issue:
    level: str          # "error" | "warning" | "note"
    check: str
    fish_id: str
    message: str


def _mad(values: Sequence[float]) -> float:
    m = st.median(values)
    return st.median([abs(v - m) for v in values]) or 0.0


def check_orientation(records) -> list[Issue]:
    out = []
    for rec in records:
        kps = getattr(rec, "keypoints", None) or {}
        a, b = kps.get("premaxilla_tip"), kps.get("caudal_base")
        if a and b and a[0] > b[0]:
            out.append(Issue(
                "error", "orientation", rec.measurements.fish_id,
                "fish faces RIGHT; the engine assumes left, so body and caudal "
                "areas (Bs, CFs) are swapped for this specimen"))
    return out


def check_landmarks_in_frame(records) -> list[Issue]:
    out = []
    for rec in records:
        kps = getattr(rec, "keypoints", None) or {}
        # getattr's default does not apply when the attribute exists and is None,
        # which it is whenever the image header could not be read.
        size = getattr(rec, "image_size", None)
        if not size:
            continue
        w, h = size
        bad = [n for n, p in kps.items()
               if not (0 <= p[0] <= w and 0 <= p[1] <= h)]
        if bad:
            out.append(Issue(
                "error", "landmark_in_frame", rec.measurements.fish_id,
                f"{len(bad)} landmark(s) outside the image ({', '.join(sorted(bad)[:4])})"
                " — the sidecar probably belongs to a different photograph"))
    return out


def check_duplicate_ids(records) -> list[Issue]:
    seen: dict[str, int] = {}
    for rec in records:
        fid = rec.measurements.fish_id
        seen[fid] = seen.get(fid, 0) + 1
    return [Issue("error", "duplicate_id", fid, f"{n} sidecars share this fish_id")
            for fid, n in seen.items() if n > 1]


def check_units(records) -> list[Issue]:
    def unit_of(rec):
        lat = rec.calibrations.get("lateral")
        return None if lat is None else ("px" if lat.method == "none" else "mm")

    units = {unit_of(r) for r in records} - {None}
    if units == {"mm", "px"}:
        px = [r.measurements.fish_id for r in records if unit_of(r) == "px"]
        return [Issue(
            "warning", "mixed_units", "",
            f"{len(px)} of {len(records)} specimens measure in PIXELS and the rest "
            f"in millimetres. Compare across rows only via the Shape or Ratios "
            f"sheet, never a raw length column.")]
    return []


def check_calibration(records, lot_of) -> list[Issue]:
    by_lot: dict[str, list[tuple[str, float]]] = {}
    for rec in records:
        lat = rec.calibrations.get("lateral")
        if lat is None or lat.method == "none":
            continue
        by_lot.setdefault(lot_of(rec.measurements.fish_id), []).append(
            (rec.measurements.fish_id, lat.px_per_mm))
    out = []
    for lot, rows in by_lot.items():
        if len(rows) < 3:
            continue
        med = st.median([v for _, v in rows])
        for fid, v in rows:
            if abs(v - med) / med > CALIB_TOLERANCE:
                out.append(Issue(
                    "error", "calibration_outlier", fid,
                    f"{v:.1f} px/mm against {med:.1f} for lot {lot or '?'} "
                    f"({(v/med - 1) * 100:+.0f}%) — every trait on this specimen is "
                    f"scaled by that factor"))
    return out


def check_shape_outliers(records) -> list[Issue]:
    """Size-corrected traits far from the group, by median/MAD."""
    codes: dict[str, list[tuple[str, float]]] = {}
    for rec in records:
        sl = rec.measurements.values.get("SL")
        if sl is None or math.isnan(sl.value) or sl.value <= 0:
            continue
        for code, mv in rec.measurements.values.items():
            if mv.unit != Unit.MM or math.isnan(mv.value) or code == "SL":
                continue
            codes.setdefault(code, []).append(
                (rec.measurements.fish_id, mv.value / sl.value))

    hits: dict[str, list[str]] = {}
    for code, rows in codes.items():
        if len(rows) < 8:            # too few to call anything an outlier
            continue
        vals = [v for _, v in rows]
        med, mad = st.median(vals), _mad(vals)
        if mad <= 0:
            continue
        for fid, v in rows:
            if abs(v - med) / (1.4826 * mad) > OUTLIER_Z:
                hits.setdefault(fid, []).append(code)
    return [Issue("warning", "shape_outlier", fid,
                  f"{len(cs)} trait(s) far from the group once size-corrected "
                  f"({', '.join(sorted(cs)[:6])}) — check the landmarks before "
                  f"treating this as biology")
            for fid, cs in sorted(hits.items())]


def check_outlines(records) -> list[Issue]:
    out = []
    for rec in records:
        polys = getattr(rec, "polygons", None) or {}
        sparse = {n: len(polys[n]) for n in FIN_POLYGONS
                  if n in polys and len(polys[n]) < FIN_POLYGON_TARGET_VERTICES}
        if sparse:
            out.append(Issue(
                "warning", "sparse_outline", rec.measurements.fish_id,
                "fin area unreliable: "
                + ", ".join(f"{n}={c}" for n, c in sorted(sparse.items()))
                + f" vertices, under {FIN_POLYGON_TARGET_VERTICES}"))
    return out


def check_completeness(records) -> list[Issue]:
    out = []
    for rec in records:
        missing = sorted({m for mv in rec.measurements.values.values()
                          for m in mv.missing_landmarks
                          if m.startswith(("keypoint:", "polygon:"))})
        if missing:
            out.append(Issue(
                "note", "incomplete", rec.measurements.fish_id,
                f"{len(missing)} input(s) not placed: "
                + ", ".join(m.split(":", 1)[1] for m in missing[:6])
                + (" …" if len(missing) > 6 else "")))
    return out


def validate(records, lot_of=lambda fid: "") -> list[Issue]:
    """Every check, most severe first."""
    issues: list[Issue] = []
    for fn in (check_duplicate_ids, check_orientation, check_landmarks_in_frame,
               check_units, check_shape_outliers, check_outlines,
               check_completeness):
        issues.extend(fn(records))
    issues.extend(check_calibration(records, lot_of))
    order = {"error": 0, "warning": 1, "note": 2}
    return sorted(issues, key=lambda i: (order[i.level], i.check, i.fish_id))


def summarise(issues: Iterable[Issue]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "note": 0}
    for i in issues:
        counts[i.level] += 1
    return counts
