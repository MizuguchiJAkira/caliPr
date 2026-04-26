"""Compute morphometric traits from polygon + keypoint annotations.

This is the Python port of MorFishJ's trait geometry (22 traits) plus the
Cornell extras (8 more). The engine is pure geometry — no image I/O, no
model calls — so it's trivially unit-testable with hand-crafted
:class:`Annotation` objects and runs identically in manual mode
(annotations from JSON sidecars) and auto mode (annotations from the
DLC + SAM model stack).

Reference lines
---------------
The schema in :mod:`fish_morpho.landmark_config` declares twelve named
reference lines (A/H/I keypoint-anchored; B/C/D/E/F/G derived from the
body_plus_caudal polygon; J/K/L derived from per-keypoint verticals and
horizontals). This module realizes them at measurement time:

* ``_body_tip_x`` / ``_caudal_tip_x`` — D and E (anterior/posterior body
  x-extrema, used by TL, SL, Hl, AO).
* ``_RefCache.body_half`` / ``caudal_half`` — the body_plus_caudal
  polygon split along line A (narrowest peduncle) via Sutherland-Hodgman
  half-plane clipping. B and C (body top/bottom) are min/max y of
  ``body_half``; F and G (caudal top/bottom) are min/max y of
  ``caudal_half``. Bs and CFs are the shoelace areas of those halves.
* ``_RefCache.eye_centroid`` — J and K are the horizontal/vertical
  through this point (eye bounding-box midpoint). Hd uses the vertical
  (line K) to pull a body-depth slice at the eye.
* Line L is implicit — the vertical at ``pectoral_insertion_upper.x``.
  PFb pulls a body-depth slice there.

Per-view calibration
--------------------
Linear and area traits that live in ``View.FRONTAL`` (currently just
mouth width, MW) use a separate calibration because they come from a
different image region with its own mirror ruler. Callers pass a
``calibrations`` dict keyed by :class:`fish_morpho.landmark_config.View`;
the engine looks up the right one for each trait. Degree traits (EMa)
don't use calibration at all.

Missing / degenerate inputs
---------------------------
Each trait declares its required polygons and keypoints in
:mod:`fish_morpho.landmark_config`. When any are absent from the
annotation the engine emits a NaN ``MeasurementValue`` whose
``missing_landmarks`` field lists what was missing (prefixed with
``polygon:`` or ``keypoint:`` so QC output can tell them apart) rather
than silently producing a wrong number. Degenerate geometry (e.g. the
vertical through the eye centroid doesn't cross the body outline)
surfaces the same way, tagged ``degenerate_geometry``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .landmark_config import (
    TRAITS,
    Trait,
    Unit,
    View,
    trait_column_order,
    trait_labels,
)
from .ruler_calibration import CalibrationResult

Point = tuple[float, float]
CalibrationsByView = Mapping[View, CalibrationResult]


# ---------------------------------------------------------------------------
# Annotation (runtime shape consumed by the engine)
# ---------------------------------------------------------------------------


@dataclass
class Annotation:
    """A single specimen's polygons + keypoints across all views.

    Polygons are keyed by name (``"body_plus_caudal"``, ``"pectoral"``,
    etc.). Each polygon is a list of ``(x, y)`` pixel-space vertices
    traced in order along the outline.

    Keypoints are keyed by name (``"eye_anterior"``, etc.). Each keypoint
    is a single ``(x, y)`` pair.

    LATERAL and FRONTAL annotations share one Annotation object — the
    engine looks up per-trait view when applying calibration, and the
    frontal keypoints simply sit in a different region of the same
    photo (or a paired mirror shot with its own ruler).
    """

    polygons: dict[str, list[Point]] = field(default_factory=dict)
    keypoints: dict[str, Point] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MeasurementValue:
    """One trait's computed value, tagged with unit and provenance."""

    key: str  # trait code, e.g. "TL"
    label: str
    value: float  # mm, mm^2, or deg; NaN when inputs missing / degenerate
    unit: str  # "mm", "mm^2", "deg"
    view: View
    missing_landmarks: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not math.isnan(self.value) and not self.missing_landmarks


@dataclass
class MeasurementSet:
    """All traits for one specimen."""

    fish_id: str
    values: dict[str, MeasurementValue] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> MeasurementValue | None:
        return self.values.get(key)

    def as_row(self, keys: list[str]) -> list[float | str]:
        """Flatten to a row of numeric values (for xlsx export)."""
        row: list[float | str] = []
        for k in keys:
            v = self.values.get(k)
            row.append(v.value if v is not None else math.nan)
        return row


# ---------------------------------------------------------------------------
# Primitive geometry
# ---------------------------------------------------------------------------


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _centroid(points: list[Point]) -> Point:
    xs = sum(p[0] for p in points)
    ys = sum(p[1] for p in points)
    n = len(points)
    return (xs / n, ys / n)


def shoelace_area(polygon: list[Point]) -> float:
    """Absolute polygon area via the Shoelace formula.

    Accepts the polygon in either winding (CW or CCW) and returns a
    non-negative value. Fewer than 3 vertices returns 0.0.
    """
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _side_of_line(p: Point, la: Point, lb: Point) -> float:
    """Signed 2x-triangle-area of (la, lb, p). Sign tells which side of
    the infinite line through ``la`` and ``lb`` point ``p`` lies on."""
    return (lb[0] - la[0]) * (p[1] - la[1]) - (lb[1] - la[1]) * (p[0] - la[0])


def _segment_line_intersection(
    p1: Point, p2: Point, la: Point, lb: Point
) -> Point | None:
    """Intersection of segment (p1, p2) with the infinite line (la, lb).

    Returns ``None`` if the segment is parallel to the line.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = la
    x4, y4 = lb
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _sutherland_hodgman(
    poly: list[Point],
    la: Point,
    lb: Point,
    keep_sign: int,
) -> list[Point]:
    """Clip ``poly`` against the infinite line (la, lb), keeping vertices
    whose :func:`_side_of_line` sign matches ``keep_sign`` (+1 or -1).

    Vertices exactly on the line (sign 0) are considered inside, so
    keypoints that were traced as polygon vertices are kept in both
    halves. This is the standard Sutherland-Hodgman polygon-clipping
    routine, specialized to a single half-plane.
    """
    if not poly or keep_sign not in (1, -1):
        return []
    out: list[Point] = []
    n = len(poly)
    for i in range(n):
        curr = poly[i]
        prev = poly[(i - 1) % n]
        curr_s = _side_of_line(curr, la, lb)
        prev_s = _side_of_line(prev, la, lb)
        curr_in = curr_s == 0 or _sign(curr_s) == keep_sign
        prev_in = prev_s == 0 or _sign(prev_s) == keep_sign
        if curr_in:
            if not prev_in:
                inter = _segment_line_intersection(prev, curr, la, lb)
                if inter is not None:
                    out.append(inter)
            out.append(curr)
        elif prev_in:
            inter = _segment_line_intersection(prev, curr, la, lb)
            if inter is not None:
                out.append(inter)
    return out


def _split_polygon_along_line_a(
    poly: list[Point], la: Point, lb: Point
) -> tuple[list[Point], list[Point]]:
    """Split ``poly`` along line A into ``(body_half, caudal_half)``.

    "Body" is the anterior side (smaller x under the left-facing-fish
    convention) and "caudal" is the posterior side. The anchors are
    derived from the polygon itself: its min-x vertex is the snout tip
    (reliably on the body side) and its max-x vertex is the caudal fin
    tip (reliably on the caudal side).

    If both anchors end up on the same side of line A — which would mean
    line A doesn't actually cut the polygon (degenerate keypoint
    placement) — returns ``(list(poly), [])`` so the caller can still do
    body-level math and the caudal traits surface as degenerate.
    """
    if len(poly) < 3:
        return [], []
    snout = min(poly, key=lambda p: p[0])
    tail = max(poly, key=lambda p: p[0])
    snout_sign = _sign(_side_of_line(snout, la, lb))
    tail_sign = _sign(_side_of_line(tail, la, lb))
    if snout_sign == 0 or tail_sign == 0 or snout_sign == tail_sign:
        return list(poly), []
    body = _sutherland_hodgman(poly, la, lb, keep_sign=snout_sign)
    caudal = _sutherland_hodgman(poly, la, lb, keep_sign=tail_sign)
    return body, caudal


def _vertical_extent_at_x(
    poly: list[Point], x: float
) -> tuple[float, float] | None:
    """Return ``(min_y, max_y)`` where a vertical line at ``x`` cuts ``poly``.

    Uses a half-open crossing convention (``x1 < x <= x2`` or
    ``x2 < x <= x1``) to avoid double-counting shared vertices. Returns
    ``None`` if the vertical doesn't cross any edge (the polygon is
    entirely to one side of the line or ``x`` exactly grazes a vertex
    without entering).
    """
    ys: list[float] = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (x1 < x <= x2) or (x2 < x <= x1):
            if x2 == x1:
                continue
            t = (x - x1) / (x2 - x1)
            ys.append(y1 + t * (y2 - y1))
    if not ys:
        return None
    return min(ys), max(ys)


# ---------------------------------------------------------------------------
# Per-specimen derived-reference cache
# ---------------------------------------------------------------------------


@dataclass
class _RefCache:
    """Lazy per-specimen cache of derived quantities.

    Each quantity is memoized on first access. This keeps a batch run
    from recomputing the body/caudal split or the eye centroid for every
    trait that needs them.
    """

    ann: Annotation
    _eye_centroid: Point | None = None
    _halves_computed: bool = False
    _body_half: list[Point] = field(default_factory=list)
    _caudal_half: list[Point] = field(default_factory=list)

    def eye_centroid(self) -> Point:
        if self._eye_centroid is None:
            ea = self.ann.keypoints["eye_anterior"]
            ep = self.ann.keypoints["eye_posterior"]
            ed = self.ann.keypoints["eye_dorsal"]
            ev = self.ann.keypoints["eye_ventral"]
            # Bounding-box midpoint: horizontal from anterior/posterior,
            # vertical from dorsal/ventral. More robust than a raw mean
            # when the 4 clicks aren't perfectly orthogonal.
            self._eye_centroid = (
                (ea[0] + ep[0]) / 2.0,
                (ed[1] + ev[1]) / 2.0,
            )
        return self._eye_centroid

    def _ensure_halves(self) -> None:
        if self._halves_computed:
            return
        poly = self.ann.polygons["body_plus_caudal"]
        la = self.ann.keypoints["peduncle_narrowest_dorsal"]
        lb = self.ann.keypoints["peduncle_narrowest_ventral"]
        self._body_half, self._caudal_half = _split_polygon_along_line_a(
            poly, la, lb
        )
        self._halves_computed = True

    def body_half(self) -> list[Point]:
        self._ensure_halves()
        return self._body_half

    def caudal_half(self) -> list[Point]:
        self._ensure_halves()
        return self._caudal_half


# ---------------------------------------------------------------------------
# Trait computers
# ---------------------------------------------------------------------------
#
# Each computer takes (Annotation, _RefCache) and returns the trait value
# in its native pre-calibration units:
#
#   * Unit.MM  → pixels (divided by px/mm at calibration time)
#   * Unit.MM2 → pixels² (divided by (px/mm)² at calibration time)
#   * Unit.DEG → degrees (no calibration)
#
# A computer may return ``math.nan`` to signal geometrically degenerate
# inputs (e.g. the body polygon was clipped empty or a vertical slice
# missed the outline). ``compute_trait`` converts that to a NaN
# MeasurementValue tagged ``degenerate_geometry``.
#
# Missing-input checking is done upstream from the Trait declarations so
# computers can assume every name they look up is present.


_TraitComputer = Callable[[Annotation, _RefCache], float]


def _body_tip_x(ann: Annotation) -> float:
    """Reference line D (anterior body extremum) x-coordinate."""
    return min(p[0] for p in ann.polygons["body_plus_caudal"])


def _caudal_tip_x(ann: Annotation) -> float:
    """Reference line E (posterior caudal extremum) x-coordinate."""
    return max(p[0] for p in ann.polygons["body_plus_caudal"])


# ---- MorFishJ traits -------------------------------------------------------


def _compute_TL(ann: Annotation, cache: _RefCache) -> float:
    # Total length: horizontal distance from D to E.
    return _caudal_tip_x(ann) - _body_tip_x(ann)


def _compute_SL(ann: Annotation, cache: _RefCache) -> float:
    # Standard length: horizontal from D to H (vertical through caudal_base).
    return ann.keypoints["caudal_base"][0] - _body_tip_x(ann)


def _compute_MBd(ann: Annotation, cache: _RefCache) -> float:
    # Maximum body depth: vertical between B (body top) and C (body bottom).
    body = cache.body_half()
    if len(body) < 3:
        return math.nan
    ys = [p[1] for p in body]
    return max(ys) - min(ys)


def _compute_Hl(ann: Annotation, cache: _RefCache) -> float:
    # Head length: horizontal from D to I (vertical through operculum_posterior).
    return ann.keypoints["operculum_posterior"][0] - _body_tip_x(ann)


def _compute_Hd(ann: Annotation, cache: _RefCache) -> float:
    # Head depth: body-half vertical extent along line K (vertical through
    # the eye centroid).
    body = cache.body_half()
    if len(body) < 3:
        return math.nan
    ex, _ = cache.eye_centroid()
    ext = _vertical_extent_at_x(body, ex)
    if ext is None:
        return math.nan
    return ext[1] - ext[0]


def _compute_Ed(ann: Annotation, cache: _RefCache) -> float:
    # Orbit diameter: Euclidean distance between the two horizontal eye
    # cardinal points. Tolerant of slight off-axis labeling.
    return _distance(
        ann.keypoints["eye_anterior"], ann.keypoints["eye_posterior"]
    )


def _compute_Eh(ann: Annotation, cache: _RefCache) -> float:
    # Eye position: vertical distance from eye centroid to C (body bottom).
    body = cache.body_half()
    if len(body) < 3:
        return math.nan
    c_y = max(p[1] for p in body)
    return c_y - cache.eye_centroid()[1]


def _compute_Snl(ann: Annotation, cache: _RefCache) -> float:
    # Snout length: horizontal from premaxilla tip to anterior orbit.
    # Coincides with AO when the premaxilla is the polygon's anterior
    # extremum, which is the normal closed-mouth case.
    return ann.keypoints["eye_anterior"][0] - ann.keypoints["premaxilla_tip"][0]


def _compute_POC(ann: Annotation, cache: _RefCache) -> float:
    # Posterior-of-orbit: horizontal from eye centroid to I.
    return ann.keypoints["operculum_posterior"][0] - cache.eye_centroid()[0]


def _compute_AO(ann: Annotation, cache: _RefCache) -> float:
    # Anterior-of-orbit: horizontal from D (anterior body tip) to the
    # anterior orbit margin.
    return ann.keypoints["eye_anterior"][0] - _body_tip_x(ann)


def _compute_EMd(ann: Annotation, cache: _RefCache) -> float:
    # Eye-mouth distance: Euclidean from eye centroid to premaxilla tip.
    return _distance(cache.eye_centroid(), ann.keypoints["premaxilla_tip"])


def _compute_EMa(ann: Annotation, cache: _RefCache) -> float:
    # Eye-mouth angle: angle (degrees) of the eye-centroid-to-premaxilla
    # vector relative to the horizontal through premaxilla_tip. Positive
    # when the eye is visually above the mouth — remember +y is down in
    # image space, so we flip dy to match the intuitive sign.
    mx, my = ann.keypoints["premaxilla_tip"]
    ex, ey = cache.eye_centroid()
    dx = ex - mx
    dy = my - ey  # positive when eye is above mouth in image space
    return math.degrees(math.atan2(dy, dx))


def _compute_Mo(ann: Annotation, cache: _RefCache) -> float:
    # Oral gape position: vertical distance from premaxilla_tip to C.
    body = cache.body_half()
    if len(body) < 3:
        return math.nan
    c_y = max(p[1] for p in body)
    return c_y - ann.keypoints["premaxilla_tip"][1]


def _compute_Jl(ann: Annotation, cache: _RefCache) -> float:
    # Maxillary jaw length: premaxilla tip → maxilla-mandible joint.
    return _distance(
        ann.keypoints["premaxilla_tip"],
        ann.keypoints["maxilla_mandible_intersection"],
    )


def _compute_Bs(ann: Annotation, cache: _RefCache) -> float:
    # Body surface area: shoelace area of the body half of body_plus_caudal.
    body = cache.body_half()
    if len(body) < 3:
        return math.nan
    return shoelace_area(body)


def _compute_CPd(ann: Annotation, cache: _RefCache) -> float:
    # Caudal peduncle depth: length of line A (the two peduncle keypoints).
    return _distance(
        ann.keypoints["peduncle_narrowest_dorsal"],
        ann.keypoints["peduncle_narrowest_ventral"],
    )


def _compute_CFd(ann: Annotation, cache: _RefCache) -> float:
    # Caudal fin depth: vertical between F (caudal top) and G (caudal bottom).
    caudal = cache.caudal_half()
    if len(caudal) < 3:
        return math.nan
    ys = [p[1] for p in caudal]
    return max(ys) - min(ys)


def _compute_CFs(ann: Annotation, cache: _RefCache) -> float:
    # Caudal fin surface area: shoelace area of the caudal half.
    caudal = cache.caudal_half()
    if len(caudal) < 3:
        return math.nan
    return shoelace_area(caudal)


def _compute_PFs(ann: Annotation, cache: _RefCache) -> float:
    # Pectoral fin surface area: shoelace area of pectoral polygon.
    return shoelace_area(ann.polygons["pectoral"])


def _compute_PFl(ann: Annotation, cache: _RefCache) -> float:
    # Pectoral fin length: upper insertion → ray tip.
    return _distance(
        ann.keypoints["pectoral_insertion_upper"],
        ann.keypoints["pectoral_ray_tip"],
    )


def _compute_PFi(ann: Annotation, cache: _RefCache) -> float:
    # Pectoral fin position: vertical from pectoral_insertion_upper to C.
    body = cache.body_half()
    if len(body) < 3:
        return math.nan
    c_y = max(p[1] for p in body)
    return c_y - ann.keypoints["pectoral_insertion_upper"][1]


def _compute_PFb(ann: Annotation, cache: _RefCache) -> float:
    # Body depth at pectoral insertion: body-half vertical extent along
    # line L (vertical through pectoral_insertion_upper).
    body = cache.body_half()
    if len(body) < 3:
        return math.nan
    pix = ann.keypoints["pectoral_insertion_upper"][0]
    ext = _vertical_extent_at_x(body, pix)
    if ext is None:
        return math.nan
    return ext[1] - ext[0]


# ---- Cornell extras --------------------------------------------------------


def _compute_LJl(ann: Annotation, cache: _RefCache) -> float:
    # Lower jaw length: lower_jaw_tip → maxilla-mandible joint.
    return _distance(
        ann.keypoints["lower_jaw_tip"],
        ann.keypoints["maxilla_mandible_intersection"],
    )


def _compute_DFh(ann: Annotation, cache: _RefCache) -> float:
    # Dorsal fin height: dorsal_base_center → dorsal_tip (definition
    # pending Cornell reference sheet — base-point semantics may tighten).
    return _distance(
        ann.keypoints["dorsal_base_center"],
        ann.keypoints["dorsal_tip"],
    )


def _compute_DFs(ann: Annotation, cache: _RefCache) -> float:
    return shoelace_area(ann.polygons["dorsal"])


def _compute_PlFl(ann: Annotation, cache: _RefCache) -> float:
    # Pelvic fin length: pelvic_base_center → pelvic_tip (definition
    # pending Cornell reference sheet).
    return _distance(
        ann.keypoints["pelvic_base_center"],
        ann.keypoints["pelvic_tip"],
    )


def _compute_PlFs(ann: Annotation, cache: _RefCache) -> float:
    return shoelace_area(ann.polygons["pelvic"])


def _compute_AFh(ann: Annotation, cache: _RefCache) -> float:
    # Anal fin height: anal_base_center → anal_tip (definition pending
    # Cornell reference sheet).
    return _distance(
        ann.keypoints["anal_base_center"],
        ann.keypoints["anal_tip"],
    )


def _compute_AFs(ann: Annotation, cache: _RefCache) -> float:
    return shoelace_area(ann.polygons["anal"])


def _compute_MW(ann: Annotation, cache: _RefCache) -> float:
    # Mouth width: frontal-view mouth_left → mouth_right.
    return _distance(
        ann.keypoints["mouth_left"], ann.keypoints["mouth_right"]
    )


TRAIT_COMPUTERS: dict[str, _TraitComputer] = {
    # MorFishJ (22)
    "TL": _compute_TL,
    "SL": _compute_SL,
    "MBd": _compute_MBd,
    "Hl": _compute_Hl,
    "Hd": _compute_Hd,
    "Ed": _compute_Ed,
    "Eh": _compute_Eh,
    "Snl": _compute_Snl,
    "POC": _compute_POC,
    "AO": _compute_AO,
    "EMd": _compute_EMd,
    "EMa": _compute_EMa,
    "Mo": _compute_Mo,
    "Jl": _compute_Jl,
    "Bs": _compute_Bs,
    "CPd": _compute_CPd,
    "CFd": _compute_CFd,
    "CFs": _compute_CFs,
    "PFs": _compute_PFs,
    "PFl": _compute_PFl,
    "PFi": _compute_PFi,
    "PFb": _compute_PFb,
    # Cornell extras (8)
    "LJl": _compute_LJl,
    "DFh": _compute_DFh,
    "DFs": _compute_DFs,
    "PlFl": _compute_PlFl,
    "PlFs": _compute_PlFs,
    "AFh": _compute_AFh,
    "AFs": _compute_AFs,
    "MW": _compute_MW,
}


# ---------------------------------------------------------------------------
# Missing-input detection + calibration
# ---------------------------------------------------------------------------


def _missing_inputs(trait: Trait, ann: Annotation) -> tuple[str, ...]:
    """List polygons/keypoints declared on ``trait`` that aren't in ``ann``.

    Names are prefixed with ``polygon:`` or ``keypoint:`` so QC output
    can tell them apart. A polygon is considered missing if it's absent
    from ``ann.polygons`` or has fewer than 3 vertices.
    """
    missing: list[str] = []
    for name in trait.required_polygons:
        poly = ann.polygons.get(name)
        if poly is None or len(poly) < 3:
            missing.append(f"polygon:{name}")
    for name in trait.required_keypoints:
        if name not in ann.keypoints:
            missing.append(f"keypoint:{name}")
    return tuple(missing)


def _apply_calibration(
    raw: float,
    trait: Trait,
    calibrations: CalibrationsByView,
) -> float:
    if trait.unit == Unit.DEG:
        return raw
    try:
        calib = calibrations[trait.view]
    except KeyError as exc:
        raise KeyError(
            f"Trait {trait.code!r} needs a calibration for view "
            f"{trait.view.value!r}, but none was provided."
        ) from exc
    if trait.unit == Unit.MM:
        return calib.px_to_mm(raw)
    if trait.unit == Unit.MM2:
        return calib.area_px_to_mm2(raw)
    raise ValueError(f"Unsupported unit for trait {trait.code!r}: {trait.unit}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_trait(
    trait: Trait,
    annotation: Annotation,
    cache: _RefCache,
    calibrations: CalibrationsByView,
) -> MeasurementValue:
    """Compute one trait, returning NaN when inputs are missing or degenerate."""
    missing = _missing_inputs(trait, annotation)
    if missing:
        return MeasurementValue(
            key=trait.code,
            label=trait.label,
            value=math.nan,
            unit=trait.unit.value,
            view=trait.view,
            missing_landmarks=missing,
        )
    computer = TRAIT_COMPUTERS.get(trait.code)
    if computer is None:
        raise KeyError(
            f"No computer registered for trait {trait.code!r}. "
            "Add a Trait to landmark_config.TRAITS and wire it into "
            "TRAIT_COMPUTERS."
        )
    raw = computer(annotation, cache)
    if math.isnan(raw):
        return MeasurementValue(
            key=trait.code,
            label=trait.label,
            value=math.nan,
            unit=trait.unit.value,
            view=trait.view,
            missing_landmarks=("degenerate_geometry",),
        )
    value = _apply_calibration(raw, trait, calibrations)
    return MeasurementValue(
        key=trait.code,
        label=trait.label,
        value=value,
        unit=trait.unit.value,
        view=trait.view,
    )


def compute_all(
    fish_id: str,
    annotation: Annotation,
    calibrations: CalibrationsByView,
    metadata: dict[str, str] | None = None,
) -> MeasurementSet:
    """Compute every declared trait for one specimen.

    ``calibrations`` must contain an entry for every view the schema
    touches: a ``View.LATERAL`` calibration always, plus a
    ``View.FRONTAL`` calibration if mouth width is being collected.
    Per-trait view lookup happens inside :func:`compute_trait`, so
    skipping the frontal calibration simply makes MW fail with a clear
    KeyError (not a silent miscalibration).
    """
    result = MeasurementSet(fish_id=fish_id, metadata=dict(metadata or {}))
    cache = _RefCache(ann=annotation)
    for trait in TRAITS:
        result.values[trait.code] = compute_trait(
            trait, annotation, cache, calibrations
        )
    return result


# ---------------------------------------------------------------------------
# Back-compat shims for export.py
# ---------------------------------------------------------------------------
#
# export.py imports ``measurement_column_order`` and ``measurement_labels``
# from this module. The canonical source of truth for both is now
# :mod:`fish_morpho.landmark_config`; these thin wrappers keep
# export.py untouched until the rest of the pipeline migration is done.


def measurement_column_order() -> list[str]:
    """Stable column ordering for xlsx export (delegates to trait_column_order)."""
    return trait_column_order()


def measurement_labels() -> dict[str, str]:
    """Header labels for xlsx export (delegates to trait_labels)."""
    return trait_labels()
