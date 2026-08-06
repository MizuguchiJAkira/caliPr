"""Anatomical limits that a predicted fin outline must obey.

A segmentation model has no anatomy. Prompted on a brook trout's pectoral fin it
returns a mask that runs forward across the gill cover; prompted on the dorsal it
dips into the shaded groove beside the fin base and swallows the specimen pin
above it. None of that is possible on a fish. It happens because the boundary the
model follows is a contrast edge, not a structure.

These are facts about the animal, not things to hope a model infers, so they
belong in code -- guaranteed, auditable, needing no training data. Each limit is
an infinite line plus the side the fin must lie on, applied with
:func:`measurement_engine.clip_polygon_to_halfplane`. Clipping rather than
rejecting matters: a mask that is right over the fin and wrong over the operculum
still carries most of the fin's area.

**Every limit carries an allowance, and the allowance is the whole design.** A
bound placed exactly at the anatomical landmark cuts real fin, because real fins
sit slightly past their own landmarks and tracings jitter. Each allowance below
was fitted by measuring how far the *hand* tracings cross the line -- they are the
fin, by definition -- then setting the bound clear of that while still catching
what the model does wrong. Clipping at zero allowance removed up to 53% of a real
hand-traced dorsal.

Encoded limits:

*Pectoral, anterior.* The fin inserts behind the gill opening but overlaps it:
all 44 specimens reach anterior of ``operculum_posterior``, median 2.7% of SL, so
the bound sits an allowance forward of it and catches only gross overhang.

*Dorsal, ventral.* Two body-outline points flanking the fin base define a chord
across the insertion -- the flat plane under the fin. Nothing below it is fin.

*Dorsal, distal.* Nothing beyond the fin's own tip is fin. This is what removes
the specimen pin, which sits above the fin and is untouched by the ventral bound.

A limit can only remove area crossing its line; it cannot repair a mask that is
wrong inside the allowed region. :func:`describe_limits` states the coverage.
"""

from __future__ import annotations

import math

from .landmark_config import FIN_KEYPOINTS
from .measurement_engine import Point, clip_polygon_to_halfplane, shoelace_area

#: How far anterior of ``operculum_posterior`` a pectoral may reach, as a fraction
#: of SL. Hand tracings: median 2.68%, p90 3.65%, max 5.71% (n=44); densely
#: re-traced fish 2.27%. SAM overhangs 5.21% there, so 4% admits the real fin on 41
#: of 44 while still catching the leak. Normalising by pectoral fin length instead
#: is no tighter (CV 42% vs 40%), so SL wins on simplicity.
PECTORAL_ANTERIOR_ALLOWANCE_SL = 0.04

#: How far below the dorsal insertion chord an outline may reach, as a fraction of
#: SL. Hand tracings cross by median 0.34%, p90 0.77%, max 1.21% (n=44); SAM's
#: shadow lobe crosses 2.95%, so 1.5% has margin on both sides.
DORSAL_VENTRAL_ALLOWANCE_SL = 0.015

#: How far beyond ``dorsal_tip`` an outline may reach, as a fraction of SL. Hand
#: tracings cross by median 0.38%, max 1.30%; SAM's pin lobe crosses 3.47%.
#:
#: Wider than the ventral allowance on purpose. This bound is anchored to a
#: *keypoint*, so in auto mode it inherits that keypoint's error -- and
#: ``dorsal_tip`` is the weakest landmark the pose model predicts, ~2.5 mm median,
#: which on a 140 mm fish is ~1.8% of SL. 2% covers hand-placed tips comfortably;
#: pass a larger value when the tip is predicted rather than clicked.
DORSAL_DISTAL_ALLOWANCE_SL = 0.02

#: Arc length to walk along the body outline, each way from the fin base, to pick
#: the two points defining the insertion chord. About half a dorsal base, so the
#: chord spans the insertion without picking up the curvature of the back.
INSERTION_FLANK_ARC_SL = 0.06

#: A clip removing more than this fraction is treated as a failed constraint
#: rather than a real violation -- bad landmarks, not a nonexistent fin.
MAX_PLAUSIBLE_REMOVAL = 0.9

Limit = tuple[Point, Point, Point, str]     # (la, lb, inside, name)


def _axis(keypoints: dict[str, Point]) -> tuple[Point, Point] | None:
    a, b = keypoints.get("premaxilla_tip"), keypoints.get("caudal_base")
    return (a, b) if a and b else None


def standard_length_px(keypoints: dict[str, Point]) -> float | None:
    ax = _axis(keypoints)
    if not ax:
        return None
    (x1, y1), (x2, y2) = ax
    return math.hypot(x2 - x1, y2 - y1) or None


def _centroid(poly: list[Point]) -> Point:
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def _offset_line(
    la: Point, lb: Point, toward: Point, distance: float
) -> tuple[Point, Point]:
    """Shift the line (la, lb) by ``distance`` in the direction of ``toward``."""
    dx, dy = lb[0] - la[0], lb[1] - la[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    mid = ((la[0] + lb[0]) / 2, (la[1] + lb[1]) / 2)
    if nx * (toward[0] - mid[0]) + ny * (toward[1] - mid[1]) < 0:
        nx, ny = -nx, -ny
    sx, sy = nx * distance, ny * distance
    return (la[0] + sx, la[1] + sy), (lb[0] + sx, lb[1] + sy)


def insertion_chord(
    body: list[Point], base: Point, sl: float,
    arc_frac: float = INSERTION_FLANK_ARC_SL,
) -> tuple[Point, Point] | None:
    """Two body-outline points flanking ``base``, one each side along the outline.

    Walking the outline outward by a fixed arc length gives a chord spanning the
    whole insertion. Fitting a line to outline vertices within a *radius* instead
    was tried and is worse -- it wrongly cut real hand-traced dorsal on 17 of 44
    specimens against 9 for a chord, because a radius window also picks up the
    curvature of the back.
    """
    n = len(body)
    if n < 3 or sl <= 0:
        return None
    want = arc_frac * sl

    # Start from the closest POINT on the outline, not the closest vertex, and
    # interpolate along edges when walking. Snapping to vertices would make the
    # chord depend on how densely the outline happens to be traced -- fine on a
    # 52-vertex body, wrong on a sparse one.
    best_i, best_t, best_d = 0, 0.0, float("inf")
    for i in range(n):
        a, b = body[i], body[(i + 1) % n]
        vx, vy = b[0] - a[0], b[1] - a[1]
        l2 = vx * vx + vy * vy
        if l2 == 0:
            continue
        t = max(0.0, min(1.0, ((base[0] - a[0]) * vx + (base[1] - a[1]) * vy) / l2))
        d = math.hypot(a[0] + t * vx - base[0], a[1] + t * vy - base[1])
        if d < best_d:
            best_i, best_t, best_d = i, t, d

    def walk(step: int) -> Point:
        """Travel ``want`` along the outline from the closest point, either way."""
        a, b = body[best_i], body[(best_i + 1) % n]
        cur = (a[0] + best_t * (b[0] - a[0]), a[1] + best_t * (b[1] - a[1]))
        # Both directions start on the same edge: forward consumes the edge's far
        # vertex, backward consumes its near one. Starting the backward walk a
        # vertex further on makes it travel forward too, and the chord collapses.
        i = best_i
        left = want
        for _ in range(n + 1):
            nxt = body[(i + 1) % n] if step > 0 else body[i]
            seg = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            if seg >= left:
                if seg == 0:
                    return cur
                f = left / seg
                return (cur[0] + f * (nxt[0] - cur[0]), cur[1] + f * (nxt[1] - cur[1]))
            left -= seg
            cur = nxt
            i = (i + 1) % n if step > 0 else (i - 1) % n
        return cur

    a, b = walk(1), walk(-1)
    return (a, b) if a != b else None


def pectoral_limits(keypoints: dict[str, Point]) -> list[Limit]:
    op = keypoints.get("operculum_posterior")
    ax = _axis(keypoints)
    if not op or not ax:
        return []
    (hx, hy), (tx, ty) = ax
    dx, dy = tx - hx, ty - hy
    n = math.hypot(dx, dy)
    if not n:
        return []
    ux, uy = dx / n, dy / n                        # anterior -> posterior
    slack = PECTORAL_ANTERIOR_ALLOWANCE_SL * n
    ox, oy = op[0] - ux * slack, op[1] - uy * slack
    la = (ox - uy * n, oy + ux * n)                # perpendicular to the body axis
    lb = (ox + uy * n, oy - ux * n)
    inside = (ox + ux * n * 0.25, oy + uy * n * 0.25)
    return [(la, lb, inside, "anterior of operculum")]


def dorsal_limits(
    keypoints: dict[str, Point], body: list[Point],
    distal_allowance_sl: float = DORSAL_DISTAL_ALLOWANCE_SL,
) -> list[Limit]:
    base = keypoints.get("dorsal_base_center")
    sl = standard_length_px(keypoints)
    if not base or not sl or len(body) < 3:
        return []
    chord = insertion_chord(body, base, sl)
    if chord is None:
        return []
    ca, cb = chord
    bc = _centroid(body)
    away = (2 * base[0] - bc[0], 2 * base[1] - bc[1])

    out: list[Limit] = []
    # ventral: push the chord toward the body, so a fin dipping slightly past its
    # own insertion is not shaved
    va, vb = _offset_line(ca, cb, bc, DORSAL_VENTRAL_ALLOWANCE_SL * sl)
    out.append((va, vb, away, "ventral of insertion"))

    # distal: nothing past the fin's own tip. Parallel to the insertion, so it
    # follows the fin's axis rather than the image.
    tip = keypoints.get("dorsal_tip")
    if tip:
        dx, dy = cb[0] - ca[0], cb[1] - ca[1]
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        ta = (tip[0] - ux * sl, tip[1] - uy * sl)
        tb = (tip[0] + ux * sl, tip[1] + uy * sl)
        # Loosen outward, i.e. further from the base than the tip already is.
        # `away` (the centroid reflected through the *base*) is not usable here:
        # the tip is normally beyond it, so offsetting toward it would tighten the
        # bound and shave the fin's own tip.
        beyond = (2 * tip[0] - base[0], 2 * tip[1] - base[1])
        ta, tb = _offset_line(ta, tb, beyond, distal_allowance_sl * sl)
        out.append((ta, tb, base, "beyond dorsal tip"))
    return out


def limits_for(
    fin: str, keypoints: dict[str, Point], body: list[Point] | None = None
) -> list[Limit]:
    """Every anatomical limit encoded for ``fin`` (empty if none)."""
    if fin == "pectoral":
        return pectoral_limits(keypoints)
    if fin == "dorsal":
        return dorsal_limits(keypoints, body or [])
    return []


def constrain_fin_polygon(
    fin: str,
    polygon: list[Point],
    keypoints: dict[str, Point],
    body: list[Point] | None = None,
) -> tuple[list[Point], float, list[str]]:
    """Clip ``polygon`` to what is anatomically possible for ``fin``.

    Returns ``(clipped, fraction_removed, limits_applied)``. Unknown fins, missing
    landmarks, and polygons that cross nothing all return the input unchanged, so
    this is safe to call unconditionally.

    A limit that would remove nearly everything is skipped: that means the limit or
    its landmarks are wrong, and returning an empty polygon would turn a bad prompt
    into a confident zero area.
    """
    if fin not in FIN_KEYPOINTS or len(polygon) < 3:
        return polygon, 0.0, []
    before = shoelace_area(polygon)
    if before <= 0:
        return polygon, 0.0, []

    current = polygon
    applied: list[str] = []
    for la, lb, inside, name in limits_for(fin, keypoints, body):
        clipped = clip_polygon_to_halfplane(current, la, lb, inside)
        if len(clipped) < 3:
            continue
        area = shoelace_area(clipped)
        if area <= 0 or 1.0 - area / before > MAX_PLAUSIBLE_REMOVAL:
            continue
        if area < shoelace_area(current) - 1e-9:
            applied.append(name)
        current = clipped
    removed = 1.0 - shoelace_area(current) / before
    return current, max(0.0, removed), applied


def describe_limits() -> dict[str, list[str]]:
    """What each fin's constraints cover, for reports and QC notes."""
    return {
        "pectoral": [
            f"not more than {PECTORAL_ANTERIOR_ALLOWANCE_SL:.0%} of SL anterior of "
            f"operculum_posterior (real fins overlap the gill cover by ~2.7%, so "
            f"this catches gross overhang only)",
        ],
        "dorsal": [
            f"not more than {DORSAL_VENTRAL_ALLOWANCE_SL:.1%} of SL below the "
            f"insertion chord between two flanking body-outline points",
            f"not more than {DORSAL_DISTAL_ALLOWANCE_SL:.0%} of SL beyond "
            f"dorsal_tip (this is what removes a specimen pin)",
        ],
        "pelvic": ["no anatomical limit encoded"],
        "anal": ["no anatomical limit encoded"],
    }
