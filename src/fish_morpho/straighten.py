"""Straightening a bent fish: measure it along its midline, as MorFishJ does.

MorFishJ (Ghilardi 2022) asks for a segmented line down the middle of a bent
fish and hands it to ImageJ's Straighten command. That fits a curve through the
clicks and resamples the photograph across it -- a strip perpendicular to the
curve at every pixel of its length, laid side by side -- so the midline becomes a
horizontal line, and every trait is then measured on the new image.

This does the same geometry to the coordinates instead of the pixels. The curve
is the one ImageJ fits (``PolygonRoi.fitSplineForStraightening``): a natural
cubic spline through the clicks, parameterised by the square root of each
segment's length, and each landmark goes where ImageJ's straightened image would
put it -- its distance along the curve becomes x, its distance off the curve,
measured square to it, becomes y. Pixel size is unchanged, so the ruler's
calibration still holds.

The photograph and the clicks on it are never changed. The midline is kept
beside them (``lateral.midline`` in the sidecar), the measurements are taken in
the straightened frame, and deleting the midline undoes it.

What follows from the geometry, in ImageJ's version as much as here:

* A straight midline is a rotation. Two clicks along a tilted fish level it,
  which is MorFishJ's other option, "Rotate".
* Off the midline, lengths along the body are stretched on the outside of a
  bend and squeezed on the inside, by the factor ``1 - curvature * offset``. A
  point on the inside of a bend farther from the curve than its radius has no
  single place on the straightened fish -- the perpendiculars cross before they
  reach it -- and is reported as folded rather than put somewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]

#: How close ``1 - curvature * offset`` may come to zero before a point counts as
#: folded: past it, a pixel's worth of error across the body moves the point a
#: body's worth along it.
FOLD_MARGIN = 0.05


def _second_derivatives(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """A natural cubic spline's second derivatives at its knots (zero at the ends).

    The tridiagonal system of ImageJ's ``SplineFitter``, solved the same way.
    """
    n = len(u)
    m2 = np.zeros(n)
    if n < 3:
        return m2
    h = np.diff(u)
    k = n - 2
    sub, diag, sup = h[:-1], 2.0 * (h[:-1] + h[1:]), h[1:]
    rhs = 6.0 * ((v[2:] - v[1:-1]) / h[1:] - (v[1:-1] - v[:-2]) / h[:-1])
    cp, dp = np.zeros(k), np.zeros(k)
    cp[0], dp[0] = sup[0] / diag[0], rhs[0] / diag[0]
    for i in range(1, k):
        den = diag[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / den
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / den
    x = np.zeros(k)
    x[-1] = dp[-1]
    for i in range(k - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    m2[1:-1] = x
    return m2


def _evaluate(u: np.ndarray, v: np.ndarray, m2: np.ndarray, at: np.ndarray) -> np.ndarray:
    j = np.clip(np.searchsorted(u, at, side="right") - 1, 0, len(u) - 2)
    h = u[j + 1] - u[j]
    a = (u[j + 1] - at) / h
    b = (at - u[j]) / h
    return a * v[j] + b * v[j + 1] + ((a ** 3 - a) * m2[j] + (b ** 3 - b) * m2[j + 1]) * h * h / 6.0


def fit_curve(nodes) -> np.ndarray:
    """The midline as ImageJ straightens along it: points about a pixel apart.

    A natural cubic spline through ``nodes``, each coordinate a function of the
    running sum of the square roots of the segment lengths; evaluated at twice
    the clicked line's length in pixels, then resampled at equal steps of as near
    one pixel as divides the curve's length.
    """
    p = np.asarray(nodes, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError("a midline needs at least two points")
    seg = np.hypot(*np.diff(p, axis=0).T)
    if seg.sum() < 2.0:
        raise ValueError("the midline's points are all in one place")
    u = np.concatenate([[0.0], np.cumsum(np.maximum(np.sqrt(seg), 0.001))])
    mx, my = _second_derivatives(u, p[:, 0]), _second_derivatives(u, p[:, 1])
    fine_n = int(seg.sum() * 2) + 1
    at = np.linspace(0.0, u[-1], fine_n)
    fine = np.column_stack([_evaluate(u, p[:, 0], mx, at), _evaluate(u, p[:, 1], my, at)])
    # equal steps along the fine curve
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(fine, axis=0).T))])
    n_out = int(round(d[-1])) + 1
    s = np.linspace(0.0, d[-1], max(n_out, 2))
    return np.column_stack([np.interp(s, d, fine[:, 0]), np.interp(s, d, fine[:, 1])])


@dataclass
class Midline:
    """A fitted midline and the straightened frame it defines.

    x in the straightened frame is distance along the curve from its first point
    (the snout end, once :meth:`oriented` has been applied), y is distance off it,
    positive toward the side a fish facing left keeps its belly -- so a straight,
    level fish comes out exactly as it went in, shifted.
    """

    nodes: list[Point]
    curve: np.ndarray = field(repr=False)
    s: np.ndarray = field(repr=False)          # arc length at each curve point
    tangent: np.ndarray = field(repr=False)    # unit tangent of each segment
    normal: np.ndarray = field(repr=False)     # unit normal of each segment
    kappa: np.ndarray = field(repr=False)      # curvature at each segment

    @classmethod
    def fit(cls, nodes) -> "Midline":
        curve = fit_curve(nodes)
        d = np.diff(curve, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        L[L == 0] = 1e-9
        tangent = d / L[:, None]
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        s = np.concatenate([[0.0], np.cumsum(L)])
        theta = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        if len(theta) > 2:
            mid = 0.5 * (s[:-1] + s[1:])
            kappa = np.gradient(theta, mid)
        else:
            kappa = np.zeros(len(theta))
        return cls([tuple(map(float, q)) for q in nodes], curve, s, tangent, normal, kappa)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def turn_deg(self) -> float:
        """How far the midline turns end to end, summed over every bend, in degrees.

        Zero for a straight fish however it is tilted. A natural spline eases off
        toward its ends, so a fish bent into a quarter circle reads a little under
        90.
        """
        th = np.unwrap(np.arctan2(self.tangent[:, 1], self.tangent[:, 0]))
        return float(np.degrees(np.abs(np.diff(th)).sum())) if len(th) > 1 else 0.0

    def oriented(self, snout: Point | None) -> "Midline":
        """This midline running from the snout end, whichever way it was drawn."""
        if snout is None:
            return self
        (s0, _), = self.project([snout])
        return self if s0 <= self.length / 2 else Midline.fit(self.nodes[::-1])

    def project(self, points) -> list[tuple[float, float]]:
        """Each point as (distance along the curve, signed distance off it).

        The nearest point of the curve is the foot of the perpendicular, which is
        what ImageJ samples along. Beyond either end the end segment is continued
        straight, so a snout tip or fin ray past the last click still has a place.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        a = self.curve[:-1]
        seg = self.curve[1:] - a
        seg_len2 = np.maximum((seg ** 2).sum(axis=1), 1e-12)
        out = []
        last = len(seg) - 1
        for p in pts:
            rel = p - a
            tau = (rel * seg).sum(axis=1) / seg_len2
            tau_c = np.clip(tau, 0.0, 1.0)
            foot = a + seg * tau_c[:, None]
            k = int(np.argmin(((p - foot) ** 2).sum(axis=1)))
            t_k = tau_c[k]
            if k == 0 and tau[0] < 0:
                t_k = tau[0]
            elif k == last and tau[last] > 1:
                t_k = tau[last]
            along = self.s[k] + t_k * math.sqrt(seg_len2[k])
            off = float(np.dot(p - (a[k] + seg[k] * t_k), self.normal[k]))
            out.append((float(along), off))
        return out

    def folded(self, points) -> list[int]:
        """Indices of the points on the inside of a bend beyond its centre.

        Tested against the curvature all along the stretch of midline within the
        point's own distance of its foot, not at the foot alone: a point past the
        centre of a bend is nearer some other part of the curve -- often an end,
        where a natural spline is straight -- and would pass a test made there.
        """
        mid = 0.5 * (self.s[:-1] + self.s[1:])
        bad = []
        for i, (along, off) in enumerate(self.project(points)):
            near = np.abs(mid - along) <= abs(off) + 1.0
            k = self.kappa[near] if near.any() else self.kappa[[int(np.argmin(np.abs(mid - along)))]]
            if (1.0 - k * off).min() < FOLD_MARGIN:
                bad.append(i)
        return bad

    def to_straight(self, points) -> list[Point]:
        """Points in the straightened frame, placed from the curve's first point."""
        x0, y0 = map(float, self.curve[0])
        return [(x0 + along, y0 + off) for along, off in self.project(points)]

    def to_image(self, along, off):
        """Where (along, off) in the straightened frame lies on the photograph.

        Vectorised over arrays; the ends are continued straight, as in
        :meth:`project`.
        """
        along = np.asarray(along, dtype=float)
        off = np.asarray(off, dtype=float)
        x = np.interp(along, self.s, self.curve[:, 0])
        y = np.interp(along, self.s, self.curve[:, 1])
        mid = 0.5 * (self.s[:-1] + self.s[1:])
        th = np.unwrap(np.arctan2(self.tangent[:, 1], self.tangent[:, 0]))
        ang = np.interp(along, mid, th)
        before, after = along < 0, along > self.length
        t0, t1 = self.tangent[0], self.tangent[-1]
        x = np.where(before, self.curve[0, 0] + along * t0[0], x)
        y = np.where(before, self.curve[0, 1] + along * t0[1], y)
        x = np.where(after, self.curve[-1, 0] + (along - self.length) * t1[0], x)
        y = np.where(after, self.curve[-1, 1] + (along - self.length) * t1[1], y)
        return x - np.sin(ang) * off, y + np.cos(ang) * off


def midline_of(block: dict | None) -> list[Point] | None:
    """The view block's midline, if it has one worth using (two points or more)."""
    raw = (block or {}).get("midline")
    if not raw or len(raw) < 2:
        return None
    try:
        return [(float(p[0]), float(p[1])) for p in raw]
    except (TypeError, ValueError, IndexError):
        return None


#: Landmarks that say which end of a midline is the snout, most reliable first.
SNOUT_LANDMARKS = ("premaxilla_tip", "lower_jaw_tip", "eye_anterior", "orbit_center",
                   "dentary_anterior")


def straighten_block(block: dict) -> tuple[dict, dict]:
    """The view block with its keypoints and outlines in the straightened frame.

    Returns ``(block, info)``. ``info`` says what was done -- the midline's
    length in pixels, how far it turns, and any points that fell in a fold --
    for the QC sheet to record. A block without a midline comes back unchanged
    with ``info`` empty.
    """
    nodes = midline_of(block)
    if not nodes:
        return block, {}
    kps = block.get("keypoints") or {}
    snout = next((tuple(kps[n]) for n in SNOUT_LANDMARKS if n in kps), None)
    ml = Midline.fit(nodes).oriented(snout)
    out = dict(block)
    names, pts = [], []
    for n, p in kps.items():
        names.append(n)
        pts.append(p)
    # Anchored where the fish is: the straightened landmarks share the clicked
    # ones' centroid. Nothing measured depends on it, but a coordinate file
    # does -- in a .tps a negative coordinate means "missing".
    anchor = pts or nodes
    straight_anchor = ml.to_straight(anchor)
    dx = float(np.mean([p[0] for p in anchor]) - np.mean([p[0] for p in straight_anchor]))
    dy = float(np.mean([p[1] for p in anchor]) - np.mean([p[1] for p in straight_anchor]))

    def moved(points):
        return [(x + dx, y + dy) for x, y in ml.to_straight(points)]
    new_kps = dict(zip(names, moved(pts))) if pts else {}
    folded = [names[i] for i in ml.folded(pts)] if pts else []
    out["keypoints"] = new_kps
    polys = {}
    for n, verts in (block.get("polygons") or {}).items():
        if verts:
            polys[n] = moved(verts)
            if ml.folded(verts):
                folded.append(n)
    if block.get("polygons") is not None:
        out["polygons"] = polys
    info = {"length_px": round(ml.length, 1), "turn_deg": round(ml.turn_deg, 1),
            "points": len(nodes), "folded": folded}
    return out, info


def straighten_image(image: np.ndarray, midline: Midline, points=(), *,
                     margin: float = 0.04, max_width: int = 2400):
    """ImageJ's straightened image of the fish, for looking at.

    Covers the midline and every point given (the landmarks and outlines, in
    photograph coordinates) with a margin; scaled down to at most ``max_width``
    pixels across. Returns ``(image, place)`` where ``place(points)`` gives
    where photograph points land on it. Needs OpenCV.
    """
    import cv2

    proj = midline.project(points) if len(points) else []
    alongs = [0.0, midline.length] + [a for a, _ in proj]
    offs = [abs(o) for _, o in proj] or [0.0]
    pad = margin * midline.length
    s_lo, s_hi = min(alongs) - pad, max(alongs) + pad
    half = max(max(offs) + pad, 0.12 * midline.length)
    scale = min(1.0, max_width / max(s_hi - s_lo, 1.0))
    w = int(math.ceil((s_hi - s_lo) * scale))
    h = int(math.ceil(2 * half * scale))
    cols = s_lo + (np.arange(w) + 0.5) / scale
    rows = -half + (np.arange(h) + 0.5) / scale
    A, O = np.meshgrid(cols, rows)
    mx, my = midline.to_image(A, O)
    out = cv2.remap(image, mx.astype(np.float32), my.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(40, 40, 40))

    def place(pts):
        return [((a - s_lo) * scale, (o + half) * scale) for a, o in midline.project(pts)]
    return out, place
