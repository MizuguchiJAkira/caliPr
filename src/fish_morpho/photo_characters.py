"""Identification characters read from the photograph, where the landmarks say to look.

Shiner keys lean on pigment and scale shape as much as on proportions: a dark spot
at the base of the caudal fin (spottail shiner), a dark stripe along the side and
onto the snout (blacknose, bridle), and scales taller than long on the front of the
body (the highscale shiners, *Luxilus*; *Cyprinella*). Melanin survives alcohol, so
these show on museum fish after the colour has gone.

``pigment`` reads how dark each place is against the pale lower flank of the same
fish: lightness (L*) is the median over a small disc, so scattered melanophores
count for what they cover and the photograph's exposure cancels. The side stripe
is the darkest disc across the side, at three places along it, since the stripe
rides higher or lower on different fish.

``scale_shape`` measures how upright the scale margins are along the front of the
side, from the shoulder girdle to below the dorsal-fin origin, in a band either
side of the head-to-tail axis: the share of the margins' strength
(``fish_scales.margin_map``) in lines nearer vertical than horizontal. Scales
taller than long have their margins near vertical.

Not here, because they failed on the shiner reference photographs: the dorsal-fin
spots (front, as in creek chub; rear membranes, as in spotfin). A fin is
translucent over whatever lies behind the fish -- white on most collections' light
boxes, black on CMN's -- and often folded, and measuring against the background
above it did not rescue the spotfin's blotch.

Every place is set from the landmarks, in the fish's own frame (snout to the end
of the vertebral column, and across it), and sized by standard length. A place the
landmarks cannot set, or that falls off the photograph, is left out rather than
guessed.
"""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

from fish_morpho import fish_scales

SNOUT = ("premaxilla_tip", "dentary_anterior")
TAIL = "vertebral_centrum_posterior"
OPERCLE = ("opercle_dorsal", "opercle_ventral", "opercle_medial", "cleithrum")

#: Names as they appear beside the landmark measurements in an identification.
PIGMENT = ("caudal spot", "snout", "side stripe")
SCALES = ("upright scale margins",)
NAMES = PIGMENT + SCALES


def _p(kp: dict, name: str):
    v = kp.get(name)
    return np.array(v[:2], float) if v else None


def frame(kp: dict):
    """(snout, u along the fish, v across it toward the belly, SL px), or None."""
    snout = next((_p(kp, n) for n in SNOUT if kp.get(n)), None)
    tail = _p(kp, TAIL)
    if snout is None or tail is None:
        return None
    sl = float(np.linalg.norm(tail - snout))
    if sl <= 0:
        return None
    u = (tail - snout) / sl
    v = np.array([-u[1], u[0]])
    back = _p(kp, "dorsal_fin_base_anterior")
    belly = _p(kp, "anal_fin_base_anterior") if back is None else None
    if back is not None and v @ (back - snout) > 0:
        v = -v
    elif belly is not None and v @ (belly - snout) < 0:
        v = -v
    elif back is None and belly is None:
        return None
    return snout, u, v, sl


def _lightness(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]


def _disc(L: np.ndarray, c, r: float) -> float | None:
    r = max(3.0, r)
    x0, y0, x1, y1 = int(c[0] - r), int(c[1] - r), int(c[0] + r) + 1, int(c[1] + r) + 1
    if x0 < 0 or y0 < 0 or x1 > L.shape[1] or y1 > L.shape[0]:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r
    return float(np.median(L[y0:y1, x0:x1][m]))


def places(kp: dict) -> dict:
    """Where each pigment is read: name -> (centre (x, y), radius px)."""
    f = frame(kp)
    if f is None:
        return {}
    snout, u, v, sl = f
    out = {}
    # on the end of the vertebral column: 0.03 SL further back is fin, over the light box
    out["caudal spot"] = (_p(kp, TAIL), 0.018 * sl)
    pm, oa = _p(kp, "premaxilla_tip"), _p(kp, "orbit_anterior")
    if pm is not None and oa is not None:
        out["snout"] = ((pm + oa) / 2, max(0.012 * sl, 0.3 * float(np.linalg.norm(oa - pm))))
    pv = _p(kp, "pelvic_fin_base_anterior")
    if pv is not None:
        # the lower flank above the pelvic fin: 60% of the way from the axis to it
        h = v @ (pv - snout)
        out["lower flank"] = (snout + (u @ (pv - snout)) * u + 0.6 * h * v, 0.025 * sl)
    return out


def pigment(image: np.ndarray, kp: dict) -> dict:
    """How dark each place is against the lower flank: (flank - place) / flank."""
    L = _lightness(image)
    where = places(kp)
    if "lower flank" not in where:
        return {}
    ref = _disc(L, *where["lower flank"])
    if not ref:
        return {}
    out = {}
    for name in ("caudal spot", "snout"):
        if name in where:
            val = _disc(L, *where[name])
            if val is not None:
                out[name] = round((ref - val) / ref, 4)
    stripe = _stripe(L, kp, ref)
    if stripe is not None:
        out["side stripe"] = round(stripe, 4)
    return out


def _stripe(L: np.ndarray, kp: dict, ref: float) -> float | None:
    """The darkest disc across the side, 0.06 SL toward the back to 0.035 SL toward
    the belly, at 35, 55 and 75% of SL; the median of the three."""
    snout, u, v, sl = frame(kp)
    best = []
    for frac in (0.35, 0.55, 0.75):
        vals = [_disc(L, snout + frac * sl * u + h * sl * v, 0.01 * sl) for h in np.arange(-0.06, 0.035, 0.0075)]
        vals = [(ref - x) / ref for x in vals if x is not None]
        if vals:
            best.append(max(vals))
    return float(np.median(best)) if best else None


def scale_shape(image: np.ndarray, kp: dict, half_band_sl: float = 0.06) -> dict:
    """Share of the scale margins' strength in near-vertical lines, front of the side."""
    f = frame(kp)
    if f is None:
        return {}
    snout, u, v, sl = f
    ops = [u @ (_p(kp, n) - snout) for n in OPERCLE if kp.get(n)]
    da = _p(kp, "dorsal_fin_base_anterior")
    if not ops or da is None:
        return {}
    t0, t1 = max(ops) + 0.03 * sl, u @ (da - snout)
    if t1 - t0 < 0.08 * sl:
        return {}
    # straighten the band: columns along the fish, rows across it, 1 px a step
    ts = np.arange(t0, t1)
    hs = np.arange(-half_band_sl * sl, half_band_sl * sl)
    mx = (snout[0] + ts[None, :] * u[0] + hs[:, None] * v[0]).astype(np.float32)
    my = (snout[1] + ts[None, :] * u[1] + hs[:, None] * v[1]).astype(np.float32)
    if mx.min() < 0 or my.min() < 0 or mx.max() >= image.shape[1] or my.max() >= image.shape[0]:
        return {}
    band = cv2.remap(_lightness(image), mx, my, cv2.INTER_LINEAR)
    size = sl / 49                                     # one lateral-line scale
    g = fish_scales._clahe(band, 3.0, size).astype(np.float32)
    s = max(1.0, 0.07 * size)
    b = cv2.GaussianBlur(g, (0, 0), s)
    dxx = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3)
    half, root = (dxx + dyy) / 2, np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
    big, small = half + root, half - root
    strength = np.clip(big - np.abs(small), 0, None)
    # the curvature is greatest ACROSS a line: for an upright line that is along x
    nx2 = np.where(root > 1e-6, 0.5 + (dxx - dyy) / (4 * np.maximum(root, 1e-6)), 0.5)
    total = float(strength.sum())
    if total <= 0:
        return {}
    return {"upright scale margins": round(float((strength * nx2).sum()) / total, 4)}


def characters(image: np.ndarray, kp: dict) -> dict:
    """``pigment`` and ``scale_shape`` together; keys from ``NAMES``."""
    kp = {k: v for k, v in kp.items() if v}
    out = pigment(image, kp)
    out.update(scale_shape(image, kp))
    return out


def dorsal_behind_pelvic(kp: dict) -> float | None:
    """How far the dorsal-fin origin is behind the pelvic-fin origin, along the
    fish, over SL: negative when the dorsal fin starts in front of the pelvics."""
    f = frame(kp)
    da, pv = _p(kp, "dorsal_fin_base_anterior"), _p(kp, "pelvic_fin_base_anterior")
    if f is None or da is None or pv is None:
        return None
    _, u, _, sl = f
    return float(u @ (da - pv)) / sl


def landmarks_key(kp: dict) -> str:
    """A short fingerprint of the landmarks characters were read with."""
    pts = {k: [round(float(v[0]), 1), round(float(v[1]), 1)] for k, v in sorted(kp.items()) if v}
    return hashlib.sha1(json.dumps(pts, sort_keys=True).encode()).hexdigest()[:16]
