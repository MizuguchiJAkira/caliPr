"""One set of landmark names for training a model across landmark schemes.

A model learns more about a fish -- where its eye is, where its jaws meet -- from
trout and alewife together than from either alone, but the studies name their
points by different schemes: caliPr's own (``landmark_config``) and BGNN 2D
(``schemes``). This is the list a cross-scheme model is trained on, and the map
from each scheme's names onto it.

Two names share one entry only where both protocols define the same anatomical
point, so that a model trained on one learns the other:

=============================  ==============================
caliPr                         BGNN 2D
=============================  ==============================
eye_anterior / posterior /     orbit_anterior / posterior /
dorsal / ventral               dorsal / ventral (bony socket in both)
lower_jaw_tip                  dentary_anterior
maxilla_mandible_intersection  mouth_posterior
caudal_base                    vertebral_centrum_posterior
dorsal_base_anterior/posterior dorsal_fin_base_anterior/posterior
anal_base_anterior/posterior   anal_fin_base_anterior/posterior
pectoral_insertion_upper       pectoral_fin_origin
caudal_crease_dorsal/ventral   (the same names)
=============================  ==============================

Everything else stays apart, even where the words look alike: caliPr's
``operculum_posterior`` is the trailing edge of the gill cover, BGNN's three
opercle points are not it. Nothing here is used to compute a trait; the schemes
module's rule (no trait across schemes) is unchanged. A study's prediction is
mapped back to its own scheme's names before anything else sees it.

Fin tips and base centres are not in the list. Where a preserved fin's tip lies
records how it dried, and it was the commonest landmark to land off the fish.
"""

from __future__ import annotations

#: The model's outputs, in order. Do not reorder within a trained model: the
#: heatmap channels are this list.
UNION: tuple[str, ...] = (
    "eye_anterior", "eye_posterior", "eye_dorsal", "eye_ventral", "orbit_center",
    "premaxilla_tip", "lower_jaw_tip", "maxilla_mandible_intersection",
    "operculum_posterior", "supraoccipital_tip",
    "pectoral_insertion_upper", "pectoral_fin_inner", "cleithrum",
    "opercle_medial", "opercle_dorsal", "opercle_ventral",
    "dorsal_base_anterior", "dorsal_base_posterior",
    "pelvic_base_center", "pelvic_fin_base_anterior", "pelvic_fin_base_posterior",
    "anal_base_anterior", "anal_base_posterior",
    "caudal_crease_dorsal", "caudal_base", "caudal_crease_ventral",
)

#: Scheme name -> union name, for the names that differ. Every union name is also
#: its own scheme name in the scheme that has it.
ALIASES: dict[str, str] = {
    "orbit_anterior": "eye_anterior",
    "orbit_posterior": "eye_posterior",
    "orbit_dorsal": "eye_dorsal",
    "orbit_ventral": "eye_ventral",
    "dentary_anterior": "lower_jaw_tip",
    "mouth_posterior": "maxilla_mandible_intersection",
    "vertebral_centrum_posterior": "caudal_base",
    "dorsal_fin_base_anterior": "dorsal_base_anterior",
    "dorsal_fin_base_posterior": "dorsal_base_posterior",
    "anal_fin_base_anterior": "anal_base_anterior",
    "anal_fin_base_posterior": "anal_base_posterior",
    "pectoral_fin_origin": "pectoral_insertion_upper",
}

_UNION_SET = frozenset(UNION)


def union_name(name: str) -> str | None:
    """The union entry a scheme's landmark trains, or None if it trains none."""
    u = ALIASES.get(name, name)
    return u if u in _UNION_SET else None


def to_union(keypoints: dict) -> dict:
    """A study's keypoints under union names; landmarks outside the union dropped.

    Two scheme names that map to one union name cannot both be present in one
    study (no scheme has both), so nothing is overwritten.
    """
    out = {}
    for name, p in keypoints.items():
        u = union_name(name)
        if u is not None:
            out[u] = p
    return out


#: How a fish is framed for the second pass, shared by training and prediction so
#: the two cannot drift apart. ``length_px`` is the fish's landmark extent
#: (snout to caudal base, roughly) after scaling; the frame is ``width`` times that
#: extent across, and at least ``height_min`` times it -- or ``height_pad`` times
#: the landmarks' own depth, if more -- top to bottom.
FRAMING = {"length_px": 700, "width": 1.45, "height_min": 0.75, "height_pad": 1.6}
#: The first pass sees the whole photograph with its longest side this long.
WHOLE_SIDE = 1024


def fish_frame(points) -> dict | None:
    """The rectangle (photograph pixels) and scale that frame a fish's points."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    length = max(xs) - min(xs)
    if length < 20:
        return None
    w = FRAMING["width"] * length
    h = max(FRAMING["height_min"] * length, FRAMING["height_pad"] * (max(ys) - min(ys)))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return {"x0": cx - w / 2, "y0": cy - h / 2, "w": w, "h": h,
            "scale": FRAMING["length_px"] / length}


def cut_frame(image, frame: dict):
    """The framed fish as an image, and a function mapping photograph points onto it.

    The part of the frame outside the photograph is mid-grey. Returns
    ``(crop, to_crop)``; :func:`from_crop` maps back.
    """
    import cv2
    import numpy as np

    H, W = image.shape[:2]
    x0, y0 = int(round(frame["x0"])), int(round(frame["y0"]))
    x1, y1 = int(round(frame["x0"] + frame["w"])), int(round(frame["y0"] + frame["h"]))
    pad = [max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W)]
    piece = image[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any(pad):
        piece = cv2.copyMakeBorder(piece, *pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    out_w = max(1, int(round((x1 - x0) * frame["scale"])))
    out_h = max(1, int(round((y1 - y0) * frame["scale"])))
    crop = cv2.resize(piece, (out_w, out_h), interpolation=cv2.INTER_AREA)
    sx, sy = out_w / max(1, x1 - x0), out_h / max(1, y1 - y0)
    frame.update({"cx0": x0, "cy0": y0, "sx": sx, "sy": sy})

    def to_crop(x, y):
        return ((x - x0) * sx, (y - y0) * sy)
    return crop, to_crop


def from_crop(frame: dict, u: float, v: float) -> tuple[float, float]:
    """A point on a :func:`cut_frame` crop, back in photograph pixels."""
    return (u / frame["sx"] + frame["cx0"], v / frame["sy"] + frame["cy0"])


#: Landmarks on the back and on the belly, for telling which way up a fish is.
DORSAL = ("supraoccipital_tip", "dorsal_base_anterior", "dorsal_base_posterior",
          "eye_dorsal", "caudal_crease_dorsal", "opercle_dorsal")
VENTRAL = ("pelvic_base_center", "pelvic_fin_base_anterior", "anal_base_anterior",
           "anal_base_posterior", "eye_ventral", "caudal_crease_ventral", "opercle_ventral")


def template(keypoint_sets) -> dict:
    """Where each landmark sits on the median fish of a training set.

    Each fish is laid snout at (0, 0) and caudal base at (1, 0), back upward
    (negative y, as in an image); the template is the median position of each
    landmark over the fish that have it. Saved with a model, it is what prediction
    fits to the first pass's points to find the whole fish from part of it.
    """
    import math
    import statistics

    acc: dict[str, list] = {}
    for kps in keypoint_sets:
        snout = kps.get("premaxilla_tip") or kps.get("lower_jaw_tip")
        tail = kps.get("caudal_base")
        if not snout or not tail:
            continue
        dx, dy = tail[0] - snout[0], tail[1] - snout[1]
        length = math.hypot(dx, dy)
        if length < 20:
            continue
        c, s = dx / length, dy / length
        laid = {n: (((x - snout[0]) * c + (y - snout[1]) * s) / length,
                    (-(x - snout[0]) * s + (y - snout[1]) * c) / length)
                for n, (x, y) in kps.items()}
        up = [laid[n][1] for n in DORSAL if n in laid]
        down = [laid[n][1] for n in VENTRAL if n in laid]
        flip = -1.0 if up and down and statistics.median(up) > statistics.median(down) else 1.0
        for n, (u, v) in laid.items():
            acc.setdefault(n, []).append((u, flip * v))
    return {n: [round(statistics.median(p[0] for p in pts), 4),
                round(statistics.median(p[1] for p in pts), 4)]
            for n, pts in acc.items() if len(pts) >= 3}


def placements(tmpl: dict, found: dict, tolerance: float = 0.12, most: int = 3) -> list:
    """The template placed on points of ``found`` that agree with one another.

    ``found`` is {union name: (x, y, confidence)}. Every pair of points far enough
    apart proposes a placement (position, size, rotation, and either facing); a
    placement's support is the points that agree with it -- within ``tolerance``
    of the fish's length. Up to ``most`` placements are returned, best supported
    first, each on different points: a photograph can hold more than one thing
    that looks like part of a fish (a number card's ``0`` is an eye to a model
    that has not seen enough of them), and which one is the fish is for the
    second pass to say. Each is refitted to its points by least squares.

    Each is ``(place, agree, length)``: ``place(name)`` is where the template puts
    that landmark in the photograph, ``agree`` the names that supported it,
    ``length`` the placed fish's snout-to-caudal-base length in pixels.
    """
    names = [n for n in found if n in tmpl]
    proposals = []
    for facing in (1.0, -1.0):
        t = {n: complex(tmpl[n][0], facing * tmpl[n][1]) for n in names}
        p = {n: complex(found[n][0], found[n][1]) for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if abs(t[b] - t[a]) < 0.15:
                    continue
                k = (p[b] - p[a]) / (t[b] - t[a])
                off = p[a] - k * t[a]
                agree = frozenset(n for n in names
                                  if abs(k * t[n] + off - p[n]) < tolerance * abs(k))
                if len(agree) >= 3:
                    proposals.append((sum(0.5 + float(found[n][2]) for n in agree),
                                      facing, agree))
    chosen = []
    for score, facing, agree in sorted(proposals, key=lambda q: -q[0]):
        if all(len(agree & other) < min(len(agree), len(other)) / 2 for _, other in chosen):
            chosen.append((facing, agree))
        if len(chosen) == most:
            break
    return [_refit(tmpl, found, facing, agree) for facing, agree in chosen]


def _refit(tmpl: dict, found: dict, facing: float, agree) -> tuple:
    t = {n: complex(tmpl[n][0], facing * tmpl[n][1]) for n in agree}
    p = {n: complex(found[n][0], found[n][1]) for n in agree}
    w = {n: 0.5 + float(found[n][2]) for n in agree}
    sw = sum(w.values())
    mt = sum(w[n] * t[n] for n in agree) / sw
    mp = sum(w[n] * p[n] for n in agree) / sw
    k = (sum(w[n] * (p[n] - mp) * (t[n] - mt).conjugate() for n in agree)
         / sum(w[n] * abs(t[n] - mt) ** 2 for n in agree))
    off = mp - k * mt

    def place(name: str) -> tuple[float, float]:
        z = k * complex(tmpl[name][0], facing * tmpl[name][1]) + off
        return (z.real, z.imag)
    return place, sorted(agree), abs(k)


def fit_template(tmpl: dict, found: dict, tolerance: float = 0.12):
    """The best supported of :func:`placements`, or None if no three points agree."""
    found_any = placements(tmpl, found, tolerance, most=1)
    return found_any[0] if found_any else None


def from_union(predicted: dict, wanted) -> dict:
    """A union-named prediction under the names a study asks for.

    ``wanted`` is the study's landmark names; one the model has no output for is
    simply absent from the result.
    """
    out = {}
    for name in wanted:
        u = union_name(name)
        if u is not None and u in predicted:
            out[name] = predicted[u]
    return out
