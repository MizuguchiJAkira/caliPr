"""Where a landmark cannot be, measured from the fish rather than assumed.

A heatmap model places every landmark it was asked for, whether or not the
structure is visible. When a fin is folded flat against the body there is nothing
in the image to find, and the argmax lands wherever the texture happens to look
fin-like -- a pelvic base on the snout, an anal fin anterior of the pelvic, a
dorsal tip on the specimen pin. Confidence catches most of that, but not all:
``ASN_42`` returned a ``dorsal_base_center`` half a fin base out of position at
**0.914** confidence, which would have gone into a workbook unchallenged.

So the checks here are not a second opinion about the image. They are facts about
where a brook trout's landmarks sit relative to its own body, and a prediction
that violates one is wrong no matter how confident it is.

**The axis.** Jonah's suggestion, and the reason this works at all: the anterior
and posterior extremes of the ``body_plus_caudal`` outline -- snout tip and the
end of the caudal fin -- are two points the segmentation finds reliably even when
every fin landmark fails. Expressed as a fraction of that span, each landmark sits
in a narrow, repeatable place: ``pelvic_base_center`` at 0.444 +- 0.013,
``anal_base_center`` at 0.692 +- 0.015, across 46 hand-labelled fish. A point far
outside the observed range is not a near miss, it is a different structure.

**Orientation.** Lab standard is head-left, and all 44 hand-labelled specimens
carrying both landmarks obey it, so a prediction with the snout behind the caudal
base means the frame is mirrored or the fish was not found. That invalidates every
axial test below, so it is reported on its own and the rest are skipped.

**Two rules were tried and removed**, both of which sound more obvious than the
one that works.

*Which side of its base a fin tip falls on.* The labelled data does not support
it: ``pelvic_tip`` sits *above* ``pelvic_base_center`` in 5 of the 6 fish that
carry both. Either preserved pelvics fold up against the flank more often than
not, or those six labels disagree with each other. Six examples cannot tell the
difference, and a rule fitted to them would reject correct predictions. This
belongs here once the fin landmarks are labelled on more than 13% of the set.

*How far a landmark sits off the body outline.* Measured on hand tracings and
applied to SAM's outline, which is not the same curve -- it wraps the fins a
tracing cuts across, and it is resampled to 52 even vertices. The rule flagged
``premaxilla_tip`` at 1.00 confidence and ``operculum_posterior`` at 0.87 on
specimens where both were correct, because the reference distribution was never
the one being tested against. Fitting it on predicted outlines instead would mean
fitting on the model's own output, which is what this module exists to avoid.

**The axis is checked before it is used.** Every position is a fraction of the
predicted outline, so an outline that is not a fish silently rescales all nineteen
of them. ``ASN_48`` segmented a lobe of foam above a small, curved specimen --
aspect 2.01 against a hand-traced 3.33-5.49 -- and eight landmarks were rejected
against an axis that was itself wrong. An outline whose bounding-box fill or
aspect falls outside what hand tracings do is refused, and nothing is checked on
that specimen. Withholding the check is the honest failure; rejecting good
landmarks on a bad axis is not.

The bands are measured, never hand-written -- ``scripts/fit_plausibility.py``
regenerates them from a dataset's own sidecars into ``plausibility.json``. A
dataset without that file is not checked, which is the right default for a taxon
nobody has measured yet: silverside fins sit nowhere near where a trout's do.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Slack each side of the observed range, as a multiple of that range. A landmark
#: whose labelled positions span 0.04 of the body is being measured far more
#: precisely than one spanning 0.13, and a flat allowance treats them alike: at a
#: flat 0.05 the ``ASN_42`` dorsal base (0.91 confidence, half a fin base out of
#: place) lands exactly on the boundary and survives.
MARGIN_FRACTION_OF_SPREAD = 0.5

#: Floor under that, so a landmark with a very tight observed range still gets
#: room for a specimen more extreme than the 46 measured so far.
MIN_MARGIN = 0.02

#: The cost either way is not symmetric. A correct landmark wrongly dropped costs
#: the labeller the seconds it takes to place it -- exactly the state it would be
#: in with no automation at all. A wrong landmark wrongly kept can reach a
#: workbook as data. Widening from a flat 0.05 to this raised false positives from
#: 1 to 4 in 553 hand-placed landmarks (0.18% -> 0.72%) and caught two confident
#: errors instead of one, including ``ASN_12`` anal_tip at 0.87 sitting on the
#: caudal peduncle. That trade is worth making again.
DEFAULT_MARGIN = MARGIN_FRACTION_OF_SPREAD

#: Smallest number of labelled examples a landmark needs before its band is
#: trusted. Below this the range says more about who labelled it than about the
#: animal.
MIN_SAMPLES = 5

BODY = "body_plus_caudal"


def _axis(polygon) -> tuple[float, float] | None:
    """``(x_anterior, length)`` from the body outline, or None if unusable."""
    if not polygon or len(polygon) < 3:
        return None
    xs = [float(p[0]) for p in polygon]
    lo, hi = min(xs), max(xs)
    return (lo, hi - lo) if hi - lo > 0 else None


def outline_shape(polygon) -> tuple[float, float] | None:
    """``(bbox_fill, aspect)`` for an outline -- how fish-shaped it is.

    A trout silhouette is a long, fat, smooth blob: it fills most of its bounding
    box and is several times longer than it is deep. A segmentation that has
    wandered off the animal is neither. These two numbers are enough to tell the
    difference and cost nothing to compute.
    """
    if not polygon or len(polygon) < 3:
        return None
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w <= 0 or h <= 0:
        return None
    n = len(polygon)
    area = abs(sum(polygon[i][0] * polygon[(i + 1) % n][1]
                   - polygon[(i + 1) % n][0] * polygon[i][1]
                   for i in range(n))) / 2
    return area / (w * h), w / h


def fit(records, margin: float = DEFAULT_MARGIN) -> dict:
    """Measure bands from ``records``, an iterable of sidecar ``lateral`` blocks.

    Returns the structure written to ``plausibility.json``. Landmarks seen fewer
    than :data:`MIN_SAMPLES` times are recorded with their count and no band, so
    the file shows what is not yet measurable rather than silently omitting it.
    """
    axial: dict[str, list[float]] = {}
    fills: list[float] = []
    aspects: list[float] = []
    used = 0
    for lat in records:
        kps = (lat or {}).get("keypoints") or {}
        poly = ((lat or {}).get("polygons") or {}).get(BODY)
        axis = _axis(poly)
        if not kps or not axis:
            continue
        x0, length = axis
        used += 1
        shape = outline_shape(poly)
        if shape:
            fills.append(shape[0])
            aspects.append(shape[1])
        for name, pt in kps.items():
            if not pt:
                continue
            axial.setdefault(name, []).append((float(pt[0]) - x0) / length)

    bands: dict[str, dict] = {}
    for name, vals in sorted(axial.items()):
        entry: dict = {"n": len(vals)}
        if len(vals) >= MIN_SAMPLES:
            lo, hi = min(vals), max(vals)
            slack = max(MIN_MARGIN, margin * (hi - lo))
            entry["axial"] = [round(lo - slack, 4), round(hi + slack, 4)]
            entry["axial_observed"] = [round(lo, 4), round(hi, 4)]
        bands[name] = entry
    out: dict = {"fish": used, "margin": margin, "landmarks": bands}
    if len(fills) >= MIN_SAMPLES:
        # What a hand-traced fish silhouette looks like. An outline outside this
        # is not a fish and must not be used as the axis -- see check().
        out["outline"] = {"n": len(fills),
                          "fill": [round(min(fills), 3), round(max(fills), 3)],
                          "aspect": [round(min(aspects), 2), round(max(aspects), 2)]}
    return out


def load(dataset_dir) -> dict | None:
    """``plausibility.json`` for a dataset, or None when it has not been fitted."""
    path = Path(dataset_dir) / "plausibility.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if data.get("landmarks") else None


def check(keypoints: dict, polygon, bands: dict | None) -> dict[str, str]:
    """Which landmarks are anatomically impossible, and why.

    Returns ``{landmark: reason}``. An empty result means nothing contradicted the
    body it was found on -- not that the prediction is correct.

    Missing bands, a missing outline and an unusable axis all return ``{}``: this
    withholds judgement rather than inventing it, so calling it unconditionally is
    safe.
    """
    if not bands or not keypoints:
        return {}
    lm = bands.get("landmarks") or {}
    axis = _axis(polygon)
    if not axis:
        return {}
    x0, length = axis

    # Every position below is a fraction of this outline, so an outline that is
    # not a fish silently rescales all nineteen of them. ASN_48 segmented a lobe
    # of foam above the specimen -- aspect 2.01 against a hand-traced 3.33-5.49 --
    # and eight landmarks were rejected on an axis that was itself wrong. Better
    # to check nothing than to check against that.
    limits = bands.get("outline")
    shape = outline_shape(polygon)
    if limits and shape:
        fill, aspect = shape
        flo, fhi = limits["fill"]
        alo, ahi = limits["aspect"]
        if not (flo <= fill <= fhi) or not (alo <= aspect <= ahi):
            return {"_axis": f"the predicted outline is not fish-shaped "
                             f"(fills {fill:.2f} of its box at {aspect:.1f}:1; "
                             f"hand tracings are {flo:.2f}-{fhi:.2f} at "
                             f"{alo:.1f}-{ahi:.1f}:1) — landmarks were not checked"}

    # Head-left is the lab standard and every hand-labelled specimen obeys it. A
    # mirrored frame makes every axial position below meaningless, so say that
    # once instead of reporting nineteen consequences of it.
    head, tail = keypoints.get("premaxilla_tip"), keypoints.get("caudal_base")
    if head and tail and float(head[0]) > float(tail[0]):
        return {"_frame": "snout is behind the caudal base — the frame is mirrored "
                          "or the subject was not found; specimens are photographed "
                          "head-left"}

    bad: dict[str, str] = {}
    for name, pt in keypoints.items():
        entry = lm.get(name)
        if not pt or not entry:
            continue
        band = entry.get("axial")
        if not band:
            continue
        pos = (float(pt[0]) - x0) / length
        lo, hi = band
        if not (lo <= pos <= hi):
            seen = entry.get("axial_observed") or band
            bad[name] = (f"{pos:.2f} along the body; every labelled fish has it "
                         f"between {seen[0]:.2f} and {seen[1]:.2f} (n={entry['n']})")
    return bad


def describe(bands: dict | None) -> list[str]:
    """What is and is not being checked, for the QC sheet and reports."""
    if not bands:
        return ["no plausibility bands fitted for this dataset — landmarks are not "
                "checked against body position"]
    lm = bands.get("landmarks") or {}
    fitted = [n for n, e in lm.items() if e.get("axial")]
    thin = [f"{n} (n={e['n']})" for n, e in lm.items() if not e.get("axial")]
    out = [f"axial position checked for {len(fitted)} landmarks, fitted on "
           f"{bands.get('fish', 0)} labelled fish with a traced outline",
           "orientation checked: specimens must be head-left"]
    if thin:
        out.append("too few labelled examples to check: " + ", ".join(sorted(thin)))
    return out
