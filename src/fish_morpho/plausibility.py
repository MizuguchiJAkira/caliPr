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

**A second axis, made of landmarks.** The outline is only there on a fish
somebody traced, and it is refused outright on the specimens where the model has
gone wrong -- so the check above is absent exactly when it is needed. The front of
the eye and the caudal base are carried by every labelled fish and come back from
the model on those specimens too, and the same landmarks sit in the same places
along them. Of the thirteen landmarks a labeller had to drag more than 250 px
back into place, this catches ten, including a ``dorsal_base_center`` on the snout
and a ``pectoral_ray_tip`` in the mirror; it flags none of the points that
labeller accepted. When more than a few landmarks fail it at once the anchor is
the likelier error than all of them, so that is reported and nothing is dropped.

**Two structures in the wrong order.** ``eye_posterior`` is behind
``eye_anterior``, ``peduncle_narrowest_ventral`` below ``peduncle_narrowest_dorsal``
-- these are not measurements, they are what the names mean, and they need
neither an outline nor an anchor. Each claim is still put to the labelled data
before it is used: one that any reviewed hand label contradicts is refused and
written into the file with the count, because a contradiction says either the
claim is wrong or the labels are, and it should be looked at rather than resolved
silently. Two fish contradict the eye claim today, both with the two eye points
clicked the wrong way round.

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

#: Slack for the fin tip-to-base bands, which rest on the six or seven fish that
#: carry a fin landmark at all rather than the forty-six that carry a head one.
#: Leave-one-out removes a sixth of the evidence, so these bands need more room
#: than the axial ones to avoid rejecting the fish they were not fitted on.
RELATIVE_MARGIN = 1.5

#: Smallest number of labelled examples a landmark needs before its band is
#: trusted. Below this the range says more about who labelled it than about the
#: animal.
MIN_SAMPLES = 5

BODY = "body_plus_caudal"

#: Fin tip and the base it belongs to. A tip's offset from its own base is a fact
#: about one fin, independent of where the ends of the fish are, so it survives
#: the outline being wrong -- which is exactly when the axial check cannot run.
FIN_PAIRS = (("dorsal", "dorsal_base_center", "dorsal_tip"),
             ("anal", "anal_base_center", "anal_tip"),
             ("pelvic", "pelvic_base_center", "pelvic_tip"),
             ("pectoral", "pectoral_insertion_upper", "pectoral_ray_tip"))

#: Scale for those offsets. The two eye points carry 46 labelled examples each and
#: come back at 0.76-0.95 confidence on specimens where every fin landmark fails,
#: so they are available precisely when the rest is not.
SCALE_POINTS = ("eye_anterior", "eye_posterior")

#: The axis made of landmarks rather than of the outline: front of the eye to the
#: caudal base. Both are carried by every labelled fish and both come back from
#: the model on specimens where the segmentation is refused -- which is precisely
#: when the outline-based check above cannot run, and when the model is at its
#: worst. Of the thirteen landmarks a labeller had to drag more than 250 px, this
#: catches ten, and flags none of the 553 points they accepted.
AXIS_POINTS = ("eye_anterior", "caudal_base")

#: How far off the eye may be, as a fraction of that axis, before the axis is not
#: believed. Measured: eye diameter runs 0.048-0.079 of snout-to-caudal-base
#: across the labelled fish. An eye outside that means one of the two anchors is
#: not the structure it is named after, and every position measured against them
#: would be wrong by the same factor -- so nothing is checked instead.
ANCHOR_EYE_SPAN = (0.04, 0.09)

#: When this many of a specimen's landmarks fail the axis test at once, the axis
#: is the likelier culprit than all of them, and nothing is dropped. A model that
#: has genuinely lost the fish fails everything; a model that put one fin in the
#: wrong place fails one.
ANCHOR_REVOLT = 6

#: Claims about where one landmark must sit relative to another, in image
#: coordinates (x rightward, y downward, specimens head-left). Unlike the bands,
#: these are not measured -- they are what the names mean, and a prediction that
#: breaks one has swapped two structures whatever the heatmap said. They are still
#: checked against the labelled data before being written into a dataset's file:
#: a claim that any reviewed hand label contradicts is refused and named, because
#: at that point either the claim is wrong or the labels are, and quietly keeping
#: it would decide which without looking.
#:
#: Fin tip claims are deliberately absent. "A tip is below its base" fails on 71
#: of 77 labelled pelvics -- preserved fins fold whichever way they dried -- and
#: the module has removed that rule once already.
ORDER_CLAIMS = (
    ("eye_anterior", "eye_posterior", "x", "behind"),
    ("eye_dorsal", "eye_ventral", "y", "below"),
    ("premaxilla_tip", "lower_jaw_tip", "y", "below"),
    ("eye_posterior", "operculum_posterior", "x", "behind"),
    ("operculum_posterior", "caudal_base", "x", "behind"),
    ("peduncle_narrowest_dorsal", "peduncle_narrowest_ventral", "y", "below"),
    ("dorsal_base_anterior", "dorsal_base_center", "x", "behind"),
    ("dorsal_base_center", "dorsal_base_posterior", "x", "behind"),
    ("anal_base_anterior", "anal_base_posterior", "x", "behind"),
    ("pelvic_base_center", "anal_base_center", "x", "behind"),
)

_AXIS_INDEX = {"x": 0, "y": 1}


def _axis(polygon) -> tuple[float, float] | None:
    """``(x_anterior, length)`` from the body outline, or None if unusable."""
    if not polygon or len(polygon) < 3:
        return None
    xs = [float(p[0]) for p in polygon]
    lo, hi = min(xs), max(xs)
    return (lo, hi - lo) if hi - lo > 0 else None


def _landmark_axis(keypoints) -> tuple[float, float] | None:
    """``(x_anterior, length)`` from the landmarks themselves, or None.

    The same shape as :func:`_axis`, so the same bands could in principle be
    applied to either -- they are not, because the two measure from different
    places and a fraction of one is not a fraction of the other.
    """
    a, b = (keypoints.get(n) for n in AXIS_POINTS)
    if not a or not b:
        return None
    x0, length = float(a[0]), float(b[0]) - float(a[0])
    return (x0, length) if length > 0 else None


def _anchor_is_credible(keypoints, length: float) -> bool:
    """Whether the two axis landmarks are plausibly the structures they name.

    One cheap test does it: an eye that is the wrong size for the fish it is on
    means the anchor has landed on something else, and every position measured
    against it inherits the error.
    """
    eye = _scale(keypoints)
    if eye is None:
        return True                       # nothing to contradict it
    lo, hi = ANCHOR_EYE_SPAN
    return lo <= eye / length <= hi


def _scale(keypoints) -> float | None:
    """Eye diameter, the one length available on a fish whose fins are invisible."""
    a, b = (keypoints.get(n) for n in SCALE_POINTS)
    if not a or not b:
        return None
    d = ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5
    return d if d > 0 else None


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


def _fit_along(records, margin: float) -> dict:
    """Bands along the eye-to-caudal-base axis, from every labelled fish.

    The outline-based bands can only be fitted on fish somebody traced. These need
    nothing but landmarks, so they are fitted on twice as many -- and, more to the
    point, they can be *checked* on a specimen whose outline was refused.
    """
    cols: dict[str, list[float]] = {}
    eye_span: list[float] = []
    used = 0
    for lat in records:
        kps = (lat or {}).get("keypoints") or {}
        axis = _landmark_axis(kps)
        if not axis:
            continue
        x0, length = axis
        used += 1
        eye = _scale(kps)
        if eye:
            eye_span.append(eye / length)
        for name, pt in kps.items():
            if pt:
                cols.setdefault(name, []).append((float(pt[0]) - x0) / length)

    lms: dict[str, dict] = {}
    for name, vals in sorted(cols.items()):
        entry: dict = {"n": len(vals)}
        if len(vals) >= MIN_SAMPLES:
            lo, hi = min(vals), max(vals)
            slack = max(MIN_MARGIN, margin * (hi - lo))
            entry["band"] = [round(lo - slack, 4), round(hi + slack, 4)]
            entry["observed"] = [round(lo, 4), round(hi, 4)]
        lms[name] = entry
    out = {"anchor": list(AXIS_POINTS), "fish": used, "landmarks": lms}
    if len(eye_span) >= MIN_SAMPLES:
        out["eye_span_observed"] = [round(min(eye_span), 4), round(max(eye_span), 4)]
    return out


def _fit_order(records) -> dict:
    """Which of :data:`ORDER_CLAIMS` the labelled fish bear out, and which they don't.

    Both halves are written down. A claim contradicted by a hand label is not
    checked against predictions -- but it is recorded, with the count, because
    "the labels disagree with this" is a finding about the labels as often as
    about the claim, and it should not disappear.
    """
    held: list[dict] = []
    refused: list[dict] = []
    for a, b, axis, relation in ORDER_CLAIMS:
        i = _AXIS_INDEX[axis]
        ok = bad = 0
        for lat in records:
            kps = (lat or {}).get("keypoints") or {}
            pa, pb = kps.get(a), kps.get(b)
            if not pa or not pb:
                continue
            greater = float(pb[i]) > float(pa[i])
            if greater is (relation in ("behind", "below")):
                ok += 1
            else:
                bad += 1
        claim = {"a": a, "b": b, "axis": axis, "relation": relation, "n": ok}
        if bad:
            refused.append({**claim, "contradicted_by": bad})
        elif ok >= MIN_SAMPLES:
            held.append(claim)
    return {"checked": held, "refused": refused}


def fit(records, margin: float = DEFAULT_MARGIN) -> dict:
    """Measure bands from ``records``, an iterable of sidecar ``lateral`` blocks.

    Returns the structure written to ``plausibility.json``. Landmarks seen fewer
    than :data:`MIN_SAMPLES` times are recorded with their count and no band, so
    the file shows what is not yet measurable rather than silently omitting it.
    """
    axial: dict[str, list[float]] = {}
    rel: dict[str, list[tuple[float, float]]] = {}
    fills: list[float] = []
    aspects: list[float] = []
    used = 0
    # Read twice: the bands below need a traced outline, which 60 of the 131
    # labelled fish carry, while the landmark axis and the order claims need only
    # landmarks and so are fitted on all of them.
    records = list(records)
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
        eye = _scale(kps)
        if eye:
            for fin, base, tip in FIN_PAIRS:
                a, b = kps.get(base), kps.get(tip)
                if a and b:
                    rel.setdefault(fin, []).append(
                        ((float(b[0]) - float(a[0])) / eye,
                         (float(b[1]) - float(a[1])) / eye))

    bands: dict[str, dict] = {}
    for name, vals in sorted(axial.items()):
        entry: dict = {"n": len(vals)}
        if len(vals) >= MIN_SAMPLES:
            lo, hi = min(vals), max(vals)
            slack = max(MIN_MARGIN, margin * (hi - lo))
            entry["axial"] = [round(lo - slack, 4), round(hi + slack, 4)]
            entry["axial_observed"] = [round(lo, 4), round(hi, 4)]
        bands[name] = entry
    relative: dict[str, dict] = {}
    for fin, base, tip in FIN_PAIRS:
        vals = rel.get(fin) or []
        entry: dict = {"n": len(vals), "base": base, "tip": tip}
        if len(vals) >= MIN_SAMPLES:
            for key, idx in (("dx", 0), ("dy", 1)):
                col = [v[idx] for v in vals]
                lo, hi = min(col), max(col)
                slack = max(MIN_MARGIN, RELATIVE_MARGIN * (hi - lo))
                entry[key] = [round(lo - slack, 3), round(hi + slack, 3)]
                entry[key + "_observed"] = [round(lo, 3), round(hi, 3)]
        relative[fin] = entry

    out: dict = {"fish": used, "margin": margin, "landmarks": bands,
                 "relative": relative,
                 "along": _fit_along(records, margin),
                 "order": _fit_order(records)}
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

    # Done first and unconditionally. A fin tip's offset from its own base needs
    # only the two eye points, so it still works on a specimen whose outline was
    # refused -- which is when the model is most likely to be wrong and least
    # likely to be checked. HRN_15 put dorsal_tip on the adipose at +4.87 eye
    # diameters against a labelled -0.44 to 1.06, on a fish the axial test could
    # not look at.
    bad: dict[str, str] = {}
    eye = _scale(keypoints)
    rel = bands.get("relative") or {}
    if eye:
        for fin, base, tip in FIN_PAIRS:
            entry = rel.get(fin) or {}
            a, b = keypoints.get(base), keypoints.get(tip)
            if not a or not b or "dx" not in entry:
                continue
            got = ((float(b[0]) - float(a[0])) / eye, (float(b[1]) - float(a[1])) / eye)
            for key, val, word in (("dx", got[0], "along the body"),
                                   ("dy", got[1], "above/below")):
                lo, hi = entry[key]
                if not (lo <= val <= hi):
                    seen = entry[key + "_observed"]
                    bad[tip] = (f"{val:+.1f} eye diameters {word} from {base}; "
                                f"labelled fish are {seen[0]:+.1f} to {seen[1]:+.1f} "
                                f"(n={entry['n']})")
                    break

    # Two structures in the wrong order. Nothing is measured here -- this is what
    # the names mean -- so it runs before anything that needs an outline or an
    # anchor, and on a prediction that has neither.
    # A mirrored frame reverses every one of them at once, so it is established
    # first and reported alone. It needs no outline either, which the version of
    # this test below it did.
    head, tail = keypoints.get("premaxilla_tip"), keypoints.get("caudal_base")
    if head and tail and float(head[0]) > float(tail[0]):
        return {**bad,
                "_frame": "snout is behind the caudal base — the frame is mirrored "
                          "or the subject was not found; specimens are photographed "
                          "head-left"}

    for claim in (bands.get("order") or {}).get("checked") or []:
        i = _AXIS_INDEX.get(claim.get("axis"), 0)
        pa, pb = keypoints.get(claim["a"]), keypoints.get(claim["b"])
        if not pa or not pb or claim["b"] in bad:
            continue
        behind = claim["relation"] in ("behind", "below")
        if (float(pb[i]) > float(pa[i])) is not behind:
            word = "behind" if claim["axis"] == "x" else "below"
            bad[claim["b"]] = (f"not {word} {claim['a']}, which it is in all "
                               f"{claim['n']} labelled fish — the two have been "
                               f"swapped")

    # The axis the landmarks themselves make. Fitted on every labelled fish rather
    # than the ones with a traced outline, and available on a specimen whose
    # outline was refused -- which is when the model is least reliable and, until
    # now, least checked.
    lax = bands.get("along") or {}
    axis_lm = _landmark_axis(keypoints)
    if lax.get("landmarks") and axis_lm and _anchor_is_credible(keypoints, axis_lm[1]):
        ax0, alen = axis_lm
        found: dict[str, str] = {}
        for name, pt in keypoints.items():
            if name in bad or not pt:
                continue
            entry = (lax["landmarks"].get(name) or {})
            band = entry.get("band")
            if not band:
                continue
            pos = (float(pt[0]) - ax0) / alen
            lo, hi = band
            if not (lo <= pos <= hi):
                seen = entry.get("observed") or band
                found[name] = (f"{pos:.2f} of the way from the eye to the caudal "
                               f"base; every labelled fish has it between "
                               f"{seen[0]:.2f} and {seen[1]:.2f} (n={entry['n']})")
        # All of them at once is a statement about the anchor, not about all of
        # them. Report that and drop nothing: a model that has lost the fish
        # entirely fails everything, one that misplaced a fin fails one.
        if len(found) >= ANCHOR_REVOLT:
            bad["_anchor"] = (f"{len(found)} landmarks are in impossible places "
                              f"relative to the eye and caudal base — those two "
                              f"are the likelier error, so nothing was dropped on "
                              f"that basis")
        else:
            bad.update(found)

    axis = _axis(polygon)
    if not axis:
        return bad
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
            # The relative findings stand -- they never used the outline.
            return {**bad,
                    "_axis": f"the predicted outline is not fish-shaped "
                             f"(fills {fill:.2f} of its box at {aspect:.1f}:1; "
                             f"hand tracings are {flo:.2f}-{fhi:.2f} at "
                             f"{alo:.1f}-{ahi:.1f}:1) — only the fin checks ran"}

    for name, pt in keypoints.items():
        if name in bad:
            continue
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
    along = bands.get("along") or {}
    along_n = [n for n, e in (along.get("landmarks") or {}).items() if e.get("band")]
    order = bands.get("order") or {}
    out = [f"axial position checked for {len(fitted)} landmarks, fitted on "
           f"{bands.get('fish', 0)} labelled fish with a traced outline",
           f"position along {' to '.join(along.get('anchor') or [])} checked for "
           f"{len(along_n)} landmarks, fitted on {along.get('fish', 0)} labelled "
           f"fish — this one needs no outline, so it also runs where the "
           f"segmentation is refused",
           f"{len(order.get('checked') or [])} pairs checked for being in the "
           f"wrong order",
           "orientation checked: specimens must be head-left"]
    for c in order.get("refused") or []:
        out.append(f"NOT checked: {c['b']} {c['relation']} {c['a']} — "
                   f"{c['contradicted_by']} labelled fish say otherwise, so either "
                   f"the rule or those labels are wrong")
    if thin:
        out.append("too few labelled examples to check: " + ", ".join(sorted(thin)))
    return out
