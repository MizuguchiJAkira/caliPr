"""Long-lived prediction process, so the labeler's Auto-label is not 7s a click.

Loading DeepLabCut and a snapshot costs ~7 seconds; the inference itself costs a
fraction of one. Running ``predict_landmarks.py`` per request pays that every
time. This loads once and then answers on stdin/stdout, so only the first request
in a session waits.

It also solves an environment problem. The labeler runs in the plain ``.venv``,
which has no torch and no DeepLabCut; the training stack lives in
``.venv-train``. Rather than merge them, the server launches this module with the
training interpreter and talks to it over pipes.

``{"detect": true, "image": ...}`` instead finds every fish in a photograph of
several (``fish_morpho.batch_fish``): a rectangle and an outline for each, no
landmarks.

Protocol — one JSON object per line, in and out::

    {"image": "/abs/path/to/fish.JPEG", "polygons": true, "view": "lateral",
     "crop": [x0, y0, x1, y1]}
    {"ok": true, "fish_id": "...", "keypoints": {...}, "confidence": {...},
     "low_confidence": [...], "polygons": {"body_plus_caudal": [[x, y], ...]},
     "implausible": {"pelvic_tip": "why it cannot be there"},
     "frame_warning": null, "elapsed": 0.21}

``implausible`` names landmarks the model placed somewhere a fish cannot have
them, checked against the dataset's own measured bands (see
:mod:`fish_morpho.plausibility`). Those are absent from ``keypoints`` — a point
the anatomy rules out is not offered, because a labeller can accept a flagged
point but cannot un-see a confident one in the wrong place.

``crop`` is optional and names the rectangle of the photograph this view lives in,
for a study that keeps one photograph per fish rather than two crops on disk. The
models were trained on crops and are still given one; it is made here, in memory.
Everything -- the resize, the outline, the plausibility check -- happens inside
that rectangle, and the offset is added back before emitting, so every coordinate
that comes out names a pixel of the photograph that was sent.

``view`` picks the model. The lateral one is loaded at start, as before; the
frontal one (mouth corners, trained separately in ``dlc_project_frontal``) only
when a frontal image is first asked for. A frontal prediction has no outline and
no plausibility check -- the bands describe positions along a fish's side -- and
its two corners come back in image order, ``mouth_left`` the further left, which
is how the model was trained to name them.

Segment Anything is loaded lazily, on the first request that asks for polygons,
because it costs several seconds and a keypoints-only pass should not pay for it.

Errors come back as ``{"ok": false, "error": "..."}`` on the same line-per-request
discipline, so one bad image cannot desynchronise the stream or take the worker
down with it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho import paths, plausibility  # noqa: E402  (needs the path above)
from fish_morpho import landmark_union as lu  # noqa: E402
from fish_morpho.landmark_config import KEYPOINTS as _KEYPOINTS, View as _View  # noqa: E402

#: The names a request gets back when it does not say which it wants: caliPr's.
CALIPR_LATERAL = [k.name for k in _KEYPOINTS if k.view == _View.LATERAL]

#: The cross-species model (trained by ``build_union_dataset.py``) predicts in two
#: passes. The first sees the whole photograph and only finds the fish; these are
#: the points it frames it by -- ones every fish has and the model finds well. The
#: second pass sees the fish cut out at a standard length and places the points.
#: A rig that photographs from farther away, or a fish in a corner of the frame,
#: then looks to the second pass like every fish it was trained on.
FRAME_POINTS = ("eye_anterior", "eye_posterior", "eye_dorsal", "eye_ventral",
                "orbit_center", "premaxilla_tip", "lower_jaw_tip",
                "maxilla_mandible_intersection", "operculum_posterior",
                "pectoral_insertion_upper", "dorsal_base_anterior",
                "dorsal_base_posterior", "pelvic_base_center",
                "pelvic_fin_base_anterior", "anal_base_anterior",
                "anal_base_posterior", "caudal_base")
#: Low on purpose. The first pass only has to say roughly where the fish's ends
#: are, and it does that well long before it is sure of it: points under 0.1 sit
#: on the right part of the fish. A frame drawn from the confident points alone
#: is often the head only, which the second pass then reads as a whole fish.
#: Strays agree with nothing and are left out (``candidate_frames``).
FRAME_MIN_CONFIDENCE = 0.05
#: How many places the first pass may propose for the fish. The second pass looks
#: at each and keeps the one whose surest landmarks it is surest of: the mean
#: confidence of its best ``FRAME_SURE_OF`` -- not of all of them, since a trout
#: has no BGNN opercle points to be sure of. A frame scoring ``FRAME_SURE`` ends
#: the search. On a frame round a number card the model still finds some points
#: at 0.6-0.8; round the fish, eight above 0.9.
FRAME_CANDIDATES = 3
FRAME_SURE_OF = 8
FRAME_SURE = 0.9

#: A point this far from where the fish's other landmarks put it -- as a fraction of
#: its length -- is on the wrong part of the fish. Generous on purpose: the template
#: is a straight median fish. A bent one puts its caudal base up to 27% of its
#: length from where a template placed on its head would (TEM_18, TEM_24, from
#: their hand labels), and other species are built differently -- a mormyrid's
#: dorsal fin starts some 15% further back than a trout's. The points it exists
#: to catch were 65-110% out: a caudal base beside the eye.
SHAPE_TOLERANCE = 0.35
#: The points the fish's placement is taken from.
SHAPE_MIN_CONFIDENCE = 0.3


def shape_outliers(ukps: dict, uconf: dict, tmpl: dict | None) -> dict:
    """Points far from where the rest of the fish puts them, with why (union names).

    The template is placed on the confident points that agree with one another;
    a point it would put somewhere else entirely is dropped. Off the fish or on
    it, a caudal base beside the eye is not a caudal base.
    """
    import math

    if not tmpl:
        return {}
    sure = {n: (*ukps[n], uconf[n]) for n in ukps if uconf.get(n, 0) >= SHAPE_MIN_CONFIDENCE}
    fit = lu.fit_template(tmpl, sure, tolerance=0.08)
    if fit is None or len(fit[1]) < 4:
        return {}
    place, agree, length = fit
    out = {}
    for n, (x, y) in ukps.items():
        if n in tmpl and n not in agree:
            d = math.dist((x, y), place(n)) / length
            if d > SHAPE_TOLERANCE:
                out[n] = (f"not where the rest of the fish puts it: {100 * d:.0f}% of the "
                          f"fish's length from where its other landmarks place it")
    return out


#: How far outside the fish's outline a point may sit and still count as on it, as
#: a fraction of the fish's length: the snout tip and the fin bases lie on the
#: edge, and an outline is never traced to the pixel.
ON_BODY_MARGIN = 0.03


def candidate_frames(names, arr, tmpl: dict | None = None, size=None) -> list:
    """Where the fish may be, for the second pass: frames, most likely first.

    With the model's template (``landmark_union.template``), the median fish is
    placed on first-pass points that agree with one another, and each frame drawn
    round the whole of it: a fish whose head was found but not its tail is still
    framed snout to tail. More than one placement can be offered -- a number
    card's ``0`` can pass for an eye -- and the second pass, which sees each up
    close, says which is the fish.
    """
    if tmpl:
        found = {n: (float(x), float(y), float(c)) for n, (x, y, c) in zip(names, arr)
                 if c >= FRAME_MIN_CONFIDENCE}
        frames = []
        for place, agree, _ in lu.placements(tmpl, found, most=FRAME_CANDIDATES):
            frame = lu.fish_frame([place(n) for n in tmpl] + [found[n][:2] for n in agree])
            if frame is None:
                continue
            # A placement whose fish would lie mostly outside the photograph is
            # not one the photograph shows.
            cx, cy = frame["x0"] + frame["w"] / 2, frame["y0"] + frame["h"] / 2
            if size and not (0 <= cx <= size[0] and 0 <= cy <= size[1]):
                continue
            frames.append(frame)
        if frames:
            return frames
    frame = robust_frame(names, arr)
    return [frame] if frame else []


def robust_frame(names, arr):
    """One frame round the first pass's points, strays dropped (no template).

    Points far from the rest -- on a pin, a card, the mirror -- are left out
    before the frame is drawn.
    """
    import numpy as np

    pts = [(float(x), float(y), float(c), n) for n, (x, y, c) in zip(names, arr)
           if n in FRAME_POINTS]
    good = [p for p in pts if p[2] >= FRAME_MIN_CONFIDENCE]
    if len(good) < 4:
        good = sorted(pts, key=lambda p: -p[2])[:6]
    xy = np.array([[p[0], p[1]] for p in good])
    if len(xy) < 2:
        return None
    med = np.median(xy, axis=0)
    d = np.hypot(*(xy - med).T)
    keep = xy[d <= max(3.0 * float(np.median(d)), 1e-6)]
    frame = lu.fish_frame(keep.tolist())
    found = {p[3] for p in good}
    # Without both ends the extent is short of the fish: give it room.
    if frame and not ("caudal_base" in found and {"premaxilla_tip", "lower_jaw_tip"} & found):
        grow = 0.35 * frame["w"]
        frame["x0"] -= grow / 2
        frame["w"] += grow
    return frame

#: The one polygon worth predicting. SAM matches a dense hand tracing on the body
#: outline (~1.6% median area error here, and it is 52 of the 82 vertices traced
#: per fish) and cannot do the fins at all — 8-35% median depending on which, with
#: the sign changing between specimens, so a correction cannot even be applied
#: consistently. Predicting them would cost more review time than tracing them.
BODY = "body_plus_caudal"

#: Keypoints that prompt it. All positive, spread head to tail: a bare point pair
#: makes SAM return part-level masks, several along the axis pin it to the animal.
BODY_PROMPT = ("premaxilla_tip", "operculum_posterior", "pectoral_insertion_upper",
               "peduncle_narrowest_dorsal", "peduncle_narrowest_ventral",
               "caudal_base")

#: Matches the median hand tracing, so a predicted outline and a traced one carry
#: the same amount of detail and the density check treats them alike.
BODY_VERTICES = 52

#: How far a body vertex may sit off the chord joining its neighbours, as a
#: fraction of standard length. Fitted, not guessed: across 55 hand-traced
#: sidecars the 95th percentile for vertices anterior to ``caudal_base`` is 2.08%
#: of SL. SAM exceeds it where it has wrapped a fin — the adipose and anal push
#: the outline out, the pelvic shadow pulls it in — and a fish's body wall does
#: neither.
MAX_BODY_SAGITTA_SL = 0.021

#: The caudal fan is left alone. Its own 95th percentile is 3.07% and its fork is
#: genuinely angular, so the same limit applied there would round off a real
#: structure. The split is at ``caudal_base``, which is where the measurement
#: engine already divides body from caudal.
SMOOTH_ITERATIONS = 12

#: Spans of the body margin between two landmarks that sit ON that margin, with
#: how far a hand tracing is allowed to bulge past the straight chord joining
#: them, as a fraction of SL. Fitted from the tracings that carry both the
#: outline and the landmarks: the observed maxima are 2.77 / 5.33 / 2.47 / 3.27%,
#: and these allowances sit just above each.
#:
#: This is what the local smoothing could not do. A fin excursion is broad, so no
#: single vertex looks spiky against its neighbours, but the whole span sits far
#: outside the chord — which is exactly how a person tracing by hand knows to cut
#: across a fin base instead of following it.
MARGIN_CHORDS = (
    # The exact fin bases, once the model predicts them. These are the chords a
    # person actually traces along, so they cut a fin off cleanly rather than
    # bounding how far it may bulge.
    ("dorsal_base_anterior", "dorsal_base_posterior", 0.004),
    ("anal_base_anterior", "anal_base_posterior", 0.004),
    # Available today. They bound the spans between landmarks the model does
    # predict, which covers the adipose and the pelvic notch but NOT the dorsal
    # fin itself: its excursion is centred on dorsal_base_center, and there is no
    # anchor forward of it to draw a chord from. That gap closes when the four
    # base endpoints above enter the model — see the README.
    ("dorsal_base_center", "peduncle_narrowest_dorsal", 0.030),   # adipose
    ("pectoral_insertion_upper", "pelvic_base_center", 0.058),    # convex belly
    ("pelvic_base_center", "anal_base_center", 0.028),
    ("anal_base_center", "peduncle_narrowest_ventral", 0.036),
)

#: How far inside a chord the margin may fall. The flank between two margin
#: landmarks is convex, so a dip inside the chord is a shadow SAM followed, not
#: anatomy — the pelvic notch in particular.
INWARD_ALLOWANCE_SL = 0.012


def mask_to_polygon(mask, n: int, scale: float):
    """Largest external contour, resampled to ``n`` points by arc length.

    Resampling by arc length rather than simplifying by tolerance gives every
    outline the same vertex count regardless of specimen size, so a predicted
    outline is directly comparable to a hand-traced one and the sparse-outline
    check applies to both on the same terms.
    """
    import cv2
    import numpy as np

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea).squeeze(1).astype(float)
    if len(c) < 4:
        return []
    closed = np.vstack([c, c[:1]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n, endpoint=False)
    return [[round(float(np.interp(x, d, closed[:, 0])) / scale, 1),
             round(float(np.interp(x, d, closed[:, 1])) / scale, 1)] for x in t]


def smooth_body(poly, kps, max_frac=MAX_BODY_SAGITTA_SL,
                iters=SMOOTH_ITERATIONS):
    """Pull body vertices back onto a smooth margin, leaving the caudal fan alone.

    SAM segments the whole animal, so its outline wraps the adipose and anal fins
    rather than crossing their bases, and dips into the shadow under the pelvic.
    A fish's body wall does neither: it is smooth between the head and the
    peduncle. This finds vertices sitting further off their neighbours' chord
    than any hand tracing puts them and eases them back, repeatedly, until none
    are left or the iteration budget runs out.

    Only vertices anterior to ``caudal_base`` are touched. The caudal fan is
    legitimately angular — it has a fork — and smoothing it would destroy a real
    structure to fix a different problem.

    Returns the polygon unchanged if the landmarks needed to orient the fish are
    missing: a constraint that cannot be applied correctly should not be applied
    approximately.
    """
    import numpy as np

    a, b = kps.get("premaxilla_tip") or kps.get("lower_jaw_tip"), kps.get("caudal_base")
    if not a or not b or len(poly) < 8:
        return poly
    P = np.array(poly, float)
    axis = np.array(b, float) - np.array(a, float)
    sl = float(np.hypot(*axis))
    if sl < 1e-6:
        return poly
    u = axis / sl
    # Position along the snout->caudal_base axis, so a specimen pinned at a
    # slight angle is split in the same place as a level one.
    t = (P - np.array(a, float)) @ u
    is_body = t < sl                       # anterior of caudal_base
    if is_body.sum() < 6:
        return poly

    # Cross the fin bases first — that is the large, structured error — then
    # smooth what is left, which is sampling noise along the margin.
    try:
        clipped, _ = clip_to_margins(P.tolist(), kps, sl)
        P = np.array(clipped, float)
    except Exception:
        pass

    limit = max_frac * sl
    for _ in range(iters):
        prev, nxt = np.roll(P, 1, 0), np.roll(P, -1, 0)
        chord = nxt - prev
        L = np.hypot(chord[:, 0], chord[:, 1])
        ap = P - prev
        sag = np.abs(chord[:, 0] * ap[:, 1] - chord[:, 1] * ap[:, 0]) / np.maximum(L, 1e-9)
        bad = is_body & (sag > limit)
        if not bad.any():
            break
        # Halfway to the midpoint of the neighbours. Moving the whole way would
        # overshoot into the body on a run of consecutive offenders; half
        # converges without flattening the genuine curve of the flank.
        P[bad] = P[bad] + 0.5 * ((prev[bad] + nxt[bad]) / 2.0 - P[bad])
    return [[round(float(x), 1), round(float(y), 1)] for x, y in P]


def clip_to_margins(poly, kps, sl):
    """Hold the outline between the chords joining landmarks on the body margin.

    SAM returns the whole animal, so its outline climbs over the dorsal and
    adipose fins and around the anal rather than crossing their bases, and it
    dips into the shadow beneath the pelvic. Both are excursions away from the
    line between two points a person would trace through.

    The span for each chord is taken as the run of vertices **along the outline**
    between the two anchors, not the vertices that project onto the chord: a
    dorsal chord is projected onto by the whole ventral margin too, and pulling
    those to it folds the fish flat.

    Each span is bounded on both sides — not further out than the chord plus the
    allowance fitted from hand tracings, and not further in than a small slack,
    because a flank between two margin landmarks is convex.
    """
    import numpy as np

    P = np.array(poly, float)
    n = len(P)
    mid = np.array([(kps["premaxilla_tip"][0] + kps["caudal_base"][0]) / 2.0,
                    (kps["premaxilla_tip"][1] + kps["caudal_base"][1]) / 2.0])
    moved = 0

    def nearest_index(pt):
        return int(np.argmin(np.hypot(P[:, 0] - pt[0], P[:, 1] - pt[1])))

    for a_name, b_name, out_frac in MARGIN_CHORDS:
        a_pt, b_pt = kps.get(a_name), kps.get(b_name)
        if not a_pt or not b_pt:
            continue                      # landmark not collected or not predicted
        ia, ib = nearest_index(a_pt), nearest_index(b_pt)
        if ia == ib:
            continue
        # Two ways round the ring; the margin is the shorter one. A span over
        # half the outline means an anchor was matched to the wrong side.
        fwd = (ib - ia) % n
        idx = ([(ia + k) % n for k in range(1, fwd)] if fwd <= n - fwd
               else [(ib + k) % n for k in range(1, n - fwd)])
        if not (2 <= len(idx) <= n // 2):
            continue

        a = np.array(a_pt, float); b = np.array(b_pt, float)
        ab = b - a; L = float(np.hypot(*ab))
        if L < 1e-6:
            continue
        nrm = np.array([-ab[1], ab[0]]) / L
        apx = mid - a
        if (ab[0] * apx[1] - ab[1] * apx[0]) / L > 0:
            nrm = -nrm                    # point it away from the fish's axis

        hi, lo = out_frac * sl, -INWARD_ALLOWANCE_SL * sl
        for i in idx:
            out = float((P[i] - a) @ nrm)
            excess = out - hi if out > hi else (out - lo if out < lo else 0.0)
            if excess:
                P[i] -= excess * nrm
                moved += 1
    return P.tolist(), moved


class _Sam:
    """Segment Anything, loaded on first use."""

    proc = model = None

    @classmethod
    def ready(cls, device: str):
        if cls.model is None:
            from transformers import SamModel, SamProcessor
            cls.proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
            cls.model = SamModel.from_pretrained("facebook/sam-vit-base")
            cls.model = cls.model.to(device).eval()
        return cls.proc, cls.model

    @classmethod
    def fish_mask(cls, rgb_small, box, device: str):
        """The fish inside ``box`` (small-image pixels) as a boolean mask, or None."""
        import torch

        proc, model = cls.ready(device)
        inputs = proc(rgb_small, input_boxes=[[box]], return_tensors="pt")
        inputs = {k: (v.to(torch.float32)
                      if getattr(v, "dtype", None) == torch.float64 else v)
                  for k, v in inputs.items()}
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, multimask_output=True)
        masks = proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0][0]
        best = int(out.iou_scores.cpu()[0][0].argmax())
        return masks[best].numpy().astype(bool)

    @classmethod
    def body_outline(cls, rgb_small, points, device: str, scale: float):
        import torch

        proc, model = cls.ready(device)
        inputs = proc(rgb_small, input_points=[[points]], return_tensors="pt")
        # The processor emits point coordinates as float64 and MPS has no such
        # dtype, so the call fails outright rather than falling back.
        inputs = {k: (v.to(torch.float32)
                      if getattr(v, "dtype", None) == torch.float64 else v)
                  for k, v in inputs.items()}
        inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, multimask_output=True)
        masks = proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0][0]
        best = int(out.iou_scores.cpu()[0][0].argmax())
        return mask_to_polygon(masks[best].numpy(), BODY_VERTICES, scale)


#: Which fins the trained outliner is offered for. Five-fold cross-validation on
#: 116 hand tracings puts the pectoral at 3.2% median area error and the anal at
#: 4.6%, both inside the spread between two hand tracings of the same fin. The
#: dorsal (11.0%) and the pelvic (6.9%, worst case +111% on a small folded fin)
#: are not offered: a predicted outline a labeller accepts becomes what the next
#: model is trained on, so the bar is not "better than nothing", it is "no worse
#: than the tracing it would replace".
OUTLINED_FINS = ("pectoral", "anal")

#: Vertices in a predicted fin outline. The hand-tracing target is 16 and the
#: median traced fin has 20, so this is a little denser than what it stands in
#: for -- enough to hold the shape, few enough to drag into place.
FIN_VERTICES = 24


class _Fin:
    """The fin outliner, loaded on first use.

    Absent on a machine that has not fetched it, which is not an error: the fins
    are hand-traced everywhere else, and that is what this stands in for.
    """

    model = _tried = None

    @classmethod
    def ready(cls, device: str):
        if cls.model is None and not cls._tried:
            cls._tried = True
            import torch

            import train_fin_segmenter as tfs
            path = paths.MODELS / "fin_seg_runs" / "fin_segmenter.pt"
            if not path.is_file():
                return None
            blob = torch.load(path, map_location="cpu", weights_only=False)
            m = tfs.FinSegmenter(pretrained=False)
            m.load_state_dict(blob["state_dict"])
            cls.model = m.to(device).eval()
        return cls.model

    @classmethod
    def outline(cls, image, kps: dict, fin: str, device: str):
        """One fin's outline in the photograph's coordinates, or None.

        The crop is framed from the fin's own base and tip and sized from the
        snout-to-caudal-base distance, which is how the model was trained. A fin
        whose base or tip is missing -- not found, or dropped as anatomically
        impossible -- gets no crop and so no outline, rather than one placed
        around a landmark that is not there.
        """
        import math

        import numpy as np
        import torch

        import build_fin_dataset as bfd
        import train_fin_segmenter as tfs

        model = cls.ready(device)
        need = (*bfd.FIN_LANDMARKS[fin], "premaxilla_tip", "caudal_base")
        if model is None or any(n not in kps for n in need):
            return None
        sl = math.dist(kps["premaxilla_tip"], kps["caudal_base"])
        base, tip = (kps[n] for n in bfd.FIN_LANDMARKS[fin])
        box = bfd.crop_box(base, tip, sl)
        crop, sx, sy = bfd.cut(image, box)
        bt = bfd.to_crop([base, tip], box, sx, sy).astype(np.float32)
        x = tfs.tensorise(crop[:, :, ::-1], bt)[None].to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        mask = tfs.largest_component(prob > 0.5)
        if mask.sum() < 16:
            return None
        poly = mask_to_polygon(mask, FIN_VERTICES, 1.0)
        # crop pixels back to the photograph's own
        return [[round(px / sx + box[0], 1), round(py / sy + box[1], 1)]
                for px, py in poly]


#: Plausibility bands, keyed by dataset directory and the file's modification
#: time. The worker is long-lived, so keying on the directory alone kept a study's
#: first answer for the whole session: copy plausibility.json into a new study
#: after its first Auto-label, or refit it, and nothing was checked until restart.
_BANDS_CACHE: dict[tuple[str, int], dict | None] = {}


def _bands_for(image: Path) -> dict | None:
    """The dataset's bands for the image being predicted, if it has been fitted.

    Images live at ``<dataset>/lateral/<stem>_L.JPEG``, so the dataset directory
    is two levels up. A dataset nobody has fitted returns None and is not checked
    — the right default for a taxon whose landmarks sit nowhere near a trout's.
    """
    dataset = image.parent.parent
    f = dataset / "plausibility.json"
    key = (str(dataset), f.stat().st_mtime_ns if f.is_file() else 0)
    if key not in _BANDS_CACHE:
        _BANDS_CACHE[key] = plausibility.load(dataset)
    return _BANDS_CACHE[key]


def _emit(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


#: What a machine without a trained model is told. The models are not in the
#: repository; they are downloaded.
NO_MODEL = ("There is no trained {view} model on this machine. Download it with:\n\n"
            "    .venv/bin/python scripts/fetch_model.py\n\n"
            "then click Auto-label again.")


class MissingModel(Exception):
    pass


class _Model:
    """One trained DeepLabCut model: its config, snapshot, scale and landmarks."""

    def __init__(self, project: Path | None, snapshot: Path | None,
                 scale: float | None, view: str):
        import ruamel.yaml

        import predict_landmarks as pl

        self.project = pl.find_project(project, view)
        self.cfg, self.snapshot = pl.find_config_and_snapshot(self.project, snapshot)
        try:
            self.split = json.loads((self.project / "split.json").read_text())
        except Exception:
            self.split = {}
        # A cross-species model is framed per fish, not shrunk by one factor.
        self.union = self.split.get("view") == "union"
        self.scale = None if self.union else pl.training_scale(self.project, scale)
        with open(self.cfg) as fh:
            self.names = list(ruamel.yaml.YAML().load(fh)["metadata"]["bodyparts"])
        self.label = pl.model_label(self.project, self.snapshot)
        self.runner = None

    def infer(self, bgr, device: str):
        """The model's (x, y, confidence) per landmark on one image, in its pixels.

        The runner is built once and the image handed over in memory -- the two
        passes would otherwise load the snapshot twice per photograph.
        """
        import cv2
        import numpy as np
        if self.runner is None:
            from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
            self.runner = get_pose_inference_runner(
                model_config=str(self.cfg), snapshot_path=str(self.snapshot), device=device)
        pred = self.runner.inference([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)])[0]
        return np.asarray(pred["bodyparts"], dtype=float).reshape(-1, 3)


def _mouth_corners_in_image_order(kps: dict, confs: dict) -> None:
    """``mouth_left`` is the corner further left in the image, as in training.

    The heatmaps can still hand back the two the other way round on an unusual
    head; mouth width is the distance between them either way, but a labeller
    reviewing the points should see each where its name says.
    """
    a, b = kps.get("mouth_left"), kps.get("mouth_right")
    if a and b and a[0] > b[0]:
        kps["mouth_left"], kps["mouth_right"] = b, a
        if "mouth_left" in confs and "mouth_right" in confs:
            confs["mouth_left"], confs["mouth_right"] = confs["mouth_right"], confs["mouth_left"]


def off_body(kps: dict, frame: dict, small, scale: float, device: str) -> dict:
    """Landmarks lying off the fish, with why: the fish is SAM's mask in the frame.

    The box is the landmarks' own extent, padded; if SAM's answer covers almost
    none of it or almost all of it, it found something other than the fish and
    nothing is judged on it.
    """
    import cv2
    import numpy as np

    xs = [p[0] for p in kps.values()]
    ys = [p[1] for p in kps.values()]
    length = frame["scale"] and lu.FRAMING["length_px"] / frame["scale"]
    pad = 0.08 * length
    box = [(min(xs) - pad) * scale, (min(ys) - pad) * scale,
           (max(xs) + pad) * scale, (max(ys) + pad) * scale]
    h, w = small.shape[:2]
    box = [max(0, box[0]), max(0, box[1]), min(w - 1, box[2]), min(h - 1, box[3])]
    mask = _Sam.fish_mask(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), box, device)
    x0, y0, x1, y1 = (int(v) for v in box)
    cover = float(mask[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0
    if not 0.12 <= cover <= 0.92:
        return {}
    r = max(1, int(round(ON_BODY_MARGIN * length * scale)))
    grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    out = {}
    for name, (x, y) in kps.items():
        u, v = int(round(x * scale)), int(round(y * scale))
        # Beyond the edge of what was looked at is out of view, not off the fish:
        # a snout past the mirror's edge is still a snout.
        if 0 <= u < w and 0 <= v < h and not grown[v, u]:
            out[name] = "not on the fish: it lies outside the specimen's outline"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="predict_worker")
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--frontal-project", type=Path, default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-confidence", type=float, default=0.6)
    args = ap.parse_args(argv)

    # Imported here, not at module scope: the ready/error handshake below should
    # report an import failure rather than the process dying before it speaks.
    try:
        import cv2
        import numpy as np
        from deeplabcut.pose_estimation_pytorch import apis

        import predict_landmarks as pl
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    try:
        models = {"lateral": _Model(args.project, args.snapshot, args.scale, "lateral")}
    except SystemExit as exc:                  # find_project and friends raise it
        missing = "no DLC project" in str(exc)
        _emit({"ready": False, "missing_model": missing,
               "error": NO_MODEL.format(view="lateral") if missing else str(exc)})
        return 1
    lat = models["lateral"]
    _emit({"ready": True, "model": lat.label, "scale": lat.scale,
           "landmarks": lat.names, "device": args.device, "union": lat.union})

    def model_for(view: str) -> _Model:
        if view not in models:
            if view != "frontal":
                raise ValueError(f"no model for the {view!r} view")
            try:
                models[view] = _Model(args.frontal_project, None, None, "frontal")
            except SystemExit as exc:          # not an Exception; must not end the worker
                if "no DLC project" in str(exc):
                    raise MissingModel(NO_MODEL.format(view="frontal")) from None
                raise RuntimeError(str(exc)) from None
        return models[view]

    tmp = Path(tempfile.mkdtemp(prefix="calipr_worker_"))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        try:
            req = json.loads(line)
            if req.get("detect"):
                # Where each fish lies in a photograph of several. No landmarks:
                # those come from a request per fish, on its own rectangle.
                src = Path(req["image"])
                im = cv2.imread(str(src))
                if im is None:
                    raise ValueError(f"unreadable image: {src.name}")
                from fish_morpho.batch_fish import find_fish
                _emit({"ok": True, "image": src.name, "fish": find_fish(im),
                       "width": im.shape[1], "height": im.shape[0],
                       "elapsed": round(time.time() - t0, 2)})
                continue
            view = req.get("view") or "lateral"
            m = model_for(view)
            src = Path(req["image"])
            if not src.is_file():
                raise FileNotFoundError(src)

            im = cv2.imread(str(src))
            if im is None:
                raise ValueError(f"unreadable image: {src.name}")
            # A study that keeps one photograph per fish sends the rectangle this
            # view lives in. Everything below works in that rectangle; the offset
            # goes back on at the end, so what comes out names pixels of the frame.
            box = req.get("crop")
            ox = oy = 0
            photo = im
            if box:
                x0, y0, x1, y1 = (int(v) for v in box)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(im.shape[1], x1), min(im.shape[0], y1)
                if x1 - x0 < 10 or y1 - y0 < 10:
                    raise ValueError(f"crop {box} is not inside the photograph")
                im, ox, oy = im[y0:y1, x0:x1], x0, y0
            frame = None
            misplaced = {}
            rename = None
            if m.union:
                # Pass 1: the whole photograph, to find the fish. Pass 2: the fish
                # framed at a standard length, to place the points.
                h, w = im.shape[:2]
                s1 = lu.WHOLE_SIDE / max(h, w)
                whole = cv2.resize(im, (round(w * s1), round(h * s1)),
                                   interpolation=cv2.INTER_AREA)
                first = m.infer(whole, args.device)
                first[:, :2] /= s1
                frames = candidate_frames(m.names, first, m.split.get("template"), (w, h))
                if not frames:
                    raise ValueError("no fish was found in this photograph")
                # The fish is cut from the whole photograph, not the view's
                # rectangle: a fish lying against the mirror has its snout on the
                # far side of the rectangle's edge, and the second pass has to see
                # it to place the points on it.
                best = None
                for cand in frames:
                    cand = dict(cand, x0=cand["x0"] + ox, y0=cand["y0"] + oy)
                    crop_img, _ = lu.cut_frame(photo, cand)
                    out = m.infer(crop_img, args.device)
                    sure = float(np.sort(out[:, 2])[::-1][:FRAME_SURE_OF].mean())
                    if best is None or sure > best[0]:
                        best = (sure, cand, out)
                    if sure >= FRAME_SURE:
                        break
                _, frame, second = best
                ukps, uconf = {}, {}
                # From here on the view's rectangle is widened to hold the whole
                # fish, so the outline and the checks below see what the second
                # pass saw -- an outline cut at the rectangle's edge puts the head
                # in front of the snout, and the anatomy check then drops it.
                H, W = photo.shape[:2]
                fx0, fy0 = int(frame["x0"]), int(frame["y0"])
                fx1, fy1 = int(frame["x0"] + frame["w"]), int(frame["y0"] + frame["h"])
                nx0, ny0 = max(0, min(ox, fx0)), max(0, min(oy, fy0))
                nx1 = min(W, max(ox + im.shape[1], fx1))
                ny1 = min(H, max(oy + im.shape[0], fy1))
                im, ox, oy = photo[ny0:ny1, nx0:nx1], nx0, ny0
                h, w = im.shape[:2]
                for name, (u, v, c) in zip(m.names, second):
                    x, y = lu.from_crop(frame, float(u), float(v))
                    x, y = x - ox, y - oy
                    ukps[name] = [round(x, 1), round(y, 1)]
                    uconf[name] = round(float(c), 3)
                # Only what this study collects. Checked under the shared names --
                # caliPr's own wherever the schemes agree, so the outline and the
                # anatomy checks below know them -- and renamed to the study's
                # names at the end.
                wanted = req.get("landmarks") or CALIPR_LATERAL
                rename = {lu.union_name(n): n for n in wanted if lu.union_name(n) in ukps}
                kps = {u: ukps[u] for u in rename}
                confs = {u: uconf[u] for u in rename}
                misplaced = {u: why for u, why in
                             shape_outliers(ukps, uconf, m.split.get("template")).items()
                             if u in rename}
                # The outline and the checks below work on a modest copy.
                scale = min(1.0, 1500.0 / max(h, w))
                small = cv2.resize(im, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            else:
                scale = m.scale
                small = cv2.resize(im, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
                # One image per call, in its own directory: the folder API is what
                # is available, and reusing a name keeps the temp dir from growing.
                for old in tmp.iterdir():
                    old.unlink()
                cv2.imwrite(str(tmp / "frame.png"), small)

                res = apis.analyze_image_folder(
                    model_cfg=str(m.cfg), images=str(tmp),
                    snapshot_path=str(m.snapshot), device=args.device,
                    progress_bar=False)
                arr = np.asarray(next(iter(res.values()))["bodyparts"]).reshape(-1, 3)

                kps, confs = {}, {}
                for name, (x, y, c) in zip(m.names, arr):
                    kps[name] = [round(float(x) / scale, 1), round(float(y) / scale, 1)]
                    confs[name] = round(float(c), 3)

            if view == "frontal":
                _mouth_corners_in_image_order(kps, confs)
                low = [n for n, c in confs.items() if c < args.min_confidence]
                kps = {n: [x + ox, y + oy] for n, (x, y) in kps.items()}
                _emit({"ok": True, "fish_id": pl.stem_of(src), "image": src.name,
                       "view": view, "keypoints": kps, "confidence": confs,
                       "low_confidence": sorted(low), "polygons": {},
                       "implausible": {}, "frame_warning": None,
                       "model": m.label,
                       "elapsed": round(time.time() - t0, 2)})
                continue
            low = [n for n, c in confs.items() if c < args.min_confidence]

            polys = {}
            if req.get("polygons"):
                # A study on BGNN's scheme marks the lower jaw, not the upper.
                names = BODY_PROMPT if "premaxilla_tip" in kps else ("lower_jaw_tip", *BODY_PROMPT)
                prompt = [[kps[n][0] * scale, kps[n][1] * scale]
                          for n in names if n in kps]
                # Fewer than three prompt points spread along the animal and SAM
                # returns a part rather than the fish; better to return no
                # outline than a plausible wrong one.
                if len(prompt) >= 3:
                    try:
                        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        poly = _Sam.body_outline(rgb, prompt, args.device, scale)
                        poly = smooth_body(poly, kps)
                        if len(poly) >= 3:
                            polys[BODY] = poly
                    except Exception as exc:
                        polys = {}
                        print(f"SAM failed: {exc}", file=sys.stderr)

            # Anatomy has the last word. A landmark outside the range every
            # labelled fish occupies is wrong however confident the heatmap was,
            # and dropping it is honest where placing it is not: ASN_42 returned a
            # dorsal base half a fin out of position at 0.914. Needs the outline
            # for its axis, so a keypoints-only request is simply not checked.
            bad = plausibility.check(kps, polys.get(BODY), _bands_for(src))
            for name, why in misplaced.items():
                bad.setdefault(name, why)
            # Both are conditions of the whole prediction, not of one landmark:
            # nothing is dropped, the labeller is told why nothing was checked.
            frame_warning = (bad.pop("_frame", None) or bad.pop("_axis", None)
                             or bad.pop("_anchor", None))
            for name in bad:
                # Keep what the model thought of a point it does not get to
                # place: "dropped, and it was 0.91 sure" and "dropped, and it
                # knew" are different facts about the model.
                bad[name] = {"why": bad[name], "confidence": confs.get(name)}
                kps.pop(name, None)
                confs.pop(name, None)
            low = [n for n in low if n not in bad]

            # A point has to be on the fish. The model's worst failures were points
            # on the board, a pin or a number card, which no anatomy band catches
            # when the band is wide; the fish's own outline does.
            if frame is not None and kps:
                try:
                    off = off_body(kps, frame, small, scale, args.device)
                except Exception as exc:
                    off = {}
                    print(f"on-body check failed: {exc}", file=sys.stderr)
                for name in off:
                    bad[name] = {"why": off[name], "confidence": confs.get(name)}
                    kps.pop(name, None)
                    confs.pop(name, None)
                low = [n for n in low if n not in bad]

            # The outline may be wanted only as the axis the check above needs.
            # It is computed either way; this decides whether it comes back as
            # something the labeller is offered and can save.
            if not req.get("emit_polygons", True):
                polys = {}

            # Fins last, and only from landmarks that survived the anatomy check:
            # a fin framed on a base the check just threw out would be an outline
            # drawn around nothing, and it would look like the others.
            for fin in req.get("fins") or ():
                if fin not in OUTLINED_FINS:
                    continue
                try:
                    poly = _Fin.outline(im, kps, fin, args.device)
                except Exception as exc:
                    poly = None
                    print(f"fin outliner failed on {fin}: {exc}", file=sys.stderr)
                if poly:
                    # crop coordinates, like everything else here: the offset
                    # back to the photograph goes on once, below.
                    polys[fin] = poly

            if rename:
                kps = {rename[n]: p for n, p in kps.items()}
                confs = {rename[n]: c for n, c in confs.items()}
                low = [rename[n] for n in low]
                bad = {rename.get(n, n): v for n, v in bad.items()}
            kps = {n: [x + ox, y + oy] for n, (x, y) in kps.items()}
            polys = {n: [[x + ox, y + oy] for x, y in poly] for n, poly in polys.items()}
            _emit({"ok": True, "fish_id": pl.stem_of(src), "image": src.name,
                   "keypoints": kps, "confidence": confs,
                   "low_confidence": sorted(low), "polygons": polys,
                   "implausible": bad, "frame_warning": frame_warning,
                   "view": view, "model": m.label,
                   "elapsed": round(time.time() - t0, 2)})
        except MissingModel as exc:
            _emit({"ok": False, "missing_model": True, "error": str(exc)})
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
