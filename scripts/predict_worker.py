"""Long-lived prediction process, so the labeler's Auto-label is not 7s a click.

Loading DeepLabCut and a snapshot costs ~7 seconds; the inference itself costs a
fraction of one. Running ``predict_landmarks.py`` per request pays that every
time. This loads once and then answers on stdin/stdout, so only the first request
in a session waits.

It also solves an environment problem. The labeler runs in the plain ``.venv``,
which has no torch and no DeepLabCut; the training stack lives in
``.venv-train``. Rather than merge them, the server launches this module with the
training interpreter and talks to it over pipes.

Protocol — one JSON object per line, in and out::

    {"image": "/abs/path/to/fish.JPEG", "polygons": true, "view": "lateral"}
    {"ok": true, "fish_id": "...", "keypoints": {...}, "confidence": {...},
     "low_confidence": [...], "polygons": {"body_plus_caudal": [[x, y], ...]},
     "implausible": {"pelvic_tip": "why it cannot be there"},
     "frame_warning": null, "elapsed": 0.21}

``implausible`` names landmarks the model placed somewhere a fish cannot have
them, checked against the dataset's own measured bands (see
:mod:`fish_morpho.plausibility`). Those are absent from ``keypoints`` — a point
the anatomy rules out is not offered, because a labeller can accept a flagged
point but cannot un-see a confident one in the wrong place.

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

from fish_morpho import plausibility  # noqa: E402  (needs the path above)

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

    a, b = kps.get("premaxilla_tip"), kps.get("caudal_base")
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


#: Plausibility bands, keyed by dataset directory. Read once per dataset: the
#: worker is long-lived and the file only changes when someone refits it.
_BANDS_CACHE: dict[str, dict | None] = {}


def _bands_for(image: Path) -> dict | None:
    """The dataset's bands for the image being predicted, if it has been fitted.

    Images live at ``<dataset>/lateral/<stem>_L.JPEG``, so the dataset directory
    is two levels up. A dataset nobody has fitted returns None and is not checked
    — the right default for a taxon whose landmarks sit nowhere near a trout's.
    """
    key = str(image.parent.parent)
    if key not in _BANDS_CACHE:
        _BANDS_CACHE[key] = plausibility.load(image.parent.parent)
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
        self.scale = pl.training_scale(self.project, scale)
        with open(self.cfg) as fh:
            self.names = list(ruamel.yaml.YAML().load(fh)["metadata"]["bodyparts"])


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
    _emit({"ready": True, "model": lat.snapshot.name, "scale": lat.scale,
           "landmarks": lat.names, "device": args.device})

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
            view = req.get("view") or "lateral"
            m = model_for(view)
            src = Path(req["image"])
            if not src.is_file():
                raise FileNotFoundError(src)

            im = cv2.imread(str(src))
            if im is None:
                raise ValueError(f"unreadable image: {src.name}")
            scale = m.scale
            small = cv2.resize(im, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            # One image per call, in its own directory: the folder API is what is
            # available, and reusing a name keeps the temp dir from growing.
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
                _emit({"ok": True, "fish_id": pl.stem_of(src), "image": src.name,
                       "view": view, "keypoints": kps, "confidence": confs,
                       "low_confidence": sorted(low), "polygons": {},
                       "implausible": {}, "frame_warning": None,
                       "model": m.snapshot.name,
                       "elapsed": round(time.time() - t0, 2)})
                continue
            low = [n for n, c in confs.items() if c < args.min_confidence]

            polys = {}
            if req.get("polygons"):
                prompt = [[kps[n][0] * scale, kps[n][1] * scale]
                          for n in BODY_PROMPT if n in kps]
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
            # Both are conditions of the whole prediction, not of one landmark:
            # nothing is dropped, the labeller is told why nothing was checked.
            frame_warning = bad.pop("_frame", None) or bad.pop("_axis", None)
            for name in bad:
                # Keep what the model thought of a point it does not get to
                # place: "dropped, and it was 0.91 sure" and "dropped, and it
                # knew" are different facts about the model.
                bad[name] = {"why": bad[name], "confidence": confs.get(name)}
                kps.pop(name, None)
                confs.pop(name, None)
            low = [n for n in low if n not in bad]

            # The outline may be wanted only as the axis the check above needs.
            # It is computed either way; this decides whether it comes back as
            # something the labeller is offered and can save.
            if not req.get("emit_polygons", True):
                polys = {}

            _emit({"ok": True, "fish_id": pl.stem_of(src), "image": src.name,
                   "keypoints": kps, "confidence": confs,
                   "low_confidence": sorted(low), "polygons": polys,
                   "implausible": bad, "frame_warning": frame_warning,
                   "view": view, "model": m.snapshot.name,
                   "elapsed": round(time.time() - t0, 2)})
        except MissingModel as exc:
            _emit({"ok": False, "missing_model": True, "error": str(exc)})
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
