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

    {"image": "/abs/path/to/fish.JPEG", "polygons": true}
    {"ok": true, "fish_id": "...", "keypoints": {...}, "confidence": {...},
     "low_confidence": [...], "polygons": {"body_plus_caudal": [[x, y], ...]},
     "elapsed": 0.21}

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


def _emit(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="predict_worker")
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-confidence", type=float, default=0.6)
    args = ap.parse_args(argv)

    # Imported here, not at module scope: the ready/error handshake below should
    # report an import failure rather than the process dying before it speaks.
    try:
        import cv2
        import numpy as np
        import ruamel.yaml
        from deeplabcut.pose_estimation_pytorch import apis

        import predict_landmarks as pl

        project = pl.find_project(args.project)
        cfg, snapshot = pl.find_config_and_snapshot(project, args.snapshot)
        scale = pl.training_scale(project, args.scale)
        yaml = ruamel.yaml.YAML()
        with open(cfg) as fh:
            conf = yaml.load(fh)
        names = list(conf["metadata"]["bodyparts"])
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    _emit({"ready": True, "model": snapshot.name, "scale": scale,
           "landmarks": names, "device": args.device})

    tmp = Path(tempfile.mkdtemp(prefix="calipr_worker_"))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        try:
            req = json.loads(line)
            src = Path(req["image"])
            if not src.is_file():
                raise FileNotFoundError(src)

            im = cv2.imread(str(src))
            if im is None:
                raise ValueError(f"unreadable image: {src.name}")
            small = cv2.resize(im, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            # One image per call, in its own directory: the folder API is what is
            # available, and reusing a name keeps the temp dir from growing.
            for old in tmp.iterdir():
                old.unlink()
            cv2.imwrite(str(tmp / "frame.png"), small)

            res = apis.analyze_image_folder(
                model_cfg=str(cfg), images=str(tmp),
                snapshot_path=str(snapshot), device=args.device,
                progress_bar=False)
            arr = np.asarray(next(iter(res.values()))["bodyparts"]).reshape(-1, 3)

            kps, confs, low = {}, {}, []
            for name, (x, y, c) in zip(names, arr):
                kps[name] = [round(float(x) / scale, 1), round(float(y) / scale, 1)]
                confs[name] = round(float(c), 3)
                if c < args.min_confidence:
                    low.append(name)

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
                        if len(poly) >= 3:
                            polys[BODY] = poly
                    except Exception as exc:
                        polys = {}
                        print(f"SAM failed: {exc}", file=sys.stderr)

            _emit({"ok": True, "fish_id": pl.stem_of(src), "image": src.name,
                   "keypoints": kps, "confidence": confs,
                   "low_confidence": sorted(low), "polygons": polys,
                   "model": snapshot.name,
                   "elapsed": round(time.time() - t0, 2)})
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
