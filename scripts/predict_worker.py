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

    {"image": "/abs/path/to/fish.JPEG"}
    {"ok": true, "fish_id": "...", "keypoints": {...}, "confidence": {...},
     "low_confidence": [...], "elapsed": 0.21}

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

            _emit({"ok": True, "fish_id": pl.stem_of(src), "image": src.name,
                   "keypoints": kps, "confidence": confs,
                   "low_confidence": sorted(low), "model": snapshot.name,
                   "elapsed": round(time.time() - t0, 2)})
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
