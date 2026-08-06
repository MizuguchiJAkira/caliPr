"""Local web server for hand-labeling lateral/frontal crops into sidecar JSON.

A dependency-free (stdlib-only) HTTP server that pairs the ``data/cornell``
lateral + frontal crops, serves a canvas-based labeling UI
(``scripts/labeling_ui/index.html``), and writes sidecar JSONs that
``fish_morpho.pipeline`` consumes directly. The landmark schema (polygon
names, keypoint names, and per-landmark labeling hints) is pulled live from
:mod:`fish_morpho.landmark_config`, so the UI always matches the measurement
engine's contract.

Run::

    python scripts/label_server.py --port 8765 \\
        --images data/cornell --out data/cornell/sidecars

Then open http://localhost:8765/ (Claude Code preview does this for you).

Endpoints
---------
  GET  /                      → the labeling UI
  GET  /api/schema            → {lateral:{polygons,keypoints,ruler}, frontal:{...}}
  GET  /api/specimens         → [{id, lateral, frontal, labeled}]
  GET  /img/lateral/<name>    → JPEG bytes
  GET  /img/frontal/<name>    → JPEG bytes
  GET  /api/sidecar/<id>      → existing sidecar JSON (404 if none)
  POST /api/save              → body is a full sidecar; writes <id>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from fish_morpho.landmark_config import (  # noqa: E402
    CALIBRATION_KEYPOINTS,
    FIN_POLYGON_TARGET_VERTICES,
    FIN_POLYGONS,
    KEYPOINTS,
    POLYGONS,
    View,
)

UI_DIR = _ROOT / "scripts" / "labeling_ui"
_ID_RE = re.compile(r"^(.*)_[LF]$")


def build_schema() -> dict:
    """Emit the per-view labeling contract straight from landmark_config."""

    def kp(items, view):
        return [
            {"name": k.name, "description": k.description, "hint": k.labeling_hint}
            for k in items
            if k.view == view
        ]

    def poly(view):
        # `target` drives the vertex counter in the labeler. Fin areas read low
        # when the outline is sparse, so the UI has to show progress toward a
        # usable density rather than just "3+ points, done".
        return [
            {
                "name": p.name,
                "description": p.description,
                "hint": p.labeling_hint,
                "target": (
                    FIN_POLYGON_TARGET_VERTICES if p.name in FIN_POLYGONS else 0
                ),
            }
            for p in POLYGONS
            if p.view == view
        ]

    return {
        "lateral": {
            "polygons": poly(View.LATERAL),
            "keypoints": kp(KEYPOINTS, View.LATERAL),
            "ruler": kp(CALIBRATION_KEYPOINTS, View.LATERAL),
        },
        "frontal": {
            "polygons": poly(View.FRONTAL),
            "keypoints": kp(KEYPOINTS, View.FRONTAL),
            # frontal ruler isn't in the schema's CALIBRATION_KEYPOINTS
            # (those are lateral-only), so synthesize a generic pair here.
            "ruler": [
                {"name": "ruler_point_a", "description": "First endpoint of a known "
                 "span on the frontal (mirror) ruler.", "hint": "Click one end of a "
                 "known mm span on the small vertical mirror ruler."},
                {"name": "ruler_point_b", "description": "Second endpoint of the "
                 "frontal ruler span.", "hint": "Click the other end; enter the mm "
                 "distance between the two points."},
            ],
        },
    }


class Handler(BaseHTTPRequestHandler):
    images_dir: Path
    out_dir: Path

    def log_message(self, *args):  # quieter console
        pass

    # -- helpers ----------------------------------------------------------
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _specimens(self):
        lat = {p.name: p for p in (self.images_dir / "lateral").glob("*.JP*G")}
        fro = {p.name: p for p in (self.images_dir / "frontal").glob("*.JP*G")}
        out = []
        for name, path in sorted(lat.items()):
            m = _ID_RE.match(path.stem)
            fid = m.group(1) if m else path.stem
            fname = path.name.replace("_L.", "_F.")
            sidecar = self.out_dir / f"{fid}.json"
            # Report lateral and frontal completion separately — a saved
            # sidecar says nothing about whether mouth width was collected.
            lat_done = fro_done = fins_done = False
            if sidecar.is_file():
                try:
                    data = json.loads(sidecar.read_text())
                    block = data.get("lateral") or {}
                    polys = block.get("polygons") or {}
                    lat_done = bool((block.get("keypoints") or {}) or polys)
                    # Fins are tracked apart from "lateral labeled": a specimen
                    # can be fully landmarked and still have fin outlines too
                    # sparse to give a trustworthy area.
                    traced = [n for n in FIN_POLYGONS if polys.get(n)]
                    fins_done = bool(traced) and all(
                        len(polys[n]) >= FIN_POLYGON_TARGET_VERTICES for n in traced
                    )
                    fkp = ((data.get("frontal") or {}).get("keypoints") or {})
                    fro_done = "mouth_left" in fkp and "mouth_right" in fkp
                except Exception:
                    lat_done = True
            out.append({
                "id": fid,
                "lateral": path.name,
                "frontal": fname if fname in fro else None,
                "labeled": sidecar.is_file(),
                "lateral_done": lat_done,
                "frontal_done": fro_done,
                "fins_done": fins_done,
            })
        return out

    def _calib_stats(self):
        """Median px/mm across saved sidecars.

        A mistyped ``known_mm`` (e.g. 10 for a 50 mm span) silently scales every
        trait for that specimen, and nothing downstream can tell — the geometry
        is self-consistent, just wrong. Comparing against the batch median is
        the cheapest way to catch it while the annotator is still on the fish.
        """
        vals = []
        for p in self.out_dir.glob("*.json"):
            try:
                cal = (json.loads(p.read_text()).get("lateral") or {}).get("calibration")
                if not cal or cal.get("mode") != "manual":
                    continue
                (ax, ay), (bx, by) = cal["point_a"], cal["point_b"]
                known = float(cal["known_mm"])
                if known > 0:
                    vals.append(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / known)
            except Exception:
                continue
        vals.sort()
        median = vals[len(vals) // 2] if vals else None
        return {"median_px_per_mm": median, "n": len(vals)}

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/" or route == "/index.html":
            html = (UI_DIR / "index.html").read_text()
            return self._send(200, html, "text/html; charset=utf-8")
        if route == "/api/schema":
            return self._send(200, build_schema())
        if route == "/api/specimens":
            return self._send(200, self._specimens())
        if route == "/api/calibstats":
            return self._send(200, self._calib_stats())
        if route.startswith("/api/autocal/"):
            name = unquote(route[len("/api/autocal/"):])
            if "/" in name or ".." in name:
                return self._send(400, {"error": "bad path"})
            path = self.images_dir / "lateral" / name
            if not path.is_file():
                return self._send(404, {"error": "image not found"})
            try:
                import cv2

                from fish_morpho.ruler_calibration import detect_tick_scale

                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                res = detect_tick_scale(img)
                return self._send(200, {
                    "px_per_mm": res.px_per_mm,
                    "confidence": res.confidence,
                    "notes": res.notes,
                })
            except Exception as exc:
                return self._send(200, {"error": str(exc)})
        if route.startswith("/api/sidecar/"):
            fid = unquote(route[len("/api/sidecar/"):])
            p = self.out_dir / f"{fid}.json"
            if not p.is_file():
                return self._send(404, {"error": "no sidecar"})
            return self._send(200, p.read_text())
        if route.startswith("/img/"):
            sub = unquote(route[len("/img/"):])  # e.g. lateral/Name_L.JPEG
            if ".." in sub:
                return self._send(400, {"error": "bad path"})
            p = self.images_dir / sub
            if not p.is_file():
                return self._send(404, {"error": "not found"})
            return self._send(200, p.read_bytes(), "image/jpeg")
        if route.startswith("/ui/"):
            name = unquote(route[len("/ui/"):])
            if "/" in name or ".." in name:
                return self._send(400, {"error": "bad path"})
            p = UI_DIR / name
            if not p.is_file():
                return self._send(404, {"error": "not found"})
            ctype = {
                ".json": "application/json",
                ".html": "text/html; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
            }.get(p.suffix, "image/jpeg")
            return self._send(200, p.read_bytes(), ctype)
        return self._send(404, {"error": "unknown route"})

    def do_POST(self):
        route = urlparse(self.path).path
        if route != "/api/save":
            return self._send(404, {"error": "unknown route"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n))
            fid = data["fish_id"]
        except Exception as exc:
            return self._send(400, {"error": f"bad payload: {exc}"})
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", fid)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / f"{safe}.json").write_text(json.dumps(data, indent=2))
        return self._send(200, {"ok": True, "saved": f"{safe}.json"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="label_server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--images", type=Path, default=_ROOT / "data" / "cornell")
    ap.add_argument("--out", type=Path, default=_ROOT / "data" / "cornell" / "sidecars")
    args = ap.parse_args(argv)
    Handler.images_dir = args.images.resolve()
    Handler.out_dir = args.out.resolve()
    Handler.out_dir.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Labeling server: http://localhost:{args.port}/  "
          f"(images={Handler.images_dir}, out={Handler.out_dir})")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
