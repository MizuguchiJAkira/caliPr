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
from urllib.parse import parse_qs, unquote, urlparse

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from fish_morpho.landmark_config import (  # noqa: E402
    CALIBRATION_KEYPOINTS,
    FIN_KEYPOINTS,
    FIN_POLYGON_TARGET_VERTICES,
    FIN_POLYGONS,
    KEYPOINTS,
    POLYGONS,
    View,
)

UI_DIR = _ROOT / "scripts" / "labeling_ui"

#: Image suffixes the labeler will list, matched case-insensitively.
#:
#: The CUMV rig writes `.JPEG`; other collections write `.jpg`, and a glob of
#: "*.JP*G" silently matches neither on a case-sensitive comparison. Silently is
#: the problem -- the specimen list just comes back empty.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def list_images(directory: Path) -> dict[str, Path]:
    """Image files in ``directory``, keyed by filename. Empty if it is missing."""
    if not directory.is_dir():
        return {}
    return {
        p.name: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }

def _heldout_ids() -> set[str]:
    """The DLC held-out specimens, if a split has been written.

    These are worth re-tracing first: they are stratified across strains, and
    because they are excluded from training, better labels there sharpen the
    held-out evaluation immediately without a retrain.
    """
    try:
        return set(json.loads((_ROOT / "dlc/split.json").read_text())["test"])
    except Exception:
        return set()


HELDOUT = _heldout_ids()


def discover_datasets(root: Path) -> dict[str, Path]:
    """Directories under ``root`` that look like a dataset: they hold a lateral/.

    Switching study in the UI beats restarting the server with different flags,
    and the folder name is the only label that is already unambiguous to whoever
    arranged the photographs.
    """
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "lateral").is_dir():
            out[d.name] = d
    return out

#: How many specimens to mark as the suggested labelling subset.
SUGGESTED_N = 40

# Case-insensitive: the catalogue prefix is typed by hand and appears as CUMV,
# CUMVFish and CUmv across the series.
_LOT_RE = re.compile(r"(?:CUMV[A-Za-z]*_(\d+))|_([A-Z]{2,4})_\d+$", re.IGNORECASE)


def _lot_of(fish_id: str) -> str:
    """Grouping key for stratification: a CUMV lot number, or a strain code."""
    m = _LOT_RE.search(fish_id)
    if not m:
        return ""
    return m.group(1) or m.group(2) or ""


def suggested_subset(ids: list[str], n: int = SUGGESTED_N) -> set[str]:
    """Pick ``n`` specimens spread round-robin across lots.

    Labelling the first n filenames alphabetically concentrates the sample in a
    handful of lots, which for a between-population comparison is close to
    worthless -- lot is confounded with locality and collection date. Taking one
    per lot in rotation spreads the same effort across every lot present, which
    is both a better sample and better training variety.
    """
    by_lot: dict[str, list[str]] = {}
    for i in sorted(ids):
        by_lot.setdefault(_lot_of(i), []).append(i)
    order: list[str] = []
    depth = 0
    while len(order) < n:
        added = False
        for lot in sorted(by_lot):
            if depth < len(by_lot[lot]):
                order.append(by_lot[lot][depth])
                added = True
                if len(order) >= n:
                    break
        if not added:
            break
        depth += 1
    return set(order)
_ID_RE = re.compile(r"^(.*)_[LF]$")


def load_profile(images_dir: Path) -> dict:
    """Per-dataset narrowing of the master schema, from ``<images_dir>/schema.json``.

    A second study on a different taxon rarely wants every structure. The alewife
    series drops the pelvic and anal fins -- the pelvic has almost no contrast
    against the flank, and the anal frays badly -- but the brook trout study still
    needs them. Editing landmark_config would break the study that is already
    validated against it, so a dataset narrows the schema instead of redefining it.

    The profile can only REMOVE. Anything a dataset adds would be absent from the
    measurement engine and could not be computed, so a typo here weakens the task
    list rather than silently inventing a landmark.
    """
    path = images_dir / "schema.json"
    if not path.is_file():
        return {}
    try:
        prof = json.loads(path.read_text())
    except Exception as exc:                       # a broken profile must be loud
        print(f"  WARNING: could not read {path}: {exc}")
        return {}
    return {
        "exclude_polygons": set(prof.get("exclude_polygons") or []),
        "exclude_keypoints": set(prof.get("exclude_keypoints") or []),
        "note": prof.get("note", ""),
    }


def build_schema(profile: dict | None = None) -> dict:
    """Emit the per-view labeling contract straight from landmark_config."""

    profile = profile or {}
    drop_poly = profile.get("exclude_polygons") or set()
    drop_kp = profile.get("exclude_keypoints") or set()

    # Excluded landmarks are MARKED, not omitted: the labeler has to be able to
    # show them struck through and let someone put one back. Consumers that read
    # schema.json directly (the standalone build, the TPS export) still filter.
    def kp(items, view):
        return [
            {"name": k.name, "description": k.description, "hint": k.labeling_hint,
             "excluded": k.name in drop_kp}
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
                "excluded": p.name in drop_poly,
            }
            for p in POLYGONS
            if p.view == view
        ]

    # Fin-retrace grouping: one fin's base keypoint, tip keypoint, and outline
    # travel together. The labeler drives its retrace mode off this rather than
    # hardcoding fin names, so adding a fin here is the only change needed.
    kp_by_name = {k.name: k for k in KEYPOINTS}

    def fin_groups():
        groups = []
        for name in FIN_POLYGONS:
            if name in drop_poly:
                continue
            base, tip = FIN_KEYPOINTS[name]
            groups.append({
                "fin": name,
                "polygon": name,
                "target": FIN_POLYGON_TARGET_VERTICES,
                "keypoints": [
                    {"name": n, "role": role,
                     "description": kp_by_name[n].description,
                     "hint": kp_by_name[n].labeling_hint}
                    for n, role in ((base, "base"), (tip, "tip"))
                    if n in kp_by_name
                ],
            })
        return groups

    return {
        "profile_note": profile.get("note", ""),
        "fin_groups": fin_groups(),
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
    datasets: dict[str, Path] = {}
    default_dataset: str = ""
    images_dir: Path          # resolved per request from ?dataset=
    out_dir: Path

    def _use(self, query: str) -> None:
        """Point this request at the dataset named in the query string."""
        name = parse_qs(query).get("dataset", [Handler.default_dataset])[0]
        base = Handler.datasets.get(name)
        if base is None:
            base = Handler.datasets.get(Handler.default_dataset)
        if base is not None:
            self.images_dir = base
            self.out_dir = base / "sidecars"

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
        lat = list_images(self.images_dir / "lateral")
        fro = list_images(self.images_dir / "frontal")
        all_ids = []
        for name, path in sorted(lat.items()):
            m0 = _ID_RE.match(Path(name).stem)
            all_ids.append(m0.group(1) if m0 else Path(name).stem)
        suggested = suggested_subset(all_ids)

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
                    # A fin is re-done only when its outline is dense enough AND
                    # its base and tip are placed. The three describe one
                    # structure and the traits mix them (PFl is base->tip, PFs is
                    # the polygon), so a dense outline with a stale tip is not
                    # finished work. Must match finState() in the labeler.
                    kps = block.get("keypoints") or {}
                    traced = [n for n in FIN_POLYGONS if polys.get(n)]
                    fins_done = bool(traced) and all(
                        len(polys[n]) >= FIN_POLYGON_TARGET_VERTICES
                        and all(k in kps for k in FIN_KEYPOINTS[n])
                        for n in traced
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
                # Epoch seconds of the committed sidecar. The labeler compares
                # this against its localStorage draft's timestamp: a draft that
                # predates the file on disk is stale and must not shadow it, or
                # an out-of-band edit (a bulk keypoint wipe, a hand fix, a pull)
                # silently reappears as the old values.
                "mtime": sidecar.stat().st_mtime if sidecar.is_file() else 0,
                "lateral_done": lat_done,
                "frontal_done": fro_done,
                "fins_done": fins_done,
                "heldout": fid in HELDOUT,
                "suggested": fid in suggested,
            })
        return out

    def _calib_stats(self, lot: str = ""):
        """Median px/mm across saved sidecars, per collection lot where possible.

        A mistyped ``known_mm`` (e.g. 10 for a 50 mm span) silently scales every
        trait for that specimen, and nothing downstream can tell — the geometry is
        self-consistent, just wrong. Comparing against a median is the cheapest way
        to catch it while the annotator is still on the fish.

        WHICH median matters. The Cornell rig holds one camera distance, so a
        whole-batch median works there. The alewife tank series does not: distance
        varies BETWEEN collection lots, from about 25 px/mm to 44, so a batch
        median flags 82 of 181 specimens as outliers when nothing is wrong with
        them. Comparing within the lot is what makes the check mean something —
        that is where a genuine mis-scale actually stands out.
        """
        vals, lot_vals = [], []
        for p in self.out_dir.glob("*.json"):
            try:
                cal = (json.loads(p.read_text()).get("lateral") or {}).get("calibration")
                if not cal or cal.get("mode") != "manual":
                    continue
                (ax, ay), (bx, by) = cal["point_a"], cal["point_b"]
                known = float(cal["known_mm"])
                if known > 0:
                    v = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / known
                    vals.append(v)
                    if lot and _lot_of(p.stem) == lot:
                        lot_vals.append(v)
            except Exception:
                continue
        # Prefer the lot, but only once it has enough specimens to have a median
        # worth trusting; otherwise fall back to the batch.
        if len(lot_vals) >= 3:
            vals = lot_vals
        vals.sort()
        median = vals[len(vals) // 2] if vals else None
        return {"median_px_per_mm": median, "n": len(vals)}

    def _set_exclusions(self):
        """Record which landmarks and outlines this study does not collect.

        Written into the dataset's own schema.json, because it is a property of
        the study rather than of a session or a machine: it has to reach the
        measurement engine, which decides that a trait needing an uncollected
        landmark gets no column at all rather than a column of blanks.
        """
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n))
        except Exception as exc:
            return self._send(400, {"error": f"bad payload: {exc}"})

        known_kp = {k.name for k in KEYPOINTS}
        known_poly = {p.name for p in POLYGONS}
        kps = [k for k in (data.get("exclude_keypoints") or []) if k in known_kp]
        polys = [p for p in (data.get("exclude_polygons") or []) if p in known_poly]

        path = self.images_dir / "schema.json"
        prof = {}
        if path.is_file():
            try:
                prof = json.loads(path.read_text())
            except Exception:
                prof = {}
        prof["exclude_keypoints"] = sorted(kps)
        prof["exclude_polygons"] = sorted(polys)
        if "note" in data:
            prof["note"] = str(data["note"])
        path.write_text(json.dumps(prof, indent=2) + "\n")

        from fish_morpho.landmark_config import traits_requiring
        return self._send(200, {
            "ok": True,
            "exclude_keypoints": prof["exclude_keypoints"],
            "exclude_polygons": prof["exclude_polygons"],
            "dropped_traits": sorted(traits_requiring(kps, polys)),
        })

    def _export(self, kind: str):
        """Build an export on demand and hand it back as a download.

        Run in-process rather than shelled out so a failure surfaces as a message
        the annotator can read, instead of a non-zero exit code in a terminal they
        are not looking at.
        """
        import io
        import zipfile
        import subprocess

        ds = self.images_dir.name
        root = _ROOT / "results" / ds
        try:
            if kind == "measurements":
                r = subprocess.run(
                    [sys.executable, str(_ROOT / "scripts/export_measurements.py"),
                     "--dataset", ds],
                    capture_output=True, text=True, timeout=600, cwd=_ROOT)
                out = root / "measurements.xlsx"
                if r.returncode != 0 or not out.is_file():
                    return self._send(500, {"error": (r.stderr or r.stdout)[-800:]})
                return self._send_file(out, f"{ds}_measurements.xlsx")

            if kind == "tps":
                r = subprocess.run(
                    [sys.executable, str(_ROOT / "scripts/export_tps.py"),
                     "--sidecars", str(self.out_dir),
                     "--images", str(self.images_dir / "lateral"),
                     "--schema-dir", str(self.images_dir),
                     "--out", str(root / "tps")],
                    capture_output=True, text=True, timeout=600, cwd=_ROOT)
                if r.returncode != 0:
                    return self._send(500, {"error": (r.stderr or r.stdout)[-800:]})
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in sorted((root / "tps").iterdir()):
                        if f.is_file():
                            z.write(f, f.name)
                return self._send_bytes(buf.getvalue(), f"{ds}_tps.zip",
                                        "application/zip")

            if kind == "overlays":
                r = subprocess.run(
                    [sys.executable, str(_ROOT / "scripts/render_overlays.py"),
                     "--dataset", ds],
                    capture_output=True, text=True, timeout=1800, cwd=_ROOT)
                if r.returncode != 0:
                    return self._send(500, {"error": (r.stderr or r.stdout)[-800:]})
                folder = root / "overlays"
                imgs = sorted(folder.glob("*.jpg")) if folder.is_dir() else []
                if not imgs:
                    return self._send(500, {"error": "nothing annotated yet"})
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in imgs:
                        z.write(f, f.name)
                return self._send_bytes(buf.getvalue(), f"{ds}_overlays.zip",
                                        "application/zip")
        except subprocess.TimeoutExpired:
            return self._send(500, {"error": "export timed out"})
        except Exception as exc:
            return self._send(500, {"error": str(exc)})
        return self._send(404, {"error": f"unknown export {kind!r}"})

    def _send_file(self, path: Path, filename: str):
        self._send_bytes(path.read_bytes(), filename,
                         "application/vnd.openxmlformats-officedocument."
                         "spreadsheetml.sheet")

    def _send_bytes(self, data: bytes, filename: str, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        route = urlparse(self.path).path
        self._use(urlparse(self.path).query)

        if route == "/api/datasets":
            names = sorted(Handler.datasets)
            return self._send(200, {
                "datasets": [
                    {"name": n,
                     "images": len(list_images(Handler.datasets[n] / "lateral")),
                     "has_frontal": (Handler.datasets[n] / "frontal").is_dir(),
                     # A dataset whose profile drops every fin polygon can never
                     # satisfy the fin-density badge, so the UI should not show it.
                     "has_fin_polygons": bool(
                         set(FIN_POLYGONS)
                         - set((load_profile(Handler.datasets[n]).get("exclude_polygons")
                                or set()))),
                     }
                    for n in names
                ],
                "default": Handler.default_dataset,
            })

        if route == "/" or route == "/index.html":
            html = (UI_DIR / "index.html").read_text()
            return self._send(200, html, "text/html; charset=utf-8")
        if route == "/api/schema":
            return self._send(200, build_schema(load_profile(self.images_dir)))
        if route == "/api/specimens":
            return self._send(200, self._specimens())
        if route == "/api/calibstats":
            return self._send(
                200, self._calib_stats(parse_qs(urlparse(self.path).query)
                                       .get("lot", [""])[0]))
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
        if route.startswith("/api/export/"):
            return self._export(route[len("/api/export/"):])

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
        self._use(urlparse(self.path).query)

        if route == "/api/schema/exclude":
            return self._set_exclusions()

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
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data",
                    help="Directory whose subfolders are datasets (each holding a "
                         "lateral/). Offered in the UI's dataset dropdown.")
    ap.add_argument("--dataset", default="",
                    help="Which one to open first. Defaults to the first found.")
    ap.add_argument("--images", type=Path, default=None,
                    help="Single-dataset mode, bypassing discovery.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.images:                       # explicit single dataset
        base = args.images.resolve()
        Handler.datasets = {base.name: base}
        Handler.default_dataset = base.name
        if args.out:
            Handler.datasets[base.name] = base
    else:
        Handler.datasets = discover_datasets(args.data_root.resolve())
        if not Handler.datasets:
            print(f"No datasets under {args.data_root} (need a lateral/ subfolder)")
            return 1
        Handler.default_dataset = (args.dataset if args.dataset in Handler.datasets
                                   else sorted(Handler.datasets)[0])
    base = Handler.datasets[Handler.default_dataset]
    Handler.images_dir = base
    Handler.out_dir = (args.out.resolve() if args.out else base / "sidecars")
    Handler.out_dir.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Labeling server: http://localhost:{args.port}/")
    for n in sorted(Handler.datasets):
        mark = "*" if n == Handler.default_dataset else " "
        print(f"  {mark} {n:12} {len(list_images(Handler.datasets[n] / 'lateral')):4d} images")

    # The specimen list is built by globbing the image directories, and a missing
    # or empty one globs to nothing -- so the UI would open to a blank list with
    # no clue why. The photographs are not in the repository (they are large, and
    # they are the museum's), so this is the normal state of a fresh clone.
    n_lat = len(list_images(Handler.images_dir / "lateral"))
    if n_lat == 0:
        print()
        print(f"  WARNING: no lateral images under {Handler.images_dir}/lateral —")
        print("  the specimen list will be empty. Photographs are not tracked in")
        print("  git; produce the crops first, e.g.")
        print("      python scripts/preprocess_jonah.py --raw-dir <raw photos> \\")
        print(f"          --out-dir {Handler.images_dir} --lateral-margin 450")
        print(f"  or point elsewhere with --images. "
              f"{len(list(Handler.out_dir.glob('*.json')))} sidecars are present.")
        print()
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
