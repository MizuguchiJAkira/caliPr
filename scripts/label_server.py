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
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from fish_morpho import auth, schemes  # noqa: E402
from fish_morpho import image_identity  # noqa: E402
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

_FP_CACHE: dict[tuple, dict | None] = {}
_FP_LOCK = threading.Lock()


def view_image(images_dir: Path, fid: str, view: str) -> Path | None:
    """The file a fish's labels for ``view`` sit on.

    A study that keeps one photograph per fish answers with that photograph for
    both views: the mirror's head-on view is part of the same frame, so the two
    share an image and a coordinate system.
    """
    folder = images_dir / view
    suffix = "_L" if view == "lateral" else "_F"
    for p in list_images(folder).values():
        if p.stem == f"{fid}{suffix}":
            return p
    if view == "frontal" and load_profile(images_dir).get("single_photo"):
        return view_image(images_dir, fid, "lateral")
    return None


def current_fingerprint(path: Path | None) -> dict | None:
    """Fingerprint of an image, cached on its size and mtime so a busy session does
    not re-hash every photograph on every click."""
    if path is None or not path.is_file():
        return None
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    with _FP_LOCK:
        if key not in _FP_CACHE:
            _FP_CACHE[key] = image_identity.fingerprint(path)
        return _FP_CACHE[key]


_SIZE_CACHE: dict[tuple, list | None] = {}


def current_size(path: Path | None) -> list | None:
    """[width, height] from the image header alone -- no hashing -- for the
    specimen list, which covers every fish on every refresh."""
    if path is None or not path.is_file():
        return None
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _SIZE_CACHE:
        from PIL import Image
        with Image.open(path) as im:
            _SIZE_CACHE[key] = [int(im.size[0]), int(im.size[1])]
    return _SIZE_CACHE[key]


def alignment(images_dir: Path, fid: str, doc: dict | None) -> dict:
    """For each labelled view: do its saved coordinates belong to the image on disk?

    ``size_changed`` is proof they do not -- the image was re-cropped or replaced
    after labelling, and every coordinate is displaced. The labeler must not draw
    or save such labels as though they fit.
    """
    manifest = image_identity.load_manifest(images_dir)
    out = {}
    for view in ("lateral", "frontal"):
        if not image_identity.has_labels(doc, view):
            continue
        cur = current_fingerprint(view_image(images_dir, fid, view))
        rec = image_identity.recorded_for(doc, manifest, fid, view)
        status = image_identity.compare(rec, cur)
        out[view] = {"status": status,
                     "detail": image_identity.describe(status, rec, cur),
                     "recorded": ({k: rec[k] for k in ("width", "height")} if rec else None),
                     "current": ({k: cur[k] for k in ("width", "height")} if cur else None)}
    return out


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

#: A study's own settings: which landmarks it collects and how its strain is read
#: from filenames (schema.json), and the anatomy bands Auto-label checks points
#: against (plausibility.json). A new study of the same fish on the same rig wants
#: both; without them Auto-label offers the outline the study turned off, checks
#: nothing, and the export has no strain column.
STUDY_SETTINGS = ("schema.json", "plausibility.json")

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
        # Narrower than exclude_polygons: the structure stays in the labelling
        # contract and is still traced by hand, but Auto-label does not offer a
        # predicted one. For the body outline that is the difference between
        # "we do not collect this" and "the model is not good enough at it yet".
        "exclude_predicted_polygons": set(
            prof.get("exclude_predicted_polygons") or []),
        # Landmarks this study adds to the master schema. They are coordinates
        # only: every trait is defined in code against a fixed name, so nothing is
        # computed from them -- they are labelled, saved, and exported.
        "extra_keypoints": [k for k in (prof.get("extra_keypoints") or [])
                            if isinstance(k, dict) and k.get("name")],
        # What this study calls a landmark. Renaming changes only what is shown:
        # the stored name is what the traits, the trained model and every sidecar
        # already written refer to.
        "labels": dict(prof.get("labels") or {}),
        # Which landmark scheme this study collects. caliPr's own unless named.
        "scheme": prof.get("scheme") or schemes.DEFAULT,
        # One photograph per fish, holding both views: the lateral fish and, in
        # the rig's mirror, its head. Nothing is cut up, and both sets of
        # landmarks are placed on the same image in the same coordinates.
        "single_photo": bool(prof.get("single_photo")),
        "note": prof.get("note", ""),
    }


def build_schema(profile: dict | None = None) -> dict:
    """Emit the per-view labeling contract straight from landmark_config."""

    profile = profile or {}
    drop_poly = profile.get("exclude_polygons") or set()
    drop_kp = profile.get("exclude_keypoints") or set()
    extra = profile.get("extra_keypoints") or []
    labels = profile.get("labels") or {}
    scheme = profile.get("scheme") or schemes.DEFAULT

    # Excluded landmarks are MARKED, not omitted: the labeler has to be able to
    # show them struck through and let someone put one back. Consumers that read
    # schema.json directly (the standalone build, the TPS export) still filter.
    def kp(items, view, extras=False):
        # A study on another scheme collects that scheme's landmarks instead of
        # caliPr's: the two name different points, and no trait is defined for it.
        if extras and schemes.get(scheme):
            return [dict(k, label=labels.get(k["name"], k["label"]),
                         excluded=k["name"] in drop_kp)
                    for k in schemes.keypoints(scheme, view.value if hasattr(view, "value") else view)]
        out = [
            {"name": k.name, "label": labels.get(k.name, k.name),
             "description": k.description, "hint": k.labeling_hint,
             "excluded": k.name in drop_kp}
            for k in items
            if k.view == view
        ]
        # This study's own landmarks, after the master ones, in the order added.
        # Only among the landmarks: the ruler points are a different kind of task,
        # and a landmark listed as one is placed as a ruler point, not a landmark.
        out += [] if not extras else [
            {"name": k["name"], "label": labels.get(k["name"], k.get("label") or k["name"]),
             "description": k.get("description")
                            or "Added for this study. Saved and exported as a coordinate; "
                               "no trait is computed from it.",
             "hint": k.get("hint") or "", "excluded": k["name"] in drop_kp, "custom": True}
            for k in extra if (k.get("view") or "lateral") == view
        ]
        return out

    def poly(view):
        if schemes.get(scheme):
            return []
        # `target` drives the vertex counter in the labeler. Fin areas read low
        # when the outline is sparse, so the UI has to show progress toward a
        # usable density rather than just "3+ points, done".
        return [
            {
                "name": p.name,
                "label": labels.get(p.name, p.name),
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
        # One photograph per fish: both views are placed on the same image, and
        # the labeler zooms to the head-on view rather than opening a crop.
        "single_photo": bool(profile.get("single_photo")),
        "scheme": scheme,
        "scheme_title": (schemes.get(scheme) or {}).get("title", "caliPr"),
        "schemes": schemes.listing(),
        # A scheme of another protocol has no fins to retrace and no traits.
        "traits": not schemes.get(scheme),
        "fin_groups": [] if schemes.get(scheme) else fin_groups(),
        "lateral": {
            "polygons": poly(View.LATERAL),
            "keypoints": kp(KEYPOINTS, View.LATERAL, extras=True),
            "ruler": kp(CALIBRATION_KEYPOINTS, View.LATERAL),
        },
        "frontal": {
            "polygons": poly(View.FRONTAL),
            "keypoints": kp(KEYPOINTS, View.FRONTAL, extras=True),
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


#: The framing each view gets when a study keeps one photograph per fish, cached
#: on the photograph's path and mtime: finding the mirror seam is one Sobel over a
#: 24-megapixel frame, and the labeler asks for it on every fish it opens.
_FRAMES: dict[tuple, dict] = {}


def view_frames(path: Path) -> dict:
    """Where each view sits in one photograph -- ``{"seam", "lateral", "frontal"}``.

    The models were trained on crops and are still given one; it is cut here, in
    memory, rather than kept on disk where it can be cut in the wrong place and
    strand the labels already placed on it. A fixed fraction of the frame cannot
    do this job: across the 131 photographs the seam runs from 0 to 0.35 of the
    width and the leftmost snout sits at 0.197, so every fixed choice either
    admits the mirrored head or cuts a real one. The seam has to be found.
    """
    try:
        st = path.stat()
    except OSError:
        return {"seam": None, "lateral": None, "frontal": None}
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _FRAMES.get(key)
    if hit is None:
        try:
            import cv2
            import preprocess_cornell as pc
            im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            hit = pc.view_frames(im) if im is not None else None
        except Exception:
            hit = None
        # Without OpenCV -- the labelling-only install -- nothing is framed and
        # the models see the whole photograph, which is what they saw before any
        # of this existed.
        if hit is None:
            hit = {"seam": None, "lateral": None, "frontal": None}
        _FRAMES.clear() if len(_FRAMES) > 64 else None
        _FRAMES[key] = hit
    return hit


def crop_for(view: str, path: Path) -> list[int] | None:
    """The rectangle to predict in, for a study that keeps one photograph per fish."""
    box = view_frames(path).get(view)
    return list(box) if box else None


#: Hand labelling works without any of this, so it is installed separately.
NO_TRAINING_STACK = (
    "Auto-label needs the training stack, which is installed separately from the "
    "labeler. In the caliPr folder:\n\n"
    "    python3.11 -m venv .venv-train\n"
    "    .venv-train/bin/pip install \"deeplabcut==3.0.1\" transformers\n"
    "    .venv/bin/python scripts/fetch_model.py\n\n"
    "then restart the labeler. Hand labelling works without it.")


XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class ExportError(Exception):
    """An export that could not be built, with a message for the annotator."""


#: What the platform's file browser is called, for the message after an export.
FILE_BROWSER = ("Finder" if sys.platform == "darwin"
                else "File Explorer" if sys.platform.startswith("win") else "the file manager")


#: Apps that take a file as a chat attachment rather than showing it. The
#: Claude desktop app registers itself for .xlsx, and on a Mac with no spreadsheet
#: app it is the default: opening an export then asked whether to attach it.
NOT_VIEWERS = ("Claude.app",)


def default_app(path: Path) -> str | None:
    """The app macOS would open ``path`` with, or None if nothing opens it."""
    js = ("ObjC.import('AppKit');var u=$.NSWorkspace.sharedWorkspace"
          f".URLForApplicationToOpenURL($.NSURL.fileURLWithPath({json.dumps(str(path))}));"
          "u.isNil()?'':u.path.js")
    r = subprocess.run(["osascript", "-l", "JavaScript", "-e", js],
                       capture_output=True, text=True, timeout=15)
    return r.stdout.strip() or None


def how_to_show(path: Path, how: str, app: str | None) -> tuple[str, str | None]:
    """Whether to open ``path`` or only select it, given the app that would open it.

    Returns (how, note): a file is only opened when a real viewer would open it;
    otherwise it is selected in the file browser, and the note says why.
    """
    if how != "open" or path.is_dir():
        return how, None
    kind = path.suffix or "these"
    if app is None:
        return "reveal", (f"Nothing on this computer opens {kind} files, so it is shown in "
                          f"{FILE_BROWSER} instead. Install an app for them to open it.")
    if Path(app).name in NOT_VIEWERS:
        return "reveal", (f"This computer would open {kind} files in {Path(app).stem}, which "
                          f"offers them to a chat instead of showing them, so it is shown in "
                          f"{FILE_BROWSER} instead. Install a spreadsheet app (Numbers, Excel "
                          f"or LibreOffice) to open it directly.")
    return "open", None


def show_on_this_computer(path: Path, how: str) -> tuple[str, str | None]:
    """Open ``path`` in its default app, or select it in the file browser.

    Returns what was actually done and, if that differs from what was asked, why.
    """
    if sys.platform == "darwin":
        how, note = how_to_show(path, how, default_app(path) if how == "open" and path.is_file() else "")
        subprocess.run(["open", "-R", str(path)] if how == "reveal" else ["open", str(path)],
                       check=True, capture_output=True, timeout=30)
        return how, note
    if sys.platform.startswith("win"):
        if how == "reveal":
            subprocess.Popen(["explorer", f"/select,{path}"])   # exits 1 even on success
        else:
            os.startfile(str(path))                             # noqa: S606 -- local file we wrote
        return how, None
    subprocess.run(["xdg-open", str(path.parent if how == "reveal" else path)],
                   check=True, capture_output=True, timeout=30)
    return how, None


class Predictor:
    """A single long-lived predict_worker subprocess, started on first use.

    The model costs ~4s to load and ~1s to run, so a process per request would
    make Auto-label feel broken. It also lives in the training environment,
    which this server's interpreter is not — hence a subprocess rather than an
    import.

    Started lazily: someone labelling by hand should not pay for a model they
    never ask for, and the server must still start on a machine with no
    training environment at all.
    """

    _proc = None
    _info: dict = {}
    _lock = threading.Lock()

    #: Interpreters to try, most specific first. A machine without the training
    #: environment simply reports Auto-label as unavailable.
    _PYTHONS = (_ROOT / ".venv-train" / "bin" / "python",
                _ROOT / ".venv" / "bin" / "python")

    @classmethod
    def _start(cls):
        worker = _ROOT / "scripts" / "predict_worker.py"
        exe = next((p for p in cls._PYTHONS if p.is_file()), None)
        if exe is None or not worker.is_file():
            return {"error": NO_TRAINING_STACK}
        try:
            proc = subprocess.Popen(
                [str(exe), str(worker)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1, cwd=str(_ROOT))
        except Exception as exc:
            return {"error": f"could not start predictor: {exc}"}
        hello = proc.stdout.readline()
        try:
            info = json.loads(hello)
        except Exception:
            proc.kill()
            return {"error": "predictor did not start (is DeepLabCut installed "
                             "in .venv-train?)"}
        if not info.get("ready"):
            proc.kill()
            err = info.get("error", "predictor failed to load")
            # A fresh install has the labelling stack and not the training one,
            # so a missing import here is the ordinary case rather than a fault.
            # Saying which package is absent helps nobody; saying what to install
            # does.
            if "ModuleNotFoundError" in err or "ImportError" in err:
                return {"error": NO_TRAINING_STACK}
            return {"error": err, "missing_model": bool(info.get("missing_model"))}
        cls._proc, cls._info = proc, info
        return info

    @classmethod
    def predict(cls, image: Path, polygons: bool = False,
                emit_polygons: bool = True, view: str = "lateral",
                crop: list[int] | None = None) -> dict:
        with cls._lock:                      # one request at a time down one pipe
            if cls._proc is None or cls._proc.poll() is not None:
                started = cls._start()
                if "error" in started:
                    return {"ok": False, **started}
            try:
                cls._proc.stdin.write(json.dumps({"image": str(image),
                                                  "polygons": polygons,
                                                  "emit_polygons": emit_polygons,
                                                  "view": view, "crop": crop})
                                      + "\n")
                cls._proc.stdin.flush()
                line = cls._proc.stdout.readline()
            except Exception as exc:
                cls._proc = None
                return {"ok": False, "error": f"predictor died: {exc}"}
            if not line:
                cls._proc = None
                return {"ok": False, "error": "predictor closed unexpectedly"}
            try:
                return json.loads(line)
            except Exception as exc:
                return {"ok": False, "error": f"bad predictor reply: {exc}"}


def specimen_gaps(data: dict, schema: dict, size, recorded) -> dict:
    """What a saved fish still lacks, measured against what its study collects."""
    lat = data.get("lateral") or {}
    fro = data.get("frontal") or {}
    kps, polys = lat.get("keypoints") or {}, lat.get("polygons") or {}
    meta = data.get("metadata") or {}
    wanted_poly = {p["name"]: p for p in schema["lateral"]["polygons"] if not p["excluded"]}
    fin_kp = {k["name"]: g["fin"] for g in schema.get("fin_groups") or [] for k in g["keypoints"]}
    # Body landmarks only: fin bases and tips are fin work, reported with the fins,
    # and the fin-base endpoints are derived from a traced outline, not clicked.
    wanted_kp = [k["name"] for k in schema["lateral"]["keypoints"]
                 if not k["excluded"] and k["name"] not in fin_kp
                 and not re.search(r"_base_(anterior|posterior)$", k["name"])]

    def scaled(block: dict) -> bool:
        cal = block.get("calibration") or {}
        return (cal.get("mode") == "ticks" and bool(cal.get("px_per_mm"))) or (
            cal.get("mode") == "manual" and bool(cal.get("known_mm")))

    fins_thin = [n for n in FIN_POLYGONS if n in wanted_poly and polys.get(n)
                 and len(polys[n]) < FIN_POLYGON_TARGET_VERTICES]
    fins_untraced = [n for n in FIN_POLYGONS if n in wanted_poly and not polys.get(n)]
    fins_no_points = sorted({fin for name, fin in fin_kp.items()
                             if fin in wanted_poly and name not in kps})
    assist = (meta.get("assist") or {}), (meta.get("assist_frontal") or {})
    started = bool(kps or polys)
    return {
        "landmarks_missing": [n for n in wanted_kp if n not in kps] if started else [],
        "no_outline": started and "body_plus_caudal" in wanted_poly
                      and len(polys.get("body_plus_caudal") or []) < 3,
        "fins_thin": fins_thin if started else [],
        "fins_untraced": fins_untraced if started else [],
        "fins_no_points": fins_no_points if started else [],
        "no_scale": started and not scaled(lat),
        "frontal_no_scale": bool(fro.get("keypoints")) and not scaled(fro),
        "unreviewed": sum(len(a.get("unreviewed") or []) for a in assist),
        "flagged": bool(meta.get("exclude_traits") or meta.get("data_note")),
        "misaligned": bool(recorded and size and started
                           and [recorded.get("width"), recorded.get("height")] != list(size)),
    }


class Handler(BaseHTTPRequestHandler):
    datasets: dict[str, Path] = {}
    default_dataset: str = ""
    images_dir: Path          # resolved per request from ?dataset=
    out_dir: Path
    session = auth.Session()  # shared across requests; never persisted
    demo_mode = False         # read-only: predictions allowed, saving refused
    #: Set by --out. Per-request dataset resolution must not clobber it: an
    #: explicit "write here" is the one instruction that should survive
    #: switching datasets, and silently ignoring it sends writes to the real
    #: sidecar directory, which is the opposite of what anyone passing it wants.
    out_override: Path | None = None
    #: Kept so the dataset list can be rebuilt on request. Without it the set of
    #: studies is frozen at startup, and a folder created afterwards is invisible
    #: until the server is restarted — which looks exactly like a bug.
    data_root: Path | None = None

    def _use(self, query: str) -> None:
        """Point this request at the dataset named in the query string."""
        name = parse_qs(query).get("dataset", [Handler.default_dataset])[0]
        base = Handler.datasets.get(name)
        if base is None:
            base = Handler.datasets.get(Handler.default_dataset)
        if base is not None:
            self.images_dir = base
            self.out_dir = Handler.out_override or (base / "sidecars")

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
        profile = load_profile(self.images_dir)
        single = profile.get("single_photo")
        schema = build_schema(profile)
        manifest = image_identity.load_manifest(self.images_dir)
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
            gaps: dict = {}
            if sidecar.is_file():
                try:
                    data = json.loads(sidecar.read_text())
                    gaps = specimen_gaps(data, schema, current_size(path),
                                         image_identity.recorded_for(data, manifest, fid, "lateral"))
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
                # One photograph per fish: the same file carries both views.
                "frontal": path.name if single else (fname if fname in fro else None),
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
                # What a labelled fish still lacks, for the search box's commands.
                "gaps": gaps,
                "heldout": fid in HELDOUT,
                "suggested": fid in suggested,
                # Size of each view's image as it is on disk now. A draft records
                # the size it was made on; the labeler discards any draft whose
                # size no longer matches, because its coordinates cannot fit.
                "sizes": {"lateral": current_size(path),
                          "frontal": current_size(path if single else fro.get(fname))},
                # A prediction is cached for this fish. The labeler applies it on
                # open, so a batch run actually reaches the fish it predicted.
                "predicted": (self.images_dir / "sidecars_auto"
                              / f"{fid}.json").is_file(),
                # The frontal model's cache is kept apart, so the labeler can
                # apply it when that view is opened.
                "predicted_frontal": (self.images_dir / "sidecars_auto" / "frontal"
                                      / f"{fid}.json").is_file(),
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

    def _set_scheme(self):
        """Switch this study to another landmark scheme, or back to caliPr's.

        Landmarks already saved are left exactly as they are: they are what a
        person clicked, and the two schemes name different points, so nothing can
        be converted. They simply stop being asked for until the scheme is
        switched back.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            name = json.loads(self.rfile.read(n) or b"{}").get("scheme") or schemes.DEFAULT
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"bad payload: {exc}"})
        if name != schemes.DEFAULT and not schemes.get(name):
            return self._send(404, {"ok": False, "error": f"no scheme {name!r}"})
        path = self.images_dir / "schema.json"
        prof = {}
        if path.is_file():
            try:
                prof = json.loads(path.read_text())
            except Exception:
                prof = {}
        if name == schemes.DEFAULT:
            prof.pop("scheme", None)
        else:
            prof["scheme"] = name
        path.write_text(json.dumps(prof, indent=2) + "\n")
        return self._send(200, {"ok": True, "scheme": name,
                                "schema": build_schema(load_profile(self.images_dir))})

    def _edit_schema_keypoint(self):
        """Add, rename or remove one of this study's own landmarks.

        Adding extends the study's schema.json; the master schema is untouched, so
        a custom landmark exists for this study and every specimen in it. Renaming
        records what to call a landmark here -- for master landmarks the stored name
        must not change, or the traits, the trained model and every sidecar already
        saved would stop referring to the same point.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"bad payload: {exc}"})
        action = data.get("action")
        path = self.images_dir / "schema.json"
        prof = {}
        if path.is_file():
            try:
                prof = json.loads(path.read_text())
            except Exception:
                prof = {}
        extra = prof.get("extra_keypoints") or []
        labels = prof.get("labels") or {}
        master = {k.name for k in KEYPOINTS} | {p.name for p in POLYGONS}
        taken = master | {k["name"] for k in extra}

        if action == "add":
            label = str(data.get("label") or "").strip()
            view = data.get("view") if data.get("view") in ("lateral", "frontal") else "lateral"
            if not 1 <= len(label) <= 60:
                return self._send(400, {"ok": False, "error": "give the landmark a name"})
            base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "landmark"
            name = base
            i = 2
            while name in taken:
                name, i = f"{base}_{i}", i + 1
            extra.append({"name": name, "label": label, "view": view,
                          "hint": str(data.get("hint") or "")})
            prof["extra_keypoints"] = extra
        elif action == "rename":
            name = str(data.get("name") or "")
            label = str(data.get("label") or "").strip()
            if name not in taken:
                return self._send(404, {"ok": False, "error": f"no landmark {name!r}"})
            if not 1 <= len(label) <= 60:
                return self._send(400, {"ok": False, "error": "give the landmark a name"})
            if label == name:
                labels.pop(name, None)            # back to its own name
            else:
                labels[name] = label
            for k in extra:
                if k["name"] == name:
                    k["label"] = label
            prof["labels"] = labels
        elif action == "remove":
            name = str(data.get("name") or "")
            if not any(k["name"] == name for k in extra):
                return self._send(400, {"ok": False,
                                        "error": "only a landmark added for this study can be removed; "
                                                 "use the exclude list for the rest"})
            placed = self._count_placed(name)
            if placed and not data.get("force"):
                return self._send(409, {"ok": False, "placed": placed,
                                        "error": f"{name} is placed on {placed} fish"})
            prof["extra_keypoints"] = [k for k in extra if k["name"] != name]
            labels.pop(name, None)
            prof["labels"] = labels
        else:
            return self._send(400, {"ok": False, "error": f"unknown action {action!r}"})

        if not prof.get("labels"):
            prof.pop("labels", None)
        if not prof.get("extra_keypoints"):
            prof.pop("extra_keypoints", None)
        path.write_text(json.dumps(prof, indent=2) + "\n")
        return self._send(200, {"ok": True, "schema": build_schema(load_profile(self.images_dir))})

    def _count_placed(self, name: str) -> int:
        """How many saved fish have this landmark placed, in either view."""
        n = 0
        for p in self.out_dir.glob("*.json"):
            try:
                doc = json.loads(p.read_text())
            except Exception:
                continue
            if any(name in ((doc.get(v) or {}).get("keypoints") or {})
                   for v in ("lateral", "frontal")):
                n += 1
        return n

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

    def _build_export(self, kind: str) -> dict:
        """Write an export under ``results/<dataset>/`` and say how to show it.

        Returns the file to download (``file``, ``name``, ``type``) and what to open
        on this computer (``show``, ``how``). Raises :class:`ExportError` with a
        message the annotator can read, and ``KeyError`` for an unknown kind.
        """
        import zipfile

        ds = self.images_dir.name
        root = _ROOT / "results" / ds

        def run(args: list[str], timeout: int) -> None:
            r = subprocess.run([sys.executable, *args], capture_output=True, text=True,
                               timeout=timeout, cwd=_ROOT)
            if r.returncode != 0:
                raise ExportError((r.stderr or r.stdout)[-800:])

        def zipped(name: str, members: list[tuple[Path, str]]) -> Path:
            target = root / name
            root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
                for f, arc in members:
                    z.write(f, arc)
            return target

        # Pass the paths, never just the dataset name. The scripts default to
        # the repository's own data/, so a server started with --data-root or
        # --images elsewhere produced exports that looked for a directory that
        # does not exist — or worse, found a same-named study in the repo.
        scheme = load_profile(self.images_dir).get("scheme")
        if kind == "measurements" and schemes.get(scheme):
            raise ExportError(
                f"This study collects the {schemes.get(scheme)['title']} landmarks. Every trait "
                f"is defined against caliPr's own landmarks, so there is nothing to measure here "
                f"— export Landmarks for R (.tps) instead, which carries all "
                f"{len(schemes.get(scheme)['landmarks'])} points in the protocol's order.")
        if kind == "measurements":
            out = root / "measurements.xlsx"
            run([str(_ROOT / "scripts/export_measurements.py"), "--dataset", ds,
                 "--images", str(self.images_dir / "lateral"),
                 "--labels", str(self.out_dir), "--out", str(out)], 600)
            if not out.is_file():
                raise ExportError("the workbook was not written")
            return {"file": out, "name": f"{ds}_measurements.xlsx", "type": XLSX,
                    "show": out, "how": "open"}

        if kind == "tps":
            run([str(_ROOT / "scripts/export_tps.py"), "--sidecars", str(self.out_dir),
                 "--images", str(self.images_dir / "lateral"),
                 "--schema-dir", str(self.images_dir), "--out", str(root / "tps")], 600)
            files = [f for f in sorted((root / "tps").iterdir()) if f.is_file()]
            zp = zipped(f"{ds}_tps.zip", [(f, f.name) for f in files])
            tps = root / "tps" / "landmarks.tps"
            return {"file": zp, "name": zp.name, "type": "application/zip",
                    "show": tps if tps.is_file() else zp, "how": "reveal"}

        if kind == "sidecars":
            # The annotations themselves, which is what training consumes and
            # what a correction lives inside. Every other export is derived
            # and cannot be trained on; without this a collaborator's
            # corrections stay on their machine.
            files = sorted(self.out_dir.glob("*.json"))
            if not files:
                raise ExportError("nothing labelled yet")
            zp = zipped(f"{ds}_sidecars.zip", [(f, f"{ds}/sidecars/{f.name}") for f in files])
            return {"file": zp, "name": zp.name, "type": "application/zip",
                    "show": zp, "how": "reveal"}

        if kind == "overlays":
            run([str(_ROOT / "scripts/render_overlays.py"), "--dataset", ds,
                 "--images", str(self.images_dir / "lateral"),
                 "--sidecars", str(self.out_dir), "--out", str(root / "overlays")], 1800)
            folder = root / "overlays"
            imgs = sorted(folder.glob("*.jpg")) if folder.is_dir() else []
            if not imgs:
                raise ExportError("nothing annotated yet")
            zp = zipped(f"{ds}_overlays.zip", [(f, f.name) for f in imgs])
            return {"file": zp, "name": zp.name, "type": "application/zip",
                    "show": folder, "how": "open"}

        raise KeyError(kind)

    def _export(self, kind: str):
        """Build an export on demand and hand it back as a download."""
        try:
            out = self._build_export(kind)
        except KeyError:
            return self._send(404, {"error": f"unknown export {kind!r}"})
        except subprocess.TimeoutExpired:
            return self._send(500, {"error": "export timed out"})
        except Exception as exc:
            return self._send(500, {"error": str(exc)})
        return self._send_bytes(out["file"].read_bytes(), out["name"], out["type"])

    def _export_and_show(self, kind: str):
        """Build an export and open it on this computer, instead of downloading it.

        A download goes wherever the browser sends it, and the browser pane inside
        the Claude app offers every download to Claude instead of opening it. The
        server only ever listens on 127.0.0.1, so it is on the annotator's own
        computer and can open the file itself: the workbook in the spreadsheet
        app, the rest selected in Finder. If nothing can open it -- no desktop --
        the page falls back to downloading.
        """
        try:
            out = self._build_export(kind)
        except KeyError:
            return self._send(404, {"ok": False, "error": f"unknown export {kind!r}"})
        except subprocess.TimeoutExpired:
            return self._send(500, {"ok": False, "error": "export timed out"})
        except Exception as exc:
            return self._send(500, {"ok": False, "error": str(exc)})
        try:
            rel = str(out["show"].relative_to(_ROOT))
        except ValueError:
            rel = str(out["show"])
        try:
            shown, note = show_on_this_computer(out["show"], out["how"])
        except Exception as exc:
            return self._send(200, {"ok": True, "path": rel, "shown": None,
                                    "error": f"could not open it here: {exc}"})
        return self._send(200, {"ok": True, "path": rel, "shown": shown,
                                "shown_in": FILE_BROWSER, "note": note})

    #: What a photograph may be. Extension alone is not enough — a .jpg that is
    #: not a JPEG produces a specimen that silently fails to load later, so the
    #: first bytes are checked too.
    IMAGE_MAGIC = (
        (b"\xff\xd8\xff", ".jpg"),          # JPEG
        (b"\x89PNG\r\n\x1a\n", ".png"),     # PNG
        (b"II*\x00", ".tif"), (b"MM\x00*", ".tif"),   # TIFF, both byte orders
    )
    MAX_UPLOAD = 300 * 1024 * 1024

    def _remove_dataset(self):
        """Take a study out of the list by moving its folder to ``data/.trash/``.

        Nothing is deleted. A study holds photographs and hand labels that took
        hours; a mistaken click must be recoverable by moving the folder back. The
        trash is not a study (it has no ``lateral/`` at its top level), so it never
        appears in the list. Exports under ``results/`` are left alone.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        if Handler.data_root is None:
            return self._send(400, {"ok": False, "error": "server was started for a single "
                                                         "dataset; restart without --images"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            name = json.loads(self.rfile.read(n) or b"{}").get("name", "")
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        Handler.datasets = discover_datasets(Handler.data_root)
        base = Handler.datasets.get(name)          # only a listed study, never a path
        if base is None:
            return self._send(404, {"ok": False, "error": f"no study named “{name}”"})
        trash = Handler.data_root / ".trash"
        trash.mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        target = trash / f"{name}-{stamp}"
        shutil.move(str(base), str(target))
        Handler.datasets = discover_datasets(Handler.data_root)
        if Handler.default_dataset not in Handler.datasets:
            Handler.default_dataset = sorted(Handler.datasets)[0] if Handler.datasets else ""
        return self._send(200, {"ok": True, "name": name,
                                "moved_to": str(target.relative_to(Handler.data_root.parent))
                                if Handler.data_root.parent in target.parents else str(target),
                                "default": Handler.default_dataset})

    def _new_dataset(self):
        """Create an empty study directory so a folder of photographs has
        somewhere to land without touching a terminal."""
        if self.demo_mode:
            return self._send(403, {"ok": False,
                                    "error": "demo mode — nothing is written"})
        if Handler.data_root is None:
            return self._send(400, {"ok": False,
                                    "error": "server was started for a single "
                                             "dataset; restart without --images"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            raw = payload.get("name", "")
            settings_from = payload.get("settings_from") or None
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        source = Handler.datasets.get(settings_from) if settings_from else None
        if settings_from and source is None:
            return self._send(400, {"ok": False,
                                    "error": f"no study named “{settings_from}” to copy settings from"})
        # A dataset name becomes a directory name, so it may not be a path.
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(str(raw)).name).strip("._-")
        if not name:
            return self._send(400, {"ok": False, "error": "give the study a name"})
        base = Handler.data_root / name
        if base.exists():
            return self._send(409, {"ok": False, "name": name,
                                    "error": f"“{name}” already exists"})
        (base / "lateral").mkdir(parents=True)
        copied = []
        for f in STUDY_SETTINGS:
            if source is not None and (source / f).is_file():
                shutil.copyfile(source / f, base / f)
                copied.append(f)
        Handler.datasets = discover_datasets(Handler.data_root)
        if Handler.default_dataset not in Handler.datasets:
            Handler.default_dataset = name
        return self._send(200, {"ok": True, "name": name, "settings_copied": copied})

    def _upload(self):
        """Accept one photograph into the current dataset's lateral/ folder.

        One file per request rather than a single multipart batch: it keeps the
        parsing trivial, lets the browser show real progress across a folder of
        200, and means one bad file fails on its own instead of taking the batch
        with it.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False,
                                    "error": "demo mode — nothing is written"})
        raw_name = self.headers.get("X-Filename", "")
        view = (self.headers.get("X-View") or "lateral").lower()
        if view not in ("lateral", "frontal"):
            return self._send(400, {"ok": False, "error": "bad view"})

        # Basename only, and a conservative character set: an uploaded name is
        # attacker-controlled in principle and a path is the one thing it must
        # never be able to be.
        name = Path(unquote(raw_name)).name
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
        if not name:
            return self._send(400, {"ok": False, "error": "no filename"})
        ext = Path(name).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            return self._send(415, {"ok": False, "name": name,
                                    "error": f"not an image extension ({ext})"})

        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return self._send(400, {"ok": False, "name": name, "error": "empty"})
        if n > self.MAX_UPLOAD:
            return self._send(413, {"ok": False, "name": name,
                                    "error": f"{n/1e6:.0f} MB exceeds the limit"})
        data = self.rfile.read(n)
        if not any(data.startswith(sig) for sig, _ in self.IMAGE_MAGIC):
            return self._send(415, {"ok": False, "name": name,
                                    "error": "contents are not a JPEG, PNG or TIFF"})

        whole = load_profile(self.images_dir).get("single_photo")
        if (self.headers.get("X-Split") or "").lower() == "mirror" and not whole:
            if view != "lateral":
                return self._send(400, {"ok": False, "name": name,
                                        "error": "a mirror split applies to lateral photographs"})
            return self._upload_split(name, data)

        dest_dir = self.images_dir / view
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest.is_file():
            # Same bytes is a re-drop of a folder already added, which is
            # ordinary; different bytes under a name already in use is not, and
            # overwriting it would replace a photograph other sidecars point at.
            if dest.read_bytes() == data:
                return self._send(200, {"ok": True, "name": name,
                                        "status": "duplicate"})
            return self._send(409, {"ok": False, "name": name,
                                    "error": "a DIFFERENT file of this name is "
                                             "already here; rename before adding"})
        dest.write_bytes(data)
        return self._send(200, {"ok": True, "name": name,
                                # This study keeps one photograph per fish, so a
                                # mirror split was not made even if asked for: the
                                # head-on view is the same frame, zoomed.
                                "status": "added_whole" if whole else "added",
                                "bytes": len(data)})

    _SPLIT_LOCK = threading.Lock()

    def _upload_split(self, name: str, data: bytes):
        """One photograph holding both views -- the fish, and its head-on view in a
        mirror on the left -- split into a lateral and a frontal image.

        The mirror detector cannot tell when it is wrong: on the 131 lab photos, the
        six cuts it made into the fish's head scored no differently from good cuts.
        So the original is always kept under ``originals/``, where the split was made
        is recorded in ``splits.json``, and a boundary where no mirror has ever been
        is not used at all -- that photo is stored whole and reported, rather than
        given a frontal view with no mouth in it.
        """
        sys.path.insert(0, str(_ROOT / "scripts"))
        import cv2
        import preprocess_cornell as pc

        stem = re.sub(r"_[LF]$", "", Path(name).stem)
        lat_dir, fro_dir = self.images_dir / "lateral", self.images_dir / "frontal"
        orig_dir = self.images_dir / "originals"
        orig = orig_dir / name
        lat_path, fro_path = lat_dir / f"{stem}_L.JPEG", fro_dir / f"{stem}_F.JPEG"

        with self._SPLIT_LOCK:
            if orig.is_file():
                if orig.read_bytes() == data:
                    return self._send(200, {"ok": True, "name": name, "status": "duplicate"})
                return self._send(409, {"ok": False, "name": name,
                                        "error": "a DIFFERENT photograph of this name was "
                                                 "already added; rename before adding"})
            sidecar = self.out_dir / f"{stem}.json"
            if sidecar.is_file():
                try:
                    doc = json.loads(sidecar.read_text())
                except Exception:
                    doc = {"lateral": {"keypoints": {"_": [0, 0]}}}
                if image_identity.has_labels(doc, "lateral") or image_identity.has_labels(doc, "frontal"):
                    return self._send(409, {"ok": False, "name": name,
                                            "error": f"{stem} already has labels; its images "
                                                     f"are not replaced"})
            for p in (lat_path, fro_path):
                if p.is_file():
                    return self._send(409, {"ok": False, "name": name,
                                            "error": f"{p.name} already exists from another "
                                                     f"source; not overwritten"})

            orig_dir.mkdir(parents=True, exist_ok=True)
            orig.write_bytes(data)
            try:
                image = pc.normalize_orientation(orig)
            except Exception as exc:
                orig.unlink(missing_ok=True)
                return self._send(415, {"ok": False, "name": name,
                                        "error": f"could not read the photograph: {exc}"})
            h, w = image.shape[:2]
            res = pc.split_composite(image)

            lat_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(lat_path), res["lateral"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if res["ok"]:
                fro_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(fro_path), res["frontal"], [cv2.IMWRITE_JPEG_QUALITY, 95])

            record = self.images_dir / "splits.json"
            try:
                splits = json.loads(record.read_text()) if record.is_file() else {}
            except Exception:
                splits = {}
            splits[stem] = {"original": f"originals/{name}", "width": w, "height": h,
                            "boundary": res["boundary"], "lateral_start": res["lateral_start"],
                            "frontal_end": res["frontal_end"], "split": res["ok"],
                            "reason": res["reason"],
                            "at": datetime.datetime.now().isoformat(timespec="seconds")}
            record.write_text(json.dumps(splits, indent=2, sort_keys=True))

        body = {"ok": True, "name": name, "stem": stem, "bytes": len(data),
                "status": "split" if res["ok"] else "stored_whole",
                "boundary_fraction": round(res["boundary"] / w, 3)}
        if not res["ok"]:
            body["reason"] = res["reason"]
        return self._send(200, body)

    def _token(self):
        raw = self.headers.get("Cookie") or ""
        for part in raw.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "calipr_token":
                return v
        return None

    def _locked(self) -> bool:
        """True when the automation is configured to require an unlock and has
        not had one. No passphrase configured means no gate — hand labelling
        must never be blocked by a feature somebody has not opted into."""
        return auth.is_configured() and not self.session.unlocked(self._token())

    def _authstate(self):
        return self._send(200, {
            "required": auth.is_configured(),
            "unlocked": self.session.unlocked(self._token()),
            "demo": self.demo_mode,
        })

    def _unlock(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        if not auth.is_configured():
            return self._send(200, {"ok": True, "message": "no passphrase set"})
        ok, msg = self.session.attempt(str(body.get("passphrase", "")))
        if not ok:
            return self._send(401, {"ok": False, "error": msg})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        # Session-scoped, not readable from JS, and not sent on cross-site
        # requests. It never leaves this machine, but there is no reason to make
        # it any more available than it has to be.
        self.send_header("Set-Cookie",
                         f"calipr_token={self.session.token}; Path=/; "
                         f"HttpOnly; SameSite=Strict")
        payload = json.dumps({"ok": True, "message": msg}).encode()
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _predict(self, fid: str):
        """Run the keypoint model on one specimen and return points + confidence.

        Deliberately does not write anything. A prediction becomes data only
        when a human has looked at it and pressed Save, at which point it is
        saved as their sidecar — so nothing here can quietly manufacture labels.

        ``?view=frontal`` runs the mouth-corner model on the frontal crop. Its
        cache is kept in ``sidecars_auto/frontal/``, apart from the lateral one,
        whose presence is what marks a fish as batch-predicted.
        """
        if self._locked():
            return self._send(401, {
                "ok": False, "locked": True,
                "error": "automated landmarking is locked on this machine"})
        scheme = load_profile(self.images_dir).get("scheme")
        if schemes.get(scheme):
            return self._send(400, {
                "ok": False, "error":
                    f"This study collects the {schemes.get(scheme)['title']} landmarks, and the "
                    f"model was trained on caliPr's. It would place points this study does not "
                    f"collect, so Auto-label is off here."})
        query = parse_qs(urlparse(self.path).query)
        view = query.get("view", ["lateral"])[0]
        if view not in ("lateral", "frontal"):
            return self._send(400, {"ok": False, "error": f"unknown view {view!r}"})
        match = None
        if view == "lateral":
            for name, path in list_images(self.images_dir / "lateral").items():
                stem = Path(name).stem
                if stem == fid or (stem.endswith("_L") and stem[:-2] == fid):
                    match = path
                    break
        else:
            match = view_image(self.images_dir, fid, "frontal")
        if match is None:
            return self._send(404, {"ok": False, "error": f"no {view} image for {fid}"})

        # A batch run writes its results here, so opening a specimen afterwards
        # returns instantly instead of paying a second of inference again. The
        # cache lives in sidecars_auto/, never beside the hand labels, and every
        # entry is marked source=predicted.
        auto = self.images_dir / "sidecars_auto"
        cache = (auto if view == "lateral" else auto / "frontal") / f"{fid}.json"
        want_cache = query.get("cache", ["1"])[0] != "0"
        res = None
        if want_cache and cache.is_file():
            try:
                doc = json.loads(cache.read_text())
                meta = doc.get("metadata") or {}
                # An entry written before the plausibility check existed holds
                # points that were never tested against the animal. Serving it
                # would quietly reinstate exactly what the check removes, so it
                # is treated as a miss and predicted again.
                if meta.get("source") == "predicted" and "implausible" in meta:
                    res = {
                        "ok": True, "fish_id": fid, "cached": True, "view": view,
                        "model": meta.get("model"),
                        "keypoints": doc[view]["keypoints"],
                        "polygons": (doc[view].get("polygons") or {}),
                        "confidence": meta.get("keypoint_confidence") or {},
                        "low_confidence": meta.get("low_confidence") or [],
                        "implausible": meta.get("implausible") or {},
                        "frame_warning": meta.get("frame_warning"),
                        "elapsed": 0.0}
            except Exception:
                res = None                 # a corrupt cache entry just re-predicts
        if res is None:
            res = self._run_model(fid, view, match, cache, want_cache)
            if not res.get("ok"):
                return self._send(503, res)

        return self._send(200, self._excluded(res))

    def _run_model(self, fid: str, view: str, match: Path, cache: Path,
                   want_cache: bool) -> dict:
        # Only the body outline is predicted, and only where the study collects
        # it. The fins are not automatable at any useful accuracy, so offering
        # them would spend review time to no end.
        prof0 = load_profile(self.images_dir)
        want_body = (view == "lateral" and "body_plus_caudal"
                     not in set(prof0.get("exclude_polygons") or ()))

        # A study may want the outline computed but not offered. The plausibility
        # check measures each landmark against the span from snout to caudal tip,
        # which the segmentation supplies -- those two extremes are the part of
        # the outline that is reliable even where the middle of it wraps a fin.
        # So "stop giving me the outline" and "stop checking the landmarks" stay
        # separate decisions.
        emit_body = "body_plus_caudal" not in set(
            prof0.get("exclude_predicted_polygons") or ())

        # A whole-frame photograph is cropped for the model in memory: the frame it
        # was trained on, without a crop on disk that can be cut in the wrong place.
        single = bool(prof0.get("single_photo"))
        crop = crop_for(view, match) if single else None
        if single and view == "frontal" and crop is None:
            # No seam, no head-on view to predict in. The mouth corners would be
            # placed somewhere in the fish's flank, confidently. Say so instead.
            return {"ok": False, "error": "the mirror's edge could not be found in "
                                          "this photograph, so the head-on view "
                                          "cannot be framed — place the two mouth "
                                          "corners by hand"}
        res = Predictor.predict(match, polygons=want_body, emit_polygons=emit_body,
                                view=view, crop=crop)
        if not res.get("ok"):
            return res

        if want_cache and not self.demo_mode:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({
                    "fish_id": fid,
                    "metadata": {"source": "predicted", "model": res.get("model"),
                                 "image": match.name,
                                 "keypoint_confidence": res.get("confidence") or {},
                                 "low_confidence": res.get("low_confidence") or [],
                                 "implausible": res.get("implausible") or {},
                                 "frame_warning": res.get("frame_warning")},
                    view: {"keypoints": res.get("keypoints") or {},
                           "polygons": res.get("polygons") or {},
                           "calibration": {"mode": "none",
                                           "notes": "predicted; not a label"}},
                }, indent=2))
            except Exception:
                pass                       # caching is an optimisation, not a duty
        return res

    def _excluded(self, res: dict) -> dict:
        # Never offer a point for a landmark this study has excluded.
        prof = load_profile(self.images_dir)
        drop = set(prof.get("exclude_keypoints") or ())
        drop_poly = set(prof.get("exclude_polygons") or ())
        if drop_poly and res.get("polygons"):
            res["polygons"] = {k: v for k, v in res["polygons"].items()
                               if k not in drop_poly}
        if drop:
            res["keypoints"] = {k: v for k, v in res["keypoints"].items()
                                if k not in drop}
            res["confidence"] = {k: v for k, v in res["confidence"].items()
                                 if k not in drop}
            res["low_confidence"] = [k for k in res["low_confidence"]
                                     if k not in drop]
        return res

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

        if route == "/api/authstate":
            return self._authstate()

        if route == "/api/datasets":
            if Handler.data_root is not None:
                found = discover_datasets(Handler.data_root)
                if found:
                    Handler.datasets = found
            # A server started against an empty data/ has no default. Without
            # this it keeps reporting none after the first study is created, and
            # the page opens to a blank list having just been told it succeeded.
            if Handler.default_dataset not in Handler.datasets:
                Handler.default_dataset = (sorted(Handler.datasets)[0]
                                           if Handler.datasets else "")
            names = sorted(Handler.datasets)
            return self._send(200, {
                "datasets": [
                    {"name": n,
                     "images": len(list_images(Handler.datasets[n] / "lateral")),
                     "has_frontal": (Handler.datasets[n] / "frontal").is_dir(),
                     "labelled": sum(1 for f in (Handler.datasets[n] / "sidecars").glob("*.json"))
                                 if (Handler.datasets[n] / "sidecars").is_dir() else 0,
                     "settings": [f for f in STUDY_SETTINGS
                                  if (Handler.datasets[n] / f).is_file()],
                     # A dataset whose profile drops every fin polygon can never
                     # satisfy the fin-density badge, so the UI should not show it.
                     "has_fin_polygons": bool(
                         not schemes.get(load_profile(Handler.datasets[n]).get("scheme"))
                         and set(FIN_POLYGONS)
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
        if route.startswith("/api/frame/"):
            # Where each view sits in one photograph. The labeler asks so it can
            # open the head-on view where the model looks for it, rather than at
            # some fraction of the frame that is right for no fish in particular.
            fid = unquote(route[len("/api/frame/"):])
            if "/" in fid or ".." in fid:
                return self._send(400, {"error": "bad path"})
            img = view_image(self.images_dir, fid, "lateral")
            if img is None:
                return self._send(404, {"error": "not found"})
            fr = view_frames(img)
            return self._send(200, {"seam": fr.get("seam"),
                                    "lateral": fr.get("lateral"),
                                    "frontal": fr.get("frontal"),
                                    "size": current_size(img)})

        if route.startswith("/api/autocal/"):
            name = unquote(route[len("/api/autocal/"):])
            if "/" in name or ".." in name:
                return self._send(400, {"error": "bad path"})
            path = self.images_dir / "lateral" / name
            if not path.is_file():
                return self._send(404, {"error": "image not found"})
            try:
                import cv2

                from fish_morpho.ruler_calibration import detect_tick_scale, locate_ticks

                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                res = detect_tick_scale(img)
                out = {"px_per_mm": res.px_per_mm, "confidence": res.confidence,
                       "notes": res.notes}
                # Where that scale puts each millimetre along the ruler, so the
                # labeler can draw it and anyone can see whether it fits.
                try:
                    out["ticks"] = locate_ticks(img, res.px_per_mm)
                except Exception as exc:
                    out["ticks_error"] = str(exc)
                return self._send(200, out)
            except Exception as exc:
                return self._send(200, {"error": str(exc)})
        if route.startswith("/api/export/"):
            return self._export(route[len("/api/export/"):])

        if route.startswith("/api/predict/"):
            return self._predict(unquote(route[len("/api/predict/"):]))

        if route.startswith("/api/sidecar/"):
            fid = unquote(route[len("/api/sidecar/"):])
            p = self.out_dir / f"{fid}.json"
            if not p.is_file():
                return self._send(404, {"error": "no sidecar"})
            try:
                doc = json.loads(p.read_text())
            except Exception as exc:
                return self._send(500, {"error": f"unreadable sidecar: {exc}"})
            # Sent with the labels, never written into them: whether each view's
            # coordinates still belong to the image on disk, and which version of
            # this file they are -- a save based on an older version is refused.
            doc["_alignment"] = alignment(self.images_dir, fid, doc)
            doc["_mtime"] = p.stat().st_mtime
            return self._send(200, doc)
        if route.startswith("/img/"):
            sub = unquote(route[len("/img/"):])  # e.g. lateral/Name_L.JPEG
            if ".." in sub:
                return self._send(400, {"error": "bad path"})
            p = self.images_dir / sub
            if not p.is_file() and sub.startswith("frontal/") and \
                    load_profile(self.images_dir).get("single_photo"):
                # one photograph per fish: the frontal view is the same frame
                p = self.images_dir / "lateral" / sub[len("frontal/"):]
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

        if route == "/api/dataset/new":
            return self._new_dataset()

        if route == "/api/dataset/remove":
            return self._remove_dataset()

        if route.startswith("/api/export/"):
            return self._export_and_show(route[len("/api/export/"):])

        if route == "/api/upload":
            return self._upload()

        if route == "/api/unlock":
            return self._unlock()

        if route == "/api/lock":
            self.session.lock()
            return self._send(200, {"ok": True})

        if route == "/api/schema/exclude":
            return self._set_exclusions()

        if route == "/api/schema/keypoint":
            return self._edit_schema_keypoint()

        if route == "/api/schema/scheme":
            return self._set_scheme()

        if route != "/api/save":
            return self._send(404, {"error": "unknown route"})
        if self.demo_mode:
            # The whole point of demo mode: the automation can be shown without
            # any path by which a prediction becomes a label.
            return self._send(403, {
                "ok": False, "demo": True,
                "error": "demo mode — saving is disabled, nothing was written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n))
            fid = data["fish_id"]
        except Exception as exc:
            return self._send(400, {"error": f"bad payload: {exc}"})
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", fid)
        data.pop("_alignment", None)
        # Labels already on disk that no longer fit their image are recoverable by
        # an exact shift -- until something overwrites them. So nothing is saved
        # for such a fish until it has been repaired.
        existing = self.out_dir / f"{safe}.json"
        data.pop("_mtime", None)
        # The version of this file the labels being saved were loaded from. If the
        # file has changed on disk since -- repaired, saved from another tab --
        # these coordinates are out of date and would overwrite the newer ones.
        base = (data.get("metadata") or {}).pop("base_mtime", None)
        if base is not None:
            now = existing.stat().st_mtime if existing.is_file() else 0.0
            if abs(now - float(base)) > 1.0:
                return self._send(409, {
                    "ok": False, "conflict": True,
                    "error": ("Not saved. This fish's saved labels changed on disk after you "
                              "opened it, so what is on screen is out of date and would "
                              "overwrite the newer version. Reload the page and reopen it. "
                              "Nothing was written.")})
        if existing.is_file():
            try:
                stale = {v: a for v, a in alignment(self.images_dir, fid,
                                                    json.loads(existing.read_text())).items()
                         if a["status"] == image_identity.SIZE_CHANGED}
            except Exception:
                stale = {}
            if stale:
                v, a = next(iter(stale.items()))
                return self._send(409, {
                    "ok": False, "misaligned": True, "view": v,
                    "error": (f"Not saved. The labels already saved for this fish do not fit "
                              f"its {v} image: {a['detail']}. Repair them first, which moves "
                              f"them exactly:  python scripts/realign_labels.py --fish {fid}")})
        meta = data.setdefault("metadata", {})
        # The labeler reports the size of the image each view's labels were placed
        # on. If the file on disk is not that size, the image changed underneath
        # the open fish and these coordinates belong to a different crop: refuse,
        # rather than write labels that are displaced from the moment they land.
        placed_on = meta.pop("placed_on", None) or {}
        images = {}
        for view in ("lateral", "frontal"):
            if not image_identity.has_labels(data, view):
                continue
            cur = current_fingerprint(view_image(self.images_dir, fid, view))
            seen = placed_on.get(view)
            if cur and seen and list(seen) != [cur["width"], cur["height"]]:
                return self._send(409, {
                    "ok": False, "misaligned": True, "view": view,
                    "error": (f"Not saved. The {view} labels were placed on a "
                              f"{seen[0]}x{seen[1]} image, but the image on disk is now "
                              f"{cur['width']}x{cur['height']} — it was re-cropped or "
                              f"replaced, so every coordinate would be displaced. "
                              f"Nothing was written.")})
            if cur:
                images[view] = cur
        if images:
            meta["images"] = images          # what these coordinates were placed on
        self.out_dir.mkdir(parents=True, exist_ok=True)
        target = self.out_dir / f"{safe}.json"
        target.write_text(json.dumps(data, indent=2))
        return self._send(200, {"ok": True, "saved": f"{safe}.json",
                                "mtime": target.stat().st_mtime})


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
    ap.add_argument("--set-password", action="store_true",
                    help="Set or change the passphrase that unlocks automated "
                         "landmarking, then exit. Stored as an scrypt hash in "
                         "~/.calipr/auth.json — outside the repository, so it "
                         "cannot be committed.")
    ap.add_argument("--clear-password", action="store_true",
                    help="Remove the passphrase, leaving the automation open on "
                         "this machine.")
    ap.add_argument("--demo", action="store_true",
                    help="Read-only: automated landmarking runs and can be shown, "
                         "but Save is refused, so a demo cannot put predictions "
                         "into the training data.")
    args = ap.parse_args(argv)

    if args.set_password:
        import getpass
        p1 = getpass.getpass("New passphrase (min 8 chars): ")
        if p1 != getpass.getpass("Repeat: "):
            print("passphrases did not match"); return 1
        try:
            where = auth.set_passphrase(p1)
        except ValueError as exc:
            print(exc); return 1
        print(f"stored an scrypt hash in {where}")
        print("The passphrase itself is not saved anywhere and cannot be "
              "recovered — only reset.")
        return 0

    if args.clear_password:
        path = auth.auth_path()
        if path.is_file():
            path.unlink(); print(f"removed {path}")
        else:
            print("no passphrase was set")
        return 0

    Handler.demo_mode = args.demo

    if args.images:                       # explicit single dataset
        base = args.images.resolve()
        Handler.datasets = {base.name: base}
        Handler.default_dataset = base.name
        if args.out:
            Handler.datasets[base.name] = base
    else:
        Handler.data_root = args.data_root.resolve()
        Handler.data_root.mkdir(parents=True, exist_ok=True)
        Handler.datasets = discover_datasets(Handler.data_root)
        # An empty data/ is a first run, not an error. Refusing to start here
        # left a new user with no way in at all: the button that creates the
        # first study is inside the page the server would not serve.
        Handler.default_dataset = (args.dataset if args.dataset in Handler.datasets
                                   else (sorted(Handler.datasets)[0]
                                         if Handler.datasets else ""))
    base = Handler.datasets.get(Handler.default_dataset)
    Handler.images_dir = base if base is not None else Handler.data_root
    Handler.out_override = args.out.resolve() if args.out else None
    Handler.out_dir = Handler.out_override or (
        (base / "sidecars") if base is not None else Handler.data_root / "sidecars")
    Handler.out_dir.mkdir(parents=True, exist_ok=True)
    if Handler.out_override:
        print(f"--out is set: EVERY dataset writes sidecars to "
              f"{Handler.out_override}")
    if args.demo:
        print("DEMO MODE — automated landmarking is live, saving is disabled.")
    if auth.is_configured():
        print(f"automated landmarking is LOCKED (passphrase set in "
              f"{auth.auth_path()})")
    else:
        print("automated landmarking is OPEN on this machine — "
              "`--set-password` to gate it.")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Labeling server: http://localhost:{args.port}/")
    for n in sorted(Handler.datasets):
        mark = "*" if n == Handler.default_dataset else " "
        print(f"  {mark} {n:12} {len(list_images(Handler.datasets[n] / 'lateral')):4d} images")

    # An empty list is the normal first run: photographs are not in the
    # repository, they are large and they belong to the collection. Say what to
    # do about it rather than leaving a blank page unexplained.
    if not Handler.datasets:
        print()
        print("  No studies yet. Open the page and use the dataset menu:")
        print("      Add folder  →  choose a folder of photographs")
        print("  It becomes a study named after that folder. Nothing else to set up.")
        print()
    elif len(list_images(Handler.images_dir / "lateral")) == 0:
        print()
        print(f"  {Handler.default_dataset} has no photographs yet. Hover it in the")
        print("  dataset menu and choose \"+ photos\", or drag a folder onto the page.")
        print(f"  ({len(list(Handler.out_dir.glob('*.json')))} sidecars are present.)")
        print()
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
