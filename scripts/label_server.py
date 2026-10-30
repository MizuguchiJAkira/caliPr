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

from fish_morpho import auth, components, paths, schemes  # noqa: E402
from fish_morpho import image_identity  # noqa: E402
from fish_morpho import darwin_core  # noqa: E402
from fish_morpho import repeatability  # noqa: E402
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
    # The Cornell rig's crops carry a _L or _F suffix; a folder someone dropped
    # on the page does not, and its files are named for the fish alone. Auto-label
    # has always accepted both, and this did not -- so on a study named the plain
    # way, everything that asks a question about a fish's photograph here (where
    # the head-on view sits, which file to serve) answered "no such fish".
    folder = images_dir / view
    suffix = "_L" if view == "lateral" else "_F"
    for p in list_images(folder).values():
        if p.stem in (f"{fid}{suffix}", fid):
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


def _version(path: Path | None) -> int | None:
    """The file's modification time in nanoseconds, or None if it is absent."""
    try:
        return path.stat().st_mtime_ns if path is not None else None
    except OSError:
        return None


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


def _read_json(path: Path):
    """A JSON file's contents, or None if it is missing or unreadable."""
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except Exception:
        return None


def _read_ruler(photo: Path) -> dict:
    """The photograph's scale, from the numbers along its ruler: ``{"px_per_mm",
    "numbers", "notes"}``, or ``{"error"}`` where none is found."""
    try:
        import cv2

        from fish_morpho.ruler_calibration import detect_numbered_scale

        r = detect_numbered_scale(cv2.imread(str(photo)))
        return {"px_per_mm": r["px_per_mm"], "numbers": r["numbers"], "notes": r["notes"]}
    except Exception as exc:                       # no numbered ruler: the scale is set by hand
        return {"error": str(exc)}


#: GBIF's backbone hangs the fish orders straight from the phylum, with no class
#: (Actinopterygii's own key, 204, matches no occurrence), so a fish order is a
#: chordate order with no class -- or, should GBIF give them one, one of these.
CHORDATA = 44
FISH_CLASSES = {"Actinopterygii", "Petromyzonti", "Cephalaspidomorphi", "Myxini", "Elasmobranchii", "Holocephali"}


def _drainage_fishes(web, dc, wkt: str, names) -> list[dict]:
    """Every fish species with museum specimens inside the outline (GBIF,
    preserved specimens), by its current name, with its family, common name, the
    names it went by and how many specimens: ``[{"key", "species", "gbif_name",
    "common", "family", "chain", "records"}]``, most records first."""
    import concurrent.futures

    search = f"{dc.GBIF}/occurrence/search"
    base = [("geometry", wkt), ("basisOfRecord", "PRESERVED_SPECIMEN"), ("limit", "0")]
    d = web.get(dc.url(search, base + [("phylumKey", str(CHORDATA)), ("facet", "orderKey"), ("facetLimit", "300")]))
    orders = [c["name"] for c in (d.get("facets") or [{}])[0].get("counts", [])]

    def is_fish(order_key):
        o = web.get(f"{dc.GBIF}/species/{order_key}")
        return o.get("class") is None or o.get("class") in FISH_CLASSES

    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        fish_orders = [o for o, fish in zip(orders, pool.map(is_fish, orders)) if fish]
    if not fish_orders:
        return []
    d = web.get(dc.url(search, base + [("facet", "speciesKey"), ("facetLimit", "1000")]
                       + [("orderKey", o) for o in fish_orders]))
    counts = (d.get("facets") or [{}])[0].get("counts", [])

    def one(c):
        sp = web.get(f"{dc.GBIF}/species/{c['name']}")
        gbif_name = sp.get("canonicalName") or sp.get("scientificName") or str(c["name"])
        t = names.resolve(gbif_name)
        return {"key": int(c["name"]), "species": t.get("current") or gbif_name, "gbif_name": gbif_name,
                "common": t.get("common"), "family": t.get("family") or sp.get("family"),
                "chain": [x["name"] for x in t.get("chain") or []], "records": c["count"]}

    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        return sorted(pool.map(one, counts), key=lambda f: -f["records"])


#: One read-and-rewrite of a study's small JSON files at a time.
_FILE_LOCK = threading.Lock()


def _update_json(path: Path, change, indent: int | None = 2) -> dict:
    """Read a JSON file, ``change`` its contents in place, write them back: under one
    lock, re-read just before the write, so two requests at once cannot each write
    over the other's entry (two drainages looked up together kept only the second)."""
    with _FILE_LOCK:
        d = _read_json(path) or {}
        change(d)
        path.write_text(json.dumps(d, indent=indent) + ("\n" if indent else ""))
        return d


#: One script loaded at a time: a request that found a half-loaded module in
#: sys.modules used it before its functions were defined (two drainages looked up
#: at once, and one failed with "no attribute 'Web'").
_SCRIPT_LOCK = threading.Lock()


def _script(name: str):
    """scripts/<name>.py as a module, loaded once."""
    import importlib.util

    with _SCRIPT_LOCK:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                del sys.modules[name]
                raise
        return sys.modules[name]


def _harvester():
    """scripts/harvest_gbif_photos.py, whose rules say which photograph of a
    specimen is usable."""
    return _script("harvest_gbif_photos")


#: What each USGS status says of a fish in a drainage it is not native to, the
#: strongest first: a fish both stocked and established there is established.
NAS_STATUS = ("established", "stocked", "collected", "unknown", "failed")


def _drainage_introduced(web, dc, huc8: str, names) -> dict:
    """The fish USGS records as not native to a drainage (Nonindigenous Aquatic
    Species database, by HUC8): current name -> ``{"status", "records",
    "nas_name", "common"}``, the status the strongest any record gives. USGS says
    whether a fish was introduced, not whether it does harm."""
    d = web.get(dc.url(dc.NAS, {"huc8": huc8}))
    by: dict[str, dict] = {}
    for r in d.get("results") or []:
        if r.get("group") != "Fishes" or not r.get("scientificName"):
            continue
        e = by.setdefault(r["scientificName"], {"statuses": set(), "records": 0, "common": r.get("commonName")})
        e["statuses"].add((r.get("status") or "unknown").lower())
        e["records"] += 1
    out = {}
    for nas_name, e in by.items():
        t = names.resolve(nas_name)
        cur = t.get("current") or nas_name
        status = next((st for st in NAS_STATUS if st in e["statuses"]), sorted(e["statuses"])[0])
        out[cur] = {"status": status, "records": e["records"], "nas_name": nas_name,
                    "common": t.get("common") or e["common"], "gbif_key": t.get("gbif_key"),
                    "chain": [c["name"] for c in t.get("chain") or []]}
    return out


def _drainage_tool():
    """scripts/drainage_candidates.py, which looks drainages and their species up."""
    return _script("drainage_candidates")


#: What a batch photograph's kept fish hold; older ones are found and landmarked
#: again. 2: each point's confidence, which the labeller reviews by.
BATCH_CACHE_VERSION = 2
#: Two boxes overlapping this much (intersection over union) are the same fish.
SAME_FISH = 0.7


def _overlap(a, b) -> float:
    """Intersection over union of two [x0, y0, x1, y1] boxes; 0 for anything else."""
    try:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0
    except (TypeError, IndexError):
        return 0.0


def _isolate_fish(photo: Path, fish: list[dict], i: int, out: Path) -> None:
    """Fish ``i``'s rectangle of a batch photograph, with every other fish in it
    painted over in the background's colour, so the landmark model -- which looks
    for one fish -- cannot take a point from the fish beside it."""
    import numpy as np
    from PIL import Image, ImageDraw

    with Image.open(photo) as im:
        im = im.convert("RGB")
        x0, y0, x1, y1 = fish[i]["box"]
        crop = im.crop((x0, y0, x1, y1))
    w, h = crop.size
    shift = lambda poly: [(x - x0, y - y0) for x, y in poly]  # noqa: E731
    pad = max(2, round(0.01 * max(w, h)))                     # fins reach past the outline
    every = Image.new("L", (w, h), 0)
    draw_every = ImageDraw.Draw(every)
    for f in fish:
        draw_every.polygon(shift(f["outline"]), fill=255, outline=255, width=pad)
    px = np.asarray(crop)
    bg = tuple(int(v) for v in np.median(px[np.asarray(every) == 0], axis=0)) \
        if (np.asarray(every) == 0).any() else (255, 255, 255)
    others = Image.new("L", (w, h), 0)
    draw_others = ImageDraw.Draw(others)
    for j, f in enumerate(fish):
        if j != i:
            draw_others.polygon(shift(f["outline"]), fill=255, outline=255, width=pad)
    mine = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mine).polygon(shift(fish[i]["outline"]), fill=255, outline=255, width=pad)
    hide = np.logical_and(np.asarray(others) > 0, np.asarray(mine) == 0)
    px = px.copy()
    px[hide] = bg
    Image.fromarray(px).save(out, quality=92)


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
        # The same for landmarks: collected, placed by hand, never offered by
        # Auto-label -- for points where the model's guesses cost more to check
        # than to click.
        "exclude_predicted_keypoints": set(
            prof.get("exclude_predicted_keypoints") or []),
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
        # The ruler span a fish starts with until it is given its own, per view.
        # The built-in 50 mm lateral default was right for the rigs it came from
        # and wrong where people mark a centimetre -- which scaled three fish 5x.
        "known_mm_default": _span_defaults(prof.get("known_mm_default")),
        # The specimens to label first, when the study has chosen them itself --
        # by species, collection and size, say -- rather than leaving it to the
        # rotation over lot numbers, which a study without lot numbers in its
        # filenames cannot use.
        "label_first": [f for f in (prof.get("label_first") or []) if isinstance(f, str)],
        # Specimens the model is never run on, so their landmarks are placed with no
        # guess to accept. Set beside fish that are corrected from the model's points,
        # they show how far accepting those points moves the shapes.
        "hand_only": [f for f in (prof.get("hand_only") or []) if isinstance(f, str)],
        # The study a batch photograph's fish are compared with: whose hand-labelled
        # fish are the reference species. This study's own when not named.
        "reference_study": str(prof.get("reference_study") or ""),
        # A study made of fish cut from another study's batch photographs, to have
        # their landmarks put right: that study's name.
        "batch_of": str(prof.get("batch_of") or ""),
        # A set of identified, hand-landmarked specimens that batch photographs are
        # compared with -- downloaded as a part of caliPr, or made by a study of its own.
        "reference_set": bool(prof.get("reference_set")),
        # Groups (by the start of their name) whose mouth is terminal: the premaxilla
        # tip and the dentary tip are one point, so a reference fish without the
        # first takes the second.
        "terminal_mouth": [g for g in (prof.get("terminal_mouth") or []) if isinstance(g, str)],
        # Counts this study takes on each fish -- scales along a row, fin rays -- as
        # whole numbers typed in, beside the landmarks: [{"name", "label", "hint"}].
        "counts": [c for c in (prof.get("counts") or [])
                   if isinstance(c, dict) and re.fullmatch(r"[a-z0-9_]+", str(c.get("name", "")))],
        "note": prof.get("note", ""),
    }


def _span_defaults(raw) -> dict:
    """{"lateral": mm, "frontal": mm}, keeping only sane positive numbers."""
    out = {}
    for view, mm in (raw or {}).items() if isinstance(raw, dict) else ():
        try:
            mm = float(mm)
        except (TypeError, ValueError):
            continue
        if view in ("lateral", "frontal") and 0 < mm < 10000:
            out[view] = mm
    return out


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
        "known_mm_default": dict(profile.get("known_mm_default") or {}),
        "counts": [{"name": c["name"], "label": str(c.get("label") or c["name"]),
                    "hint": str(c.get("hint") or "")} for c in profile.get("counts") or []],
        "batch_of": profile.get("batch_of", ""),
        # Landmarks placed by hand here: Auto-label does not offer them, and a
        # guess for one still sitting unreviewed in a draft is dropped on opening.
        "exclude_predicted_keypoints": sorted(profile.get("exclude_predicted_keypoints") or ()),
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


#: Which fins Auto-label offers an outline for. Kept in step with
#: ``predict_worker.OUTLINED_FINS``, which is where the reasoning lives; repeated
#: here because the server decides what to ask for and must not need torch to do it.
OUTLINED_FINS = ("pectoral", "anal")

#: Hand labelling works without any of this, so it is installed separately.
NO_TRAINING_STACK = (
    "Auto-label needs PyTorch, DeepLabCut and the trained models, which are installed "
    "separately from the labeler: about 2.3 GB to download. Hand labelling and export "
    "work without them.")


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
    _PYTHONS = (paths.TRAINING_PYTHON,
                _ROOT / ".venv" / "bin" / "python")

    @classmethod
    def _start(cls):
        worker = _ROOT / "scripts" / "predict_worker.py"
        exe = next((p for p in cls._PYTHONS if p.is_file()), None)
        if exe is None or not worker.is_file():
            return {"error": NO_TRAINING_STACK, "install": True}
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
                return {"error": NO_TRAINING_STACK, "install": True}
            # A missing model is what the installer's last step fetches.
            return {"error": err, "missing_model": bool(info.get("missing_model")),
                    "install": bool(info.get("missing_model"))}
        cls._proc, cls._info = proc, info
        return info

    @classmethod
    def predict(cls, image: Path, polygons: bool = False,
                emit_polygons: bool = True, view: str = "lateral",
                crop: list[int] | None = None,
                fins: list[str] | None = None,
                landmarks: list[str] | None = None) -> dict:
        with cls._lock:                      # one request at a time down one pipe
            # A model pinned since the worker started is used from the next
            # request on, without restarting the labeler.
            now = current_model_label("lateral")
            if (cls._proc is not None and cls._proc.poll() is None and now
                    and cls._info.get("model") not in (None, now)):
                cls._proc.kill()
                cls._proc = None
            if cls._proc is None or cls._proc.poll() is not None:
                started = cls._start()
                if "error" in started:
                    return {"ok": False, **started}
            return cls._ask({"image": str(image), "polygons": polygons,
                             "emit_polygons": emit_polygons, "view": view, "crop": crop,
                             "fins": fins or [], "landmarks": landmarks})

    @classmethod
    def detect(cls, image: Path) -> dict:
        """Every fish in a photograph of several: a rectangle and an outline each."""
        with cls._lock:
            if cls._proc is None or cls._proc.poll() is not None:
                started = cls._start()
                if "error" in started:
                    return {"ok": False, **started}
            return cls._ask({"detect": True, "image": str(image)})

    @classmethod
    def _ask(cls, request: dict) -> dict:
        """One request down the pipe and its answer. The caller holds the lock."""
        try:
            cls._proc.stdin.write(json.dumps(request) + "\n")
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


def current_model(view: str = "lateral"):
    """(project, label) of the model Auto-label would load now, or None."""
    import predict_landmarks as pl
    return pl.current_model(view)


def current_model_label(view: str = "lateral") -> str | None:
    found = current_model(view)
    return found[1] if found else None


def union_model_in_use() -> bool:
    """Whether Auto-label's side-view model is the cross-species one."""
    import predict_landmarks as pl
    found = current_model("lateral")
    return bool(found) and found[0].parent.name == pl.UNION_PROJECT


class AutolabelInstall:
    """``install_autolabel.py``, run from the page and watched by it.

    One install at a time, whichever tab asked. Its output goes to a log beside
    the environment it builds, and the page shows the latest line as progress.
    """

    _proc = None
    _lock = threading.Lock()
    LOG = "autolabel-install.log"

    @classmethod
    def log_path(cls) -> Path:
        return paths.MODELS / cls.LOG

    @classmethod
    def command(cls) -> list[str]:
        return [sys.executable, "-u", str(_ROOT / "scripts" / "install_autolabel.py")]

    @staticmethod
    def installed() -> bool:
        return paths.TRAINING_PYTHON.is_file()

    @classmethod
    def start(cls) -> dict:
        with cls._lock:
            if cls._proc is not None and cls._proc.poll() is None:
                return cls.status()
            paths.MODELS.mkdir(parents=True, exist_ok=True)
            log = open(cls.log_path(), "w")
            cls._proc = subprocess.Popen(
                cls.command(), stdout=log, stderr=subprocess.STDOUT, cwd=str(paths.MODELS))
            log.close()
            # The predictor that failed to start is retried on the next request.
            Predictor._proc = None
        return cls.status()

    @classmethod
    def status(cls) -> dict:
        proc = cls._proc
        lines = []
        try:
            lines = [ln for ln in cls.log_path().read_text(errors="replace").splitlines()
                     if ln.strip()]
        except OSError:
            pass
        running = proc is not None and proc.poll() is None
        return {"installed": cls.installed(),
                "running": running,
                "finished": proc is not None and not running,
                "ok": proc is not None and not running and proc.returncode == 0,
                "line": lines[-1].strip() if lines else "",
                "tail": lines[-12:] if proc is not None and not running else []}



class BatchInstall(AutolabelInstall):
    """``install_component.py batch``: the reference set batch identification compares
    fish with, downloaded into the data folder and watched the same way."""

    _proc = None
    _lock = threading.Lock()
    LOG = "batch-install.log"

    @classmethod
    def command(cls) -> list[str]:
        return [sys.executable, "-u", str(_ROOT / "scripts" / "install_component.py"), "batch",
                "--data", str(Handler.data_root or paths.WORK / "data")]

    @staticmethod
    def installed() -> bool:
        return batch_installed()


def autolabel_installed() -> bool:
    """The environment the model runs in, and a trained side-view model."""
    return paths.TRAINING_PYTHON.is_file() and current_model("lateral") is not None


def reference_sets() -> list[str]:
    """Studies marked as a reference set (``"reference_set": true`` in schema.json),
    with photographs or without: a downloaded set holds landmarks and credits only."""
    dirs = dict(Handler.datasets)
    root = Handler.data_root
    if root is not None and root.is_dir():
        dirs.update({d.name: d for d in root.iterdir() if d.is_dir()})
    return sorted(n for n, d in dirs.items()
                  if (d / "schema.json").is_file() and load_profile(d).get("reference_set"))


def batch_installed() -> bool:
    """Batch identification runs on Auto-label and a reference set to compare with."""
    return autolabel_installed() and bool(reference_sets())


def components_state() -> dict:
    """Each optional part: installed on this machine, and released for download."""
    auto = autolabel_installed()
    refs = reference_sets()
    return {
        "autolabel": {"installed": auto, "released": components.released("autolabel")},
        "batch": {"installed": auto and bool(refs), "released": components.released("batch"),
                  "needs_autolabel": not auto, "reference_sets": refs},
    }


#: What the batch routes answer on a machine without batch identification.
BATCH_MISSING = ("Batch identification is not installed. It needs Auto-label and a reference "
                 "set of identified, landmarked specimens, which are downloaded separately "
                 "from the labeler.")

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


# What identifies a calibration: while these are unchanged it is the same scale.
_CAL_IDENTITY = ("mode", "px_per_mm", "point_a", "point_b", "known_mm")


def _carry_unmanaged(data: dict, before: dict) -> None:
    """Keep what other tools wrote inside the records the labeler rebuilds.

    The page rebuilds each review record (metadata.assist, assist_frontal) and
    each calibration from the fields it knows, so a field another tool added
    inside one was erased by the next save of that fish: 61 head-on records say
    their mouth corners were approved in bulk, and at what confidence, and 61
    head-on scales that the ruler detector placed them. Every change saving
    itself would erase them from every fish touched. They are carried across
    while the record is the same one -- the same prediction, the same scale. A
    new prediction or a re-placed ruler is a new record, and starts clean.
    """
    meta, had = data.get("metadata") or {}, before.get("metadata") or {}
    for key in ("assist", "assist_frontal"):
        new, old = meta.get(key), had.get(key)
        if (isinstance(new, dict) and isinstance(old, dict)
                and (new.get("model"), new.get("predicted_at"))
                == (old.get("model"), old.get("predicted_at"))):
            for k, v in old.items():
                new.setdefault(k, v)
    for view in ("lateral", "frontal"):
        new = (data.get(view) or {}).get("calibration")
        old = (before.get(view) or {}).get("calibration")
        if (isinstance(new, dict) and isinstance(old, dict)
                and all(new.get(k) == old.get(k) for k in _CAL_IDENTITY)):
            for k, v in old.items():
                new.setdefault(k, v)


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
        suggested = (set(profile["label_first"]) & set(all_ids) if profile.get("label_first")
                     else suggested_subset(all_ids))
        hand_only = set(profile.get("hand_only") or ())

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
                "hand_only": fid in hand_only,
                # Size of each view's image as it is on disk now. A draft records
                # the size it was made on; the labeler discards any draft whose
                # size no longer matches, because its coordinates cannot fit.
                "sizes": {"lateral": current_size(path),
                          "frontal": current_size(path if single else fro.get(fname))},
                # When each view's image file last changed. The labeler puts it in
                # the image's address: a browser reuses an image it has already
                # shown at the same address, whatever the cache headers say, so a
                # photograph turned in place would otherwise go on showing the old.
                "image_version": {"lateral": _version(path),
                                  "frontal": _version(path if single else fro.get(fname))},
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

    #: What a study's own example consists of; nothing else is served from there.
    REFERENCE_FILES = ("reference.json", "reference_base.jpg", "reference_annot.jpg")

    def _set_reference(self):
        """Make a labelled fish this study's example in the reference panel.

        Built from the fish's SAVED landmarks into ``<study>/reference/``, so the
        example is the study's own -- its species, its scheme, its convention --
        rather than the one brook trout every study used to be shown.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            fid = str(json.loads(self.rfile.read(n) or b"{}").get("id", ""))
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        if not fid or fid != Path(fid).name or fid.startswith("."):
            return self._send(400, {"ok": False, "error": "no such specimen"})
        if not (self.out_dir / f"{fid}.json").is_file():
            return self._send(400, {"ok": False, "error": "Save this fish first: the example "
                                                         "is made from its saved landmarks."})
        img = view_image(self.images_dir, fid, "lateral")
        if img is None:
            return self._send(404, {"ok": False, "error": f"no photograph for “{fid}”"})
        prof = load_profile(self.images_dir)
        out = self.images_dir / "reference"
        r = subprocess.run(
            [sys.executable, str(_ROOT / "scripts/make_reference.py"), "--specimen", fid,
             "--sidecars", str(self.out_dir), "--image", str(img), "--out", str(out),
             "--skip-keypoints", ",".join(sorted(prof.get("exclude_keypoints") or ())),
             "--skip-polygons", ",".join(sorted(prof.get("exclude_polygons") or ()))],
            capture_output=True, text=True, timeout=120, cwd=_ROOT)
        if r.returncode != 0:
            return self._send(400, {"ok": False, "error": (r.stderr or r.stdout).strip()[-400:]})
        return self._send(200, {"ok": True, "specimen": fid})

    def _reference_file(self, name: str):
        """One of this study's example files, or 404 when it has none."""
        if name not in self.REFERENCE_FILES:
            return self._send(404, {"error": "unknown file"})
        path = self.images_dir / "reference" / name
        if not path.is_file():
            return self._send(404, {"error": "this study has no example of its own"})
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if name.endswith(".json") else "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _set_known_mm_default(self):
        """Make a ruler span the study's default for one view.

        Kept in the study's schema.json, beside its other settings, so it holds for
        everyone labelling that study. A fish that already has a span keeps it.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            view, mm = str(data.get("view", "")), float(data.get("known_mm"))
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        if view not in ("lateral", "frontal") or not 0 < mm < 10000:
            return self._send(400, {"ok": False, "error": "give a view and a span in mm"})
        path = self.images_dir / "schema.json"
        prof = {}
        if path.is_file():
            try:
                prof = json.loads(path.read_text())
            except Exception:
                return self._send(500, {"ok": False, "error": "this study's schema.json "
                                                            "could not be read; nothing written"})
        spans = _span_defaults(prof.get("known_mm_default"))
        spans[view] = mm
        prof["known_mm_default"] = spans
        path.write_text(json.dumps(prof, indent=2) + "\n")
        return self._send(200, {"ok": True, "known_mm_default": spans})

    def _fish_ids(self) -> list[str]:
        """Every photograph's specimen id in this study, as the list names them."""
        out = []
        for name in sorted(list_images(self.images_dir / "lateral")):
            m = _ID_RE.match(Path(name).stem)
            out.append(m.group(1) if m else Path(name).stem)
        return out

    def _dwc(self):
        """The study's Darwin Core records, and what the filenames suggest.

        Suggestions are sent apart from the records: they are shown, and written
        only when someone accepts them.
        """
        ids = self._fish_ids()
        records = darwin_core.load(self.images_dir)
        suggestions = {}
        for fid in ids:
            s = darwin_core.suggest(fid, self.images_dir.name)
            s = {t: v for t, v in s.items() if not (records.get(fid) or {}).get(t)}
            if s:
                suggestions[fid] = s
        return self._send(200, {
            "terms": [{"term": t, "help": h} for t, h in darwin_core.TERMS],
            "ids": ids, "records": {f: r for f, r in records.items()},
            "suggestions": suggestions, "file": darwin_core.FILENAME})

    def _dwc_update(self):
        """Set or clear terms: ``{"changes": {fish_id: {term: value}}}``."""
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            changes = json.loads(self.rfile.read(n) or b"{}").get("changes") or {}
            assert isinstance(changes, dict)
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        known = set(self._fish_ids()) | set(darwin_core.load(self.images_dir))
        stray = sorted(set(changes) - known)
        if stray:
            return self._send(404, {"ok": False, "error": f"no photograph for “{stray[0]}”"})
        try:
            records = darwin_core.update(self.images_dir, changes)
        except ValueError as exc:
            return self._send(400, {"ok": False, "error": str(exc)})
        return self._send(200, {"ok": True, "records": records})

    def _dwc_import(self):
        """Read a CSV or Excel sheet into the study's records.

        ``{"name": file, "data": base64, "apply": bool}``. Without ``apply`` it
        only reports what would change -- which column named the photographs,
        which columns became which terms, what matched nothing -- so the person
        importing sees that before anything is written.
        """
        import base64
        n = int(self.headers.get("Content-Length", 0))
        if n > 30 * 1024 * 1024:
            return self._send(413, {"ok": False, "error": "that file is too large to be a "
                                                        "table of specimens"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            name = str(body.get("name") or "table.csv")
            rows = darwin_core.read_table(name, base64.b64decode(body.get("data") or ""))
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"could not read that file: {exc}"})
        if not rows:
            return self._send(400, {"ok": False, "error": "that file has no rows"})
        existing = darwin_core.load(self.images_dir)
        plan = darwin_core.plan_import(rows, self._fish_ids(), existing)
        plan["rows"] = len(rows)
        if not plan["key"] or not plan["named"]:
            return self._send(400, {"ok": False, "plan": plan, "error": (
                "No row in that file names a photograph in this study. It needs a "
                "column of specimen ids or filenames, e.g. fish_id or local_filename.")})
        if not body.get("apply"):
            return self._send(200, {"ok": True, "plan": plan, "applied": False})
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        records = darwin_core.update(self.images_dir, plan["changes"])
        return self._send(200, {"ok": True, "plan": plan, "applied": True, "records": records})

    @staticmethod
    def _round_brief(folder: Path) -> dict | None:
        """What the page may know about a blind round: never which fish is which."""
        r = repeatability.read_round(folder)
        if not r:
            return None
        return {"study": r["study"], "round": r.get("round"), "n": len(r["fish"]),
                "done": len(repeatability.done_codes(folder)), "operator": r.get("operator", ""),
                "created": r.get("created", ""), "same_as": r.get("same_as")}

    def _relabel_rounds(self):
        """The open study's blind re-label rounds and how far each has got."""
        here = repeatability.read_round(self.images_dir)
        study = (self.images_dir.parent / here["study"]) if here else self.images_dir
        rounds = []
        for r in repeatability.rounds_of(study):
            rounds.append({**self._round_brief(r["folder"]), "name": r["folder"].name})
        labelled = sum(1 for f in (study / "sidecars").glob("*.json")) \
            if (study / "sidecars").is_dir() else 0
        return self._send(200, {"study": study.name, "in_round": self._round_brief(self.images_dir),
                                "rounds": rounds, "labelled": labelled})

    def _relabel_new(self):
        """Set up a blind re-label round: ``{"n", "operator", "same_as"}``."""
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            count = int(body.get("n") or 0)
            same = body.get("same_as")
            same = int(same) if same not in (None, "") else None
            operator = str(body.get("operator") or "")[:40]
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        try:
            r = repeatability.create_round(self.images_dir, count, operator, same_as=same)
        except (ValueError, OSError) as exc:
            return self._send(400, {"ok": False, "error": str(exc)})
        if Handler.data_root is not None:
            Handler.datasets = discover_datasets(Handler.data_root) or Handler.datasets
        name = Path(r["folder"]).name
        Handler.datasets.setdefault(name, Path(r["folder"]))
        return self._send(200, {"ok": True, "name": name, "round": r["round"], "n": r["n"]})

    def _relabel_stats(self):
        """ICC and %ME per trait so far, computed in a subprocess like every export."""
        import tempfile
        here = repeatability.read_round(self.images_dir)
        study = (self.images_dir.parent / here["study"]) if here else self.images_dir
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "me.json"
            r = subprocess.run(
                [sys.executable, str(_ROOT / "scripts/measurement_error.py"), "--dataset",
                 study.name, "--data-root", str(study.parent), "--json", str(out)],
                capture_output=True, text=True, timeout=300, cwd=_ROOT)
            if r.returncode != 0 or not out.is_file():
                return self._send(500, {"ok": False, "error": (r.stderr.strip().splitlines()
                                                               or ["could not compute"])[-1]})
            return self._send(200, {"ok": True, **json.loads(out.read_text())})

    def _straighten_preview(self):
        """The fish as its midline straightens it: ``{"id", "block"}`` -> a JPEG.

        ``block`` is the lateral view as it stands in the labeler, saved or not --
        its midline, landmarks and outlines. Drawn in a subprocess, as every
        image operation here is, and nothing is written to the study.
        """
        import base64
        import tempfile
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            fid, block = str(body.get("id") or ""), body.get("block") or {}
            assert isinstance(block, dict)
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        if not fid or fid != Path(fid).name or fid.startswith("."):
            return self._send(400, {"ok": False, "error": f"no such specimen “{fid}”"})
        photo = view_image(self.images_dir, fid, "lateral")
        if photo is None:
            return self._send(404, {"ok": False, "error": f"no photograph for “{fid}”"})
        with tempfile.TemporaryDirectory() as tmp:
            lab, out, info = Path(tmp) / "labels.json", Path(tmp) / "s.jpg", Path(tmp) / "i.json"
            lab.write_text(json.dumps(block))
            r = subprocess.run(
                [sys.executable, str(_ROOT / "scripts/straighten_preview.py"), "--image",
                 str(photo), "--labels", str(lab), "--out", str(out), "--info", str(info)],
                capture_output=True, text=True, timeout=120, cwd=_ROOT)
            if r.returncode != 0 or not out.is_file():
                return self._send(400, {"ok": False, "error": (r.stderr.strip().splitlines()
                                                               or ["could not straighten"])[-1]})
            return self._send(200, {"ok": True, "info": json.loads(info.read_text()),
                                    "jpeg": base64.b64encode(out.read_bytes()).decode()})

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

    def _new_scheme(self):
        """Start a landmark scheme of your own, optionally from one that exists.

        A scheme defined in code is a published protocol and is never edited here:
        adding a point to it would make this lab's files stop matching everyone
        else's under that protocol's name, which is the one thing a named scheme
        exists to prevent. Copying it says plainly that what follows is a
        different scheme, and the copy records what it came from.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"bad payload: {exc}"})
        title = str(data.get("title") or "").strip()
        if not 1 <= len(title) <= 80:
            return self._send(400, {"ok": False, "error": "give the scheme a name"})
        base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "scheme"
        name, i = base, 2
        while name in schemes.all_schemes() or name == schemes.DEFAULT:
            name, i = f"{base}_{i}", i + 1

        src = data.get("copy_from")
        landmarks: list = []
        note = "Defined in this labeler."
        source = ""
        if src and src != schemes.DEFAULT:
            got = schemes.get(src)
            if not got:
                return self._send(404, {"ok": False, "error": f"no scheme {src!r}"})
            landmarks = [tuple(k) for k in got["landmarks"]]
            source = f"copied from {got['title']}"
            note = (f"Started as a copy of {got['title']} and diverges from it as soon as "
                    f"a landmark is added, so its files are not that protocol's files.")
        elif src == schemes.DEFAULT:
            # caliPr's own landmarks as a starting point. The copy is a plain list
            # of points like any other scheme: no trait follows it across.
            landmarks = [(k.name, k.name.replace("_", " "), "caliPr", k.labeling_hint or "")
                         for k in KEYPOINTS if k.view == View.LATERAL]
            source = "copied from caliPr's own landmarks"
            note = ("Started as a copy of caliPr's landmarks. No trait is computed for a "
                    "scheme, so these are coordinates only.")
        doc = {"title": title, "source": source, "note": note,
               "view": "lateral", "landmarks": landmarks}
        try:
            schemes.save(name, doc)
        except Exception as exc:
            return self._send(500, {"ok": False, "error": f"could not write the scheme: {exc}"})
        return self._send(200, {"ok": True, "scheme": name,
                                "schemes": schemes.listing()})

    def _edit_scheme_keypoint(self, scheme: str, data: dict):
        """Append a landmark to a scheme of your own, or rename one.

        **Appended, never inserted.** TPS identifies a landmark by its row, so
        putting one in the middle would silently redefine every file already
        exported under this scheme -- including other people's, if they follow it.
        Adding at the end leaves rows 1..N meaning exactly what they meant.

        Removal is refused for the same reason: taking one out renumbers
        everything after it.
        """
        doc = schemes.get(scheme)
        if not doc:
            return self._send(404, {"ok": False, "error": f"no scheme {scheme!r}"})
        if not doc.get("editable"):
            return self._send(409, {
                "ok": False, "builtin": True,
                "error": f"{doc['title']} is a published protocol, so it is not edited here. "
                         f"Make a copy of it and add to that."})
        landmarks = [tuple(k) for k in doc["landmarks"]]
        action = data.get("action")
        if action == "add":
            label = str(data.get("label") or "").strip()
            if not 1 <= len(label) <= 60:
                return self._send(400, {"ok": False, "error": "give the landmark a name"})
            base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "landmark"
            taken = {k[0] for k in landmarks}
            name, i = base, 2
            while name in taken:
                name, i = f"{base}_{i}", i + 1
            landmarks.append((name, label, str(data.get("group") or "added"),
                              str(data.get("hint") or "")))
        elif action == "rename":
            name = str(data.get("name") or "")
            label = str(data.get("label") or "").strip()
            if not any(k[0] == name for k in landmarks):
                return self._send(404, {"ok": False, "error": f"no landmark {name!r}"})
            if not 1 <= len(label) <= 60:
                return self._send(400, {"ok": False, "error": "give the landmark a name"})
            landmarks = [(k[0], label, k[2], k[3]) if k[0] == name else k for k in landmarks]
        elif action == "remove":
            return self._send(409, {
                "ok": False,
                "error": "a scheme numbers its landmarks, and TPS identifies one by its "
                         "row — removing this one would renumber every landmark after it "
                         "and silently redefine every file already exported. Rename it "
                         "instead, or start a new scheme."})
        else:
            return self._send(400, {"ok": False, "error": f"unknown action {action!r}"})
        schemes.save(scheme, dict(doc, landmarks=landmarks))
        return self._send(200, {"ok": True,
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
        # A study following a scheme collects that scheme's landmarks and nothing
        # else, so a landmark added here has to go into the scheme or it would be
        # written down and never asked for.
        prof0 = load_profile(self.images_dir)
        if schemes.get(prof0.get("scheme")):
            return self._edit_scheme_keypoint(prof0["scheme"], data)
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

    def _build_export(self, kind: str, opts: dict | None = None) -> dict:
        """Write an export under ``results/<dataset>/`` and say how to show it.

        Returns the file to download (``file``, ``name``, ``type``) and what to open
        on this computer (``show``, ``how``). Raises :class:`ExportError` with a
        message the annotator can read, and ``KeyError`` for an unknown kind.
        """
        import zipfile

        ds = self.images_dir.name
        root = paths.WORK / "results" / ds

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
        # A study on another landmark scheme exports its landmark coordinates
        # as its measurements (scripts/scheme_export.py); the same call serves it.
        if kind == "measurements":
            # What the export preview chose: columns to leave out, and whether to
            # keep only the fish with a value in every remaining column.
            opts = opts or {}
            codes = [str(c) for c in opts.get("leave_out") or []
                     if re.fullmatch(r"[A-Za-z0-9_]+", str(c))]
            # The CSV is what R reads. The workbook is still written beside it,
            # because its About, QC and Validation sheets have no place in a CSV.
            # Shown in the file browser rather than opened: a spreadsheet app
            # opening a CSV reinterprets it, and saving it back is how a column
            # of IDs becomes dates.
            out = root / "measurements.xlsx"
            csv_out = root / "measurements.csv"
            run([str(_ROOT / "scripts/export_measurements.py"), "--dataset", ds,
                 "--images", str(self.images_dir / "lateral"),
                 "--labels", str(self.out_dir), "--out", str(out), "--csv", str(csv_out),
                 *(["--leave-out", ",".join(codes)] if codes else []),
                 *(["--complete-only"] if opts.get("complete_only") else [])], 600)
            if not csv_out.is_file() or not out.is_file():
                raise ExportError("the measurements were not written")
            # Both are always written; the preview says which one to hand over.
            if opts.get("format") == "xlsx":
                return {"file": out, "name": f"{ds}_measurements.xlsx", "type": XLSX,
                        "show": out, "how": "open"}
            return {"file": csv_out, "name": f"{ds}_measurements.csv",
                    "type": "text/csv; charset=utf-8", "show": csv_out, "how": "reveal"}

        # A fish with a midline goes out straightened unless the export says not.
        straight = [] if (opts or {}).get("straighten") is False else ["--straighten"]
        if kind == "tps" and (opts or {}).get("format") == "per_specimen":
            # One CSV per specimen in ImageJ's layout, the folder a Procrustes
            # workflow in R reads in as a series. Shown as the folder itself.
            units = "mm" if (opts or {}).get("units") == "mm" else "cm"
            folder = root / "landmarks_by_specimen"
            run([str(_ROOT / "scripts/export_tps.py"), "--sidecars", str(self.out_dir),
                 "--images", str(self.images_dir / "lateral"),
                 "--schema-dir", str(self.images_dir), "--out", str(root / "tps"),
                 "--per-specimen", str(folder), "--units", units, *straight], 600)
            files = sorted(f for f in folder.iterdir() if f.is_file()) if folder.is_dir() else []
            if not any(f.suffix == ".csv" and f.name not in ("specimens.csv", "landmark_key.csv")
                       for f in files):
                raise ExportError("nothing labelled yet")
            zp = zipped(f"{ds}_landmarks_by_specimen.zip",
                        [(f, f"{ds}_landmarks_by_specimen/{f.name}") for f in files])
            return {"file": zp, "name": zp.name, "type": "application/zip",
                    "show": folder, "how": "open"}

        if kind == "tps":
            run([str(_ROOT / "scripts/export_tps.py"), "--sidecars", str(self.out_dir),
                 "--images", str(self.images_dir / "lateral"),
                 "--schema-dir", str(self.images_dir), "--out", str(root / "tps"),
                 *straight], 600)
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

    def _export_preview(self):
        """The measurements as they would be exported, for the export preview.

        Built to a scratch workbook and read back, so the preview shows exactly
        what the file will hold: every trait column with its count of blanks,
        and every specimen's values. Nothing under results/ is touched.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            js = Path(tmp) / "preview.json"
            try:
                r = subprocess.run(
                    [sys.executable, str(_ROOT / "scripts/export_measurements.py"),
                     "--dataset", self.images_dir.name,
                     "--images", str(self.images_dir / "lateral"),
                     "--labels", str(self.out_dir), "--out", str(Path(tmp) / "m.xlsx"),
                     "--preview-json", str(js)],
                    capture_output=True, text=True, timeout=600, cwd=_ROOT)
            except subprocess.TimeoutExpired:
                return self._send(200, {"ok": False, "error": "the preview timed out"})
            if r.returncode != 0 or not js.is_file():
                return self._send(200, {"ok": False,
                                        "error": (r.stderr or r.stdout)[-800:]})
            data = json.loads(js.read_text())
        return self._send(200, {"ok": True, **data})

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
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            opts = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            opts = {}
        try:
            out = self._build_export(kind, opts if isinstance(opts, dict) else {})
        except KeyError:
            return self._send(404, {"ok": False, "error": f"unknown export {kind!r}"})
        except subprocess.TimeoutExpired:
            return self._send(500, {"ok": False, "error": "export timed out"})
        except Exception as exc:
            return self._send(500, {"ok": False, "error": str(exc)})
        try:
            rel = str(out["show"].relative_to(paths.WORK))
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

    def _remove_specimen(self):
        """Take photographs out of a study, with everything that belongs to them.

        ``{"id": fish}`` or ``{"ids": [fish, ...]}``. Moved, never deleted: each
        photograph (and a separate head-on crop, where the study has one), its
        saved labels and its cached predictions go to one folder,
        ``data/.trash/<study>--<fish or N-photos>--<time>/``, keeping their paths
        inside the study, so restoring is moving that folder's contents back.
        Every id is checked before anything moves: one bad id moves nothing.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            ids = body.get("ids") if body.get("ids") is not None else [body.get("id", "")]
            ids = [str(i) for i in ids]
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        if not ids:
            return self._send(400, {"ok": False, "error": "no specimens given"})
        study = self.images_dir
        plan: list[Path] = []
        labelled = 0
        for fid in ids:
            # A specimen id names files, so it may never be a path.
            if not fid or fid != Path(fid).name or fid.startswith("."):
                return self._send(400, {"ok": False, "error": f"no such specimen “{fid}”"})
            found: list[Path] = []
            for view in ("lateral", "frontal"):
                p = view_image(study, fid, view)
                if p is not None and p.parent == study / view and p not in found:
                    found.append(p)
            if not found:
                return self._send(404, {"ok": False, "error": f"no photograph for “{fid}”"})
            sidecar = self.out_dir / f"{fid}.json"
            if sidecar.is_file():
                found.append(sidecar)
                labelled += 1
            for cache in (study / "sidecars_auto" / f"{fid}.json",
                          study / "sidecars_auto" / "frontal" / f"{fid}.json"):
                if cache.is_file():
                    found.append(cache)
            plan += found
        trash = (Handler.data_root or study.parent) / ".trash"
        what = ids[0] if len(ids) == 1 else f"{len(ids)}-photos"
        target = trash / f"{study.name}--{what}--{datetime.datetime.now():%Y%m%d-%H%M%S}"
        for f in plan:
            try:
                rel = f.relative_to(study)
            except ValueError:                      # labels kept outside the study (--out)
                rel = Path("sidecars") / f.name
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(target / rel))
        if len(ids) > 1:
            (target / "REMOVED.txt").write_text(
                f"Removed from {study.name} on {datetime.datetime.now():%Y-%m-%d %H:%M}: "
                f"{len(ids)} photographs, {labelled} of them labelled.\n"
                "To restore one, move its files back to the same place inside the study.\n\n"
                + "\n".join(ids) + "\n")
        root = Handler.data_root.parent if Handler.data_root else study.parent.parent
        return self._send(200, {"ok": True, "ids": ids, "id": ids[0], "removed": len(ids),
                                "labelled": labelled if len(ids) > 1 else bool(labelled),
                                "files": len(plan),
                                "moved_to": str(target.relative_to(root))
                                if root in target.parents else str(target)})

    #: What a specimen may be called: it names files, so no path, no leading dot,
    #: and not ending in the rig's _L / _F view suffix, which would be read off it.
    _NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")

    def _rename_specimen(self):
        """Give a photograph's specimen a new name, and everything of it with it.

        ``{"id": old, "new_id": new}``. Renamed together, or not at all: the
        photograph (and a separate head-on crop), the saved labels (whose fish_id
        changes too), the cached predictions, and every table that names the fish
        -- its Darwin Core record, groups.csv, the image fingerprints, the study's
        worked example, and any re-label round it is in. The name carries the
        strain (``..._TEM_18``), so a rename can move a fish between groups.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            old, new = str(body.get("id") or ""), str(body.get("new_id") or "").strip()
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        study = self.images_dir
        if repeatability.read_round(study):
            return self._send(400, {"ok": False, "error": (
                "Codes in a blind re-label round are not renamed: the round finds its fish "
                "by them. Rename the fish in the study itself.")})
        if not old or old != Path(old).name or old.startswith("."):
            return self._send(400, {"ok": False, "error": f"no such specimen “{old}”"})
        if new == old:
            return self._send(400, {"ok": False, "error": "that is its name already"})
        if not self._NAME_RE.match(new) or re.search(r"_[LF]$", new):
            return self._send(400, {"ok": False, "error": (
                f"“{new}” cannot be a name: letters, digits, _ . - only, starting with a letter "
                "or digit, and not ending in _L or _F (those mark the view).")})
        if view_image(study, new, "lateral") or (self.out_dir / f"{new}.json").exists():
            return self._send(409, {"ok": False, "error": f"“{new}” is already a specimen here"})

        moves: list[tuple[Path, Path]] = []
        for view, suffix in (("lateral", "_L"), ("frontal", "_F")):
            p = view_image(study, old, view)
            if p is None or p.parent != study / view:
                continue
            stem = new + suffix if p.stem == old + suffix else new
            if (p, p.with_name(stem + p.suffix)) not in moves:
                moves.append((p, p.with_name(stem + p.suffix)))
        if not moves:
            return self._send(404, {"ok": False, "error": f"no photograph for “{old}”"})
        sidecar = self.out_dir / f"{old}.json"
        for cache in (study / "sidecars_auto" / f"{old}.json",
                      study / "sidecars_auto" / "frontal" / f"{old}.json"):
            if cache.is_file():
                moves.append((cache, cache.with_name(f"{new}.json")))
        if any(dst.exists() for _, dst in moves):
            return self._send(409, {"ok": False, "error": f"a file named for “{new}” is already here"})

        done: list[tuple[Path, Path]] = []
        try:
            for src, dst in moves:
                src.rename(dst)
                done.append((src, dst))
            if sidecar.is_file():
                doc = json.loads(sidecar.read_text())
                doc["fish_id"] = new
                (self.out_dir / f"{new}.json").write_text(json.dumps(doc, indent=2))
                sidecar.unlink()
        except Exception as exc:
            for src, dst in reversed(done):                 # put back what moved
                try:
                    dst.rename(src)
                except OSError:
                    pass
            return self._send(500, {"ok": False, "error": f"nothing renamed: {exc}"})

        updated = []
        # the tables that name it
        recs = darwin_core.load(study)
        if old in recs:
            recs[new] = recs.pop(old)
            darwin_core.save(study, recs)
            updated.append("Darwin Core record")
        groups = study / "groups.csv"
        if groups.is_file():
            import csv
            import io
            rows = list(csv.reader(io.StringIO(groups.read_text(encoding="utf-8-sig"))))
            if any(r and r[0].strip() == old for r in rows):
                buf = io.StringIO()
                csv.writer(buf).writerows([[new, *r[1:]] if r and r[0].strip() == old else r
                                           for r in rows])
                groups.write_text(buf.getvalue(), encoding="utf-8")
                updated.append("groups.csv")
        manifest = study / image_identity.MANIFEST
        if manifest.is_file():
            m = image_identity.load_manifest(study)
            if old in m:
                m[new] = m.pop(old)
                manifest.write_text(json.dumps(m, indent=2, sort_keys=True))
                updated.append("image fingerprints")
        ref = study / "reference" / "reference.json"
        if ref.is_file():
            try:
                r = json.loads(ref.read_text())
                if r.get("specimen") == old:
                    r["specimen"] = new
                    ref.write_text(json.dumps(r))
                    updated.append("the study's example")
            except Exception:
                pass
        for rnd in repeatability.rounds_of(study):
            if old in rnd["fish"].values():
                f = rnd["folder"] / repeatability.ROUND_FILE
                doc = json.loads(f.read_text())
                doc["fish"] = {c: (new if x == old else x) for c, x in doc["fish"].items()}
                f.write_text(json.dumps(doc, indent=2) + "\n")
                updated.append(f"re-label round {rnd.get('round')}")
        return self._send(200, {"ok": True, "id": new, "was": old, "files": len(moves)
                                + (1 if (self.out_dir / f"{new}.json").is_file() else 0),
                                "updated": updated})

    _ORIENT_LOCK = threading.Lock()
    #: The net turn of each photograph turned in the labeler, by its path in the
    #: study, and the folder that keeps each one as it was before its first turn.
    ORIENTATION_FILE = "orientation.json"
    UNTURNED_DIR = "unturned"

    def _orient_specimen(self):
        """Turn a photograph a quarter turn, or mirror it, with every point on it.

        ``{"id": fish, "view": "lateral" | "frontal", "op": "rotate_cw" |
        "rotate_ccw" | "mirror" | "flip"}``. For photographs that came in on their
        side, upside down or facing right: every study is labelled head left, back
        up, which is the orientation the landmark model and the exports expect.

        Turned together, or not at all: the image file; the saved labels of every
        view drawn on it (both views, where one photograph holds both); the
        fingerprints that record which image the labels were placed on; and the
        cached Auto-label predictions. Nothing is lost: the photograph as it was
        before its first turn is kept under ``unturned/``, and each new image is
        cut from that original by the net turn recorded in ``orientation.json``,
        so a JPEG is re-encoded once however many times it is turned, and turning
        it back to the start puts the original file back exactly. A mirrored
        photograph is recorded as mirrored in its labels' metadata: a mirror shows
        the fish's other side, which matters for anything left-right.
        """
        if self.demo_mode:
            return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            fid = str(body.get("id") or "")
            view = str(body.get("view") or "lateral")
            op = str(body.get("op") or "")
        except Exception:
            return self._send(400, {"ok": False, "error": "bad payload"})
        study = self.images_dir
        if repeatability.read_round(study):
            return self._send(400, {"ok": False, "error": (
                "Photographs in a blind re-label round are not turned. Turn the fish in the "
                "study itself.")})
        if not fid or fid != Path(fid).name or fid.startswith("."):
            return self._send(400, {"ok": False, "error": f"no such specimen “{fid}”"})
        if op not in image_identity.ORIENT_OPS or view not in ("lateral", "frontal"):
            return self._send(400, {"ok": False, "error": f"cannot turn by “{op}”"})
        photo = view_image(study, fid, view)
        if photo is None or photo.parent not in (study / "lateral", study / "frontal"):
            return self._send(404, {"ok": False, "error": f"no {view} photograph for “{fid}”"})
        views = [v for v in ("lateral", "frontal") if view_image(study, fid, v) == photo]

        from PIL import Image, ImageOps

        with self._ORIENT_LOCK:
            sidecar = self.out_dir / f"{fid}.json"
            doc = json.loads(sidecar.read_text()) if sidecar.is_file() else None
            if doc is not None:
                misfit = {v: a for v, a in alignment(study, fid, doc).items()
                          if v in views and a["status"] == image_identity.SIZE_CHANGED}
                if misfit:
                    return self._send(409, {"ok": False, "error": (
                        f"The saved {', '.join(misfit)} labels of {fid} no longer fit its "
                        f"photograph ({next(iter(misfit.values()))['detail']}). Turning would "
                        f"move them further; they need repairing first.")})
            record_file = study / self.ORIENTATION_FILE
            try:
                records = json.loads(record_file.read_text()) if record_file.is_file() else {}
            except Exception:
                records = {}
            key = f"{photo.parent.name}/{photo.name}"
            state = image_identity.orientation_after(records.get(key), op)
            original = study / self.UNTURNED_DIR / photo.parent.name / photo.name
            with Image.open(photo) as im:
                w, h = ImageOps.exif_transpose(im).size       # as the page shows it
            new_w, new_h = image_identity.oriented_size(op, w, h)

            # the new image, written beside the old and swapped in once all is ready
            tmp = photo.with_name(f".{photo.name}.turning")     # no image suffix: never listed
            restored = state == {"turns": 0, "mirrored": False}
            try:
                if not original.is_file():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(photo, original)
                if restored:
                    shutil.copy2(original, tmp)
                else:
                    with Image.open(original) as im:
                        out = image_identity.apply_orientation(ImageOps.exif_transpose(im), state)
                        if out.size != (new_w, new_h):
                            raise RuntimeError(f"turned image is {out.size}, expected {(new_w, new_h)}")
                        fmt = "PNG" if photo.suffix.lower() == ".png" else "JPEG"
                        out.convert("RGB" if fmt == "JPEG" else out.mode).save(
                            tmp, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                return self._send(500, {"ok": False, "error": f"the photograph was not turned: {exc}"})

            # every coordinate drawn on it, in memory first
            moved = 0
            writes: list[tuple[Path, dict]] = []
            fp = image_identity.fingerprint(tmp)
            if doc is not None:
                meta = doc.setdefault("metadata", {})
                for v in views:
                    moved += image_identity.orient_view(doc, v, op, w, h)
                    if image_identity.has_labels(doc, v) or (meta.get("images") or {}).get(v):
                        meta.setdefault("images", {})[v] = fp
                    if restored:
                        (meta.get("orientation") or {}).pop(v, None)
                    else:
                        meta.setdefault("orientation", {})[v] = dict(state)
                if meta.get("orientation") == {}:
                    meta.pop("orientation")
                writes.append((sidecar, doc))
            auto = study / "sidecars_auto"
            for v in views:
                cache = (auto if v == "lateral" else auto / "frontal") / f"{fid}.json"
                if cache.is_file():
                    try:
                        c = json.loads(cache.read_text())
                        image_identity.orient_view(c, v, op, w, h)
                        writes.append((cache, c))
                    except Exception:
                        cache.unlink(missing_ok=True)       # unreadable: predicted afresh
            manifest = image_identity.load_manifest(study)
            if fid in manifest and any(v in manifest[fid] for v in views):
                for v in views:
                    if v in manifest[fid]:
                        manifest[fid][v] = fp
                writes.append((study / image_identity.MANIFEST, manifest))

            os.replace(tmp, photo)
            for path, data in writes:
                part = path.with_name(f".{path.name}.turning")
                part.write_text(json.dumps(data, indent=2,
                                           sort_keys=path.name == image_identity.MANIFEST))
                os.replace(part, path)
            if restored:
                records.pop(key, None)
                original.unlink(missing_ok=True)
                for d in (original.parent, original.parent.parent):    # unturned/<view>, unturned
                    try:
                        d.rmdir()
                    except OSError:
                        break                                          # not empty: others kept
            else:
                records[key] = state
            record_file.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        return self._send(200, {"ok": True, "id": fid, "op": op, "views": views,
                                "size": [new_w, new_h], "mirrored": state["mirrored"],
                                "turns": state["turns"], "restored": restored, "moved": moved})

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
        """Accept one photograph into the current dataset's lateral/ folder -- or,
        with ``X-View: batch``, its batch/ folder of photographs of several fish.

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
        if view not in ("lateral", "frontal", "batch"):
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

        whole = load_profile(self.images_dir).get("single_photo") and view != "batch"
        if (self.headers.get("X-Split") or "").lower() == "mirror" and not whole and view != "batch":
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
        if repeatability.read_round(self.images_dir):
            return self._send(403, {
                "ok": False, "error": ("This is a blind re-label round: every landmark is placed "
                                       "by hand, so the model's guesses cannot shape the "
                                       "measurement error it is here to find.")})
        if fid in (load_profile(self.images_dir).get("hand_only") or ()):
            return self._send(403, {
                "ok": False, "hand_only": True, "error": (
                    f"{fid} is kept from the model: every landmark is placed by hand, so it "
                    f"can show how far accepting the model's points moves the shapes.")})
        scheme = load_profile(self.images_dir).get("scheme")
        # The cross-species model learned BGNN's landmarks as well as caliPr's and
        # answers under whichever names the study uses; the older model knows only
        # caliPr's.
        if schemes.get(scheme) and not union_model_in_use():
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
        # ?cache=only: what has already been predicted, or nothing -- never a model
        # run. Opening a fish asks this way: it must not wait seconds (minutes, on a
        # machine short of memory) for a model, and a cached guess made by a model
        # since replaced is not served either; the Auto-label button predicts afresh.
        cache_only = query.get("cache", [""])[0] == "only"
        res = None
        if want_cache and cache.is_file():
            try:
                doc = json.loads(cache.read_text())
                meta = doc.get("metadata") or {}
                # An entry written before the plausibility check existed holds
                # points that were never tested against the animal. Serving it
                # would quietly reinstate exactly what the check removes, so it
                # is treated as a miss and predicted again.
                # An entry made when the fin outliner was not installed holds no
                # fin outlines, which looks exactly like an entry where it found
                # none. Recording what was asked for tells the two apart, so
                # fetching the outliner later is enough to get them -- without
                # it, the first Auto-label a machine ever ran would decide
                # forever that this fish has no fins.
                asked = meta.get("fins_asked")
                stale = asked is not None and not set(self._wanted_fins(view)) <= set(asked)
                # Made by a model since replaced: predicted again by the one in use.
                now = current_model_label(view)
                stale = stale or bool(now and meta.get("model") != now)
                if meta.get("source") == "predicted" and "implausible" in meta and not stale:
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
        if res is None and cache_only:
            return self._send(200, {"ok": False, "cached": False, "fish_id": fid, "view": view,
                                    "error": "no prediction by the current model is saved "
                                             "for this fish"})
        if res is None:
            res = self._run_model(fid, view, match, cache, want_cache)
            if not res.get("ok"):
                return self._send(503, res)

        return self._send(200, self._excluded(res))

    def _wanted_fins(self, view: str) -> list[str]:
        """Fins to ask the outliner for: the ones it is offered for, minus any
        this study does not collect or does not want predicted.

        A fin outline the labeller accepts becomes training data for the next
        model, so this stays a short list on purpose -- see ``OUTLINED_FINS`` in
        predict_worker.py for what earned a place on it.
        """
        if view != "lateral":
            return []
        prof = load_profile(self.images_dir)
        drop = (set(prof.get("exclude_polygons") or ())
                | set(prof.get("exclude_predicted_polygons") or ()))
        return [f for f in OUTLINED_FINS if f not in drop]

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
        want_fins = self._wanted_fins(view)
        # The names this study collects its landmarks under, which the
        # cross-species model answers in; the older model ignores them.
        wanted = list(schemes.study_landmarks(self.images_dir)[0]) if view == "lateral" else None
        res = Predictor.predict(match, polygons=want_body, emit_polygons=emit_body,
                                view=view, crop=crop, fins=want_fins, landmarks=wanted)
        if not res.get("ok"):
            return res

        if want_cache and not self.demo_mode:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({
                    "fish_id": fid,
                    "metadata": {"source": "predicted", "model": res.get("model"),
                                 "image": match.name,
                                 "fins_asked": self._wanted_fins(view),
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
        # Never offer a point for a landmark this study has excluded -- nor for
        # one it collects but places by hand (``exclude_predicted_keypoints``):
        # where the model's guesses are poor, reviewing them costs more than
        # clicking, and an accepted poor guess reads as a checked one.
        prof = load_profile(self.images_dir)
        drop = set(prof.get("exclude_keypoints") or ()) | set(
            prof.get("exclude_predicted_keypoints") or ())
        # The same for outlines, and for one cached before the study stopped
        # wanting it: a run only withholds what the study refuses at that moment.
        drop_poly = set(prof.get("exclude_polygons") or ()) | set(
            prof.get("exclude_predicted_polygons") or ())
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

    # -- photographs of several fish --------------------------------------

    def _batch_reference_dir(self) -> Path:
        """The study whose hand-labelled fish the batch is compared with: the one
        schema.json names as ``reference_study``, or this one."""
        name = load_profile(self.images_dir).get("reference_study")
        if not name:
            return self.images_dir
        root = Handler.data_root
        beside = root / name if root is not None and "/" not in name else None
        return Handler.datasets.get(name) or (beside if beside and beside.is_dir() else self.images_dir)

    def _batch_photos(self):
        out = []
        cache = self.images_dir / "batch_cache"
        for name, path in list_images(self.images_dir / "batch").items():
            doc = _read_json(cache / f"{path.stem}.json")
            out.append({"name": name, "fish": len(doc["fish"]) if doc else None})
        return self._send(200, {"ok": True, "photos": out,
                                "reference": self._batch_reference_dir().name})

    def _batch_fish(self, name: str):
        """Every fish in one batch photograph, its landmarks, and how typical its
        shape is of each reference species, best first.

        Finding the fish and placing their landmarks is slow and is kept, beside
        the photograph, until the photograph or the model changes. The scores are
        not kept: they are worked out again on every request, so a fish labelled in
        the reference study since counts at once.
        """
        path = self.images_dir / "batch" / name
        if "/" in name or ".." in name or not path.is_file():
            return self._send(404, {"ok": False, "error": f"no batch photograph {name!r}"})
        query = parse_qs(urlparse(self.path).query)
        refresh = query.get("refresh", ["0"])[0] == "1"
        cache = self.images_dir / "batch_cache" / f"{path.stem}.json"
        st = path.stat()
        key = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "model": current_model_label("lateral"),
               "version": BATCH_CACHE_VERSION}
        doc = None if refresh else _read_json(cache)
        fresh = not doc or doc.get("key") != key
        # ?cache=only: the saved landmarks, scored, or nothing -- never a model run.
        # The page asks this way first, so it says it is finding and landmarking
        # fish only when it is.
        if fresh and query.get("cache", [""])[0] == "only":
            return self._send(200, {"ok": False, "cached": False})
        if fresh:
            if self._locked():
                return self._send(401, {"ok": False, "locked": True,
                                        "error": "automated landmarking is locked on this machine"})
            found = Predictor.detect(path)
            if not found.get("ok"):
                return self._send(503, found)
            ref_dir = self._batch_reference_dir()
            wanted = list(schemes.study_landmarks(ref_dir)[0])
            crops = cache.parent / path.stem
            crops.mkdir(parents=True, exist_ok=True)
            fish = []
            for i, f in enumerate(found["fish"]):
                crop_path = crops / f"fish_{i + 1}.jpg"
                _isolate_fish(path, found["fish"], i, crop_path)
                res = Predictor.predict(crop_path, polygons=False, emit_polygons=False,
                                        landmarks=wanted)
                x0, y0 = f["box"][:2]
                kps = {n: [p[0] + x0, p[1] + y0] for n, p in (res.get("keypoints") or {}).items() if p}
                fish.append({**f, "keypoints": kps,
                             "confidence": res.get("confidence") or {},
                             "low_confidence": res.get("low_confidence") or [],
                             "error": None if res.get("ok") else res.get("error")})
            doc = {"key": key, "width": found["width"], "height": found["height"], "fish": fish}
            cache.write_text(json.dumps(doc))
        if "ruler" not in doc:
            doc["ruler"] = _read_ruler(path)
            cache.write_text(json.dumps(doc))
        return self._send(200, {**self._batch_scored(name, doc), "cached": not fresh})

    def _batch_scored(self, name: str, doc: dict) -> dict:
        from fish_morpho import batch_fish, grouping, species_match, taxonomy

        ref_dir = self._batch_reference_dir()
        pattern = grouping.filename_pattern(ref_dir)
        table = grouping.load_group_table(ref_dir)
        present = {Path(n).stem for n in list_images(ref_dir / "lateral")}
        # A fish whose photograph has left the study is not compared -- except in a
        # downloaded reference set, which carries landmarks and credits, no photographs.
        landmarks_only = load_profile(ref_dir).get("reference_set") and not present
        ref = [r for r in species_match.load_reference(
                   ref_dir, lambda fid: grouping.resolve(fid, table=table, pattern=pattern),
                   load_profile(ref_dir).get("terminal_mouth"))
               if landmarks_only or r.fid in present]
        # a scale set by hand wins; else the one read from the ruler's numbers
        scale = (_read_json(self.images_dir / "batch_scale.json") or {}).get(name)
        ruler = doc.get("ruler") or {}
        ppm = species_match.px_per_mm(scale) or ruler.get("px_per_mm")
        scale_from = "hand" if species_match.px_per_mm(scale) else ("ruler" if ruler.get("px_per_mm") else None)
        # each reference species by its current name, and the names it went by
        names = taxonomy.Names(self._taxonomy_cache())
        taxa = {sp: names.resolve(sp) for sp in sorted({r.species for r in ref})}
        sizes = species_match.sizes_for(taxa)
        for sp, t in taxa.items():                     # a table entry under the current name will do
            if sp not in sizes and t.get("current"):
                hit = species_match.sizes_for([t["current"]]).get(t["current"])
                if hit:
                    sizes[sp] = hit
        fish = []
        for i, (f, (fid, fixed)) in enumerate(zip(doc["fish"], self._batch_corrections(name, doc))):
            kps = fixed or f["keypoints"]
            m = (species_match.identify(kps, ref, ppm, sizes=sizes) if fixed or not f.get("error")
                 else {"ok": False, "error": f["error"]})
            sl_px = species_match.standard_length_px(kps)
            fish.append({"n": i + 1, "box": f["box"], "outline": f["outline"],
                         "keypoints": kps, "low_confidence": [] if fixed else f.get("low_confidence", []),
                         "sl_mm": m.get("sl_mm"),
                         "sl_px": round(sl_px) if sl_px else None,
                         "too_small": bool(sl_px and sl_px < batch_fish.SMALLEST_SL_PX),
                         "fid": fid, "checked": bool(fixed), "match": m})
        counts: dict[str, int] = {}
        for r in ref:
            counts[r.species] = counts.get(r.species, 0) + 1
        drainage = (_read_json(self.images_dir / "batch_drainage.json") or {}).get(name)
        if drainage:
            by_key = {s["key"]: s for s in drainage["species"]}
            keys = drainage.get("reference_keys") or {}
            # the species recorded from the drainage first: see species_match.with_drainage
            records = {sp: (by_key.get(keys.get(sp)) or {}).get("records", 0) for sp in counts}
            for f in fish:
                species_match.with_drainage(f["match"], records)
            mine = {keys.get(sp) for sp in counts}
            shiner_keys = {s["key"] for s in drainage["species"]}
            intro = drainage.get("introduced") or {}
            fishes = [{**f, "reference": f["key"] in mine, "shiner": f["key"] in shiner_keys,
                       "introduced": intro.get(f["species"]) or intro.get(f["gbif_name"])}
                      for f in drainage.get("fishes") or []]
            # introduced fish USGS records there that no museum specimen on GBIF does
            listed = {f["species"] for f in fishes} | {f["gbif_name"] for f in fishes}
            fishes += [{"key": e.get("gbif_key"), "species": cur, "gbif_name": e["nas_name"], "common": e.get("common"),
                        "family": None, "chain": e.get("chain") or [cur], "records": 0, "reference": False,
                        "shiner": False, "introduced": e, "usgs_only": True}
                       for cur, e in intro.items() if cur not in listed and e["nas_name"] not in listed]
            for f in fish:                             # the reference species' own status there
                for x in f["match"].get("matches") or []:
                    t = taxa.get(x["species"]) or {}
                    x["introduced"] = intro.get(t.get("current")) or intro.get(x["species"].replace("_", " "))
            drainage = {**{k: v for k, v in drainage.items() if k not in ("species", "reference_keys", "fishes")},
                        "fishes": fishes,
                        "others": sorted(({"species": s.get("current") or s["species"], "gbif_name": s["species"],
                                           "common": s.get("common"), "chain": s.get("chain") or [s["species"]],
                                           "key": s["key"], "records": s["records"]}
                                          for s in drainage["species"] if s["key"] not in mine),
                                         key=lambda s: -s["records"])}
        return {"ok": True, "photo": name, "width": doc["width"], "height": doc["height"],
                "scale": scale, "ruler": ruler, "scale_from": scale_from, "px_per_mm": ppm,
                "drainage": drainage, "taxa": taxa,
                "fish": fish, "smallest_sl_px": batch_fish.SMALLEST_SL_PX,
                "reference": {"study": ref_dir.name, "fish": len(ref), "species": counts}}

    # -- where a batch photograph's lot was caught ------------------------

    def _batch_drainage(self):
        """The drainage a batch photograph's lot comes from -- from coordinates, a
        HUC8 code, or the lot's own GBIF record -- and the shiners museums have
        recorded from it (GBIF, preserved specimens). Shown beside the shape scores,
        never folded into them: a range record is itself a past identification.
        Every answer from the web is kept in ``drainage_cache.json``."""
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            name = str(data["photo"])
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"bad payload: {exc}"})
        path = self.images_dir / "batch_drainage.json"
        doc = _read_json(self.images_dir / "batch_cache" / f"{Path(name).stem}.json")
        if data.get("clear"):
            _update_json(path, lambda known: known.pop(name, None))
            return self._send(200, self._batch_scored(name, doc) if doc else {"ok": True})
        dc = _drainage_tool()
        web = dc.Web(self.images_dir / "drainage_cache.json")
        try:
            huc8, where = str(data.get("huc8") or "").strip(), {}
            if not huc8:
                lat, lon = data.get("lat"), data.get("lon")
                source = "clicked on the map" if data.get("source") == "clicked on the map" else "typed in"
                if lat is None or lon is None:
                    lat, lon, source = self._lot_coordinates(web, dc, name)
                h = dc.huc8_of(web, float(lat), float(lon))
                if not h:
                    return self._send(404, {"ok": False, "error": f"{lat}, {lon} is in no US drainage (HUC8)"})
                huc8 = h["huc8"]
                where = {"lat": float(lat), "lon": float(lon), "source": source,
                         "name": h.get("name"), "states": h.get("states")}
            elif not re.fullmatch(r"\d{8}", huc8):
                return self._send(400, {"ok": False, "error": "a HUC8 code is eight digits"})
            else:
                where = {"source": "HUC8 typed in"}
            wkt, outline, stand_in = dc.huc8_outline(web, huc8)
            keys = dc.genus_keys(web, (*dc.SHINER_GENERA, "Hudsonius"))
            species = dc.drainage_species(web, wkt, keys.values())
            from fish_morpho import taxonomy

            names = taxonomy.Names(self._taxonomy_cache())
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                resolved = list(pool.map(lambda sp: names.resolve(sp["species"]), species))
            for sp, t in zip(species, resolved):
                sp["current"], sp["common"] = t.get("current") or sp["species"], t.get("common")
                sp["chain"] = [c["name"] for c in t.get("chain") or []]
            fishes = _drainage_fishes(web, dc, wkt, names)
            introduced = _drainage_introduced(web, dc, huc8, names)
            ref = self._batch_reference_dir()
            names = sorted(self._reference_species(ref))
            ref_keys = {}
            for sp in names:
                m = web.get(dc.url(f"{dc.GBIF}/species/match", {"name": sp.replace("_", " ")}))
                if m.get("usageKey"):
                    ref_keys[sp] = m.get("acceptedUsageKey") or m["usageKey"]
        except Exception as exc:
            web.save()
            return self._send(503, {"ok": False, "error": f"the drainage could not be looked up: {exc}"})
        web.save()
        entry = {"huc8": huc8, **where, "approximate": stand_in or None, "species": species, "fishes": fishes,
                 "introduced": introduced, "reference_keys": ref_keys, "outline": outline}
        _update_json(path, lambda known: known.__setitem__(name, entry))
        return self._send(200, self._batch_scored(name, doc) if doc else {"ok": True})

    def _batch_specimens(self):
        """Museum specimens of a species with a photograph, for the gallery: GBIF's
        preserved-specimen records with a still image, one per lot, by the same rules
        as the reference set's (no fresh tissue vouchers, no NEON or BOLD field
        images). ``?taxonKey=`` the species' GBIF key; ``&here=1&photo=`` only from
        the drainage of that batch photograph's lot.

        Nothing is downloaded: the page shows each photograph from GBIF's image
        service. What is kept (``gallery_cache.json``) is the list -- museum, number,
        year, place, licence and the photograph's address -- a few hundred bytes a
        specimen."""
        import hashlib

        q = parse_qs(urlparse(self.path).query)
        try:
            key = int(q["taxonKey"][0])
        except (KeyError, ValueError):
            return self._send(400, {"ok": False, "error": "taxonKey is a GBIF species key"})
        here = q.get("here", ["0"])[0] == "1"
        huc8 = None
        if here:
            known = (_read_json(self.images_dir / "batch_drainage.json") or {}).get(q.get("photo", [""])[0])
            if not known:
                return self._send(404, {"ok": False, "error": "no drainage looked up for this photograph"})
            huc8 = known["huc8"]
        path = self.images_dir / "gallery_cache.json"
        cache = _read_json(path) or {}
        ck = f"{key}|{huc8 or 'all'}"
        if ck not in cache:
            dc, hv = _drainage_tool(), _harvester()
            web = dc.Web(None)
            params = [("taxonKey", str(key)), ("basisOfRecord", "PRESERVED_SPECIMEN"),
                      ("mediaType", "StillImage"), ("limit", "300")]
            try:
                if huc8:
                    wkt, _, _ = dc.huc8_outline(dc.Web(self.images_dir / "drainage_cache.json"), huc8)
                    params.append(("geometry", wkt))
                d = web.get(dc.url(f"{dc.GBIF}/occurrence/search", params))
            except Exception as exc:
                return self._send(503, {"ok": False, "error": f"GBIF could not be asked: {exc}"})
            lots, items = set(), []
            for r in d.get("results") or []:
                m = hv.photograph(r)
                lot = ((r.get("institutionCode") or "").upper(), r.get("catalogNumber") or str(r["key"]))
                if not m or lot in lots:
                    continue
                lots.add(lot)
                h = hashlib.md5(m["identifier"].encode()).hexdigest()
                img = f"{dc.GBIF}/image/cache/{{size}}/occurrence/{r['key']}/media/{h}"
                items.append({
                    "occurrence": r["key"], "institution": lot[0], "catalog": lot[1], "year": r.get("year"),
                    "place": ", ".join(v for v in (r.get("locality"), r.get("county"), r.get("stateProvince"),
                                                   r.get("countryCode")) if v),
                    "identified_as": r.get("scientificName"),
                    "thumb": img.format(size="300x"), "large": img.format(size="1200x"),
                    "original": m["identifier"], "license": m.get("license"),
                    "credit": m.get("rightsHolder") or m.get("creator") or lot[0]})
                if len(items) >= 60:
                    break
            cache[ck] = {"count": d.get("count", 0), "items": items,
                         "fetched": datetime.date.today().isoformat()}
            _update_json(path, lambda c: c.__setitem__(ck, cache[ck]), indent=None)
        return self._send(200, {"ok": True, **cache[ck], "here": bool(huc8)})

    def _taxonomy_cache(self) -> Path:
        """One cache of resolved names for every study: a name means the same fish in each."""
        return (Handler.data_root or self.images_dir.parent) / "taxonomy_cache.json"

    def _lot_coordinates(self, web, dc, name: str):
        """Where the lot in a batch photograph was caught, from its GBIF record:
        the study's photo_manifest.csv row for the photograph (its name less
        ``_lot``) gives the occurrence."""
        import csv

        fid = re.sub(r"_lot$", "", Path(name).stem)
        man = self.images_dir / "photo_manifest.csv"
        row = next((r for r in csv.DictReader(man.open()) if r.get("fish_id") == fid), None) if man.is_file() else None
        if not row or not row.get("gbif_occurrence"):
            raise ValueError("no coordinates given and no GBIF record for this photograph's lot")
        occ = web.get(f"{dc.GBIF}/occurrence/{row['gbif_occurrence']}")
        if occ.get("decimalLatitude") is None:
            where = ", ".join(v for v in (occ.get("locality") or row.get("locality"), occ.get("county") or row.get("county"),
                                          occ.get("stateProvince") or row.get("state")) if v)
            raise ValueError(f"the lot's GBIF record ({row['gbif_occurrence']}) has no coordinates; it says "
                             f"\"{where}\" -- click the map where that is, or type its latitude and longitude")
        return occ["decimalLatitude"], occ["decimalLongitude"], f"GBIF record {row['gbif_occurrence']} of the lot"

    def _reference_species(self, ref_dir: Path) -> set[str]:
        from fish_morpho import grouping

        pattern, table = grouping.filename_pattern(ref_dir), grouping.load_group_table(ref_dir)
        out = set()
        for img in list_images(ref_dir / "lateral"):
            sp = grouping.resolve(Path(img).stem, table=table, pattern=pattern)
            if sp:
                out.add(sp)
        return out

    # -- putting a batch fish's points right ------------------------------

    def _batch_study_dir(self) -> Path:
        """Where this study's batch fish go to have their points put right: a study
        of their own beside it. What is corrected there scores the fish and trains
        the landmark model, and never becomes a reference fish."""
        return self.images_dir.parent / f"{self.images_dir.name}_batch"

    def _batch_corrections(self, name: str, doc: dict) -> list[tuple]:
        """For each fish found in batch photograph ``name``: its id in the batch
        study and its saved points there, in the photograph's coordinates -- or
        ``(None, None)``, ``(id, None)`` while it has none. A fish is known by where
        it lies, not by its number, so finding the fish again cannot hand one
        fish's corrected points to another."""
        bdir = self._batch_study_dir()
        mine = {fid: e for fid, e in (_read_json(bdir / "batch_fish.json") or {}).items()
                if isinstance(e, dict) and e.get("photo") == name}
        out = []
        for f in doc["fish"]:
            fid = next((k for k, e in mine.items() if _overlap(e.get("box"), f["box"]) >= SAME_FISH), None)
            fixed = None
            if fid:
                kp = ((_read_json(bdir / "sidecars" / f"{fid}.json") or {}).get("lateral") or {}).get("keypoints")
                if kp:
                    x0, y0 = mine[fid]["box"][:2]
                    fixed = {n: [p[0] + x0, p[1] + y0] for n, p in kp.items() if p}
            out.append((fid, fixed))
        return out

    def _batch_correct(self):
        """Put every fish of a batch photograph into the batch study -- cut out,
        with the model's points waiting on it -- so they can be opened in the
        labeller one after another; answers with the study and the fish asked for.

        Nothing already there is replaced: not a fish's photograph, which its saved
        points are placed on, and not a guess from the model in use."""
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            name, want = str(data["photo"]), int(data["n"])
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"bad payload: {exc}"})
        photo = self.images_dir / "batch" / name
        doc = _read_json(self.images_dir / "batch_cache" / f"{Path(name).stem}.json")
        if "/" in name or ".." in name or not photo.is_file() or not doc:
            return self._send(404, {"ok": False, "error": f"no fish found yet in {name!r}"})
        if not 1 <= want <= len(doc["fish"]):
            return self._send(400, {"ok": False, "error": f"no fish {want} in {name}"})
        bdir = self._ensure_batch_study()
        manifest = _read_json(bdir / "batch_fish.json") or {}
        model = (doc.get("key") or {}).get("model")
        ids = []
        for i, (f, (fid, _)) in enumerate(zip(doc["fish"], self._batch_corrections(name, doc))):
            if fid is None:
                k = i + 1
                while f"{photo.stem}_fish_{k}" in manifest:
                    k += 1
                fid = f"{photo.stem}_fish_{k}"
                manifest[fid] = {"photo": name, "box": f["box"]}
            img = bdir / "lateral" / f"{fid}.jpg"
            if not img.is_file():
                _isolate_fish(photo, doc["fish"], i, img)
            auto = bdir / "sidecars_auto" / f"{fid}.json"
            had = (_read_json(auto) or {}).get("metadata") or {}
            if f["keypoints"] and (had.get("model") != model or not had.get("keypoint_confidence")):
                x0, y0 = manifest[fid]["box"][:2]
                auto.parent.mkdir(parents=True, exist_ok=True)
                # In the labeller's own cache form, so the fish opens with these
                # points on it -- the ones it was scored on -- waiting for review.
                auto.write_text(json.dumps({
                    "fish_id": fid,
                    "metadata": {"source": "predicted", "model": model, "image": img.name,
                                 "from": f"batch photograph {name}", "fins_asked": [],
                                 # what the labeller counts as waiting for review
                                 "keypoint_confidence": f.get("confidence") or {},
                                 "low_confidence": f.get("low_confidence") or [],
                                 "implausible": {}, "frame_warning": None},
                    "lateral": {"keypoints": {k: [p[0] - x0, p[1] - y0] for k, p in f["keypoints"].items()},
                                "polygons": {},
                                "calibration": {"mode": "none", "notes": "predicted; not a label"}},
                }, indent=2))
            ids.append(fid)
        _update_json(bdir / "batch_fish.json", lambda d: d.update(manifest))
        Handler.datasets = discover_datasets(Handler.data_root) if Handler.data_root else Handler.datasets
        Handler.datasets.setdefault(bdir.name, bdir)
        return self._send(200, {"ok": True, "dataset": bdir.name, "fish": ids[want - 1]})

    def _ensure_batch_study(self) -> Path:
        """The batch study, made the first time it is wanted: the reference study's
        settings -- its landmarks, its counts -- with each fish's group being the
        photograph it came from. Fins and outlines are not offered: what is put
        right here is the landmarks."""
        bdir = self._batch_study_dir()
        (bdir / "lateral").mkdir(parents=True, exist_ok=True)
        (bdir / "sidecars").mkdir(exist_ok=True)
        if not (bdir / "schema.json").is_file():
            ref = self._batch_reference_dir()
            prof = _read_json(ref / "schema.json") or {}
            for key in ("label_first", "reference_study", "batch_of"):
                prof.pop(key, None)
            prof.update({
                "note": (f"Fish cut from the batch photographs of {self.images_dir.name}, to have "
                         f"the model's landmarks put right. Their corrected points score them on "
                         f"the batch page and train the landmark model; they are never reference fish."),
                "batch_of": self.images_dir.name,
                "group_from_filename": r"^(.+)_fish_\d+$",
                "exclude_predicted_polygons": sorted(
                    set(prof.get("exclude_predicted_polygons") or ()) | {"body_plus_caudal", *OUTLINED_FINS}),
            })
            (bdir / "schema.json").write_text(json.dumps(prof, indent=2) + "\n")
            if (ref / "plausibility.json").is_file() and not (bdir / "plausibility.json").exists():
                shutil.copyfile(ref / "plausibility.json", bdir / "plausibility.json")
        return bdir

    def _batch_scale(self):
        """Set a batch photograph's ruler -- two points and the millimetres between
        them -- or clear it; answers with the photograph scored again."""
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            name = str(data["photo"])
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"bad payload: {exc}"})
        path = self.images_dir / "batch_scale.json"
        if data.get("clear"):
            scales = _update_json(path, lambda d: d.pop(name, None))
        else:
            try:
                cal = {"mode": "manual", "point_a": [float(v) for v in data["point_a"]],
                       "point_b": [float(v) for v in data["point_b"]],
                       "known_mm": float(data["known_mm"])}
            except Exception as exc:
                return self._send(400, {"ok": False, "error": f"bad ruler: {exc}"})
            if cal["known_mm"] <= 0 or cal["point_a"] == cal["point_b"]:
                return self._send(400, {"ok": False, "error": "the ruler needs two points and a span"})
            scales = _update_json(path, lambda d: d.__setitem__(name, cal))
        doc = _read_json(self.images_dir / "batch_cache" / f"{Path(name).stem}.json")
        if not doc:
            return self._send(200, {"ok": True, "scale": scales.get(name)})
        return self._send(200, self._batch_scored(name, doc))

    #: What is kept of a museum specimen put beside the batch photograph: where GBIF
    #: serves its photograph and how it is cited. Never the photograph itself.
    COMPARE_FIELDS = ("occurrence", "taxon", "name", "common", "institution", "catalog", "year",
                      "place", "identified_as", "credit", "license", "thumb", "large", "original")

    def _batch_compare(self):
        """The museum specimens kept to compare with the batch photographs, in the
        order they were kept. GET lists them; POST ``{"add": specimen}`` or
        ``{"remove": occurrence}`` changes the list and answers with it."""
        path = self.images_dir / "batch_compare.json"
        if self.command == "GET":
            return self._send(200, {"ok": True, "items": (_read_json(path) or {}).get("items", [])})
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            if "add" in data:
                it = {k: data["add"][k] for k in self.COMPARE_FIELDS if data["add"].get(k) not in (None, "")}
                it["occurrence"] = int(it["occurrence"])
                if not all(str(it.get(k, "")).startswith(("https://", "http://"))
                           for k in ("thumb", "large", "original") if k in it):
                    raise ValueError("photograph addresses must be web addresses")
                change = lambda d: d.__setitem__("items", [x for x in d.get("items", [])
                                                           if x.get("occurrence") != it["occurrence"]] + [it])
            else:
                gone = int(data["remove"])
                change = lambda d: d.__setitem__("items", [x for x in d.get("items", [])
                                                           if x.get("occurrence") != gone])
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"bad payload: {exc}"})
        return self._send(200, {"ok": True, "items": _update_json(path, change)["items"]})

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

        if route.startswith("/api/batch/") and not batch_installed():
            return self._send(409, {"ok": False, "not_installed": "batch", "error": BATCH_MISSING})

        if route == "/api/authstate":
            return self._authstate()

        if route == "/api/components":
            return self._send(200, components_state())

        if route == "/api/autolabel/install":
            return self._send(200, AutolabelInstall.status())

        if route == "/api/install/batch":
            return self._send(200, BatchInstall.status())

        if route == "/api/dwc":
            return self._dwc()

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
                     "relabel": self._round_brief(Handler.datasets[n]),
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
        if route == "/batch":
            return self._send(200, (UI_DIR / "batch.html").read_text(), "text/html; charset=utf-8")
        if route == "/api/batch/photos":
            return self._batch_photos()
        if route == "/api/batch/specimens":
            return self._batch_specimens()
        if route == "/api/batch/compare":
            return self._batch_compare()
        if route.startswith("/api/batch/fish/"):
            return self._batch_fish(unquote(route[len("/api/batch/fish/"):]))
        if route == "/api/schema":
            return self._send(200, {**build_schema(load_profile(self.images_dir)),
                                    "relabel": self._round_brief(self.images_dir)})
        if route == "/api/relabel":
            return self._relabel_rounds()
        if route == "/api/relabel/stats":
            return self._relabel_stats()
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
        if route.startswith("/api/reference/"):
            return self._reference_file(unquote(route[len("/api/reference/"):]))
        if route == "/api/export/preview":
            return self._export_preview()
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

        if route.startswith("/api/batch/") and not batch_installed():
            return self._send(409, {"ok": False, "not_installed": "batch", "error": BATCH_MISSING})

        if route == "/api/dataset/new":
            return self._new_dataset()

        if route == "/api/specimen/remove":
            return self._remove_specimen()

        if route == "/api/specimen/rename":
            return self._rename_specimen()
        if route == "/api/specimen/orient":
            return self._orient_specimen()

        if route == "/api/dataset/remove":
            return self._remove_dataset()

        if route.startswith("/api/export/"):
            return self._export_and_show(route[len("/api/export/"):])

        if route == "/api/upload":
            return self._upload()

        if route == "/api/unlock":
            return self._unlock()

        if route in ("/api/autolabel/install", "/api/install/batch"):
            # Only the labeler's own page may start a 2 GB download, not any
            # site the browser happens to have open.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self._send(403, {"ok": False, "error": "not from this page"})
            name = "autolabel" if route == "/api/autolabel/install" else "batch"
            # A part not released yet has nothing to download.
            if not components.released(name):
                title = (components.manifest().get(name) or {}).get("title", name)
                return self._send(403, {"ok": False, "not_released": name,
                                        "error": f"{title} is not available to download yet."})
            if name == "batch" and not autolabel_installed():
                return self._send(409, {"ok": False, "error": "Install Auto-label first: "
                                        "batch identification finds and landmarks the fish with it."})
            installer = AutolabelInstall if name == "autolabel" else BatchInstall
            return self._send(200, {"ok": True, **installer.start()})

        if route == "/api/lock":
            self.session.lock()
            return self._send(200, {"ok": True})

        if route == "/api/schema/exclude":
            return self._set_exclusions()

        if route == "/api/schema/keypoint":
            return self._edit_schema_keypoint()

        if route == "/api/schema/scheme":
            return self._set_scheme()
        if route == "/api/schema/known_mm":
            return self._set_known_mm_default()
        if route == "/api/dwc":
            return self._dwc_update()
        if route == "/api/dwc/import":
            return self._dwc_import()
        if route == "/api/straighten/preview":
            return self._straighten_preview()
        if route == "/api/relabel/new":
            return self._relabel_new()
        if route == "/api/reference":
            return self._set_reference()
        if route == "/api/schemes/new":
            return self._new_scheme()
        if route == "/api/batch/compare":
            if self.demo_mode:
                return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
            return self._batch_compare()
        if route == "/api/batch/scale":
            if self.demo_mode:
                return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
            return self._batch_scale()
        if route == "/api/batch/drainage":
            if self.demo_mode:
                return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
            return self._batch_drainage()
        if route == "/api/batch/correct":
            if self.demo_mode:
                return self._send(403, {"ok": False, "error": "demo mode — nothing is written"})
            return self._batch_correct()

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

        # Metadata this labeler does not manage is carried across rather than
        # dropped. The page builds the outgoing metadata from scratch -- strain,
        # source, and what it knows about the prediction -- so anything a script
        # wrote there was erased by the next save of that fish. coordinate_history
        # is the record of a shift being applied exactly, written by the un-split
        # migration and by realign_labels; losing it leaves coordinates that moved
        # with nothing saying they did, or why.
        if existing.is_file():
            try:
                before = json.loads(existing.read_text())
            except Exception:
                before = {}
            had = before.get("metadata") or {}
            for key, was in had.items():
                if key not in meta:
                    meta[key] = was
            _carry_unmanaged(data, before)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        target = self.out_dir / f"{safe}.json"
        target.write_text(json.dumps(data, indent=2))
        return self._send(200, {"ok": True, "saved": f"{safe}.json",
                                "mtime": target.stat().st_mtime})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="label_server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)),
                    help="Port to serve on (default: $PORT, else 8765).")
    ap.add_argument("--data-root", type=Path, default=paths.WORK / "data",
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
        schemes.use_data_root(base.parent)
    else:
        Handler.data_root = args.data_root.resolve()
        Handler.data_root.mkdir(parents=True, exist_ok=True)
        Handler.datasets = discover_datasets(Handler.data_root)
        # Schemes of your own live beside the studies, not inside one: a scheme is
        # a protocol, and row N means the same landmark in every study that
        # follows it.
        schemes.use_data_root(Handler.data_root)
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
