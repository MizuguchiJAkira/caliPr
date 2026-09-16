"""Which image a set of coordinates was placed on.

A landmark is only meaningful on the pixels it was clicked on. Preprocessing
crops each raw photograph into a lateral and a frontal view, and the lateral
crop runs from ``boundary - lateral_margin`` to the right edge -- so changing
the margin, or the boundary, and re-running moves every pixel of the fish while
leaving any saved coordinates exactly where they were. That happened to
``HRN_4``: labelled on a crop starting at x=1656, re-cropped on 2026-08-04 to
start at x=1206, and from then on every landmark and the whole outline sat 450
px to the left of the fish, in the ruler, with nothing anywhere to say so.

The defence is to record the image a label was made against and compare it with
the image on disk whenever the label is used. Two properties are kept:

* **width and height.** A lateral crop always extends to the right edge of the
  photograph, so moving its start changes its width -- any re-crop shows up
  here. It survives a lossless or lossy re-save of the same crop, which leaves
  the coordinates valid.
* **sha256 of the file.** Catches everything the size does not: a different
  photograph of identical dimensions, a 180 degree rotation, an edited image.
  It also changes on a harmless re-encode, so a hash mismatch at an unchanged
  size is reported as a warning to look, not as proof the labels are wrong.

A size mismatch *is* proof: coordinates placed on a crop of one width cannot
describe a crop of another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

#: How a recorded image compares with the one on disk.
OK = "ok"
SIZE_CHANGED = "size_changed"          # the labels cannot line up; refuse them
CONTENT_CHANGED = "content_changed"    # same size, different bytes; check by eye
UNRECORDED = "unrecorded"              # labelled before fingerprints existed
MISSING = "missing"                    # the image itself is gone

MANIFEST = "image_fingerprints.json"


def fingerprint(path: Path) -> dict | None:
    """``{"width", "height", "sha256"}`` for an image file, or None if absent."""
    path = Path(path)
    if not path.is_file():
        return None
    from PIL import Image                       # only here: pure-file callers skip it

    with Image.open(path) as im:                # reads the header, not the pixels
        width, height = im.size
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {"width": int(width), "height": int(height), "sha256": digest.hexdigest()}


def compare(recorded: dict | None, current: dict | None) -> str:
    """One of the module constants: how ``current`` differs from ``recorded``."""
    if current is None:
        return MISSING
    if not recorded:
        return UNRECORDED
    if (recorded.get("width"), recorded.get("height")) != (current["width"], current["height"]):
        return SIZE_CHANGED
    if recorded.get("sha256") and recorded["sha256"] != current["sha256"]:
        return CONTENT_CHANGED
    return OK


def load_manifest(dataset_dir: Path) -> dict:
    """``{fish_id: {view: fingerprint}}`` recorded for labels saved before this module."""
    path = Path(dataset_dir) / MANIFEST
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except Exception:
        return {}


def recorded_for(sidecar: dict | None, manifest: dict, fish_id: str, view: str) -> dict | None:
    """The fingerprint a view's labels were made against: the sidecar's own record
    first, then the manifest backfilled for older labels."""
    own = (((sidecar or {}).get("metadata") or {}).get("images") or {}).get(view)
    return own or (manifest.get(fish_id) or {}).get(view)


def describe(status: str, recorded: dict | None, current: dict | None) -> str:
    """A sentence for the labeller."""
    if status == SIZE_CHANGED:
        return (f"the image is now {current['width']}x{current['height']} but these labels "
                f"were placed on a {recorded['width']}x{recorded['height']} image — it has "
                f"been re-cropped, and every coordinate is displaced")
    if status == CONTENT_CHANGED:
        return ("the image file has changed since these labels were saved, at the same size "
                "— check that the landmarks still sit on the fish")
    if status == MISSING:
        return "the image for these labels is missing"
    if status == UNRECORDED:
        return "no record of which image these labels were placed on"
    return "labels match the image they were placed on"


def shift_view(doc: dict, view: str, dx: float, dy: float = 0.0) -> int:
    """Move every coordinate stored for ``view`` by ``(dx, dy)``, in place.

    Covers the three places a sidecar keeps image coordinates: landmarks, outline
    vertices, and the two points of a manual ruler calibration. Returns how many
    points moved. A uniform shift changes no distance, area or angle, so every
    measured trait is unaffected -- what it repairs is the pairing of each
    coordinate with the pixels under it, which is what the labeler draws and what
    the landmark model is trained on.
    """
    block = doc.get(view) or {}
    moved = 0

    def mv(pt):
        nonlocal moved
        moved += 1
        return [round(float(pt[0]) + dx, 1), round(float(pt[1]) + dy, 1)]

    for name, pt in list((block.get("keypoints") or {}).items()):
        if pt:
            block["keypoints"][name] = mv(pt)
    for name, poly in list((block.get("polygons") or {}).items()):
        if poly:
            block["polygons"][name] = [mv(q) for q in poly]
    cal = block.get("calibration") or {}
    for key in ("point_a", "point_b"):
        if cal.get(key):
            cal[key] = mv(cal[key])
    return moved


def max_x(doc: dict, view: str) -> float | None:
    """Largest x of any coordinate stored for ``view``, or None if there are none."""
    block = doc.get(view) or {}
    xs = [p[0] for p in (block.get("keypoints") or {}).values() if p]
    xs += [q[0] for poly in (block.get("polygons") or {}).values() for q in (poly or [])]
    cal = block.get("calibration") or {}
    xs += [cal[k][0] for k in ("point_a", "point_b") if cal.get(k)]
    return max(xs) if xs else None


def has_labels(doc: dict | None, view: str) -> bool:
    block = (doc or {}).get(view) or {}
    return bool(block.get("keypoints") or block.get("polygons"))
