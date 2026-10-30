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

    Covers every place a sidecar keeps image coordinates: landmarks, outline
    vertices, the two points of a manual ruler calibration, and the midline a
    bent fish is straightened along. Returns how many
    points moved. A uniform shift changes no distance, area or angle, so every
    measured trait is unaffected -- what it repairs is the pairing of each
    coordinate with the pixels under it, which is what the labeler draws and what
    the landmark model is trained on.
    """
    return _map_view(doc, view, lambda x, y: (x + dx, y + dy))


#: The ways a photograph can be turned. Each is its own exact inverse but the
#: quarter turns, which undo each other.
ORIENT_OPS = ("rotate_cw", "rotate_ccw", "mirror", "flip")


def orient_point(op: str, x: float, y: float, w: float, h: float) -> tuple[float, float]:
    """Where the point (x, y) of a w x h image lands when the image is turned by ``op``.

    Coordinates are continuous, from 0 to w across: a pixel's left edge at x lands
    at w - x, so the mapping is exact, with no half-pixel drift.
    """
    if op == "rotate_cw":           # a quarter turn clockwise: the left edge becomes the top
        return h - y, x
    if op == "rotate_ccw":          # anticlockwise: the top edge becomes the left
        return y, w - x
    if op == "mirror":              # left and right swapped
        return w - x, y
    if op == "flip":                # top and bottom swapped
        return x, h - y
    raise ValueError(f"unknown orientation {op!r}")


def oriented_size(op: str, w: int, h: int) -> tuple[int, int]:
    """The image's (width, height) after ``op``."""
    return (h, w) if op in ("rotate_cw", "rotate_ccw") else (w, h)


def orient_view(doc: dict, view: str, op: str, w: float, h: float) -> int:
    """Turn every coordinate stored for ``view`` with its image, in place.

    The same coverage as :func:`shift_view`. A turn or a mirror changes no
    distance, area or angle, so every trait is unaffected; each point keeps the
    anatomy it marks. Names stay right after a mirror too: ``mouth_left`` is the
    fish's own left corner, and the point moves with that corner's pixels.
    """
    return _map_view(doc, view, lambda x, y: orient_point(op, x, y, w, h))


def orientation_after(state: dict | None, op: str) -> dict:
    """The net turn of an image after one more ``op``.

    ``{"turns": k, "mirrored": m}`` means: mirror left-right if m, then k quarter
    turns clockwise -- every combination of turns and mirrors is one of these
    eight. Keeping the net turn, rather than the steps, lets each new image be cut
    from the untouched original, so a JPEG is re-encoded once however many times
    it is turned, and turning back to the start returns the original exactly.
    """
    k = int((state or {}).get("turns", 0)) % 4
    m = bool((state or {}).get("mirrored", False))
    if op == "rotate_cw":
        k += 1
    elif op == "rotate_ccw":
        k -= 1
    elif op == "mirror":            # H R^k = R^-k H
        k, m = -k, not m
    elif op == "flip":              # flip = a half turn and a mirror
        k, m = 2 - k, not m
    else:
        raise ValueError(f"unknown orientation {op!r}")
    return {"turns": k % 4, "mirrored": m}


def apply_orientation(image, state: dict):
    """A PIL image turned by a net orientation (see :func:`orientation_after`)."""
    from PIL import Image

    if state.get("mirrored"):
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    for _ in range(int(state.get("turns", 0)) % 4):
        image = image.transpose(Image.Transpose.ROTATE_270)     # PIL's 270 is clockwise
    return image


def _map_view(doc: dict, view: str, fn) -> int:
    """Replace every coordinate (x, y) stored for ``view`` with ``fn(x, y)``."""
    block = doc.get(view) or {}
    moved = 0

    def mv(pt):
        nonlocal moved
        moved += 1
        x, y = fn(float(pt[0]), float(pt[1]))
        return [round(x, 1), round(y, 1)]

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
    # The midline a bent fish is straightened along names pixels too.
    if block.get("midline"):
        block["midline"] = [mv(q) for q in block["midline"]]
    # Where the model first put each point it was corrected on. Not a label, so
    # not counted, but it names a pixel like one: left behind by the 2026-09-21
    # un-split, the labeller drew every correction line from ~1000 px off.
    assist = (doc.get("metadata") or {}).get("assist" if view == "lateral"
                                             else f"assist_{view}") or {}
    for rec in (assist.get("corrected") or {}).values():
        for key in ("from", "to"):
            if rec.get(key):
                x, y = fn(float(rec[key][0]), float(rec[key][1]))
                rec[key] = [round(x, 1), round(y, 1)]
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
