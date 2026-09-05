"""Tests for the contributor round-trip: labeler export -> sidecars -> training.

The failure this guards against is not a crash. It is a resized photograph:
labels drawn on a shrunken copy import cleanly, land inside the frame, and shift
every coordinate by a constant factor that no later check can see. So most of
these assert on *rejection*, and one asserts that a bundled photograph arrives
byte-identical, because "close enough" is exactly the bug.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


imp = _load("import_standalone_labels")

try:
    dlc = _load("build_dlc_dataset")
except ModuleNotFoundError:          # pandas/cv2 live in the training env only
    dlc = None
needs_dlc = pytest.mark.skipif(dlc is None,
                               reason="build_dlc_dataset needs the training env")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _png(path: Path, w: int, h: int) -> bytes:
    """A real PNG of the given size, so PIL reports genuine dimensions."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        import zlib
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    import zlib
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes([(x * 7) % 256, (y * 5) % 256, 0] * 1)
                   * 1 + bytes([0, 0, 0]) * (w - 1) for y in range(h) for x in [0])
    body = zlib.compress(raw)
    blob = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", body) + chunk(b"IEND", b""))
    path.write_bytes(blob)
    return blob


KPS = {"premaxilla_tip": [10, 20], "caudal_base": [90, 25]}


def _doc(name: str, w: int, h: int, raw: bytes | None):
    fp = None
    if raw is not None:
        fp = {"bytes": len(raw), "modified": 0,
              "sha256": hashlib.sha256(raw).hexdigest(), "crc32": 0}
    return {"format": "calipr-landmarks/1", "landmark_order": list(KPS),
            "exported": "2026-01-01T00:00:00Z",
            "specimens": {name: {"width": w, "height": h,
                                 "keypoints": dict(KPS), "file": fp}}}


def _bundle(path: Path, doc: dict, name: str, raw: bytes) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("labels.json", json.dumps(doc))
        z.writestr(f"images/{name}", raw)
    return path


# --------------------------------------------------------------------------
# the fallback hash must agree with the labeler's JavaScript
# --------------------------------------------------------------------------

def test_fnv1a_matches_the_browser():
    """Pinned against the value the labeler's JS produced for bytes 0..255.

    A browser without SubtleCrypto falls back to this hash. If the two
    implementations drift, every such contributor's import fails verification
    for no real reason.
    """
    assert imp.fnv1a(bytes(range(256))) == "fnv1a:90a458c5eb75b064"


def test_digest_picks_the_style_the_export_used():
    data = b"abc"
    assert imp.digest(data, "fnv1a:0") == imp.fnv1a(data)
    assert imp.digest(data, "a" * 64) == hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# a bundle carries the photograph, unmodified
# --------------------------------------------------------------------------

def test_bundle_extracts_the_photograph_byte_for_byte(tmp_path):
    name = "IMG_0042.png"
    raw = _png(tmp_path / name, 100, 60)
    (tmp_path / name).unlink()
    bundle = _bundle(tmp_path / "b.zip", _doc(name, 100, 60, raw), name, raw)

    images, out = tmp_path / "imgs", tmp_path / "side"
    rc = imp.main(["--labels", str(bundle), "--images", str(images),
                   "--out", str(out), "--annotator", "R. Chen"])

    assert rc == 0
    assert (images / name).read_bytes() == raw, "photograph must not be re-encoded"
    sidecar = json.loads((out / "IMG_0042.json").read_text())
    assert sidecar["metadata"]["annotator"] == "R. Chen"
    assert sidecar["metadata"]["image"] == name
    assert sidecar["lateral"]["keypoints"]["premaxilla_tip"] == [10, 20]
    # No ruler in the standalone, so the series must declare itself scale-free
    # rather than be rejected for a missing calibration.
    assert sidecar["lateral"]["calibration"]["mode"] == "none"


def test_prefix_namespaces_a_contributor(tmp_path):
    name = "IMG_0042.png"
    raw = _png(tmp_path / name, 100, 60)
    (tmp_path / name).unlink()
    bundle = _bundle(tmp_path / "b.zip", _doc(name, 100, 60, raw), name, raw)
    images, out = tmp_path / "imgs", tmp_path / "side"

    imp.main(["--labels", str(bundle), "--images", str(images), "--out", str(out),
              "--prefix", "rchen"])

    assert (images / "rchen_IMG_0042.png").is_file()
    assert (out / "rchen_IMG_0042.json").is_file()


# --------------------------------------------------------------------------
# rejections: each of these was silently accepted before
# --------------------------------------------------------------------------

def test_resized_photograph_is_rejected(tmp_path):
    """The headline case: labels drawn on a half-size copy."""
    name = "fish.png"
    images = tmp_path / "imgs"
    images.mkdir()
    _png(images / name, 200, 120)                      # the full-size original
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps(_doc(name, 100, 60, None)))   # labelled at half

    out = tmp_path / "side"
    rc = imp.main(["--labels", str(labels), "--images", str(images),
                   "--out", str(out)])

    assert rc == 1
    assert not list(out.glob("*.json"))


def test_reencoded_photograph_is_rejected(tmp_path):
    """Identical byte count and dimensions, different pixels.

    Only the content hash can catch this one — size and dimensions both agree,
    which is what makes it the case worth having a hash for at all.
    """
    name = "fish.png"
    images = tmp_path / "imgs"
    images.mkdir()
    here = _png(images / name, 100, 60)

    # One flipped byte deep in the pixel data: same length, same header, so the
    # image still opens at 100x60 and only the digest differs.
    labelled = bytearray(here)
    labelled[-10] ^= 0xFF
    labelled = bytes(labelled)
    assert len(labelled) == len(here) and labelled != here

    labels = tmp_path / "l.json"
    labels.write_text(json.dumps(_doc(name, 100, 60, labelled)))

    out = tmp_path / "side"
    rc = imp.main(["--labels", str(labels), "--images", str(images),
                   "--out", str(out)])

    assert rc == 1
    assert not list(out.glob("*.json"))


def test_matching_photograph_passes_verification(tmp_path):
    """The control: the same guard must not reject a correct import."""
    name = "fish.png"
    images = tmp_path / "imgs"
    images.mkdir()
    raw = _png(images / name, 100, 60)
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps(_doc(name, 100, 60, raw)))

    out = tmp_path / "side"
    rc = imp.main(["--labels", str(labels), "--images", str(images), "--out", str(out)])

    assert rc == 0
    assert (out / "fish.json").is_file()


def test_labels_only_without_the_photograph_is_rejected(tmp_path):
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps(_doc("absent.png", 100, 60, b"x")))
    out = tmp_path / "side"

    rc = imp.main(["--labels", str(labels), "--images", str(tmp_path / "imgs"),
                   "--out", str(out)])

    assert rc == 1
    assert not list(out.glob("*.json"))


def test_bundled_photo_never_overwrites_a_different_one(tmp_path):
    name = "IMG_0042.png"
    images = tmp_path / "imgs"
    images.mkdir()
    mine = _png(images / name, 100, 60)
    theirs = _png(tmp_path / "theirs.png", 120, 80)
    assert mine != theirs
    bundle = _bundle(tmp_path / "b.zip", _doc(name, 120, 80, theirs), name, theirs)

    out = tmp_path / "side"
    rc = imp.main(["--labels", str(bundle), "--images", str(images),
                   "--out", str(out)])

    assert rc == 1
    assert (images / name).read_bytes() == mine, "existing photograph must survive"


def test_unknown_landmark_is_rejected(tmp_path):
    name = "fish.png"
    images = tmp_path / "imgs"
    images.mkdir()
    raw = _png(images / name, 100, 60)
    doc = _doc(name, 100, 60, raw)
    doc["specimens"][name]["keypoints"]["dorsal_spike"] = [5, 5]
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps(doc))

    out = tmp_path / "side"
    rc = imp.main(["--labels", str(labels), "--images", str(images), "--out", str(out)])

    assert rc == 1
    assert not list(out.glob("*.json"))


def test_existing_sidecar_is_not_clobbered(tmp_path):
    name = "fish.png"
    images = tmp_path / "imgs"
    images.mkdir()
    raw = _png(images / name, 100, 60)
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps(_doc(name, 100, 60, raw)))
    out = tmp_path / "side"
    out.mkdir()
    (out / "fish.json").write_text('{"mine": true}')

    rc = imp.main(["--labels", str(labels), "--images", str(images), "--out", str(out)])

    assert rc == 1
    assert json.loads((out / "fish.json").read_text()) == {"mine": True}


# --------------------------------------------------------------------------
# the training set must accept a contributor's naming, not just the trout rig
# --------------------------------------------------------------------------

@needs_dlc
@pytest.mark.parametrize("filename", [
    "Salvelinus_fontinalis_ASN_10_L.JPEG",   # trout rig
    "1932_CUMV_33050_01_Shad.jpg",           # alewife series
    "IMG_0042.png",                          # a contributor's phone
])
def test_find_image_resolves_every_naming_convention(tmp_path, filename):
    (tmp_path / filename).write_bytes(b"x")
    stem = Path(filename).stem
    fid = stem[:-2] if stem.endswith("_L") else stem

    assert dlc.find_image(tmp_path, fid) == tmp_path / filename


@needs_dlc
def test_find_image_prefers_the_recorded_filename(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"y")

    assert dlc.find_image(tmp_path, "a", recorded="b.jpg") == tmp_path / "b.jpg"


@needs_dlc
def test_find_image_returns_none_rather_than_guessing(tmp_path):
    (tmp_path / "unrelated.jpg").write_bytes(b"x")

    assert dlc.find_image(tmp_path, "missing") is None
