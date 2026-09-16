"""A re-crop must never leave saved labels pointing at the wrong pixels.

HRN_4 was labelled on a lateral crop starting at x=1656 and re-cropped to start
at x=1206; every coordinate then sat 450 px into the ruler. These reproduce that
and check each way out: refused, moved exactly, or refused when a shift would be
a guess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import preprocess_cornell as pc  # noqa: E402

BASE = "Salvelinus_fontinalis_HRN_4"
W, H = 900, 300


def _photo(seed=0):
    """A smooth 'photograph' whose columns are still distinguishable.

    Smooth because JPEG is: raw per-pixel noise does not survive a round trip
    within a few grey levels, so a noise image would make every crop look like a
    different photograph -- which the guard then correctly refuses to move.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (H // 12, W // 12, 3), dtype=np.uint8)
    return cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """process_one on a synthetic photo, with orientation and split under our control."""
    photo = {"img": _photo()}
    raw = tmp_path / "raw.JPG"
    raw.write_bytes(b"not read")
    monkeypatch.setattr(pc, "normalize_orientation", lambda _p: photo["img"].copy())

    def run(margin=0, shift=False, boundary=300):
        return pc.process_one(raw, tmp_path, "HRN", 4, boundary_override=boundary,
                              lateral_margin=margin, shift_labels=shift)

    def label(lat=None, fro=None):
        (tmp_path / "sidecars").mkdir(exist_ok=True)
        doc = {"fish_id": BASE, "metadata": {}}
        if lat is not None:
            doc["lateral"] = lat
        if fro is not None:
            doc["frontal"] = fro
        (tmp_path / "sidecars" / f"{BASE}.json").write_text(json.dumps(doc))

    def sidecar():
        return json.loads((tmp_path / "sidecars" / f"{BASE}.json").read_text())

    return type("Rig", (), dict(run=staticmethod(run), label=staticmethod(label),
                                sidecar=staticmethod(sidecar), photo=photo, dir=tmp_path))


LAT = {"keypoints": {"premaxilla_tip": [17, 150]},
       "polygons": {"body_plus_caudal": [[8, 100], [500, 100], [500, 200], [8, 200]]},
       "calibration": {"mode": "manual", "point_a": [230, 40], "point_b": [480, 42],
                       "known_mm": 10}}


def test_unlabelled_fish_is_cropped_freely(rig):
    rig.run(margin=0)
    rig.run(margin=100)                       # no sidecar: nothing to protect
    assert (rig.dir / "lateral" / f"{BASE}_L.JPEG").is_file()


def test_same_crop_again_is_allowed(rig):
    rig.run(margin=0)
    rig.label(lat=LAT)
    rig.run(margin=0)                         # identical start: labels still fit
    assert rig.sidecar()["lateral"] == LAT


def test_recrop_under_labels_is_refused_and_nothing_is_written(rig):
    rig.run(margin=0)
    rig.label(lat=LAT)
    before = (rig.dir / "lateral" / f"{BASE}_L.JPEG").read_bytes()
    with pytest.raises(pc.LabelledImageError, match=r"displace every label by \+100 px"):
        rig.run(margin=100)
    assert (rig.dir / "lateral" / f"{BASE}_L.JPEG").read_bytes() == before
    assert rig.sidecar()["lateral"] == LAT


def test_shift_labels_moves_every_coordinate_exactly(rig):
    rig.run(margin=0)
    rig.label(lat=LAT)
    rig.run(margin=100, shift=True)
    lat = rig.sidecar()["lateral"]
    assert lat["keypoints"]["premaxilla_tip"] == [117, 150]
    assert lat["polygons"]["body_plus_caudal"][0] == [108, 100]
    assert lat["calibration"]["point_a"] == [330, 40]      # the ruler clicks move too
    assert lat["calibration"]["point_b"] == [580, 42]
    hist = rig.sidecar()["metadata"]["coordinate_history"][-1]
    assert hist["dx"] == 100 and hist["points"] == 7
    assert list((rig.dir / "sidecars").glob(f"{BASE}.pre-recrop-*.json.bak"))


def test_shifted_labels_land_on_the_same_pixels(rig):
    """The point of the shift: each coordinate still names the same raw pixel."""
    rig.run(margin=0)
    rig.label(lat=LAT)
    old = cv2.imread(str(rig.dir / "lateral" / f"{BASE}_L.JPEG"))
    x, y = LAT["keypoints"]["premaxilla_tip"]
    rig.run(margin=100, shift=True)
    new = cv2.imread(str(rig.dir / "lateral" / f"{BASE}_L.JPEG"))
    nx, ny = rig.sidecar()["lateral"]["keypoints"]["premaxilla_tip"]
    assert np.abs(old[y, x].astype(int) - new[int(ny), int(nx)].astype(int)).max() <= 12


def test_a_different_photograph_is_refused_even_with_shift(rig):
    """Two raw files mapped to one specimen are not related by a shift at all."""
    rig.run(margin=0)
    rig.label(lat=LAT)
    rig.photo["img"] = _photo(seed=99)
    with pytest.raises(pc.LabelledImageError, match="not a slice of this photograph"):
        rig.run(margin=100, shift=True)
    assert rig.sidecar()["lateral"] == LAT


def test_frontal_crop_that_would_cut_off_a_label_is_refused(rig):
    rig.run(margin=0, boundary=300)
    rig.label(fro={"keypoints": {"mouth_left": [250, 100], "mouth_right": [290, 100]}})
    with pytest.raises(pc.LabelledImageError, match="cutting off a labelled point"):
        rig.run(boundary=200)


def test_fingerprints_are_recorded_for_labelled_fish_only(rig):
    rig.run(margin=0)
    assert not (rig.dir / "image_fingerprints.json").is_file()
    rig.label(lat=LAT)
    rig.run(margin=0)
    fp = json.loads((rig.dir / "image_fingerprints.json").read_text())[BASE]["lateral"]
    assert (fp["width"], fp["height"]) == (W - 300, H)
