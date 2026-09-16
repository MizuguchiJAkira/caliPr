"""Displaced labels are found, and moved back only by a shift the image confirms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import realign_labels as R  # noqa: E402

FID = "Salvelinus_fontinalis_HRN_4"
W, H = 2400, 800
# A dark, fish-shaped body on bright foam, and the outline that traces it. Curved on
# purpose: a rectangle's top and bottom edges still line up after a sideways shift,
# so a displaced outline keeps most of its contrast -- a real fish does not.
_t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
FISH = [[int(round(1200 + 600 * np.cos(a))), int(round(400 + 110 * np.sin(a)))] for a in _t]
SNOUT = min(FISH, key=lambda q: q[0])


def _dataset(tmp_path, outline_dx=0, width=W):
    (tmp_path / "lateral").mkdir()
    (tmp_path / "sidecars").mkdir()
    img = np.full((H, width, 3), 235, np.uint8)
    cv2.fillPoly(img, [np.array(FISH, np.int32)], (40, 40, 40))
    cv2.imwrite(str(tmp_path / "lateral" / f"{FID}_L.JPEG"), img)
    doc = {"fish_id": FID, "metadata": {},
           "lateral": {"keypoints": {"premaxilla_tip": [SNOUT[0] + outline_dx, SNOUT[1]]},
                       "polygons": {R.BODY: [[x + outline_dx, y] for x, y in FISH]},
                       "calibration": {"mode": "manual", "point_a": [100 + outline_dx, 50],
                                       "point_b": [700 + outline_dx, 50], "known_mm": 50}}}
    (tmp_path / "sidecars" / f"{FID}.json").write_text(json.dumps(doc))
    return tmp_path


def _doc(ds):
    return json.loads((ds / "sidecars" / f"{FID}.json").read_text())


def test_audit_passes_labels_on_the_fish(tmp_path):
    [row] = R.audit(_dataset(tmp_path))
    assert row["result"] == "aligned"
    assert abs(row["proposed_dx"]) <= R.ALIGNED_WITHIN


def test_audit_finds_displaced_labels_and_the_shift_back(tmp_path):
    [row] = R.audit(_dataset(tmp_path, outline_dx=-450))
    assert row["result"] == "DISPLACED"
    assert abs(row["proposed_dx"] - 450) <= 8
    # Displaced, the outline keeps only a sliver of its edge on the fish. The
    # absolute threshold is calibrated to real photographs (aligned 20.6-37.6);
    # this synthetic fish is far higher contrast, so assert the ratio the method
    # rests on: HRN_4 scored 5.4 against 26.3, this 15.1 against 153.4.
    assert row["contrast_now"] < 0.25 * row["contrast_best"]


def test_legacy_labels_need_an_explicit_shift(tmp_path, capsys):
    ds = _dataset(tmp_path, outline_dx=-450)
    assert R.repair(ds, FID, None, dry_run=False) == 2
    assert "pass --dx" in capsys.readouterr().out
    assert _doc(ds)["lateral"]["polygons"][R.BODY][0] == [FISH[0][0] - 450, FISH[0][1]]  # untouched


def test_repair_moves_every_coordinate_and_keeps_a_backup(tmp_path):
    ds = _dataset(tmp_path, outline_dx=-450)
    assert R.repair(ds, FID, 450, dry_run=False) == 0
    lat = _doc(ds)["lateral"]
    assert lat["polygons"][R.BODY] == [[float(x), float(y)] for x, y in FISH]
    assert lat["keypoints"]["premaxilla_tip"] == [float(SNOUT[0]), float(SNOUT[1])]
    assert lat["calibration"]["point_a"] == [100, 50]                    # ruler clicks too
    meta = _doc(ds)["metadata"]
    assert meta["coordinate_history"][-1]["dx"] == 450
    assert meta["images"]["lateral"]["width"] == W                       # now recorded
    assert list((ds / "sidecars").glob(f"{FID}.pre-realign-*.json.bak"))
    assert R.audit(ds)[0]["result"] == "aligned"


def test_a_shift_that_misses_the_fish_is_refused(tmp_path, capsys):
    ds = _dataset(tmp_path, outline_dx=-450)
    assert R.repair(ds, FID, 900, dry_run=False) == 3
    assert "REFUSED" in capsys.readouterr().out
    assert _doc(ds)["lateral"]["polygons"][R.BODY][0] == [FISH[0][0] - 450, FISH[0][1]]


def test_dry_run_writes_nothing(tmp_path):
    ds = _dataset(tmp_path, outline_dx=-450)
    before = (ds / "sidecars" / f"{FID}.json").read_text()
    assert R.repair(ds, FID, 450, dry_run=True) == 0
    assert (ds / "sidecars" / f"{FID}.json").read_text() == before


def test_recorded_size_gives_the_exact_shift(tmp_path):
    """Once a label's image size is on record, a re-crop is pure arithmetic."""
    ds = _dataset(tmp_path)
    assert R.record_fingerprints(ds) == 0
    # the photo is re-cropped 300 px further left: wider by 300, fish 300 px right
    img = np.full((H, W + 300, 3), 235, np.uint8)
    cv2.fillPoly(img, [np.array([[x + 300, y] for x, y in FISH], np.int32)], (40, 40, 40))
    cv2.imwrite(str(ds / "lateral" / f"{FID}_L.JPEG"), img)
    [row] = R.audit(ds)
    assert row["fingerprint"] == "size_changed" and row["exact_dx"] == 300
    assert R.repair(ds, FID, None, dry_run=False) == 0
    assert _doc(ds)["lateral"]["polygons"][R.BODY][0] == [float(FISH[0][0] + 300), float(FISH[0][1])]


def test_fingerprints_are_not_recorded_while_anything_is_displaced(tmp_path, capsys):
    ds = _dataset(tmp_path, outline_dx=-450)
    assert R.record_fingerprints(ds) == 1
    assert "repair these first" in capsys.readouterr().out
    assert not (ds / "image_fingerprints.json").is_file()
