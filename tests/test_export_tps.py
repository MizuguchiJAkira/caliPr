"""The R material: the .tps file and the ImageJ-shaped table beside it.

Coordinates are an input to geomorph, not something to read in a spreadsheet, so
this is where they are written -- and the two files in this folder use opposite y
conventions on purpose, which is the thing most worth a test.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def exported(tmp_path):
    """One fish with a scale and one without, run through the export."""
    study = tmp_path / "study"
    images, sidecars, out = study / "lateral", study / "sidecars", tmp_path / "tps"
    images.mkdir(parents=True)
    sidecars.mkdir()
    for fid, calib in (("F1", {"mode": "ticks", "px_per_mm": 10.0}), ("F2", None)):
        Image.new("RGB", (800, 600), (40, 40, 40)).save(images / f"{fid}_L.JPEG")
        block = {"keypoints": {"premaxilla_tip": [100.0, 500.0],
                               "eye_anterior": [250.0, 400.0]}}
        # placed on one fish only, so the other's row has to say so rather than
        # inventing a coordinate -- and the landmark is not dropped outright,
        # which is what happens when nobody placed it on anybody
        if fid == "F1":
            block["keypoints"]["caudal_base"] = [700.0, 450.0]
        if calib:
            block["calibration"] = calib
        (sidecars / f"{fid}.json").write_text(json.dumps(
            {"fish_id": fid, "lateral": block}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/export_tps.py"),
                        "--sidecars", str(sidecars), "--images", str(images),
                        "--schema-dir", str(study), "--out", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    return out


def _imagej(out):
    with (out / "landmarks_imagej.csv").open() as f:
        return list(csv.reader(f))


def test_the_table_is_shaped_like_imagejs_multi_measure(exported):
    """One row per landmark per specimen, named, with the photograph repeated:
    what ImageJ writes, so a caliPr series can be pooled with an ImageJ one."""
    rows = _imagej(exported)
    assert rows[0] == ["landmark", "Label", "X", "Y", "units"]
    body = {(r[0], r[1]): r for r in rows[1:]}
    snout = body[("premaxilla_tip", "F1_L.JPEG")]
    # millimetres, because this one has a scale
    assert snout[2:] == ["10.000", "50.000", "mm"]
    # y is NOT flipped here: ImageJ measures downward from the top of the image
    assert float(snout[3]) == 500.0 / 10.0


def test_coordinates_are_pixels_when_the_specimen_has_no_scale(exported):
    rows = {(r[0], r[1]): r for r in _imagej(exported)[1:]}
    assert rows[("premaxilla_tip", "F2_L.JPEG")][2:] == ["100.000", "500.000", "px"]


def test_a_landmark_nobody_placed_is_an_empty_row(exported):
    """read.csv reads that as NA, which is what estimate.missing wants."""
    rows = {r[1]: r for r in _imagej(exported)[1:] if r[0] == "caudal_base"}
    assert rows["F1_L.JPEG"][2:4] == ["70.000", "45.000"]
    assert rows["F2_L.JPEG"][2:4] == ["", ""]


def _tps_blocks(out):
    """``{image: [[x, y], ...]}`` from the .tps file."""
    blocks, rows = {}, []
    for ln in (out / "landmarks.tps").read_text().splitlines():
        if ln.startswith("LM="):
            rows = []
        elif ln.startswith("IMAGE="):
            blocks[ln.split("=", 1)[1]] = rows
        elif ln.startswith(("ID=", "SCALE=")) or not ln.strip():
            continue
        else:
            rows.append([float(v) for v in ln.split()])
    return blocks


def test_the_two_files_state_the_same_point_in_opposite_conventions(exported):
    """The one thing that silently ruins an analysis if it is got wrong.

    ``readland.tps`` expects Cartesian y, so the .tps measures up from the bottom
    of the photograph. ImageJ measures down from the top, and the .csv matches
    ImageJ so a series digitised in each can be pooled. Mixing them mirrors a
    specimen, and Procrustes will happily fit the mirrored one.
    """
    tps = _tps_blocks(exported)["F2_L.JPEG"]          # the one with no scale
    csv_rows = [r for r in _imagej(exported)[1:] if r[1] == "F2_L.JPEG"]
    # both files list a specimen's landmarks in the study's order, so row k is
    # the same point in each
    k = next(i for i, r in enumerate(csv_rows) if r[0] == "premaxilla_tip")
    assert tps[k] == [100.0, 600.0 - 500.0]
    assert [float(csv_rows[k][2]), float(csv_rows[k][3])] == [100.0, 500.0]
    assert csv_rows[k][4] == "px"


def test_a_part_scaled_series_goes_to_tps_in_pixels_throughout(exported):
    """Half a series in millimetres and half in pixels would make centroid size
    incomparable between them, which is worse than having no millimetres at all.
    The .csv keeps each specimen's own units and says which per row."""
    assert "SCALE=" not in (exported / "landmarks.tps").read_text()
    scaled = [r for r in _imagej(exported)[1:]
              if r[1] == "F1_L.JPEG" and r[0] == "premaxilla_tip"][0]
    assert scaled[2:] == ["10.000", "50.000", "mm"]     # px/10, and still y down
    assert _tps_blocks(exported)["F1_L.JPEG"][
        next(i for i, r in enumerate(_imagej(exported)[1:])
             if r[1] == "F1_L.JPEG" and r[0] == "premaxilla_tip")] == [100.0, 100.0]


def test_a_landmark_nobody_placed_on_a_fish_is_marked_missing_in_both(exported):
    """TPS has its own convention for that and read.csv has another; a made-up
    coordinate would be indistinguishable from a real one in either."""
    tps = _tps_blocks(exported)["F2_L.JPEG"]
    csv_rows = [r for r in _imagej(exported)[1:] if r[1] == "F2_L.JPEG"]
    k = next(i for i, r in enumerate(csv_rows) if r[0] == "caudal_base")
    assert tps[k] == [-1.0, -1.0]
    assert csv_rows[k][2:4] == ["", ""]


def test_the_workbook_does_not_also_carry_the_coordinates(tmp_path):
    """One home for them. Two would drift apart, and the workbook is not where
    anyone reads coordinates anyway."""
    sys.path.insert(0, str(ROOT / "src"))
    import inspect

    from fish_morpho import export

    assert "landmarks" not in inspect.signature(export.export_to_xlsx).parameters
    assert not hasattr(export, "_write_landmarks_sheet")
