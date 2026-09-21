"""Smoke tests for xlsx export against the new Annotation-based engine."""

import json
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from fish_morpho.export import ExportRecord, export_to_xlsx
from fish_morpho.landmark_config import View
from fish_morpho.measurement_engine import Annotation, compute_all
from fish_morpho.ruler_calibration import CalibrationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _full_annotation() -> Annotation:
    """Synthetic fish with all 5 polygons + 21 keypoints present."""
    return Annotation(
        polygons={
            "body_plus_caudal": [
                (0, 50),
                (20, 20),
                (60, 10),
                (120, 10),
                (150, 15),
                (160, 30),
                (180, 10),
                (200, 50),
                (180, 90),
                (160, 70),
                (150, 85),
                (120, 90),
                (60, 90),
                (20, 80),
            ],
            "pectoral": [(40, 60), (55, 75), (35, 80), (30, 70)],
            "dorsal": [(80, 10), (100, -5), (115, 10)],
            "pelvic": [(90, 90), (100, 100), (85, 95)],
            "anal": [(130, 90), (145, 100), (125, 95)],
        },
        keypoints={
            "eye_anterior": (20, 45),
            "eye_posterior": (30, 45),
            "eye_dorsal": (25, 40),
            "eye_ventral": (25, 50),
            "premaxilla_tip": (0, 50),
            "maxilla_mandible_intersection": (15, 55),
            "lower_jaw_tip": (0, 55),
            "operculum_posterior": (45, 55),
            "pectoral_insertion_upper": (40, 60),
            "pectoral_ray_tip": (55, 75),
            "peduncle_narrowest_dorsal": (160, 30),
            "peduncle_narrowest_ventral": (160, 70),
            "caudal_base": (165, 50),
            "dorsal_base_center": (100, 10),
            "dorsal_tip": (100, -5),
            "pelvic_base_center": (95, 90),
            "pelvic_tip": (100, 100),
            "anal_base_center": (135, 90),
            "anal_tip": (145, 100),
            "mouth_left": (1000, 500),
            "mouth_right": (1050, 500),
        },
    )


def _calibs() -> dict:
    return {
        View.LATERAL: CalibrationResult(
            px_per_mm=10.0, method="manual", confidence=1.0, notes="lateral test"
        ),
        View.FRONTAL: CalibrationResult(
            px_per_mm=5.0, method="manual", confidence=1.0, notes="frontal test"
        ),
    }


def _record(fish_id: str, locality: str) -> ExportRecord:
    ms = compute_all(
        fish_id=fish_id,
        annotation=_full_annotation(),
        calibrations=_calibs(),
        metadata={"locality": locality, "collection_date": "2025-07-14"},
    )
    calib_map = _calibs()
    return ExportRecord(
        measurements=ms,
        calibrations={
            View.LATERAL.value: calib_map[View.LATERAL],
            View.FRONTAL.value: calib_map[View.FRONTAL],
        },
        image_filename=f"{fish_id}.jpg",
    )


# ---------------------------------------------------------------------------
# Full, populated records
# ---------------------------------------------------------------------------


def test_export_writes_its_three_sheets(tmp_path: Path):
    records = [
        _record("BKT-001", "Hogan's Brook"),
        _record("BKT-002", "Six Mile Creek"),
    ]
    out = tmp_path / "results.xlsx"
    export_to_xlsx(records, out)
    assert out.exists()

    wb = openpyxl.load_workbook(out)
    # About first, because it is what opens: what the file is, its units, and
    # whether any check fired. Validation only appears when the caller ran one.
    assert {"About", "Measurements", "Ratios", "Shape", "QC"} <= set(wb.sheetnames)
    assert wb.sheetnames[0] == "About"

    meas = wb["Measurements"]
    rows = list(meas.iter_rows(values_only=True))
    header = rows[0]

    # Metadata columns at the front, then the per-row units flag.
    # group sits second: it is what a comparative study sorts and filters by.
    assert header[:5] == ("fish_id", "group", "locality", "collection_date",
                          "image_filename")
    assert header[5] == "units"
    # One column per trait after those.
    from fish_morpho.landmark_config import TRAITS
    assert len(header) == 6 + len(TRAITS)   # 5 metadata + units

    # Every row states its own units. A workbook can hold both a specimen shot
    # with a ruler and one shot without, and pixels sitting silently under an
    # "(mm)" header is a mistake nothing downstream could detect.
    u = header.index("units")
    assert all(r[u] in ("mm", "px") for r in rows[1:])

    # Row ordering preserved.
    assert len(rows) == 3  # header + 2 fish
    assert rows[1][header.index("fish_id")] == "BKT-001"
    assert rows[2][header.index("fish_id")] == "BKT-002"
    assert rows[1][header.index("locality")] == "Hogan's Brook"

    # TL column — must be a rounded number for a fully-populated fixture.
    tl_col = next(i for i, h in enumerate(header) if str(h).startswith("TL "))
    assert isinstance(rows[1][tl_col], (int, float))
    assert rows[1][tl_col] == pytest.approx(20.0)

    # EMa is a degree trait; unit text should appear in the header.
    ema_col = next(i for i, h in enumerate(header) if str(h).startswith("EMa "))
    assert "(deg)" in str(header[ema_col])
    assert isinstance(rows[1][ema_col], (int, float))


def test_export_qc_sheet_has_row_per_view(tmp_path: Path):
    records = [_record("BKT-001", "Hogan's Brook")]
    out = tmp_path / "qc.xlsx"
    export_to_xlsx(records, out)

    wb = openpyxl.load_workbook(out)
    qc = wb["QC"]
    qc_rows = list(qc.iter_rows(values_only=True))
    assert qc_rows[0][0] == "fish_id"
    assert "calibration_method" in qc_rows[0]
    # Header + one row per view (lateral, frontal).
    assert len(qc_rows) == 3


# ---------------------------------------------------------------------------
# Sparse / missing inputs → blank cells on measurements, missing list on QC
# ---------------------------------------------------------------------------


def test_export_handles_missing_measurements_as_blank(tmp_path: Path):
    # Empty annotation — every trait is missing, every cell should be blank.
    ann = Annotation()
    calibs = {
        View.LATERAL: CalibrationResult(
            px_per_mm=10.0, method="manual", confidence=1.0
        ),
        View.FRONTAL: CalibrationResult(
            px_per_mm=5.0, method="manual", confidence=1.0
        ),
    }
    ms = compute_all(
        fish_id="BKT-003",
        annotation=ann,
        calibrations=calibs,
        metadata={"locality": "Nowhere"},
    )
    rec = ExportRecord(
        measurements=ms,
        calibrations={View.LATERAL.value: calibs[View.LATERAL]},
        image_filename="BKT-003.jpg",
    )

    out = tmp_path / "sparse.xlsx"
    export_to_xlsx([rec], out)

    wb = openpyxl.load_workbook(out)
    rows = list(wb["Measurements"].iter_rows(values_only=True))
    header = rows[0]
    # Every trait column should be blank for an empty annotation.
    from fish_morpho.landmark_config import TRAITS
    blanks = sum(
        1 for cell in rows[1][len(header) - len(TRAITS):] if cell in ("", None)
    )
    assert blanks == len(TRAITS)

    # QC sheet should report at least one missing polygon/keypoint label.
    qc_rows = list(wb["QC"].iter_rows(values_only=True))
    missing_col = qc_rows[0].index("missing_landmarks")
    assert "polygon:body_plus_caudal" in str(qc_rows[1][missing_col])


def test_tps_exports_a_study_s_own_landmarks_and_its_names(tmp_path):
    """A landmark the study added is a coordinate on every specimen, which is what
    TPS carries; the names file says both what it is stored as and what it is called."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import export_tps

    study = tmp_path / "study"
    study.mkdir()
    (study / "schema.json").write_text(json.dumps({
        "exclude_keypoints": ["lower_jaw_tip"],
        "extra_keypoints": [{"name": "adipose_base", "label": "Adipose base", "view": "lateral"},
                            {"name": "cheek_spot", "label": "Cheek spot", "view": "frontal"}],
        "labels": {"premaxilla_tip": "snout tip"}}))
    order = export_tps.landmark_order(study)
    assert order[-1] == "adipose_base"            # last, after the master landmarks
    assert "cheek_spot" not in order              # frontal: not in a lateral TPS
    assert "lower_jaw_tip" not in order
    labels = export_tps.landmark_labels(study)
    assert labels["premaxilla_tip"] == "snout tip"
    assert labels["adipose_base"] == "Adipose base"      # what the study calls its own


def test_the_landmarks_sheet_is_shaped_like_imagejs_multi_measure(tmp_path):
    """One row per landmark per specimen, named, with the photograph repeated:
    what ImageJ writes, so a caliPr series can be pooled with an ImageJ one."""
    import openpyxl

    ann = Annotation()
    ann.keypoints["premaxilla_tip"] = (100.0, 500.0)
    ann.keypoints["eye_anterior"] = (250.0, 400.0)
    calib = CalibrationResult(px_per_mm=10.0, method="manual", confidence=1.0)
    ms = compute_all("F1", ann, {View.LATERAL: calib, View.FRONTAL: calib})
    rec = ExportRecord(measurements=ms, calibrations={"lateral": calib},
                       image_filename="F1.jpg", keypoints=dict(ann.keypoints))
    out = tmp_path / "wb.xlsx"
    export_to_xlsx([rec], out, landmarks=["premaxilla_tip", "eye_anterior", "caudal_base"],
                   landmark_labels={"premaxilla_tip": "snout tip"})
    rows = list(openpyxl.load_workbook(out)["Landmarks"].iter_rows(values_only=True))
    assert rows[0] == ("landmark", "Label", "X", "Y", "units")
    # the study's landmark order, its own names, and mm because this one has a scale
    assert rows[1] == ("snout tip", "F1.jpg", 10.0, 50.0, "mm")
    assert rows[2] == ("eye_anterior", "F1.jpg", 25.0, 40.0, "mm")
    # y is NOT flipped here: ImageJ measures downward from the top of the image
    assert rows[1][3] == 500.0 / 10.0
    # a landmark nobody placed is an empty row, which read.csv reads as NA
    assert rows[3] == ("caudal_base", "F1.jpg", None, None, "mm")


def test_landmark_coordinates_are_pixels_when_the_specimen_has_no_scale(tmp_path):
    import openpyxl

    ann = Annotation()
    ann.keypoints["premaxilla_tip"] = (100.0, 500.0)
    free = CalibrationResult(px_per_mm=1.0, method="none", confidence=0.0)
    ms = compute_all("F2", ann, {View.LATERAL: free})
    rec = ExportRecord(measurements=ms, calibrations={}, image_filename="F2.jpg",
                       keypoints=dict(ann.keypoints))
    out = tmp_path / "wb2.xlsx"
    export_to_xlsx([rec], out, landmarks=["premaxilla_tip"])
    [_, row] = list(openpyxl.load_workbook(out)["Landmarks"].iter_rows(values_only=True))
    assert row == ("premaxilla_tip", "F2.jpg", 100.0, 500.0, "px")


def test_a_workbook_without_a_landmark_order_has_no_landmarks_sheet(tmp_path):
    import openpyxl

    ms = compute_all("F3", Annotation(), {})
    rec = ExportRecord(measurements=ms, calibrations={}, image_filename="F3.jpg")
    out = tmp_path / "wb3.xlsx"
    export_to_xlsx([rec], out)
    assert "Landmarks" not in openpyxl.load_workbook(out).sheetnames


def test_the_tps_folder_carries_the_same_table_for_r(tmp_path):
    """Same coordinates, same order, ImageJ's shape — next to the .tps."""
    import subprocess
    import sys as _sys

    study = tmp_path / "study"
    (study / "lateral").mkdir(parents=True)
    (study / "sidecars").mkdir()
    from PIL import Image
    Image.new("RGB", (1000, 600), (200, 200, 200)).save(study / "lateral" / "F1_L.JPEG")
    (study / "sidecars" / "F1.json").write_text(json.dumps({
        "fish_id": "F1",
        "lateral": {"keypoints": {"premaxilla_tip": [100, 500], "eye_anterior": [250, 400]},
                    "calibration": {"mode": "manual", "point_a": [0, 0], "point_b": [100, 0],
                                    "known_mm": 10.0}}}))
    out = tmp_path / "tps"
    r = subprocess.run([_sys.executable, str(Path(__file__).resolve().parent.parent
                                             / "scripts/export_tps.py"),
                        "--sidecars", str(study / "sidecars"), "--images", str(study / "lateral"),
                        "--schema-dir", str(study), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = (out / "landmarks_imagej.csv").read_text().strip().split("\n")
    assert lines[0] == "landmark,Label,X,Y,units"
    rows = {l.split(",")[0]: l.split(",") for l in lines[1:]}
    assert rows["premaxilla_tip"][1] == "F1_L.JPEG"
    assert rows["premaxilla_tip"][2:5] == ["10.000", "50.000", "mm"]   # px/mm = 10, y downward
    # the .tps beside it flips y into Cartesian: the two must not be mixed
    tps = (out / "landmarks.tps").read_text()
    assert "100.0000 100.0000" in tps                                   # 600 - 500
    assert "arrayspecs" in (out / "load_landmarks.R").read_text()
