"""Smoke tests for xlsx export against the new Annotation-based engine."""

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


def test_export_writes_two_sheets(tmp_path: Path):
    records = [
        _record("BKT-001", "Hogan's Brook"),
        _record("BKT-002", "Six Mile Creek"),
    ]
    out = tmp_path / "results.xlsx"
    export_to_xlsx(records, out)
    assert out.exists()

    wb = openpyxl.load_workbook(out)
    assert set(wb.sheetnames) == {"Measurements", "QC"}

    meas = wb["Measurements"]
    rows = list(meas.iter_rows(values_only=True))
    header = rows[0]

    # Metadata columns at the front.
    assert header[:4] == ("fish_id", "locality", "collection_date", "image_filename")
    # One column per trait after metadata.
    from fish_morpho.landmark_config import TRAITS
    assert len(header) == 4 + len(TRAITS)

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
