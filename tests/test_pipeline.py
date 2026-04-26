"""End-to-end pipeline test in manual mode.

Creates a pair of placeholder image files (manual calibration never
actually reads them) and matching JSON sidecars in the new nested
polygon + keypoint schema, then runs the pipeline and verifies the
exported xlsx contains a row per fish plus QC rows.
"""

import json
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from fish_morpho.pipeline import discover_specimens, run


# ---------------------------------------------------------------------------
# Sidecar payload — mirrors examples/sample_sidecar.json but trimmed down
# ---------------------------------------------------------------------------


def _sidecar_payload(fish_id: str) -> dict:
    """Minimal-but-complete sidecar covering all 5 polygons + 21 keypoints."""
    return {
        "fish_id": fish_id,
        "metadata": {
            "locality": "Hogan's Brook",
            "collection_date": "2025-07-14",
        },
        "lateral": {
            "polygons": {
                "body_plus_caudal": [
                    [0, 50],
                    [20, 20],
                    [60, 10],
                    [120, 10],
                    [150, 15],
                    [160, 30],
                    [180, 10],
                    [200, 50],
                    [180, 90],
                    [160, 70],
                    [150, 85],
                    [120, 90],
                    [60, 90],
                    [20, 80],
                ],
                "pectoral": [[40, 60], [55, 75], [35, 80], [30, 70]],
                "dorsal": [[80, 10], [100, -5], [115, 10]],
                "pelvic": [[90, 90], [100, 100], [85, 95]],
                "anal": [[130, 90], [145, 100], [125, 95]],
            },
            "keypoints": {
                "eye_anterior": [20, 45],
                "eye_posterior": [30, 45],
                "eye_dorsal": [25, 40],
                "eye_ventral": [25, 50],
                "premaxilla_tip": [0, 50],
                "maxilla_mandible_intersection": [15, 55],
                "lower_jaw_tip": [0, 55],
                "operculum_posterior": [45, 55],
                "pectoral_insertion_upper": [40, 60],
                "pectoral_ray_tip": [55, 75],
                "peduncle_narrowest_dorsal": [160, 30],
                "peduncle_narrowest_ventral": [160, 70],
                "caudal_base": [165, 50],
                "dorsal_base_center": [100, 10],
                "dorsal_tip": [100, -5],
                "pelvic_base_center": [95, 90],
                "pelvic_tip": [100, 100],
                "anal_base_center": [135, 90],
                "anal_tip": [145, 100],
            },
            "calibration": {
                "mode": "manual",
                "point_a": [0, 500],
                "point_b": [1000, 500],
                "known_mm": 100.0,
            },
        },
        "frontal": {
            "keypoints": {
                "mouth_left": [0, 0],
                "mouth_right": [50, 0],
            },
            "calibration": {
                "mode": "manual",
                "point_a": [0, 700],
                "point_b": [100, 700],
                "known_mm": 20.0,
            },
        },
    }


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_pipeline_manual_mode_end_to_end(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    for fid in ("BKT-0001", "BKT-0002"):
        (images / f"{fid}.jpg").write_bytes(b"\x00")  # placeholder
        (labels / f"{fid}.json").write_text(json.dumps(_sidecar_payload(fid)))

    out = tmp_path / "results.xlsx"
    returned = run(
        images_dir=images,
        labels_dir=labels,
        output_path=out,
        mode="manual",
        model_config=None,
    )
    assert returned == out
    assert out.exists()

    wb = openpyxl.load_workbook(out)
    assert set(wb.sheetnames) == {"Measurements", "QC"}

    rows = list(wb["Measurements"].iter_rows(values_only=True))
    header = rows[0]
    assert "fish_id" in header
    assert len(rows) == 3  # header + 2 fish

    id_col = header.index("fish_id")
    ids = {row[id_col] for row in rows[1:]}
    assert ids == {"BKT-0001", "BKT-0002"}

    # TL should be a clean, populated number (no blank, no NaN).
    tl_col = next(i for i, h in enumerate(header) if str(h).startswith("TL "))
    tl_values = [row[tl_col] for row in rows[1:]]
    assert all(isinstance(v, (int, float)) for v in tl_values)
    assert all(v > 0 for v in tl_values)

    # QC sheet: header + lateral/frontal rows for each of the two fish.
    qc_rows = list(wb["QC"].iter_rows(values_only=True))
    assert qc_rows[0][0] == "fish_id"
    assert len(qc_rows) == 1 + 2 * 2


# ---------------------------------------------------------------------------
# Discovery error paths
# ---------------------------------------------------------------------------


def test_discover_specimens_errors_on_orphan_sidecar(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (labels / "lonely.json").write_text(json.dumps(_sidecar_payload("x")))
    with pytest.raises(FileNotFoundError):
        discover_specimens(images, labels)


def test_discover_specimens_skips_image_without_sidecar(tmp_path: Path, caplog):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "no_sidecar.jpg").write_bytes(b"\x00")
    (images / "paired.jpg").write_bytes(b"\x00")
    (labels / "paired.json").write_text(json.dumps(_sidecar_payload("paired")))

    with caplog.at_level("WARNING", logger="fish_morpho.pipeline"):
        specs = discover_specimens(images, labels)

    assert [s.fish_id for s in specs] == ["paired"]
    assert any("no_sidecar" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Sidecar validation
# ---------------------------------------------------------------------------


def test_pipeline_rejects_sidecar_without_lateral_block(tmp_path: Path):
    from fish_morpho.pipeline import process_specimen, SpecimenInput

    sidecar = {
        "fish_id": "no-lateral",
        "frontal": {
            "keypoints": {"mouth_left": [0, 0], "mouth_right": [10, 0]},
            "calibration": {
                "mode": "manual",
                "point_a": [0, 0],
                "point_b": [10, 0],
                "known_mm": 5.0,
            },
        },
    }
    spec = SpecimenInput(
        fish_id="no-lateral",
        image_path=tmp_path / "x.jpg",
        sidecar_path=tmp_path / "x.json",
        sidecar=sidecar,
    )
    with pytest.raises(ValueError, match="lateral"):
        process_specimen(spec)


def test_auto_mode_requires_model_config(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    with pytest.raises(ValueError, match="model-config"):
        run(
            images_dir=images,
            labels_dir=None,
            output_path=tmp_path / "out.xlsx",
            mode="auto",
            model_config=None,
        )


def test_manual_mode_requires_labels_dir(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    with pytest.raises(ValueError, match="labels"):
        run(
            images_dir=images,
            labels_dir=None,
            output_path=tmp_path / "out.xlsx",
            mode="manual",
            model_config=None,
        )
