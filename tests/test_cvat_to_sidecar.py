"""Unit tests for the CVAT XML → sidecar converter script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "cvat_to_sidecar.py"


def _load_script_module():
    """Import scripts/cvat_to_sidecar.py as a module for unit testing.

    The module must be registered in ``sys.modules`` before
    ``exec_module`` runs, otherwise ``@dataclass`` can't introspect its
    own defining module and blows up with ``NoneType has no __dict__``.
    """
    spec = importlib.util.spec_from_file_location("cvat_to_sidecar", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cvat_to_sidecar"] = mod
    spec.loader.exec_module(mod)
    return mod


cvs = _load_script_module()
View = cvs.View


# ---------------------------------------------------------------------------
# Fixture: synthetic CVAT exports
# ---------------------------------------------------------------------------


def _write_lateral_xml(tmp_path: Path) -> Path:
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta></meta>
  <image id="0" name="BKT-001.jpg" width="2000" height="1500">
    <polygon label="body_plus_caudal" source="manual" occluded="0"
             points="120.5,345.1;205.3,300.2;300.0,290.0;200.0,400.0" z_order="0"/>
    <polygon label="pectoral" source="manual" occluded="0"
             points="395,370;430,395;470,430;400,420"/>
    <points label="eye_anterior" source="manual" occluded="0" points="190.5,335.1"/>
    <points label="eye_posterior" source="manual" occluded="0" points="230.0,335.0"/>
    <points label="premaxilla_tip" source="manual" occluded="0" points="120.5,345.1"/>
    <points label="mouth_left" source="manual" occluded="0" points="1420,210"/>
    <polygon label="mystery_fin" source="manual" occluded="0"
             points="1,1;2,2;3,3"/>
  </image>
  <image id="1" name="BKT-002.jpg" width="2000" height="1500">
    <points label="eye_anterior" source="manual" occluded="0" points="150.0,400.0"/>
  </image>
</annotations>
"""
    path = tmp_path / "lateral.xml"
    path.write_text(xml)
    return path


def _write_frontal_xml(tmp_path: Path) -> Path:
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta></meta>
  <image id="0" name="BKT-001.jpg" width="800" height="600">
    <points label="mouth_left" source="manual" occluded="0" points="1420,210"/>
    <points label="mouth_right" source="manual" occluded="0" points="1478,208"/>
  </image>
</annotations>
"""
    path = tmp_path / "frontal.xml"
    path.write_text(xml)
    return path


# ---------------------------------------------------------------------------
# Points-string parser
# ---------------------------------------------------------------------------


def test_parse_points_multi():
    assert cvs._parse_points("1,2;3.5,4.5") == [[1.0, 2.0], [3.5, 4.5]]


def test_parse_points_single():
    assert cvs._parse_points("10,20") == [[10.0, 20.0]]


def test_parse_points_rejects_malformed():
    with pytest.raises(ValueError):
        cvs._parse_points("1,2,3")


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def test_parse_lateral_xml_filters_by_view(tmp_path: Path):
    path = _write_lateral_xml(tmp_path)
    images = cvs.parse_cvat_xml(path, View.LATERAL)

    assert {i.name for i in images} == {"BKT-001.jpg", "BKT-002.jpg"}
    first = next(i for i in images if i.name == "BKT-001.jpg")

    # Valid polygons kept.
    assert set(first.polygons.keys()) == {"body_plus_caudal", "pectoral"}
    assert first.polygons["body_plus_caudal"][0] == [120.5, 345.1]
    assert len(first.polygons["body_plus_caudal"]) == 4

    # Valid lateral keypoints kept.
    assert set(first.keypoints.keys()) == {
        "eye_anterior",
        "eye_posterior",
        "premaxilla_tip",
    }
    assert first.keypoints["eye_anterior"] == [190.5, 335.1]

    # Frontal keypoint shouldn't end up in the lateral export.
    assert "mouth_left" not in first.keypoints
    assert "mouth_left" in first.skipped_wrong_view

    # Unknown label surfaced separately.
    assert "mystery_fin" in first.unknown_labels


def test_parse_frontal_xml(tmp_path: Path):
    path = _write_frontal_xml(tmp_path)
    images = cvs.parse_cvat_xml(path, View.FRONTAL)

    assert len(images) == 1
    img = images[0]
    assert set(img.keypoints.keys()) == {"mouth_left", "mouth_right"}
    assert img.keypoints["mouth_right"] == [1478.0, 208.0]
    assert img.polygons == {}


# ---------------------------------------------------------------------------
# Sidecar writing (fresh + merge)
# ---------------------------------------------------------------------------


def test_write_sidecars_creates_fresh_lateral(tmp_path: Path):
    xml = _write_lateral_xml(tmp_path)
    out = tmp_path / "sidecars"
    images = cvs.parse_cvat_xml(xml, View.LATERAL)
    written = cvs.write_sidecars(
        images=images,
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="none",
        calibration_map={},
        merge=False,
    )
    assert {p.name for p in written} == {"BKT-001.json", "BKT-002.json"}

    loaded = json.loads((out / "BKT-001.json").read_text())
    assert loaded["fish_id"] == "BKT-001"
    assert set(loaded["lateral"]["polygons"].keys()) == {
        "body_plus_caudal",
        "pectoral",
    }
    assert loaded["lateral"]["keypoints"]["premaxilla_tip"] == [120.5, 345.1]
    # No calibration under --calibration-mode=none.
    assert "calibration" not in loaded["lateral"]
    # No frontal block yet.
    assert "frontal" not in loaded


def test_write_sidecars_merges_frontal_into_existing(tmp_path: Path):
    lateral_xml = _write_lateral_xml(tmp_path)
    frontal_xml = _write_frontal_xml(tmp_path)
    out = tmp_path / "sidecars"

    # First pass: write lateral sidecars.
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(lateral_xml, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="none",
        calibration_map={},
        merge=False,
    )
    # Second pass: merge frontal into BKT-001 only.
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(frontal_xml, View.FRONTAL),
        out_dir=out,
        view=View.FRONTAL,
        calibration_mode="none",
        calibration_map={},
        merge=True,
    )

    merged = json.loads((out / "BKT-001.json").read_text())
    assert "lateral" in merged and "frontal" in merged
    assert merged["frontal"]["keypoints"]["mouth_right"] == [1478.0, 208.0]
    # BKT-002 wasn't in the frontal export — keeps its lateral-only sidecar.
    only_lateral = json.loads((out / "BKT-002.json").read_text())
    assert "frontal" not in only_lateral


def test_calibration_mode_auto_inserts_stub(tmp_path: Path):
    xml = _write_lateral_xml(tmp_path)
    out = tmp_path / "sidecars"
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(xml, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="auto",
        calibration_map={},
        merge=False,
    )
    loaded = json.loads((out / "BKT-001.json").read_text())
    assert loaded["lateral"]["calibration"] == {"mode": "auto"}


def test_calibration_json_overrides_mode(tmp_path: Path):
    xml = _write_lateral_xml(tmp_path)
    out = tmp_path / "sidecars"
    calib_map = {
        "BKT-001": {
            "lateral": {
                "mode": "manual",
                "point_a": [0, 500],
                "point_b": [1000, 500],
                "known_mm": 100.0,
            }
        }
    }
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(xml, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="auto",  # overridden for BKT-001 by the map
        calibration_map=calib_map,
        merge=False,
    )
    bkt1 = json.loads((out / "BKT-001.json").read_text())
    assert bkt1["lateral"]["calibration"]["mode"] == "manual"
    assert bkt1["lateral"]["calibration"]["known_mm"] == 100.0

    # BKT-002 falls through to --calibration-mode auto.
    bkt2 = json.loads((out / "BKT-002.json").read_text())
    assert bkt2["lateral"]["calibration"] == {"mode": "auto"}


# ---------------------------------------------------------------------------
# Calibration via CVAT ruler keypoints (ruler_point_a + ruler_point_b)
# ---------------------------------------------------------------------------


def _write_lateral_xml_with_ruler(tmp_path: Path) -> Path:
    """Variant fixture that includes two ruler calibration keypoints.

    Positions are chosen so the span is 800 px horizontally → at
    known_mm=80 that's a clean 10 px/mm scale.
    """
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta></meta>
  <image id="0" name="BKT-RULER.jpg" width="2000" height="1500">
    <polygon label="body_plus_caudal" source="manual" occluded="0"
             points="120.5,345.1;205.3,300.2;300.0,290.0;200.0,400.0"/>
    <points label="eye_anterior" source="manual" occluded="0" points="190.5,335.1"/>
    <points label="ruler_point_a" source="manual" occluded="0" points="100,1200"/>
    <points label="ruler_point_b" source="manual" occluded="0" points="900,1200"/>
  </image>
</annotations>
"""
    path = tmp_path / "lateral_ruler.xml"
    path.write_text(xml)
    return path


def test_ruler_points_land_in_calibration_points_not_keypoints(tmp_path: Path):
    """Ruler clicks must be parsed into ``calibration_points``, kept out of
    the anatomical ``keypoints`` dict, and not flagged as unknown."""
    xml = _write_lateral_xml_with_ruler(tmp_path)
    images = cvs.parse_cvat_xml(xml, View.LATERAL)
    assert len(images) == 1
    p = images[0]

    assert set(p.calibration_points.keys()) == {"ruler_point_a", "ruler_point_b"}
    assert p.calibration_points["ruler_point_a"] == [100.0, 1200.0]
    assert p.calibration_points["ruler_point_b"] == [900.0, 1200.0]
    assert "ruler_point_a" not in p.keypoints
    assert "ruler_point_b" not in p.keypoints
    assert "ruler_point_a" not in p.unknown_labels
    assert "ruler_point_b" not in p.unknown_labels


def test_known_mm_synthesizes_manual_calibration(tmp_path: Path):
    """With ruler clicks + --known-mm, the sidecar gets a manual calibration
    block with point_a/point_b/known_mm — exactly the shape the pipeline
    consumes."""
    xml = _write_lateral_xml_with_ruler(tmp_path)
    out = tmp_path / "sidecars"
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(xml, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="none",
        calibration_map={},
        merge=False,
        known_mm=80.0,
    )
    loaded = json.loads((out / "BKT-RULER.json").read_text())
    cal = loaded["lateral"]["calibration"]
    assert cal["mode"] == "manual"
    assert cal["point_a"] == [100.0, 1200.0]
    assert cal["point_b"] == [900.0, 1200.0]
    assert cal["known_mm"] == 80.0


def test_ruler_points_without_known_mm_fall_back_to_mode(tmp_path: Path):
    """Labeler clicked the ruler but the operator forgot --known-mm → we
    can't synthesize anything, so fall back to the --calibration-mode
    default rather than writing a half-formed calibration block."""
    xml = _write_lateral_xml_with_ruler(tmp_path)
    out = tmp_path / "sidecars"
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(xml, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="auto",
        calibration_map={},
        merge=False,
        known_mm=None,
    )
    loaded = json.loads((out / "BKT-RULER.json").read_text())
    assert loaded["lateral"]["calibration"] == {"mode": "auto"}


def test_calibration_json_beats_ruler_points(tmp_path: Path):
    """Explicit per-fish JSON override wins over CVAT ruler clicks, so you
    can patch individual specimens without throwing away the rest of the
    batch's automatic calibration."""
    xml = _write_lateral_xml_with_ruler(tmp_path)
    out = tmp_path / "sidecars"
    calib_map = {
        "BKT-RULER": {
            "lateral": {
                "mode": "manual",
                "point_a": [0, 500],
                "point_b": [1000, 500],
                "known_mm": 100.0,
            }
        }
    }
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(xml, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="none",
        calibration_map=calib_map,
        merge=False,
        known_mm=80.0,  # present but should be ignored in favor of the override
    )
    cal = json.loads((out / "BKT-RULER.json").read_text())["lateral"]["calibration"]
    assert cal["known_mm"] == 100.0
    assert cal["point_b"] == [1000, 500]


def test_only_one_ruler_point_does_not_synthesize(tmp_path: Path):
    """A labeler who dropped only ruler_point_a must NOT produce a half-baked
    calibration block; the converter falls through to --calibration-mode."""
    xml_str = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta></meta>
  <image id="0" name="BKT-HALF.jpg" width="2000" height="1500">
    <polygon label="body_plus_caudal" source="manual" occluded="0"
             points="0,0;10,0;10,10;0,10"/>
    <points label="ruler_point_a" source="manual" occluded="0" points="100,1200"/>
  </image>
</annotations>
"""
    xml_path = tmp_path / "half_ruler.xml"
    xml_path.write_text(xml_str)
    out = tmp_path / "sidecars"
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(xml_path, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="none",
        calibration_map={},
        merge=False,
        known_mm=80.0,
    )
    loaded = json.loads((out / "BKT-HALF.json").read_text())
    assert "calibration" not in loaded["lateral"]


def test_degenerate_ruler_span_skipped(tmp_path: Path):
    """If both ruler points landed at the same pixel the span is zero — we
    refuse to synthesize a divide-by-zero calibration and fall through."""
    xml_str = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta></meta>
  <image id="0" name="BKT-DEGEN.jpg" width="2000" height="1500">
    <polygon label="body_plus_caudal" source="manual" occluded="0"
             points="0,0;10,0;10,10;0,10"/>
    <points label="ruler_point_a" source="manual" occluded="0" points="500,500"/>
    <points label="ruler_point_b" source="manual" occluded="0" points="500,500"/>
  </image>
</annotations>
"""
    xml_path = tmp_path / "degen_ruler.xml"
    xml_path.write_text(xml_str)
    out = tmp_path / "sidecars"
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(xml_path, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="auto",
        calibration_map={},
        merge=False,
        known_mm=80.0,
    )
    loaded = json.loads((out / "BKT-DEGEN.json").read_text())
    # Fell through to --calibration-mode auto rather than synthesizing garbage.
    assert loaded["lateral"]["calibration"] == {"mode": "auto"}


def test_negative_known_mm_rejected(tmp_path: Path):
    xml = _write_lateral_xml_with_ruler(tmp_path)
    out = tmp_path / "sidecars"
    with pytest.raises(ValueError, match="known-mm"):
        cvs.write_sidecars(
            images=cvs.parse_cvat_xml(xml, View.LATERAL),
            out_dir=out,
            view=View.LATERAL,
            calibration_mode="none",
            calibration_map={},
            merge=False,
            known_mm=-1.0,
        )


def test_ruler_synthesized_sidecar_round_trips_through_pipeline(tmp_path: Path):
    """End-to-end: a CVAT export with ruler clicks + --known-mm should
    produce a sidecar ``process_specimen`` can measure. TL must come out
    at the expected millimeter length given the ruler's pixel span."""
    from fish_morpho.pipeline import SpecimenInput, process_specimen

    # 2000x1500 image, body polygon 0..200 px in x → TL = 200 px wide.
    # Ruler span: 0 px to 1000 px → 1000 px = known_mm.
    # At known_mm=100 → px_per_mm = 10 → TL = 200/10 = 20 mm.
    xml_str = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta></meta>
  <image id="0" name="BKT-E2E-RULER.jpg" width="2000" height="1500">
    <polygon label="body_plus_caudal" source="manual" occluded="0"
             points="0,50;200,50;200,90;0,90"/>
    <points label="eye_anterior" source="manual" occluded="0" points="20,45"/>
    <points label="ruler_point_a" source="manual" occluded="0" points="0,1200"/>
    <points label="ruler_point_b" source="manual" occluded="0" points="1000,1200"/>
  </image>
</annotations>
"""
    xml_path = tmp_path / "lateral.xml"
    xml_path.write_text(xml_str)
    out = tmp_path / "sidecars"
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(xml_path, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="none",
        calibration_map={},
        merge=False,
        known_mm=100.0,
    )
    sidecar_path = out / "BKT-E2E-RULER.json"
    sidecar = json.loads(sidecar_path.read_text())
    spec = SpecimenInput(
        fish_id="BKT-E2E-RULER",
        image_path=tmp_path / "BKT-E2E-RULER.jpg",
        sidecar_path=sidecar_path,
        sidecar=sidecar,
    )
    record = process_specimen(spec)
    assert record.measurements.values["TL"].value == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# End-to-end: converter output round-trips through the pipeline
# ---------------------------------------------------------------------------


def test_converter_output_loads_in_pipeline(tmp_path: Path):
    """A CVAT export plus a hand-supplied calibration should produce a
    sidecar that ``process_specimen`` can consume without errors."""
    from fish_morpho.pipeline import SpecimenInput, process_specimen

    # A minimum-viable lateral CVAT export: one polygon, a handful of
    # keypoints. Any missing shapes will show up as NaN traits later,
    # which is the correct behavior — we only need the loader to accept
    # the file shape here.
    xml_str = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta></meta>
  <image id="0" name="BKT-E2E.jpg" width="2000" height="1500">
    <polygon label="body_plus_caudal" source="manual" occluded="0"
             points="0,50;200,50;200,90;0,90"/>
    <points label="eye_anterior" source="manual" occluded="0" points="20,45"/>
  </image>
</annotations>
"""
    xml_path = tmp_path / "lateral.xml"
    xml_path.write_text(xml_str)
    out = tmp_path / "sidecars"
    calib_map = {
        "BKT-E2E": {
            "lateral": {
                "mode": "manual",
                "point_a": [0, 1000],
                "point_b": [1000, 1000],
                "known_mm": 100.0,
            }
        }
    }
    cvs.write_sidecars(
        images=cvs.parse_cvat_xml(xml_path, View.LATERAL),
        out_dir=out,
        view=View.LATERAL,
        calibration_mode="none",
        calibration_map=calib_map,
        merge=False,
    )

    sidecar_path = out / "BKT-E2E.json"
    sidecar = json.loads(sidecar_path.read_text())
    spec = SpecimenInput(
        fish_id="BKT-E2E",
        image_path=tmp_path / "BKT-E2E.jpg",  # placeholder, unused in manual mode
        sidecar_path=sidecar_path,
        sidecar=sidecar,
    )
    record = process_specimen(spec)
    # TL needs only body_plus_caudal; it should be a clean 20 mm.
    assert record.measurements.values["TL"].value == pytest.approx(20.0)
    # Other traits NaN out cleanly since most keypoints are absent.
    assert record.measurements.values["SL"].missing_landmarks
