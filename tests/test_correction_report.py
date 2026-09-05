"""Tests for the correction-feedback aggregation.

The point of the report is to keep three kinds of evidence apart. A test suite
that only checked the arithmetic would miss the thing most likely to go wrong,
which is corrections and acceptances quietly getting pooled.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cr = _load("correction_report")


def _sidecar(path: Path, fid: str, corrected=None, accepted=(), unreviewed=()):
    doc = {
        "fish_id": fid,
        "metadata": {
            "source": "hand-labeled",
            "assisted_by": "snapshot-best-170.pt",
            "assist": {
                "model": "snapshot-best-170.pt",
                "accepted": list(accepted),
                "corrected": corrected or {},
                "unreviewed": list(unreviewed),
            },
        },
        "lateral": {"keypoints": {}},
    }
    (path / f"{fid}.json").write_text(json.dumps(doc))


def test_only_assisted_sidecars_are_read(tmp_path):
    """A hand label with no assist block is not evidence about the model."""
    _sidecar(tmp_path, "assisted", accepted=["eye_dorsal"])
    (tmp_path / "plain.json").write_text(json.dumps(
        {"fish_id": "plain", "metadata": {"source": "hand-labeled"},
         "lateral": {"keypoints": {}}}))

    rows = cr.load(tmp_path)

    assert [fid for fid, _ in rows] == ["assisted"]


def test_corrupt_sidecar_is_skipped_not_fatal(tmp_path):
    _sidecar(tmp_path, "good", accepted=["eye_dorsal"])
    (tmp_path / "bad.json").write_text("{not json")

    assert [fid for fid, _ in cr.load(tmp_path)] == ["good"]


def test_correction_distances_are_read_per_landmark(tmp_path, capsys):
    _sidecar(tmp_path, "f1", corrected={
        "dorsal_tip": {"from": [0, 0], "to": [30, 40], "px": 50.0, "conf": 0.35}})
    _sidecar(tmp_path, "f2", corrected={
        "dorsal_tip": {"from": [0, 0], "to": [60, 80], "px": 100.0, "conf": 0.4}})

    cr.main(["--sidecars", str(tmp_path)])
    out = capsys.readouterr().out

    assert "dorsal_tip" in out
    assert "2 assisted specimen(s)" in out
    assert "75.0" in out, "median of 50 and 100"
    assert "100.0" in out, "worst"


def test_px_per_mm_converts(tmp_path, capsys):
    _sidecar(tmp_path, "f1", corrected={
        "dorsal_tip": {"from": [0, 0], "to": [0, 20], "px": 20.0, "conf": 0.3}})

    cr.main(["--sidecars", str(tmp_path), "--px-per-mm", "10"])

    assert "2.0" in capsys.readouterr().out


def test_correction_rate_excludes_unreviewed_points(tmp_path, capsys):
    """Unreviewed means nobody looked. Counting it as agreement would make an
    untouched batch look like a validated one."""
    _sidecar(tmp_path, "f1",
             corrected={"dorsal_tip": {"from": [0, 0], "to": [3, 4],
                                       "px": 5.0, "conf": 0.3}},
             accepted=["dorsal_tip_other"],
             unreviewed=["a", "b", "c", "d", "e", "f", "g", "h"])

    cr.main(["--sidecars", str(tmp_path), "--json", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    data = json.loads((tmp_path / "r.json").read_text())

    # one corrected, one accepted -> two points actually reviewed -> 50%
    assert "1/2 reviewed points needed moving (50%)" in out
    assert data["landmarks"]["dorsal_tip"]["correction_rate"] == 1.0
    assert data["landmarks"]["dorsal_tip"]["unreviewed"] == 0
    assert "8 predicted point(s) were never reviewed" in out


def test_accepted_and_corrected_are_never_pooled(tmp_path, capsys):
    _sidecar(tmp_path, "f1",
             corrected={"dorsal_tip": {"from": [0, 0], "to": [3, 4],
                                       "px": 5.0, "conf": 0.3}},
             accepted=["eye_dorsal", "premaxilla_tip"])

    cr.main(["--sidecars", str(tmp_path), "--json", str(tmp_path / "r.json")])
    data = json.loads((tmp_path / "r.json").read_text())

    assert data["landmarks"]["eye_dorsal"] == {
        "corrected": 0, "accepted": 1, "unreviewed": 0,
        "median": None, "worst": None, "correction_rate": 0.0}
    assert data["landmarks"]["dorsal_tip"]["corrected"] == 1
    assert data["landmarks"]["dorsal_tip"]["accepted"] == 0
    # An accepted landmark must never acquire a correction distance.
    assert data["landmarks"]["eye_dorsal"]["median"] is None


def test_confidence_of_corrected_points_is_reported(tmp_path, capsys):
    """Whether corrections land on flagged points is the check on the gate."""
    _sidecar(tmp_path, "f1", corrected={
        "a": {"from": [0, 0], "to": [1, 0], "px": 1.0, "conf": 0.30},
        "b": {"from": [0, 0], "to": [1, 0], "px": 1.0, "conf": 0.95}})

    cr.main(["--sidecars", str(tmp_path)])

    assert "1/2 corrections were on points the model had already flagged" \
        in capsys.readouterr().out


def test_no_assisted_sidecars_is_not_an_error(tmp_path, capsys):
    assert cr.main(["--sidecars", str(tmp_path)]) == 0
    assert "No assisted sidecars" in capsys.readouterr().out
