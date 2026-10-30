"""Landmarks a study places by hand: collected, but never offered by Auto-label.

``exclude_predicted_keypoints`` in a study's schema.json -- for points where the
model's guesses cost more to check than to click, as the brook trout study found
for the caudal points. Tested: the setting reaches the labeler, and a
prediction is stripped of those points (and their confidences) before it is
offered, while the rest of it is offered as before.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load("label_server", SCRIPTS / "label_server.py")


def _study(tmp_path, **prof):
    (tmp_path / "lateral").mkdir()
    (tmp_path / "schema.json").write_text(json.dumps(prof))
    return tmp_path


def test_the_setting_reaches_the_labeler(tmp_path):
    st = _study(tmp_path, exclude_predicted_keypoints=["caudal_base"])
    prof = ls.load_profile(st)
    assert prof["exclude_predicted_keypoints"] == {"caudal_base"}
    assert ls.build_schema(prof)["exclude_predicted_keypoints"] == ["caudal_base"]
    # still collected: the landmark stays in the list, not struck through
    lat = {k["name"]: k for k in ls.build_schema(prof)["lateral"]["keypoints"]}
    assert "caudal_base" in lat and not lat["caudal_base"].get("excluded")


def test_a_prediction_is_offered_without_them(tmp_path):
    st = _study(tmp_path, exclude_predicted_keypoints=["caudal_base"],
                exclude_keypoints=["peduncle_narrowest_dorsal"])
    h = ls.Handler.__new__(ls.Handler)
    h.images_dir = st
    res = h._excluded({"ok": True, "polygons": {},
                       "keypoints": {"caudal_base": [1, 2], "premaxilla_tip": [3, 4],
                                     "peduncle_narrowest_dorsal": [5, 6]},
                       "confidence": {"caudal_base": 0.3, "premaxilla_tip": 0.9,
                                      "peduncle_narrowest_dorsal": 0.4},
                       "low_confidence": ["caudal_base", "peduncle_narrowest_dorsal"]})
    assert set(res["keypoints"]) == {"premaxilla_tip"}
    assert set(res["confidence"]) == {"premaxilla_tip"} and res["low_confidence"] == []
