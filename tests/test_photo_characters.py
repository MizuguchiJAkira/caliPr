"""Characters read from the photograph where the landmarks say to look, and their
use beside shape in identification.

A synthetic fish: a grey body on a pale ground, its landmarks laid out like a
shiner's, with a caudal spot, a dark snout, a side stripe or upright scale margins
painted on where a test wants them.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fish_morpho import photo_characters as pc  # noqa: E402
from fish_morpho import species_match as sm  # noqa: E402

KP = {"premaxilla_tip": (80, 200), "orbit_anterior": (120, 192), "orbit_posterior": (150, 192),
      "orbit_center": (135, 192), "opercle_dorsal": (200, 170), "cleithrum": (210, 235),
      "dorsal_fin_base_anterior": (380, 130), "dorsal_fin_base_posterior": (440, 135),
      "pelvic_fin_base_anterior": (360, 262), "anal_fin_base_anterior": (500, 255),
      "vertebral_centrum_posterior": (700, 200)}
SL = 620


def body(spot=False, snout=False, stripe=False, margins=None):
    img = np.full((400, 900, 3), 235, np.uint8)
    cv2.ellipse(img, (400, 200), (330, 75), 0, 0, 360, (175, 175, 175), -1)
    if spot:
        cv2.circle(img, (700, 200), 12, (80, 80, 80), -1)
    if snout:
        cv2.circle(img, (100, 196), 14, (90, 90, 90), -1)
    if stripe:
        cv2.rectangle(img, (150, 185), (690, 195), (90, 90, 90), -1)
    if margins == "upright":
        for x in range(230, 380, 14):
            cv2.line(img, (x, 160), (x, 240), (120, 120, 120), 2)
    elif margins == "level":
        for y in range(160, 240, 14):
            cv2.line(img, (230, y), (380, y), (120, 120, 120), 2)
    return img


def test_a_caudal_spot_reads_dark_and_a_plain_tail_does_not():
    assert pc.pigment(body(spot=True), KP)["caudal spot"] > 0.4
    assert abs(pc.pigment(body(), KP)["caudal spot"]) < 0.05


def test_snout_and_side_stripe():
    plain, marked = pc.pigment(body(), KP), pc.pigment(body(snout=True, stripe=True), KP)
    assert marked["snout"] > 0.3 > abs(plain["snout"])
    assert marked["side stripe"] > 0.4 > plain["side stripe"]


def test_upright_margins_score_above_level_ones():
    up = pc.scale_shape(body(margins="upright"), KP)["upright scale margins"]
    level = pc.scale_shape(body(margins="level"), KP)["upright scale margins"]
    assert up > 0.7 and level < 0.3


def test_dorsal_origin_against_the_pelvic_origin():
    assert math.isclose(pc.dorsal_behind_pelvic(KP), 20 / SL, rel_tol=1e-6)
    ahead = dict(KP, dorsal_fin_base_anterior=(330, 130))
    assert pc.dorsal_behind_pelvic(ahead) < 0


def test_what_the_landmarks_cannot_place_is_left_out():
    no_pelvic = {k: v for k, v in KP.items() if k != "pelvic_fin_base_anterior"}
    assert pc.pigment(body(), no_pelvic) == {}                 # no lower flank to compare with
    assert pc.characters(body(), {"orbit_center": (1, 1)}) == {}
    assert pc.dorsal_behind_pelvic(no_pelvic) is None


def test_the_fingerprint_follows_the_landmarks():
    moved = dict(KP, vertebral_centrum_posterior=(705, 200))
    assert pc.landmarks_key(KP) == pc.landmarks_key(dict(KP))
    assert pc.landmarks_key(KP) != pc.landmarks_key(moved)


def _ref(rng, n=12):
    """Two species alike in shape; A carries a caudal spot, B does not."""
    def kp():
        return {k: (x + rng.normal(0, 1.5), y + rng.normal(0, 1.5)) for k, (x, y) in KP.items()}
    a = [sm.RefFish(f"A_{i}", "A", kp(), 50.0, {"caudal spot": 0.45 + rng.normal(0, 0.05)}) for i in range(n)]
    b = [sm.RefFish(f"B_{i}", "B", kp(), 50.0, {"caudal spot": 0.0 + rng.normal(0, 0.05)}) for i in range(n)]
    return a + b


def test_a_photograph_character_tells_apart_what_shape_cannot():
    rng = np.random.default_rng(3)
    ref = _ref(rng)
    q = {k: (x + rng.normal(0, 1.5), y + rng.normal(0, 1.5)) for k, (x, y) in KP.items()}
    m = sm.match(q, ref, query_chars={"caudal spot": 0.47})
    assert "caudal spot" in m["traits"]
    assert m["matches"][0]["species"] == "A" and m["matches"][1]["score"] < 0.05
    assert "caudal spot" not in sm.match(q, ref)["traits"]       # not read from this fish


def test_a_character_goes_with_a_landmark_set_aside():
    rng = np.random.default_rng(4)
    ref = _ref(rng)
    q = dict(KP)
    m = sm.match(q, ref, query_chars={"caudal spot": 0.47}, set_aside=("vertebral_centrum_posterior",))
    assert "caudal spot" not in (m.get("traits") or [])


def test_characters_read_with_other_landmarks_are_not_used(tmp_path):
    study = tmp_path / "shiners"
    (study / "sidecars").mkdir(parents=True)
    side = {"fish_id": "Notropis_x_1", "lateral": {"keypoints": {k: list(v) for k, v in KP.items()}}}
    (study / "sidecars" / "Notropis_x_1.json").write_text(json.dumps(side))
    (study / sm.CHARACTERS_FILE).write_text(json.dumps({"fish": {"Notropis_x_1": {
        "landmarks": pc.landmarks_key(KP), "characters": {"caudal spot": 0.4}}}}))
    ref = sm.load_reference(study, lambda f: "Notropis_x")
    assert ref[0].chars == {"caudal spot": 0.4}
    side["lateral"]["keypoints"]["vertebral_centrum_posterior"] = [690, 200]
    (study / "sidecars" / "Notropis_x_1.json").write_text(json.dumps(side))
    assert sm.load_reference(study, lambda f: "Notropis_x")[0].chars == {}


def test_the_script_reads_a_study_and_keeps_what_has_not_changed(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "photo_characters_script", Path(__file__).resolve().parent.parent / "scripts" / "photo_characters.py")
    script = importlib.util.module_from_spec(spec)
    sys.modules["photo_characters_script"] = script
    spec.loader.exec_module(script)
    study = tmp_path / "shiners"
    (study / "sidecars").mkdir(parents=True)
    (study / "lateral").mkdir()
    cv2.imwrite(str(study / "lateral" / "Notropis_x_1.jpg"), body(spot=True))
    (study / "sidecars" / "Notropis_x_1.json").write_text(json.dumps(
        {"fish_id": "Notropis_x_1", "lateral": {"keypoints": {k: list(v) for k, v in KP.items()}}}))
    (study / "sidecars" / "Notropis_x_2.json").write_text(json.dumps(
        {"fish_id": "Notropis_x_2", "lateral": {"keypoints": {k: list(v) for k, v in KP.items()}}}))
    assert script.refresh(study) == {"read": 1, "kept": 0, "fish": 1}   # x_2 has no photograph
    doc = json.loads((study / sm.CHARACTERS_FILE).read_text())
    assert doc["fish"]["Notropis_x_1"]["characters"]["caudal spot"] > 0.4
    assert script.refresh(study) == {"read": 0, "kept": 1, "fish": 1}
