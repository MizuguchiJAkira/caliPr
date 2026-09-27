"""A landmark off the fish is an error, and a shift moves every stored pixel.

ASN_48 and ASN_50 carried a dorsal fin base the model had put in the corner of
the frame; one was accepted with a keypress and measured a dorsal fin 109 mm tall
on an 81 mm trout. And the un-split migration moved every landmark but not the
record of where the model first put each corrected one, so the labeller drew
those correction lines ~1000 px off.
"""

from __future__ import annotations

from types import SimpleNamespace

from fish_morpho.image_identity import shift_view
from fish_morpho.validation import check_landmarks_on_body, validate


def _rec(fid, kps):
    return SimpleNamespace(keypoints=kps, image_size=(6000, 4000),
                           measurements=SimpleNamespace(fish_id=fid, values={}),
                           calibrations={})


FISH = {"premaxilla_tip": (1800, 1650), "caudal_base": (3500, 1450),
        "dorsal_base_center": (2700, 1400), "dorsal_tip": (2750, 1300),
        "pelvic_tip": (2900, 1980), "lower_jaw_tip": (1790, 1720)}


def test_landmarks_on_the_fish_pass():
    assert check_landmarks_on_body([_rec("ok", FISH)]) == []


def test_a_corner_guess_is_an_error_naming_the_landmark():
    bad = dict(FISH, dorsal_base_center=(1040, 97))
    [issue] = check_landmarks_on_body([_rec("ASN_48", bad)])
    assert issue.level == "error" and issue.check == "landmark_off_body"
    assert "dorsal_base_center" in issue.message and "dorsal_tip" not in issue.message


def test_it_runs_with_the_other_checks():
    bad = dict(FISH, dorsal_base_center=(1040, 97))
    assert any(i.check == "landmark_off_body" for i in validate([_rec("ASN_50", bad)]))


def test_the_head_on_view_is_not_judged_against_the_side_view():
    # The mouth corners are clicked in the mirror, to the left of the fish.
    kps = dict(FISH, mouth_left=(400, 900), mouth_right=(470, 900))
    assert check_landmarks_on_body([_rec("ok", kps)]) == []


def test_without_the_body_line_nothing_is_judged():
    assert check_landmarks_on_body([_rec("x", {"dorsal_base_center": (0, 0)})]) == []


def test_a_shift_moves_the_models_first_guesses_with_the_labels():
    doc = {"lateral": {"keypoints": {"caudal_base": [2500, 1450]}},
           "metadata": {"assist": {"corrected": {
               "caudal_base": {"from": [2519, 1456], "to": [2500, 1450], "px": 19.9}}}}}
    moved = shift_view(doc, "lateral", 998)
    rec = doc["metadata"]["assist"]["corrected"]["caudal_base"]
    assert doc["lateral"]["keypoints"]["caudal_base"] == [3498, 1450]
    assert rec["from"] == [3517, 1456] and rec["to"] == [3498, 1450]
    assert rec["px"] == 19.9 and moved == 1         # a distance does not move; not a label


def test_a_frontal_shift_leaves_the_lateral_record_alone():
    doc = {"frontal": {"keypoints": {"mouth_left": [10, 10]}},
           "metadata": {"assist": {"corrected": {"caudal_base": {"from": [1, 1], "to": [2, 2]}}},
                        "assist_frontal": {"corrected": {"mouth_left": {"from": [5, 5],
                                                                        "to": [10, 10]}}}}}
    shift_view(doc, "frontal", 100)
    assert doc["metadata"]["assist"]["corrected"]["caudal_base"]["to"] == [2, 2]
    assert doc["metadata"]["assist_frontal"]["corrected"]["mouth_left"]["to"] == [110, 10]
