"""The cross-species landmark list, and the median fish prediction frames a fish by."""

from __future__ import annotations

import math

from fish_morpho import landmark_union as lu

#: A plain fish, snout at the origin, caudal base at (1, 0), back upward.
FISH = {
    "premaxilla_tip": (0.0, 0.0), "eye_anterior": (0.05, -0.02), "eye_posterior": (0.10, -0.02),
    "operculum_posterior": (0.22, 0.02), "supraoccipital_tip": (0.20, -0.09),
    "pectoral_insertion_upper": (0.21, 0.08), "dorsal_base_anterior": (0.47, -0.10),
    "dorsal_base_posterior": (0.60, -0.08), "pelvic_base_center": (0.51, 0.13),
    "anal_base_anterior": (0.74, 0.10), "anal_base_posterior": (0.87, 0.06),
    "caudal_base": (1.0, 0.0),
}


def placed(fish: dict, scale: float, angle: float, dx: float, dy: float, mirror=False) -> dict:
    """The fish as a photograph might show it: moved, sized, turned, perhaps facing right."""
    c, s = math.cos(angle), math.sin(angle)
    out = {}
    for n, (x, y) in fish.items():
        if mirror:
            x = -x
        out[n] = (scale * (c * x - s * y) + dx, scale * (s * x + c * y) + dy)
    return out


def test_scheme_names_train_the_same_output():
    assert lu.to_union({"orbit_anterior": (1, 2), "dentary_anterior": (3, 4)}) == {
        "eye_anterior": (1, 2), "lower_jaw_tip": (3, 4)}
    assert lu.from_union({"eye_anterior": (1, 2)}, ["orbit_anterior", "mouth_left"]) == {
        "orbit_anterior": (1, 2)}


def test_the_template_is_the_same_fish_however_it_was_photographed():
    shots = [placed(FISH, 2400, 0.0, 1100, 1900),
             placed(FISH, 900, 0.3, 50, 400),
             placed(FISH, 1600, 0.1, 3000, 1500, mirror=True)]
    t = lu.template(shots)
    for n, (x, y) in FISH.items():
        assert math.isclose(t[n][0], x, abs_tol=1e-3) and math.isclose(t[n][1], y, abs_tol=1e-3), n


def test_the_template_finds_the_whole_fish_from_part_of_it_and_ignores_strays():
    t = lu.template([placed(FISH, 1000, 0.0, 0, 0)] * 3)
    truth = placed(FISH, 2000, 0.05, 800, 1500)
    # The first pass found the head and the dorsal fin, nothing of the tail, and put
    # two points on a pin far from the fish.
    found = {n: (*truth[n], 0.9) for n in ("premaxilla_tip", "eye_anterior", "eye_posterior",
                                           "operculum_posterior", "dorsal_base_anterior")}
    found["caudal_base"] = (300.0, 300.0, 0.4)
    found["anal_base_posterior"] = (320.0, 310.0, 0.3)
    place, agree, length = lu.fit_template(t, found)
    assert set(agree) == {"premaxilla_tip", "eye_anterior", "eye_posterior",
                          "operculum_posterior", "dorsal_base_anterior"}
    assert math.dist(place("caudal_base"), truth["caudal_base"]) < 0.01 * 2000
    assert math.isclose(length, 2000, rel_tol=0.01)


def test_a_fish_facing_the_other_way_is_placed_too():
    t = lu.template([placed(FISH, 1000, 0.0, 0, 0)] * 3)
    truth = placed(FISH, 1500, 0.0, 4000, 1000, mirror=True)
    found = {n: (*p, 0.8) for n, p in truth.items() if n != "caudal_base"}
    place, agree, _ = lu.fit_template(t, found)
    assert math.dist(place("caudal_base"), truth["caudal_base"]) < 0.01 * 1500


def test_nothing_is_placed_on_points_that_agree_on_nothing():
    t = lu.template([placed(FISH, 1000, 0.0, 0, 0)] * 3)
    assert lu.fit_template(t, {"premaxilla_tip": (0, 0, 0.9), "caudal_base": (5, 5, 0.9)}) is None


def test_adding_fish_keeps_every_held_out_fish_held_out():
    import importlib.util
    import sys
    from pathlib import Path

    import pytest
    pytest.importorskip("pandas")
    pytest.importorskip("cv2")
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("build_union_dataset",
                                                  scripts / "build_union_dataset.py")
    bud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bud)

    fish = lambda g, n: [{"fish_id": f"{g}_{i}", "strain": g} for i in range(n)]  # noqa: E731
    before = {"fish_train": [f"trout_{i}" for i in range(17)] + [f"shad_{i}" for i in range(4)],
              "fish_test": ["trout_17", "trout_18", "trout_19", "shad_4"]}
    train, test = bud.keep_split(fish("trout", 20) + fish("shad", 25), before, 0.15, 17)
    test_ids = {s["fish_id"] for s in test}
    assert set(before["fish_test"]) <= test_ids
    assert not set(before["fish_train"]) & test_ids
    # the new shad are split so the group's held-out share stays near 15%
    assert sum(s["strain"] == "shad" for s in test) == round(25 * 0.15)
    assert len(train) + len(test) == 45
