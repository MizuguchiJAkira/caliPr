"""How typical a fish's shape is of each reference species.

Made-up species with known shapes: a fish drawn from one scores high for it and
low for the other; held-out fish of a species spread their scores over 0-1, as a
typicality should; a fish compared at its own size is matched where growth alone
would mislead; and the caudal-crease points never enter the comparison.
"""

from __future__ import annotations

import math
import statistics

import numpy as np

from fish_morpho import species_match as sm

NAMES = list(sm.RELIABLE) + ["caudal_crease_dorsal", "caudal_crease_ventral"]
#: a shiner-ish layout: snout at x=0, caudal base at x=100
BASE = {"dentary_anterior": (0, 2), "premaxilla_tip": (0, 0), "mouth_posterior": (6, 3),
        "orbit_anterior": (7, -2), "orbit_posterior": (13, -2), "orbit_dorsal": (10, -5),
        "orbit_ventral": (10, 1), "dorsal_fin_base_anterior": (48, -16),
        "pectoral_fin_origin": (22, 8), "anal_fin_base_anterior": (64, 12),
        "vertebral_centrum_posterior": (100, 0),
        "caudal_crease_dorsal": (98, -6), "caudal_crease_ventral": (98, 6)}


def fish(rng, deep=1.0, snout=1.0, sl_mm=50.0, noise=0.6, growth=0.0, scale=4.0, crease_shift=0.0):
    """A fish: BASE made deeper or longer-snouted, plus growth with size, jittered."""
    out = {}
    for n, (x, y) in BASE.items():
        if x < 14:
            x *= snout
        y *= deep * (1 + growth * math.log(sl_mm / 50))
        if n.startswith("caudal_crease"):
            x += crease_shift
        out[n] = ((x + rng.normal(0, noise)) * scale, (y + rng.normal(0, noise)) * scale)
    return out


def ref_set(rng, n=12, **kw):
    a = [sm.RefFish(f"A_{i}", "A", fish(rng, deep=1.0, **kw), 50.0) for i in range(n)]
    b = [sm.RefFish(f"B_{i}", "B", fish(rng, deep=1.35, snout=1.3, **kw), 50.0) for i in range(n)]
    return a + b


def test_chi2_tail():
    assert math.isclose(sm.chi2_sf(3.0, 2), math.exp(-1.5))
    assert math.isclose(sm.chi2_sf(3.0, 4), math.exp(-1.5) * 2.5)
    assert sm.chi2_sf(0, 4) == 1.0
    for k, x95 in ((1, 3.8415), (3, 7.8147), (5, 11.0705), (9, 16.919)):   # odd: the 5 % points
        assert abs(sm.chi2_sf(x95, k) - 0.05) < 1e-3


def test_a_reference_fish_with_a_gap_still_counts():
    rng = np.random.default_rng(6)
    ref = ref_set(rng)
    for r in ref[::6]:                                   # a sixth of them lack one point each
        r.kp.pop("pectoral_fin_origin")
    m = sm.match(fish(rng, deep=1.35, snout=1.3), ref)
    assert m["ok"] and m["reference_fish"] == len(ref)
    assert "pectoral_fin_origin" in m["landmarks"] and m["matches"][0]["species"] == "B"


def test_measurements_are_used_where_every_species_has_them():
    rng = np.random.default_rng(8)
    ref = ref_set(rng)
    m = sm.match(fish(rng), ref)
    assert set(m["traits"]) == {"snout length / orbit", "peduncle depth / SL"}
    for r in ref:
        if r.species == "B":
            r.kp.pop("premaxilla_tip")                   # no B fish has a snout length now
    assert sm.match(fish(rng), ref)["traits"] == ["peduncle depth / SL"]


def test_a_fish_scores_high_for_its_own_species():
    rng = np.random.default_rng(1)
    ref = ref_set(rng)
    m = sm.match(fish(rng, deep=1.35, snout=1.3), ref)
    assert m["ok"] and m["matches"][0]["species"] == "B"
    scores = {x["species"]: x["score"] for x in m["matches"]}
    assert scores["B"] > 0.2 > scores["A"]
    assert all(0 <= s <= 1 for s in scores.values())


def test_held_out_fish_spread_over_the_whole_range():
    rng = np.random.default_rng(2)
    ref = ref_set(rng, n=20)
    own = []
    for r in ref:
        m = sm.match(r.kp, [o for o in ref if o.fid != r.fid])
        own.append(next(x["score"] for x in m["matches"] if x["species"] == r.species))
    assert 0.3 < statistics.median(own) < 0.75          # a typicality, not a certainty
    assert sum(s < 0.25 for s in own) / len(own) < 0.45


def test_the_caudal_crease_is_never_shape():
    rng = np.random.default_rng(3)
    ref = ref_set(rng)
    q = fish(rng, crease_shift=40)                       # a crease far along the tail
    m = sm.match(q, ref)
    assert "caudal_crease_dorsal" not in m["landmarks"]
    assert m["matches"][0]["species"] == "A"


def test_a_fish_is_compared_at_its_own_size():
    rng = np.random.default_rng(4)
    # both species deepen as they grow, B the deeper at any one length; the
    # reference fish of both are small
    a = [sm.RefFish(f"A_{i}", "A", fish(rng, sl_mm=sl, growth=0.6, noise=0.3), sl)
         for i, sl in enumerate(np.linspace(30, 55, 16))]
    b = [sm.RefFish(f"B_{i}", "B", fish(rng, deep=1.25, sl_mm=sl, growth=0.6, noise=0.3), sl)
         for i, sl in enumerate(np.linspace(30, 55, 16))]
    big_a = fish(rng, sl_mm=90, growth=0.6, noise=0.3)  # as deep as a small B, from growth alone
    shape_only = sm.match(big_a, a + b)
    sized = sm.match(big_a, a + b, 90.0)
    assert shape_only["matches"][0]["species"] == "B"
    assert sized["size_matched"] and sized["matches"][0]["species"] == "A"
    assert all(x["size_note"] for x in sized["matches"])  # 90 mm is beyond every reference fish


def test_too_little_to_compare_is_said():
    rng = np.random.default_rng(5)
    ref = ref_set(rng)
    q = {n: p for n, p in fish(rng).items() if n in ("dentary_anterior", "orbit_anterior",
                                                     "vertebral_centrum_posterior")}
    m = sm.match(q, ref)
    assert not m["ok"] and "landmarks" in m["error"]
    assert not sm.match(fish(rng), ref[:2])["ok"]        # no species with enough fish


def test_a_terminal_mouth_lends_its_dentary_tip(tmp_path):
    import json
    (tmp_path / "sidecars").mkdir()
    rng = np.random.default_rng(9)
    for fid in ("Luxilus_cornutus_1", "Notropis_heterolepis_1"):
        kp = {n: list(p) for n, p in fish(rng).items() if n != "premaxilla_tip"}
        (tmp_path / "sidecars" / f"{fid}.json").write_text(json.dumps({"fish_id": fid, "lateral": {"keypoints": kp}}))
    ref = {r.fid: r for r in sm.load_reference(tmp_path, lambda f: f.rsplit("_", 1)[0], ["Luxilus"])}
    assert ref["Luxilus_cornutus_1"].kp["premaxilla_tip"] == ref["Luxilus_cornutus_1"].kp["dentary_anterior"]
    assert "premaxilla_tip" not in ref["Notropis_heterolepis_1"].kp        # its snout overhangs


def test_standard_length_and_ruler():
    assert sm.standard_length_px({"premaxilla_tip": (0, 0), "vertebral_centrum_posterior": (30, 40)}) == 50
    assert sm.standard_length_px({"dentary_anterior": (0, 0), "vertebral_centrum_posterior": (3, 4)}) == 5
    assert sm.standard_length_px({"orbit_anterior": (0, 0)}) is None
    assert sm.px_per_mm({"mode": "manual", "point_a": [0, 0], "point_b": [200, 0], "known_mm": 10}) == 20
    assert sm.px_per_mm({"mode": "none"}) is None
    assert sm.px_per_mm(None) is None


def test_a_misplaced_point_is_left_out_and_said():
    rng = np.random.default_rng(11)
    ref = ref_set(rng)
    q = fish(rng)                                        # an A, its dorsal origin set on the nape
    q["dorsal_fin_base_anterior"] = (24 * 4.0, -14 * 4.0)
    assert sm.match(q, ref)["matches"][0]["score"] < sm.NO_MATCH
    left = sm.misplaced(q, ref)
    assert [x["landmark"] for x in left] == ["dorsal_fin_base_anterior"] and left[0]["times"] > sm.MISPLACED
    m = sm.identify(q, ref)
    assert m["left_out"] == left and "dorsal_fin_base_anterior" not in m["landmarks"]
    assert m["matches"][0]["species"] == "A" and m["matches"][0]["score"] > 0.05
    assert m["with_all"] < sm.NO_MATCH and not m["no_match"]


def test_points_in_place_are_kept():
    rng = np.random.default_rng(12)
    ref = ref_set(rng)
    kept = sum(not sm.misplaced(fish(rng, deep=d, snout=s), ref) for d, s in [(1.0, 1.0), (1.35, 1.3)] * 15)
    assert kept >= 28                                    # of 30


def test_like_no_species():
    rng = np.random.default_rng(13)
    ref = ref_set(rng)
    m = sm.identify(fish(rng, deep=2.2, snout=0.6), ref)
    assert m["ok"] and m["no_match"]
    assert not sm.identify(fish(rng), ref)["no_match"]


def test_identify_measures_length_from_the_points_it_keeps():
    rng = np.random.default_rng(14)
    ref = ref_set(rng)
    q = fish(rng, noise=0.0)                             # SL 100 units of 4 px
    assert math.isclose(sm.identify(q, ref, px_per_mm=8.0)["sl_mm"], 50.0, rel_tol=0.02)
    assert sm.identify(q, ref)["sl_mm"] is None


def test_scores_within_a_tie_are_named_together():
    rng = np.random.default_rng(15)
    ref = ref_set(rng)
    twins = ref + [sm.RefFish(r.fid + "_twin", "A2", r.kp, r.sl_mm) for r in ref if r.species == "A"]
    m = sm.identify(fish(rng), twins)                 # A and its exact copy A2 cannot be told apart
    assert set(m["tied"][:2]) == {"A", "A2"} and "B" not in m["tied"]
    assert sm.identify(fish(rng, deep=1.35, snout=1.3), ref)["tied"] == ["B"]


def test_body_depth_over_peduncle_depth_counts_once_every_species_has_it():
    rng = np.random.default_rng(16)
    ref = ref_set(rng)

    def depth(kp, d):                                    # greatest depth d units, crease 12 apart
        kp["body_depth_dorsal"] = (40 * 4.0, (-d / 2 + rng.normal(0, 0.5)) * 4.0)
        kp["body_depth_ventral"] = (40 * 4.0, (d / 2 + rng.normal(0, 0.5)) * 4.0)
        return kp
    q = depth(fish(rng), 30)
    assert "body depth / peduncle depth" not in sm.match(q, ref)["traits"]   # no reference fish has it
    for r in ref:
        depth(r.kp, 30 if r.species == "A" else 42)
    m = sm.match(q, ref)
    assert "body depth / peduncle depth" in m["traits"] and m["matches"][0]["species"] == "A"
    assert math.isclose(sm._depth_over_peduncle({"body_depth_dorsal": (0, 0), "body_depth_ventral": (0, 30),
                                                  "caudal_crease_dorsal": (90, 0), "caudal_crease_ventral": (90, 12)}), 2.5)


def test_a_reference_fish_without_a_ruler_does_not_turn_size_off():
    rng = np.random.default_rng(17)
    ref = ref_set(rng)
    for i, r in enumerate(ref):
        r.sl_mm = 40.0 + i                               # lengths to fit growth to
    ref[0].sl_mm = None
    m = sm.match(fish(rng), ref, query_sl_mm=50.0)
    assert m["size_matched"] and m["reference_fish"] == len(ref) - 1


def test_a_species_like_none_is_not_tied():
    rng = np.random.default_rng(18)
    ref = ref_set(rng)
    m = sm.identify(fish(rng), ref)
    assert all(x["score"] >= sm.NO_MATCH for x in m["matches"] if x["species"] in m["tied"])


def test_length_counts_only_near_and_past_a_species_record():
    assert sm.size_factor(50, 100) == 1.0 and sm.size_factor(85, 100) == 1.0
    assert math.isclose(sm.size_factor(100, 100), sm.SIZE_AT_RECORD)
    assert sm.size_factor(92, 100) < 1.0 and sm.size_factor(110, 100) < 0.02
    assert sm.size_factor(None, 100) == 1.0 and sm.size_factor(50, None) == 1.0


def test_a_fish_longer_than_a_species_gets_loses_it():
    rng = np.random.default_rng(19)
    ref = ref_set(rng)
    for r in ref:
        r.sl_mm = 40.0 + rng.uniform(0, 20)
    q = fish(rng, noise=0.4)                             # an A by shape, 50 mm long (100 units of 4 px at 8 px/mm)
    sizes = {"A": {"max_sl_mm": 45, "max_tl_mm": 54, "max_source": "test"}, "B": {"max_sl_mm": 200}}
    plain, sized = sm.identify(q, ref, px_per_mm=8.0), sm.identify(q, ref, px_per_mm=8.0, sizes=sizes)
    a0 = next(x for x in plain["matches"] if x["species"] == "A")
    a1 = next(x for x in sized["matches"] if x["species"] == "A")
    assert a1["shape_score"] == a0["score"] and a1["score"] < 0.2 * a0["score"] and a1["size"]["factor"] < 0.1
    assert "shape_score" not in next(x for x in sized["matches"] if x["species"] == "B")


def test_sizes_are_found_by_any_name_gbif_knows():
    got = sm.sizes_for(["Notropis_hudsonius", "Hudsonius hudsonius", "Notropis_cornutus", "Notropis_muskoka",
                        "Notropis kanawha", "Notropis_nonesuch"])
    assert got["Notropis_hudsonius"]["max_sl_mm"] == got["Hudsonius hudsonius"]["max_sl_mm"]
    assert got["Notropis_cornutus"]["name"] == "Luxilus cornutus"
    assert got["Notropis_muskoka"]["name"] == "Notropis heterolepis"
    assert "Notropis kanawha" not in got and "Notropis_nonesuch" not in got   # ambiguous; unknown


def test_species_recorded_from_the_drainage_come_first():
    rng = np.random.default_rng(20)
    ref = ref_set(rng)
    q = fish(rng)                                        # an A by shape
    m = sm.with_drainage(sm.identify(q, ref), {"B": 9})  # but only B is recorded where it was caught
    assert [x["species"] for x in m["matches"]] == ["B", "A"]
    assert m["better_elsewhere"]["species"] == "A" and m["matches"][1]["drainage_records"] == 0
    a = next(x for x in m["matches"] if x["species"] == "A")
    assert a["score"] == next(x for x in sm.identify(q, ref)["matches"] if x["species"] == "A")["score"]  # scores untouched
    m = sm.with_drainage(sm.identify(q, ref), {"A": 3, "B": 9})
    assert m["matches"][0]["species"] == "A" and m["better_elsewhere"] is None
    m = sm.with_drainage(sm.identify(q, ref), {})
    assert m["none_here"] and m["matches"][0]["species"] == "A"
