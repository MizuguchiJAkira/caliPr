"""Operator IDs, and measurement error from blind re-label rounds.

Every landmark carries who placed it, and a subset of fish labelled again blind
gives each trait's ICC and %ME. What is tested: the statistics against Shrout &
Fleiss's worked example and against the identities that define them; a round
draws across the study's groups, hides which fish is which from everything the
page can see, never offers the reference fish, and never touches the originals;
predictions are refused inside a round; the analysis pairs each re-label with its
original through the same pipeline; and the workbook carries the table, the
operators, and who determined each Darwin Core measurement.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import random
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fish_morpho import operators, repeatability
from fish_morpho.repeatability import create_round, f_ppf, trait_error

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
openpyxl = pytest.importorskip("openpyxl")
Image = pytest.importorskip("PIL.Image")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


payload = _load("_pipeline_payload", TESTS / "test_pipeline.py")._sidecar_payload


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

SHROUT_FLEISS = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]


def test_the_statistics_match_shrout_and_fleiss():
    t = trait_error(SHROUT_FLEISS)
    assert (t.n, t.k) == (6, 4)
    assert t.icc1 == pytest.approx(0.17, abs=0.005)
    assert t.icc31 == pytest.approx(0.71, abs=0.005)
    assert (t.icc1_lo, t.icc1_hi) == (pytest.approx(-0.13, abs=0.01), pytest.approx(0.72, abs=0.01))
    assert t.pct_me == pytest.approx(100 * (1 - t.icc1))           # the identity %ME rests on


def test_f_quantiles_match_the_tables():
    assert f_ppf(0.95, 5, 10) == pytest.approx(3.326, abs=1e-3)
    assert f_ppf(0.975, 10, 10) == pytest.approx(3.717, abs=1e-3)
    assert f_ppf(0.99, 3, 20) == pytest.approx(4.938, abs=1e-3)


def test_identical_measurements_have_no_error():
    t = trait_error([[10.0, 10.0], [12.0, 12.0], [15.0, 15.0], [11.0, 11.0]])
    assert t.icc1 == pytest.approx(1) and t.pct_me == pytest.approx(0) and t.tem == 0
    assert t.verdict == "excellent"


def test_a_systematic_shift_is_error_to_icc1_but_not_to_icc31():
    rows = [[v, v + 2.0 + 0.05 * i] for i, v in enumerate([10, 14, 18, 22, 26, 30])]
    t = trait_error(rows)
    assert t.bias == pytest.approx(2.125, abs=0.01)
    assert t.icc31 > 0.999 and t.icc1 < t.icc31 - 0.02


def test_too_few_fish_give_no_coefficients():
    t = trait_error([[1.0, 1.1], [2.0, 2.1]])
    assert t.icc1 is None and t.verdict == ""


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def test_operators_are_read_per_landmark():
    sc = {"lateral": {"operators": {"premaxilla_tip": "JC", "caudal_base": "JC",
                                    "eye_anterior": "AB", "dorsal_tip": "model"}},
          "frontal": {"operators": {"mouth_left": "AB"}}}
    assert operators.people(sc) == ["AB", "JC"] or operators.people(sc) == ["JC", "AB"]
    assert operators.summary(sc).endswith("1 unreviewed model point")
    assert operators.determined_by(sc, ["premaxilla_tip", "caudal_base"]) == "JC"
    assert operators.determined_by(sc, ["premaxilla_tip", "dorsal_tip"]) == "JC; model"
    assert operators.determined_by(sc, ["nothing"]) == ""
    assert operators.summary({"lateral": {}}) == ""


# ---------------------------------------------------------------------------
# a study with labelled fish
# ---------------------------------------------------------------------------

def _scaled(sc: dict, f: float, noise: float = 0.0, rng=None) -> dict:
    """The fish f times larger on the same ruler, each point jittered by noise px."""
    out = copy.deepcopy(sc)
    lat = out["lateral"]
    j = (lambda: rng.uniform(-noise, noise)) if noise else (lambda: 0.0)
    lat["keypoints"] = {n: [p[0] * f + j(), p[1] * f + j()] for n, p in lat["keypoints"].items()}
    lat["polygons"] = {n: [[q[0] * f + j(), q[1] * f + j()] for q in v]
                       for n, v in lat["polygons"].items()}
    return out


@pytest.fixture
def study(tmp_path):
    data = tmp_path / "data"
    st = data / "trout"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    (st / "schema.json").write_text(json.dumps({"group_from_filename": "_([A-Z]{3})_\\d+$"}))
    for i in range(12):
        fid = f"T_{'ABC'[i % 3] * 3}_{i}"
        Image.new("RGB", (400, 300), (90, 90, 90)).save(st / "lateral" / f"{fid}_L.jpg")
        sc = _scaled(payload(fid), 1 + 0.07 * i)
        sc["lateral"]["operators"] = {n: "JC" for n in sc["lateral"]["keypoints"]}
        (st / "sidecars" / f"{fid}.json").write_text(json.dumps(sc))
    return data, st


def test_a_round_draws_across_groups_and_hides_who_is_who(study):
    data, st = study
    r = create_round(st, 6, "AB", seed=1)
    folder = Path(r["folder"])
    assert folder.name == "trout.relabel-1" and r["n"] == 6
    groups = {fid.split("_")[1] for fid in r["fish"].values()}
    assert groups == {"AAA", "BBB", "CCC"}                       # two of each strain
    assert sorted(p.name for p in (folder / "lateral").iterdir()) == [f"{c}_L.jpg" for c in sorted(r["fish"])]
    assert not list((folder / "sidecars").iterdir())             # nothing to copy from
    assert (folder / "schema.json").is_file()
    assert json.loads((folder / "round.json").read_text())["operator"] == "AB"
    # the originals are untouched and the photographs are linked, not moved
    assert len(list((st / "lateral").iterdir())) == 12
    again = create_round(st, 0, "CD", same_as=1, seed=2)
    assert sorted(again["fish"].values()) == sorted(r["fish"].values())
    assert again["round"] == 2 and set(again["fish"]) != set(r["fish"])  # new codes


def test_the_reference_fish_is_never_offered(study):
    data, st = study
    (st / "reference").mkdir()
    (st / "reference" / "reference.json").write_text(json.dumps({"specimen": "T_AAA_0"}))
    r = create_round(st, 12, "AB", seed=3)
    assert "T_AAA_0" not in r["fish"].values() and r["n"] == 11


def test_a_round_is_not_made_from_a_round(study):
    data, st = study
    r = create_round(st, 4, "AB", seed=1)
    with pytest.raises(ValueError):
        create_round(Path(r["folder"]), 3, "AB")


def _relabel(st, r, noise, seed=0):
    """Label every fish of round r again, as a person would: close, not exact."""
    rng = random.Random(seed)
    folder = Path(r["folder"])
    for code, fid in r["fish"].items():
        orig = json.loads((st / "sidecars" / f"{fid}.json").read_text())
        i = int(fid.split("_")[-1])
        sc = _scaled(payload(code), 1 + 0.07 * i, noise, rng)
        sc["fish_id"] = code
        sc["lateral"]["calibration"] = orig["lateral"]["calibration"]
        sc["lateral"]["operators"] = {n: r["operator"] for n in sc["lateral"]["keypoints"]}
        (folder / "sidecars" / f"{code}.json").write_text(json.dumps(sc))


def test_the_analysis_pairs_each_relabel_with_its_original(study):
    data, st = study
    r = create_round(st, 9, "AB", seed=4)
    _relabel(st, r, noise=0.4)
    [d] = repeatability.analyse(st)
    assert len(d.fish) == 9 and len(d.sessions) == 2
    assert "operator JC" in d.sessions[0] and "operator AB" in d.sessions[1]
    sl = d.traits["SL"]
    assert sl.n == 9 and sl.k == 2
    assert sl.icc1 > 0.95 and 0 <= sl.pct_me < 5 and sl.verdict in ("excellent", "good")


def test_noisier_relabelling_shows_more_error(study):
    data, st = study
    r1 = create_round(st, 9, "AB", seed=5)
    _relabel(st, r1, noise=0.2, seed=1)
    tight = repeatability.analyse(st)[0].traits["Ed"].pct_me
    for f in (Path(r1["folder"]) / "sidecars").iterdir():
        f.unlink()
    _relabel(st, r1, noise=3.0, seed=1)
    loose = repeatability.analyse(st)[0].traits["Ed"].pct_me
    assert loose > tight


def test_a_round_counts_once_three_fish_are_done(study):
    data, st = study
    r = create_round(st, 6, "AB", seed=6)
    folder = Path(r["folder"])
    _relabel(st, r, noise=0.4)
    for f in sorted((folder / "sidecars").iterdir())[2:]:
        f.unlink()
    assert repeatability.analyse(st) == []


def test_the_workbook_carries_the_error_and_the_operators(study, tmp_path):
    data, st = study
    r = create_round(st, 9, "AB", seed=7)
    _relabel(st, r, noise=0.4)
    exporter = _load("export_measurements", SCRIPTS / "export_measurements.py")
    out = tmp_path / "m.xlsx"
    assert exporter.main(["--images", str(st / "lateral"), "--labels", str(st / "sidecars"),
                          "--out", str(out)]) == 0
    wb = openpyxl.load_workbook(out)
    rows = list(wb["Measurement error"].iter_rows(values_only=True))
    head = next(i for i, row in enumerate(rows) if row[0] == "trait")
    table = {row[0]: row for row in rows[head + 1:] if row[0] and row[0] not in ("METHOD", "")}
    assert "SL" in table and table["SL"][3] == 9 and table["SL"][5] > 0.95
    assert any("Bailey" in str(row[1]) for row in rows)
    about = [row[0] for row in wb["About"].iter_rows(values_only=True)]
    assert "MEASUREMENT ERROR" in about
    qc = list(wb["QC"].iter_rows(values_only=True))
    col = qc[0].index("operators")
    assert all(row[col] == "JC" for row in qc[1:])


def test_a_study_without_rounds_gets_no_sheet(study, tmp_path):
    data, st = study
    exporter = _load("export_measurements", SCRIPTS / "export_measurements.py")
    out = tmp_path / "m.xlsx"
    assert exporter.main(["--images", str(st / "lateral"), "--labels", str(st / "sidecars"),
                          "--out", str(out)]) == 0
    assert "Measurement error" not in openpyxl.load_workbook(out).sheetnames


def test_darwin_core_says_who_determined_each_measurement(study, tmp_path):
    data, st = study
    from fish_morpho import darwin_core
    darwin_core.update(st, {"T_AAA_0": {"catalogNumber": "1"}})
    exporter = _load("export_measurements", SCRIPTS / "export_measurements.py")
    out = tmp_path / "m.xlsx"
    assert exporter.main(["--images", str(st / "lateral"), "--labels", str(st / "sidecars"),
                          "--out", str(out)]) == 0
    mof = list(openpyxl.load_workbook(out)["MeasurementOrFact"].iter_rows(values_only=True))
    col = mof[0].index("measurementDeterminedBy")
    sl = next(row for row in mof if row[0] == "T_AAA_0" and row[1].endswith("(SL)"))
    assert sl[col] == "JC"


def test_the_r_table_says_who_placed_the_landmarks(study, tmp_path):
    data, st = study
    out = tmp_path / "tps"
    import subprocess
    r = subprocess.run([sys.executable, str(SCRIPTS / "export_tps.py"), "--sidecars",
                        str(st / "sidecars"), "--images", str(st / "lateral"),
                        "--schema-dir", str(st), "--out", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    sp = list(csv.DictReader((out / "specimens.csv").open()))
    assert {row["operators"] for row in sp} == {"JC"}


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------

@pytest.fixture
def srv(study):
    data, st = study
    ls = _load("label_server", SCRIPTS / "label_server.py")
    ls.Handler.data_root = data
    ls.Handler.datasets = ls.discover_datasets(data)
    ls.Handler.default_dataset = "trout"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    ls.Handler.session = ls.auth.Session()
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield ls, f"http://127.0.0.1:{server.server_address[1]}", data, st
    server.shutdown()


def _call(url, route, ds="trout", body=None):
    req = urllib.request.Request(f"{url}{route}?dataset={ds}",
                                 method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_server_starts_a_round_and_never_says_who_is_who(srv):
    ls, url, data, st = srv
    code, r = _call(url, "/api/relabel/new", body={"n": 6, "operator": "AB"})
    assert code == 200 and r["ok"] and r["name"] == "trout.relabel-1" and r["n"] == 6
    everything = json.dumps([_call(url, "/api/datasets")[1],
                             _call(url, "/api/relabel", ds=r["name"])[1],
                             _call(url, "/api/schema", ds=r["name"])[1],
                             _call(url, "/api/specimens", ds=r["name"])[1]])
    assert "T_AAA" not in everything and "T_BBB" not in everything      # no fish named
    ds = {d["name"]: d for d in _call(url, "/api/datasets")[1]["datasets"]}
    assert ds["trout.relabel-1"]["relabel"] == {**ds["trout.relabel-1"]["relabel"],
                                               "study": "trout", "round": 1, "n": 6, "done": 0}
    assert ds["trout"]["relabel"] is None
    info = _call(url, "/api/relabel")[1]
    assert info["rounds"][0]["name"] == "trout.relabel-1" and info["in_round"] is None


def test_no_prediction_inside_a_round(srv):
    ls, url, data, st = srv
    ls.auth.load_config = lambda: {}                                  # unlocked on this machine
    _call(url, "/api/relabel/new", body={"n": 4, "operator": "AB"})
    code, r = _call(url, "/api/predict/R1-1", ds="trout.relabel-1")
    assert code in (401, 403) and not r.get("ok")


def test_the_server_reports_the_error_so_far(srv):
    ls, url, data, st = srv
    code, r = _call(url, "/api/relabel/new", body={"n": 9, "operator": "AB"})
    rnd = json.loads((data / r["name"] / "round.json").read_text())
    _relabel(st, dict(rnd, folder=str(data / r["name"])), noise=0.4)
    code, s = _call(url, "/api/relabel/stats")
    assert code == 200 and s["ok"]
    [d] = s["designs"]
    sl = next(t for t in d["traits"] if t["code"] == "SL")
    assert d["fish"] == 9 and sl["icc1"] > 0.95


def test_old_labels_can_be_credited_to_their_operator(tmp_path):
    ao = _load("attribute_operator", SCRIPTS / "attribute_operator.py")
    doc = {"metadata": {"assist": {"unreviewed": ["anal_base_center"]}},
           "lateral": {"keypoints": {"premaxilla_tip": [1, 2], "anal_base_center": [3, 4],
                                     "caudal_base": [5, 6]},
                       "calibration": {"mode": "manual", "point_a": [0, 0], "point_b": [9, 0]},
                       "operators": {"caudal_base": "AB"}}}
    assert ao.attribute(doc, "JC") == 4
    assert doc["lateral"]["operators"] == {"premaxilla_tip": "JC", "anal_base_center": "model",
                                           "caudal_base": "AB", "ruler_point_a": "JC",
                                           "ruler_point_b": "JC"}
    data = tmp_path / "data"
    (data / "s" / "sidecars").mkdir(parents=True)
    (data / "s" / "sidecars" / "F.json").write_text(json.dumps(
        {"lateral": {"keypoints": {"premaxilla_tip": [1, 2]}}}))
    assert ao.main(["--dataset", "s", "--operator", "JC", "--data-root", str(data)]) == 0
    assert "operators" not in json.loads((data / "s" / "sidecars" / "F.json").read_text())["lateral"]
    assert ao.main(["--dataset", "s", "--operator", "JC", "--data-root", str(data), "--write"]) == 0
    assert json.loads((data / "s" / "sidecars" / "F.json").read_text())["lateral"]["operators"] == {
        "premaxilla_tip": "JC"}
    assert list(data.glob(".labels-backup-*/F.json"))
