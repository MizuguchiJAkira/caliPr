"""Straightening a bent fish along its midline, as MorFishJ does.

MorFishJ hands a midline to ImageJ's Straighten and measures on the image that
comes back; caliPr puts the landmarks where that image would put them and
measures those. What is tested: the geometry (a level line changes nothing, a
tilted one is a rotation, an arc's length is recovered, the ends are continued,
the drawn direction does not matter, a point past the centre of a bend is caught);
that the measurements of a fish tilted or bent on the photograph come out as for
the same fish lying straight; that the export says which fish were straightened
and keeps the coordinates as clicked for the checks against the photograph; that
realigning a fish's labels moves its midline; the R export's straightened
coordinates; and the server's preview.
"""

from __future__ import annotations

import base64
import copy
import csv
import importlib.util
import json
import math
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from fish_morpho import image_identity
from fish_morpho.straighten import Midline, straighten_block

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_a_level_midline_changes_nothing():
    ml = Midline.fit([(100, 500), (900, 500)])
    assert ml.length == pytest.approx(800)
    assert ml.turn_deg == pytest.approx(0, abs=1e-6)
    assert ml.to_straight([(300, 480), (700, 530)]) == [
        pytest.approx((300, 480)), pytest.approx((700, 530))]


def test_a_tilted_midline_is_a_rotation():
    a = math.radians(30)

    def tilt(x, y):
        return (100 + x * math.cos(a) - y * math.sin(a), 500 + x * math.sin(a) + y * math.cos(a))
    ml = Midline.fit([tilt(0, 0), tilt(800, 0)])
    got = ml.to_straight([tilt(200, -20), tilt(600, 30)])
    x0, y0 = tilt(0, 0)
    assert [(x - x0, y - y0) for x, y in got] == [pytest.approx((200, -20)),
                                                   pytest.approx((600, 30))]


def test_an_arcs_length_and_offsets_are_recovered():
    R = 1000
    ml = Midline.fit([(R * math.sin(t), R - R * math.cos(t))
                      for t in np.linspace(0, math.pi / 2, 7)])
    assert ml.length == pytest.approx(math.pi / 2 * R, rel=1e-3)
    t = math.pi / 4
    on = (R * math.sin(t), R - R * math.cos(t))
    inside = ((R - 50) * math.sin(t), R - (R - 50) * math.cos(t))
    (s1, o1), (s2, o2) = ml.project([on, inside])
    assert s1 == pytest.approx(t * R, rel=2e-3) and s2 == pytest.approx(s1, abs=1)
    assert o1 == pytest.approx(0, abs=0.5) and o2 == pytest.approx(50, abs=0.5)


def test_the_ends_are_continued_straight():
    ml = Midline.fit([(100, 500), (900, 500)])
    (before, _), (after, _) = ml.project([(40, 500), (950, 510)])
    assert before == pytest.approx(-60) and after == pytest.approx(850)


def test_the_drawn_direction_does_not_matter():
    kps = {"premaxilla_tip": [110, 500], "caudal_base": [880, 505]}
    one, _ = straighten_block({"midline": [[100, 500], [500, 530], [900, 500]], "keypoints": kps})
    two, _ = straighten_block({"midline": [[900, 500], [500, 530], [100, 500]], "keypoints": kps})
    for n in kps:
        assert one["keypoints"][n] == pytest.approx(two["keypoints"][n], abs=1e-6)
    assert one["keypoints"]["caudal_base"][0] > one["keypoints"]["premaxilla_tip"][0]


def test_a_point_past_the_centre_of_a_bend_is_reported():
    R = 1000
    ml = Midline.fit([(R * math.sin(t), R - R * math.cos(t))
                      for t in np.linspace(0, math.pi / 2, 7)])
    t = math.pi / 4
    on, n = np.array((R * math.sin(t), R - R * math.cos(t))), np.array((-math.sin(t), math.cos(t)))
    assert ml.folded([tuple(on + 0.5 * R * n), tuple(on - 0.5 * R * n)]) == []
    assert ml.folded([tuple(on + 1.2 * R * n)]) == [0]


def test_a_midline_needs_two_distinct_points():
    with pytest.raises(ValueError):
        Midline.fit([(1, 1)])
    with pytest.raises(ValueError):
        Midline.fit([(1, 1), (1, 1)])
    block = {"midline": [[5, 5]], "keypoints": {"a": [1, 2]}}
    assert straighten_block(block) == (block, {})            # one point: not a midline


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

payload = _load("_pipeline_payload", TESTS / "test_pipeline.py")._sidecar_payload
openpyxl = pytest.importorskip("openpyxl")


def _map_block(block: dict, f, ruler: bool = True) -> dict:
    """Every coordinate of a lateral block moved by f(x, y) -> (x, y).

    The ruler moves with a rigid motion of the whole photograph, and stays put
    when only the fish is bent: a ruler does not bend.
    """
    out = copy.deepcopy(block)
    out["keypoints"] = {n: list(f(*p)) for n, p in block["keypoints"].items()}
    out["polygons"] = {n: [list(f(*q)) for q in v] for n, v in block["polygons"].items()}
    cal = out.get("calibration") or {}
    for k in ("point_a", "point_b"):
        if ruler and cal.get(k):
            cal[k] = list(f(*cal[k]))
    return out


def _measure(tmp_path, sidecars: dict):
    from fish_morpho.pipeline import run
    images, labels = tmp_path / "study" / "lateral", tmp_path / "study" / "sidecars"
    images.mkdir(parents=True)
    labels.mkdir()
    for fid, sc in sidecars.items():
        (images / f"{fid}.jpg").write_bytes(b"\x00")
        (labels / f"{fid}.json").write_text(json.dumps(sc))
    out = run(images_dir=images, labels_dir=labels, output_path=tmp_path / "m.xlsx",
              mode="manual", model_config=None)
    wb = openpyxl.load_workbook(out)
    rows = list(wb["Measurements"].iter_rows(values_only=True))
    hdr = list(rows[0])
    u = hdr.index("units")
    table = {r[0]: {str(h).split(" — ")[0]: v for h, v in zip(hdr[u + 1:], r[u + 1:])}
             for r in rows[1:]}
    return table, wb


def test_a_tilted_fish_with_a_midline_measures_as_if_level(tmp_path):
    a = math.radians(20)

    def tilt(x, y):
        return (400 + (x - 100) * math.cos(a) - (y - 50) * math.sin(a),
                300 + (x - 100) * math.sin(a) + (y - 50) * math.cos(a))
    level = payload("LEVEL")
    tilted = payload("TILTED")
    tilted["lateral"] = _map_block(level["lateral"], tilt)
    fixed = payload("FIXED")
    fixed["lateral"] = _map_block(level["lateral"], tilt)
    fixed["lateral"]["midline"] = [list(tilt(-20, 50)), list(tilt(220, 50))]
    table, wb = _measure(tmp_path, {"LEVEL": level, "TILTED": tilted, "FIXED": fixed})

    # Without a midline the "horizontal" traits read short, as MorFishJ's would
    # on an image it had not been told to rotate.
    assert table["TILTED"]["SL"] < 0.95 * table["LEVEL"]["SL"]
    for code, v in table["LEVEL"].items():
        if v in (None, ""):
            assert table["FIXED"][code] in (None, "")
        else:
            assert table["FIXED"][code] == pytest.approx(v, rel=1e-3, abs=2e-3), code

    qc = list(wb["QC"].iter_rows(values_only=True))
    col = qc[0].index("straightened")
    fixed_row = next(r for r in qc if r[0] == "FIXED" and r[2] == "lateral")
    level_row = next(r for r in qc if r[0] == "LEVEL" and r[2] == "lateral")
    assert "2-point midline" in fixed_row[col] and "turning 0°" in fixed_row[col]
    assert not level_row[col]
    about = [r[0] for r in wb["About"].iter_rows(values_only=True)]
    assert "STRAIGHTENED" in about


def test_a_bent_fish_measures_as_if_straight(tmp_path):
    # The level fish laid along an arc: x becomes distance along the arc, y the
    # distance off it -- a fish bent through about 40 degrees.
    R = 300.0

    def bend(x, y):
        th = x / R
        return (R * math.sin(th) - (y - 50) * math.sin(th) + 60,
                R - R * math.cos(th) + (y - 50) * math.cos(th) + 50)
    level = payload("LEVEL")
    bent = payload("BENT")
    bent["lateral"] = _map_block(level["lateral"], bend, ruler=False)
    fixed = payload("FIXED")
    fixed["lateral"] = _map_block(level["lateral"], bend, ruler=False)
    fixed["lateral"]["midline"] = [list(bend(x, 50)) for x in np.linspace(-20, 220, 7)]
    table, _ = _measure(tmp_path, {"LEVEL": level, "BENT": bent, "FIXED": fixed})
    assert table["BENT"]["SL"] < 0.97 * table["LEVEL"]["SL"]
    assert table["FIXED"]["SL"] == pytest.approx(table["LEVEL"]["SL"], rel=0.01)
    assert table["FIXED"]["TL"] == pytest.approx(table["LEVEL"]["TL"], rel=0.01)
    assert table["FIXED"]["Hl"] == pytest.approx(table["LEVEL"]["Hl"], rel=0.02)


def test_the_checks_see_the_landmarks_as_clicked(tmp_path):
    from fish_morpho.pipeline import SpecimenInput, process_specimen
    sc = payload("F")
    sc["lateral"]["midline"] = [[-20, 60], [220, 40]]
    img = tmp_path / "F.jpg"
    img.write_bytes(b"\x00")
    rec = process_specimen(SpecimenInput(fish_id="F", image_path=img, sidecar_path=tmp_path / "F.json",
                                         sidecar=sc))
    assert rec.keypoints["premaxilla_tip"] == pytest.approx((0, 50))       # as clicked
    assert rec.measurements.metadata["straightened"].startswith("along a 2-point midline")


def test_realigning_labels_moves_the_midline():
    doc = {"lateral": {"keypoints": {"a": [10, 10]}, "midline": [[0, 0], [100, 5]]}}
    image_identity.shift_view(doc, "lateral", 7, -3)
    assert doc["lateral"]["midline"] == [[7, -3], [107, 2]]


# ---------------------------------------------------------------------------
# the R export and the server
# ---------------------------------------------------------------------------

def test_the_r_export_can_straighten(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    st = tmp_path / "study"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    a = math.radians(25)
    rot = lambda x, y: (300 + x * math.cos(a) - y * math.sin(a), 300 + x * math.sin(a) + y * math.cos(a))
    for fid in ("F1", "F2"):
        Image.new("RGB", (900, 900)).save(st / "lateral" / f"{fid}_L.JPEG")
        block = {"keypoints": {"premaxilla_tip": list(rot(0, 0)), "caudal_base": list(rot(400, 0))}}
        if fid == "F1":
            block["midline"] = [list(rot(-10, 0)), list(rot(410, 0))]
        (st / "sidecars" / f"{fid}.json").write_text(json.dumps({"fish_id": fid, "lateral": block}))
    folder = tmp_path / "by_specimen"
    r = subprocess.run([sys.executable, str(SCRIPTS / "export_tps.py"),
                        "--sidecars", str(st / "sidecars"), "--images", str(st / "lateral"),
                        "--schema-dir", str(st), "--out", str(tmp_path / "tps"),
                        "--per-specimen", str(folder), "--units", "mm", "--straighten"],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    sp = {row["ID"]: row for row in csv.DictReader((folder / "specimens.csv").open())}
    assert sp["F1"]["straightened"] == "yes" and sp["F2"]["straightened"] == ""
    key = [row[1] for row in csv.reader((folder / "landmark_key.csv").open())][1:]
    rows = list(csv.reader((folder / "F1.csv").open()))[1:]
    snout = rows[key.index("premaxilla_tip")]
    tail = rows[key.index("caudal_base")]
    assert float(snout[3]) == pytest.approx(float(tail[3]), abs=1e-3)   # level now
    assert float(tail[2]) - float(snout[2]) == pytest.approx(400, abs=1e-2)
    rows2 = list(csv.reader((folder / "F2.csv").open()))[1:]
    assert float(rows2[key.index("caudal_base")][3]) != pytest.approx(
        float(rows2[key.index("premaxilla_tip")][3]), abs=1)             # as clicked, still tilted


@pytest.fixture
def srv(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("cv2")
    ls = _load("label_server", SCRIPTS / "label_server.py")
    data = tmp_path / "data"
    st = data / "trout"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    Image.new("RGB", (800, 400), (90, 110, 120)).save(st / "lateral" / "T_1.jpg")
    ls.Handler.data_root = data
    ls.Handler.datasets = {"trout": st}
    ls.Handler.default_dataset = "trout"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", st
    server.shutdown()


def _preview(url, body):
    req = urllib.request.Request(url + "/api/straighten/preview?dataset=trout", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_server_draws_the_straightened_fish(srv):
    url, st = srv
    block = {"midline": [[80, 250], [400, 180], [720, 250]],
             "keypoints": {"premaxilla_tip": [90, 250], "caudal_base": [700, 245]}}
    code, r = _preview(url, {"id": "T_1", "block": block})
    assert code == 200 and r["ok"], r
    assert base64.b64decode(r["jpeg"])[:2] == b"\xff\xd8"            # a JPEG
    assert r["info"]["turn_deg"] > 10 and r["info"]["folded"] == []
    assert not list((st / "sidecars").iterdir())                     # nothing written


def test_the_server_refuses_what_it_cannot_straighten(srv):
    url, _ = srv
    assert _preview(url, {"id": "T_1", "block": {"midline": [[1, 1]]}})[0] == 400
    assert _preview(url, {"id": "../trout/lateral/T_1", "block": {}})[0] == 400
    assert _preview(url, {"id": "nobody", "block": {"midline": [[0, 0], [9, 9]]}})[0] == 404
