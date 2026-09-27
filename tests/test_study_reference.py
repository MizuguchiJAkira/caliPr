"""A study's own example in the reference panel.

Every study used to be shown the same brook trout, including studies of other
fish on other landmark schemes. Now any saved fish can be made its study's
example ("Use as example"), built from its saved landmarks into
<study>/reference/. Tested: the builder takes any photograph and a fish without
fin outlines, leaves out what the study does not collect, and the server makes
and serves the files -- and refuses what it should.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("cv2")
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


KPS = {"dentary_anterior": [100, 300], "orbit_center": [220, 260],
       "caudal_crease_dorsal": [700, 250], "cleithrum": [330, 320]}


@pytest.fixture
def study(tmp_path):
    st = tmp_path / "data" / "minnows"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    (st / "schema.json").write_text(json.dumps({"scheme": "bgnn_2d",
                                                "exclude_keypoints": ["cleithrum"]}))
    Image.new("RGB", (800, 500), (120, 120, 120)).save(st / "lateral" / "M_01.jpg")
    Image.new("RGB", (800, 500), (120, 120, 120)).save(st / "lateral" / "M_02.jpg")
    (st / "sidecars" / "M_01.json").write_text(json.dumps(
        {"fish_id": "M_01", "lateral": {"keypoints": KPS}}))
    return st


def test_the_builder_takes_any_photo_and_skips_what_is_not_collected(study, tmp_path):
    mr = _load("make_reference", SCRIPTS / "make_reference.py")
    ref = mr.build("M_01", study / "sidecars", study / "lateral", out_dir=tmp_path / "ref",
                   image_path=study / "lateral" / "M_01.jpg", skip_keypoints=("cleithrum",))
    assert ref["specimen"] == "M_01" and ref["polygons"] == {}
    assert set(ref["keypoints"]) == {"dentary_anterior", "orbit_center", "caudal_crease_dorsal"}
    for f in ("reference.json", "reference_base.jpg", "reference_annot.jpg"):
        assert (tmp_path / "ref" / f).is_file()
    # never enlarged past the photograph: the crop is narrower than 3600 px
    assert ref["w"] <= 800


@pytest.fixture
def srv(study):
    ls = _load("label_server", SCRIPTS / "label_server.py")
    ls.Handler.data_root = study.parent
    ls.Handler.datasets = {"minnows": study}
    ls.Handler.default_dataset = "minnows"
    ls.Handler.images_dir = study
    ls.Handler.out_dir = study / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield ls, f"http://127.0.0.1:{server.server_address[1]}", study
    server.shutdown()


def _post(url, body):
    req = urllib.request.Request(url + "/api/reference?dataset=minnows", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(url, name):
    try:
        with urllib.request.urlopen(url + f"/api/reference/{name}?dataset=minnows", timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_a_study_has_no_example_until_one_is_made(srv):
    _, url, st = srv
    assert _get(url, "reference.json")[0] == 404


def test_a_saved_fish_becomes_the_studys_example(srv):
    _, url, st = srv
    code, r = _post(url, {"id": "M_01"})
    assert code == 200 and r["ok"], r
    code, body = _get(url, "reference.json")
    ref = json.loads(body)
    assert code == 200 and ref["specimen"] == "M_01"
    assert "cleithrum" not in ref["keypoints"]              # the study hides it
    assert _get(url, "reference_base.jpg")[0] == 200
    assert (st / "reference" / "reference_annot.jpg").is_file()


def test_what_it_refuses(srv):
    ls, url, st = srv
    assert _post(url, {"id": "M_02"})[0] == 400              # no saved labels
    assert _post(url, {"id": "../minnows/sidecars/M_01"})[0] == 400
    assert _get(url, "../schema.json")[0] == 404
    assert _get(url, "anything.txt")[0] == 404
    ls.Handler.demo_mode = True
    assert _post(url, {"id": "M_01"})[0] == 403
    assert not (st / "reference").exists()
