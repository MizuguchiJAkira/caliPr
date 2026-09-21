"""Schemes of your own: created here, appended to, and never renumbered.

A scheme is a protocol. Its landmarks are numbered, TPS identifies one by its
row, and every rule below follows from that single fact.
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
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))

from fish_morpho import schemes  # noqa: E402

FID = "fish_1"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load("label_server")


@pytest.fixture
def srv(tmp_path, monkeypatch):
    data = tmp_path / "data"
    study = data / "study"
    (study / "lateral").mkdir(parents=True)
    (study / "sidecars").mkdir()
    Image.new("RGB", (400, 200), (90, 90, 90)).save(study / "lateral" / f"{FID}.JPEG")
    (study / "schema.json").write_text("{}")
    schemes.use_data_root(data)
    ls.Handler.datasets = {"study": study}
    ls.Handler.default_dataset = "study"
    ls.Handler.images_dir = study
    ls.Handler.out_dir = study / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.data_root = data
    ls.Handler.demo_mode = False
    monkeypatch.setattr(ls.Handler, "_locked", lambda self: False)
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", data, study
    server.shutdown()
    schemes.use_data_root(None)


def _post(url, path, payload):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def _names(schema):
    return [k["name"] for k in schema["lateral"]["keypoints"]]


def test_a_scheme_of_your_own_is_created_and_listed(srv):
    url, data, _ = srv
    r = _post(url, "/api/schemes/new?dataset=study", {"title": "Notropis working set"})
    assert r["ok"] and r["scheme"] == "notropis_working_set"
    assert (data / "schemes" / "notropis_working_set.json").is_file()
    names = {s["name"]: s for s in r["schemes"]}
    assert names["notropis_working_set"]["editable"] is True
    # and the two that come with caliPr are not editable
    assert names["calipr"]["editable"] is False and names["bgnn_2d"]["editable"] is False


def test_it_can_start_from_one_that_exists(srv):
    url, _, _ = srv
    r = _post(url, "/api/schemes/new?dataset=study",
              {"title": "BGNN plus ours", "copy_from": "bgnn_2d"})
    assert r["ok"]
    copy = schemes.get(r["scheme"])
    assert len(copy["landmarks"]) == len(schemes.BUILTIN["bgnn_2d"]["landmarks"])
    # and it says what it came from, because from the first added point its files
    # are no longer that protocol's files
    assert "BGNN" in copy["source"] and "diverges" in copy["note"]


def test_a_landmark_is_appended_after_every_existing_one(srv):
    """Never inserted. TPS identifies a landmark by its row, so renumbering would
    silently redefine every file already exported under the scheme."""
    url, _, _ = srv
    made = _post(url, "/api/schemes/new?dataset=study",
                 {"title": "Mine", "copy_from": "bgnn_2d"})
    _post(url, "/api/schema/scheme?dataset=study", {"scheme": made["scheme"]})
    before = _names(ls.build_schema(ls.load_profile(ls.Handler.images_dir)))
    r = _post(url, "/api/schema/keypoint?dataset=study",
              {"action": "add", "label": "Adipose fin origin"})
    after = _names(r["schema"])
    assert after[:len(before)] == before          # nothing already there moved
    assert after[-1] == "adipose_fin_origin"
    assert r["schema"]["lateral"]["keypoints"][-1]["label"].startswith(f"{len(after)}. ")


def test_a_published_protocol_is_not_edited_in_place(srv):
    """Adding to it would make this lab's files stop matching everyone else's
    under that protocol's name, which is what a named scheme exists to prevent."""
    url, _, _ = srv
    _post(url, "/api/schema/scheme?dataset=study", {"scheme": "bgnn_2d"})
    r = _post(url, "/api/schema/keypoint?dataset=study",
              {"action": "add", "label": "Nope"})
    assert not r["ok"] and r["builtin"] is True and "copy" in r["error"]
    assert len(schemes.BUILTIN["bgnn_2d"]["landmarks"]) == 23


def test_removing_a_landmark_from_a_scheme_is_refused(srv):
    """It would renumber everything after it."""
    url, _, _ = srv
    made = _post(url, "/api/schemes/new?dataset=study",
                 {"title": "Mine", "copy_from": "bgnn_2d"})
    _post(url, "/api/schema/scheme?dataset=study", {"scheme": made["scheme"]})
    r = _post(url, "/api/schema/keypoint?dataset=study",
              {"action": "remove", "name": "cleithrum"})
    assert not r["ok"] and "renumber" in r["error"]
    assert len(schemes.get(made["scheme"])["landmarks"]) == 23


def test_renaming_changes_the_label_and_not_the_row(srv):
    url, _, _ = srv
    made = _post(url, "/api/schemes/new?dataset=study",
                 {"title": "Mine", "copy_from": "bgnn_2d"})
    _post(url, "/api/schema/scheme?dataset=study", {"scheme": made["scheme"]})
    before = _names(ls.build_schema(ls.load_profile(ls.Handler.images_dir)))
    r = _post(url, "/api/schema/keypoint?dataset=study",
              {"action": "rename", "name": "cleithrum", "label": "Shoulder girdle"})
    assert r["ok"] and _names(r["schema"]) == before
    lm = [k for k in r["schema"]["lateral"]["keypoints"] if k["name"] == "cleithrum"][0]
    assert lm["label"].endswith("Shoulder girdle")


def test_a_malformed_scheme_file_does_not_stop_the_others_listing(srv):
    url, data, _ = srv
    _post(url, "/api/schemes/new?dataset=study", {"title": "Good one"})
    (data / "schemes" / "broken.json").write_text("{not json")
    listed = {s["name"] for s in schemes.listing()}
    assert "good_one" in listed and "broken" not in listed
