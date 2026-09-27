"""Taking one photograph out of a study.

Moved, never deleted: the photograph, its saved labels and its cached prediction
go to data/.trash/<study>--<fish>--<time>/ with their paths inside the study
kept, so restoring is moving them back. Tested: exactly that fish's files move,
the others stay, and nothing outside the study can be named.
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
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def srv(tmp_path):
    ls = _load("label_server", SCRIPTS / "label_server.py")
    data = tmp_path / "data"
    st = data / "minnows"
    for d in ("lateral", "sidecars", "sidecars_auto"):
        (st / d).mkdir(parents=True)
    for fid in ("M_01", "M_02"):
        Image.new("RGB", (60, 40), (90, 90, 90)).save(st / "lateral" / f"{fid}.jpg")
    (st / "sidecars" / "M_01.json").write_text(json.dumps({"fish_id": "M_01", "lateral": {}}))
    (st / "sidecars_auto" / "M_01.json").write_text("{}")
    ls.Handler.data_root = data
    ls.Handler.datasets = {"minnows": st}
    ls.Handler.default_dataset = "minnows"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield ls, f"http://127.0.0.1:{server.server_address[1]}", data, st
    server.shutdown()


def _remove(url, fid):
    req = urllib.request.Request(url + "/api/specimen/remove?dataset=minnows", method="POST",
                                 data=json.dumps({"id": fid}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_photo_and_everything_of_its_goes_to_the_trash(srv):
    _, url, data, st = srv
    code, r = _remove(url, "M_01")
    assert code == 200 and r["ok"] and r["labelled"] and r["files"] == 3
    [moved] = list((data / ".trash").iterdir())
    assert moved.name.startswith("minnows--M_01--")
    assert (moved / "lateral" / "M_01.jpg").is_file()
    assert (moved / "sidecars" / "M_01.json").is_file()
    assert (moved / "sidecars_auto" / "M_01.json").is_file()
    assert not (st / "lateral" / "M_01.jpg").exists() and not (st / "sidecars" / "M_01.json").exists()
    assert (st / "lateral" / "M_02.jpg").is_file()          # the other fish is untouched


def test_an_unlabelled_photo_moves_alone(srv):
    _, url, data, st = srv
    code, r = _remove(url, "M_02")
    assert code == 200 and not r["labelled"] and r["files"] == 1


def test_nothing_outside_the_study_can_be_named(srv):
    _, url, data, st = srv
    assert _remove(url, "../minnows/lateral/M_02")[0] == 400
    assert _remove(url, "no_such_fish")[0] == 404
    assert (st / "lateral" / "M_02.jpg").is_file()


def test_demo_mode_removes_nothing(srv):
    ls, url, data, st = srv
    ls.Handler.demo_mode = True
    assert _remove(url, "M_01")[0] == 403
    assert (st / "lateral" / "M_01.jpg").is_file()


def _remove_many(url, ids):
    req = urllib.request.Request(url + "/api/specimen/remove?dataset=minnows", method="POST",
                                 data=json.dumps({"ids": ids}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_a_batch_goes_to_one_folder_with_a_list(srv):
    _, url, data, st = srv
    code, r = _remove_many(url, ["M_01", "M_02"])
    assert code == 200 and r["removed"] == 2 and r["labelled"] == 1
    [moved] = list((data / ".trash").iterdir())
    assert moved.name.startswith("minnows--2-photos--")
    assert (moved / "lateral" / "M_01.jpg").is_file() and (moved / "lateral" / "M_02.jpg").is_file()
    assert "M_01" in (moved / "REMOVED.txt").read_text()
    assert not list((st / "lateral").iterdir())


def test_one_bad_id_in_a_batch_moves_nothing(srv):
    _, url, data, st = srv
    assert _remove_many(url, ["M_01", "no_such_fish"])[0] == 404
    assert (st / "lateral" / "M_01.jpg").is_file() and (st / "sidecars" / "M_01.json").is_file()
    assert not (data / ".trash").exists() or not list((data / ".trash").iterdir())
