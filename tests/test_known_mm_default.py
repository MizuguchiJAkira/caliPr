"""A study's own default ruler span.

The built-in 50 mm lateral span was the wrong starting value on a rig where the
ruler points are placed a centimetre apart, and three fish were saved 5x too
small that way. A study can now set its own, kept in its schema.json; a fish with
no span of its own starts with it, and one that has a span keeps it.
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
    st = tmp_path / "data" / "trout"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    (st / "schema.json").write_text(json.dumps({"exclude_keypoints": ["dorsal_tip"]}))
    ls.Handler.data_root = tmp_path / "data"
    ls.Handler.datasets = {"trout": st}
    ls.Handler.default_dataset = "trout"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield ls, f"http://127.0.0.1:{server.server_address[1]}", st
    server.shutdown()


def _set(url, body):
    req = urllib.request.Request(url + "/api/schema/known_mm?dataset=trout", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_default_is_kept_in_the_studys_schema(srv):
    ls, url, st = srv
    code, r = _set(url, {"view": "lateral", "known_mm": 10})
    assert code == 200 and r["known_mm_default"] == {"lateral": 10.0}
    prof = json.loads((st / "schema.json").read_text())
    assert prof["known_mm_default"] == {"lateral": 10.0}
    assert prof["exclude_keypoints"] == ["dorsal_tip"]           # the rest of the file is kept


def test_the_page_is_given_it(srv):
    ls, url, st = srv
    _set(url, {"view": "frontal", "known_mm": 5})
    with urllib.request.urlopen(url + "/api/schema?dataset=trout", timeout=30) as r:
        schema = json.loads(r.read())
    assert schema["known_mm_default"] == {"frontal": 5.0}


def test_each_view_keeps_its_own(srv):
    ls, url, st = srv
    _set(url, {"view": "lateral", "known_mm": 10})
    _, r = _set(url, {"view": "frontal", "known_mm": 5})
    assert r["known_mm_default"] == {"lateral": 10.0, "frontal": 5.0}


def test_nonsense_is_refused_and_nothing_written(srv):
    ls, url, st = srv
    before = (st / "schema.json").read_text()
    assert _set(url, {"view": "lateral", "known_mm": -3})[0] == 400
    assert _set(url, {"view": "dorsal", "known_mm": 10})[0] == 400
    assert _set(url, {"view": "lateral", "known_mm": "ten"})[0] == 400
    assert (st / "schema.json").read_text() == before


def test_demo_mode_writes_nothing(srv):
    ls, url, st = srv
    ls.Handler.demo_mode = True
    assert _set(url, {"view": "lateral", "known_mm": 10})[0] == 403
