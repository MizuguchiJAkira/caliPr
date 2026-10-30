"""Counts a study takes on each fish, typed in beside the landmarks.

A study lists them in schema.json (``counts``); the page is given each one's name,
label and hint, and a count with a name that is not a plain identifier is refused
rather than written into every sidecar under a broken key.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
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
def schema_of(tmp_path):
    ls = _load("label_server", SCRIPTS / "label_server.py")
    st = tmp_path / "data" / "shiners"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    ls.Handler.data_root = tmp_path / "data"
    ls.Handler.datasets = {"shiners": st}
    ls.Handler.default_dataset = "shiners"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def get(schema):
        (st / "schema.json").write_text(json.dumps(schema))
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}"
                                    "/api/schema?dataset=shiners", timeout=30) as r:
            return json.loads(r.read())

    yield get
    server.shutdown()


def test_a_study_lists_its_counts(schema_of):
    s = schema_of({"counts": [{"name": "anterior_dorsolateral_scales", "label": "Anterior dorso-lateral scales",
                               "hint": "Gilbert (1964)"}, {"name": "anal_rays"}]})
    assert s["counts"] == [
        {"name": "anterior_dorsolateral_scales", "label": "Anterior dorso-lateral scales", "hint": "Gilbert (1964)"},
        {"name": "anal_rays", "label": "anal_rays", "hint": ""}]


def test_a_count_needs_a_plain_name(schema_of):
    s = schema_of({"counts": [{"name": "bad name"}, {"label": "no name"}, "not a dict", {"name": "ok_1"}]})
    assert [c["name"] for c in s["counts"]] == ["ok_1"]


def test_no_counts_is_none(schema_of):
    assert schema_of({})["counts"] == []
