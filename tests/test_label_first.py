"""A study that names its own specimens to label first.

The labeller marks a subset to label first, drawn round-robin over the lot numbers
in the filenames. A study whose filenames carry none -- ``Genus_species_N`` --
would get the first names alphabetically, all one species. Such a study lists its
own in schema.json (``label_first``), and those are the ones marked.
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
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
FISH = ["Luxilus_cornutus_1", "Luxilus_cornutus_2", "Notropis_rubellus_1", "Notropis_rubellus_2"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def study(tmp_path):
    ls = _load("label_server", SCRIPTS / "label_server.py")
    st = tmp_path / "data" / "shiners"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    for f in FISH:
        Image.new("RGB", (40, 20), "white").save(st / "lateral" / f"{f}.jpg")
    ls.Handler.data_root = tmp_path / "data"
    ls.Handler.datasets = {"shiners": st}
    ls.Handler.default_dataset = "shiners"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def suggested(schema=None):
        if schema is not None:
            (st / "schema.json").write_text(json.dumps(schema))
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}"
                                    "/api/specimens?dataset=shiners", timeout=30) as r:
            return sorted(s["id"] for s in json.loads(r.read()) if s["suggested"])

    suggested.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield ls, suggested
    server.shutdown()


def test_the_study_names_its_own(study):
    _, suggested = study
    assert suggested({"label_first": ["Notropis_rubellus_2", "Luxilus_cornutus_1"]}) == [
        "Luxilus_cornutus_1", "Notropis_rubellus_2"]


def test_a_name_with_no_photograph_is_left_out(study):
    _, suggested = study
    assert suggested({"label_first": ["Notropis_rubellus_1", "Notropis_rubellus_99"]}) == [
        "Notropis_rubellus_1"]


def test_without_a_list_the_rotation_over_lots_decides(study):
    ls, suggested = study
    assert suggested({}) == sorted(ls.suggested_subset(FISH))


def test_a_hand_only_fish_is_never_predicted(study):
    ls, suggested = study
    suggested({"hand_only": ["Notropis_rubellus_1"]})
    ls.auth.is_configured = lambda: False                 # no unlock on this machine
    with urllib.request.urlopen(suggested.url + "/api/specimens?dataset=shiners",
                                timeout=30) as r:
        marked = {s["id"]: s["hand_only"] for s in json.loads(r.read())}
    assert marked["Notropis_rubellus_1"] and not marked["Notropis_rubellus_2"]
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(suggested.url + "/api/predict/Notropis_rubellus_1"
                               "?dataset=shiners&cache=only", timeout=30)
    assert e.value.code == 403 and json.loads(e.value.read())["hand_only"]
