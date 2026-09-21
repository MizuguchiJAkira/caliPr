"""Auto-label offers an outline for two fins, and says so everywhere after.

The accuracy case for the pectoral and the anal is in docs/what-we-tried.md. The
case that matters here is the other one: a predicted outline a labeller accepts
becomes what the next model trains on, so every route out of one has to keep
saying where it came from.
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
FID = "Salvelinus_fontinalis_TXD_9"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load("label_server")


@pytest.fixture
def srv(tmp_path, monkeypatch):
    study = tmp_path / "study"
    (study / "lateral").mkdir(parents=True)
    (study / "sidecars").mkdir()
    Image.new("RGB", (1200, 400), (90, 90, 90)).save(study / "lateral" / f"{FID}_L.JPEG")
    (study / "schema.json").write_text(json.dumps({}))
    asked = []

    def fake(image, polygons=False, emit_polygons=True, view="lateral", crop=None,
             fins=None):
        asked.append(list(fins or []))
        return {"ok": True, "keypoints": {"premaxilla_tip": [50.0, 100.0]},
                "confidence": {"premaxilla_tip": 0.9}, "low_confidence": [],
                "polygons": {f: [[1, 1], [9, 1], [9, 9]] for f in (fins or [])},
                "implausible": {}, "frame_warning": None, "model": "x.pt",
                "elapsed": 0.1}

    monkeypatch.setattr(ls.Predictor, "predict", classmethod(lambda cls, *a, **k: fake(*a, **k)))
    ls.Handler.datasets = {"study": study}
    ls.Handler.default_dataset = "study"
    ls.Handler.images_dir = study
    ls.Handler.out_dir = study / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    monkeypatch.setattr(ls.Handler, "_locked", lambda self: False)
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", study, asked
    server.shutdown()


def _json(url, path):
    try:
        with urllib.request.urlopen(url + path) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def test_only_the_two_fins_that_earned_it_are_offered(srv):
    """The dorsal and the pelvic are measured too: 11.0% and 6.9% median area
    error against 3.2% and 4.6%, and a pelvic worst case of +111%. The bar is not
    better than nothing -- it is no worse than the tracing it stands in for."""
    url, _, asked = srv
    _json(url, f"/api/predict/{FID}?dataset=study")
    assert asked == [["pectoral", "anal"]]


def test_a_study_can_refuse_them(srv):
    url, study, asked = srv
    (study / "schema.json").write_text(json.dumps(
        {"exclude_predicted_polygons": ["anal"], "exclude_polygons": ["pectoral"]}))
    _json(url, f"/api/predict/{FID}?dataset=study")
    assert asked[-1] == []


def test_the_frontal_view_is_not_asked_for_fins(srv):
    """There are no fins in the head-on view, and its model knows two points."""
    url, study, asked = srv
    (study / "frontal").mkdir()
    Image.new("RGB", (300, 400), (90, 90, 90)).save(study / "frontal" / f"{FID}_F.JPEG")
    _json(url, f"/api/predict/{FID}?dataset=study&view=frontal")
    assert asked[-1] == []


def test_a_cache_made_without_the_outliner_is_not_served_forever(srv):
    """Otherwise the first Auto-label a machine ever ran decides, permanently,
    that this fish has no fins -- and fetching the outliner later changes nothing,
    because a cached entry with no outlines looks like one where it found none."""
    url, study, asked = srv
    (study / "schema.json").write_text(json.dumps({"exclude_polygons": ["pectoral", "anal"]}))
    _json(url, f"/api/predict/{FID}?dataset=study")
    assert asked[-1] == []
    cached = json.loads((study / "sidecars_auto" / f"{FID}.json").read_text())
    assert cached["metadata"]["fins_asked"] == []

    # the outliner arrives: the entry is re-predicted rather than served
    (study / "schema.json").write_text(json.dumps({}))
    got = _json(url, f"/api/predict/{FID}?dataset=study")
    assert asked[-1] == ["pectoral", "anal"]
    assert set(got["polygons"]) == {"pectoral", "anal"}

    # and an entry that did ask for them is still served from cache
    before = len(asked)
    _json(url, f"/api/predict/{FID}?dataset=study")
    assert len(asked) == before


def test_an_outline_the_model_drew_is_not_trained_on(tmp_path):
    """The whole point of the flag. An outline nobody redrew is this model's own
    output; training on it teaches the next one to repeat what it gets wrong."""
    bfd = _load("build_fin_dataset")
    study = tmp_path / "study"
    (study / "lateral").mkdir(parents=True)
    (study / "sidecars").mkdir()
    Image.new("RGB", (2000, 800), (90, 90, 90)).save(study / "lateral" / f"{FID}_L.JPEG")
    ring = [[900 + 60 * (i % 2) + i * 4, 400 + 40 * (i % 3)] for i in range(20)]
    doc = {
        "fish_id": FID,
        "lateral": {
            "keypoints": {"premaxilla_tip": [200, 400], "caudal_base": [1800, 400],
                          "pectoral_insertion_upper": [900, 420],
                          "pectoral_ray_tip": [1010, 430],
                          "anal_base_center": [1300, 500], "anal_tip": [1400, 520]},
            "polygons": {"pectoral": ring, "anal": [[p[0] + 400, p[1]] for p in ring]},
        },
        "metadata": {"assist": {"polygons_from_model": ["pectoral"]}},
    }
    (study / "sidecars" / f"{FID}.json").write_text(json.dumps(doc))
    rows = bfd.build(study, tmp_path / "out")
    assert {r["fin"] for r in rows if not r.get("no_crop")} == {"anal"}


def test_the_workbook_says_which_areas_came_from_the_model(tmp_path):
    """Once a fin area is in a spreadsheet nothing else records where it came
    from, so the QC note is the last place it can be said."""
    sys.path.insert(0, str(ROOT / "src"))
    from fish_morpho import pipeline

    doc = {"fish_id": FID,
           "lateral": {"keypoints": {}, "polygons": {"anal": [[1, 1], [2, 2], [3, 1]]}},
           "metadata": {"assist": {"polygons_from_model": ["anal", "pelvic"]}}}
    spec = type("S", (), {"fish_id": FID, "sidecar": doc,
                          "image_path": Path("x.JPEG")})()
    note = pipeline._model_outline_note(spec)
    assert "anal" in note and "pelvic" not in note      # only outlines that exist
