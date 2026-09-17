"""Auto-label on the frontal view runs the frontal model, and keeps apart from lateral.

The model is not run here -- the predictor is replaced -- so what is tested is the
routing: the frontal crop is what gets predicted, its cache sits apart from the
lateral one (whose presence marks a fish as batch-predicted), and the snapshot a
project pins is the one loaded.
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
FID = "Salvelinus_fontinalis_ASN_1"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load("label_server")


@pytest.fixture
def srv(tmp_path, monkeypatch):
    study = tmp_path / "data" / "study"
    for view, suffix, w in (("lateral", "_L", 900), ("frontal", "_F", 300)):
        (study / view).mkdir(parents=True)
        Image.new("RGB", (w, 200), (90, 90, 90)).save(study / view / f"{FID}{suffix}.JPEG")
    (study / "sidecars").mkdir()
    calls = []

    def fake(image, polygons=False, emit_polygons=True, view="lateral"):
        calls.append({"image": Path(image).name, "polygons": polygons, "view": view})
        kps = ({"mouth_left": [100.0, 120.0], "mouth_right": [180.0, 121.0]} if view == "frontal"
               else {"premaxilla_tip": [50.0, 100.0]})
        return {"ok": True, "keypoints": kps, "confidence": {k: 0.9 for k in kps},
                "low_confidence": [], "polygons": {}, "implausible": {},
                "frame_warning": None, "model": f"{view}.pt", "elapsed": 0.1}

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
    yield f"http://127.0.0.1:{server.server_address[1]}", study, calls
    server.shutdown()


def _get(url, path):
    try:
        with urllib.request.urlopen(url + path) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_frontal_predicts_the_frontal_crop_with_no_outline(srv):
    url, study, calls = srv
    code, r = _get(url, f"/api/predict/{FID}?dataset=study&view=frontal")
    assert code == 200 and r["ok"] and set(r["keypoints"]) == {"mouth_left", "mouth_right"}
    assert calls == [{"image": f"{FID}_F.JPEG", "polygons": False, "view": "frontal"}]


def test_frontal_cache_is_kept_apart_from_the_lateral_one(srv):
    url, study, calls = srv
    _get(url, f"/api/predict/{FID}?dataset=study&view=frontal")
    cached = study / "sidecars_auto" / "frontal" / f"{FID}.json"
    assert cached.is_file() and "frontal" in json.loads(cached.read_text())
    assert not (study / "sidecars_auto" / f"{FID}.json").exists()
    # so a frontal prediction does not mark the fish as batch-predicted
    code, specs = _get(url, "/api/specimens?dataset=study")
    assert [s["predicted"] for s in specs] == [False]
    # and asking again is served from that cache, not the model
    code, r = _get(url, f"/api/predict/{FID}?dataset=study&view=frontal")
    assert r["cached"] and r["keypoints"]["mouth_left"] == [100.0, 120.0] and len(calls) == 1


def test_lateral_is_unchanged(srv):
    url, study, calls = srv
    code, r = _get(url, f"/api/predict/{FID}?dataset=study")
    assert code == 200 and calls[0]["view"] == "lateral" and calls[0]["image"] == f"{FID}_L.JPEG"
    assert (study / "sidecars_auto" / f"{FID}.json").is_file()


def test_a_fish_with_no_frontal_crop_says_so(srv):
    url, study, calls = srv
    (study / "frontal" / f"{FID}_F.JPEG").unlink()
    code, r = _get(url, f"/api/predict/{FID}?dataset=study&view=frontal")
    assert code == 404 and "frontal" in r["error"] and not calls


def test_an_unknown_view_is_refused(srv):
    url, _, calls = srv
    code, _ = _get(url, f"/api/predict/{FID}?dataset=study&view=dorsal")
    assert code == 400 and not calls


def _project(tmp_path, names):
    train = tmp_path / "dlc-models-pytorch" / "iteration-0" / "shuffle1" / "train"
    train.mkdir(parents=True)
    (train / "pytorch_config.yaml").write_text("x: 1\n")
    for n in names:
        (train / n).write_bytes(b"")
    return tmp_path, train


def test_a_pinned_snapshot_is_the_one_loaded(tmp_path):
    pl = _load("predict_landmarks")
    project, train = _project(tmp_path, ["snapshot-best-030.pt", "snapshot-200.pt"])
    assert pl.find_config_and_snapshot(project, None)[1].name == "snapshot-best-030.pt"
    (project / pl.PIN).write_text(json.dumps({"snapshot": "snapshot-200.pt"}))
    assert pl.find_config_and_snapshot(project, None)[1].name == "snapshot-200.pt"
    (project / pl.PIN).write_text(json.dumps({"snapshot": "snapshot-999.pt"}))
    with pytest.raises(SystemExit):
        pl.find_config_and_snapshot(project, None)


def test_the_last_snapshot_is_the_latest_epoch_not_the_last_name(tmp_path):
    pl = _load("predict_landmarks")
    project, _ = _project(tmp_path, ["snapshot-200.pt", "snapshot-1000.pt"])
    assert pl.find_config_and_snapshot(project, None)[1].name == "snapshot-1000.pt"


def test_mouth_corners_come_back_in_image_order():
    pytest.importorskip("numpy")
    w = _load("predict_worker")
    kps = {"mouth_left": [300.0, 10.0], "mouth_right": [100.0, 12.0]}
    confs = {"mouth_left": 0.4, "mouth_right": 0.9}
    w._mouth_corners_in_image_order(kps, confs)
    assert kps == {"mouth_left": [100.0, 12.0], "mouth_right": [300.0, 10.0]}
    assert confs == {"mouth_left": 0.9, "mouth_right": 0.4}     # each follows its point


def test_plausibility_bands_added_after_a_first_prediction_are_used(tmp_path):
    pytest.importorskip("numpy")
    w = _load("predict_worker")
    image = tmp_path / "study" / "lateral" / "x_L.JPEG"
    image.parent.mkdir(parents=True)
    assert w._bands_for(image) is None
    (tmp_path / "study" / "plausibility.json").write_text(json.dumps({"landmarks": {"a": [0, 1]}}))
    assert w._bands_for(image) == {"landmarks": {"a": [0, 1]}}
