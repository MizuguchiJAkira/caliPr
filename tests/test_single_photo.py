"""A study that keeps one photograph per fish, holding both views.

The Cornell rig photographs a fish beside a mirror showing its head. That frame
used to be cut into a lateral and a frontal image; it no longer is. Both views
are now placed on the same photograph, in the same coordinates, and the models
are given a crop made in memory instead of one stored on disk.
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load("label_server")

FID = "Salvelinus_fontinalis_TXD_9"


def _photo(w, h, seam=None):
    """A frame like the rig's: a dark vertical mirror edge, or none at all."""
    im = Image.new("RGB", (w, h), (90, 90, 90))
    if seam is not None:
        for x in range(seam - 10, seam):        # the detector clears the frame by 10 px
            for y in range(h):
                im.putpixel((x, y), (0, 0, 0))
    return im


@pytest.fixture
def srv(tmp_path, monkeypatch):
    study = tmp_path / "study"
    (study / "lateral").mkdir(parents=True)
    (study / "sidecars").mkdir()
    _photo(1200, 400, seam=264).save(study / "lateral" / f"{FID}_L.JPEG")
    (study / "schema.json").write_text(json.dumps({"single_photo": True}))
    calls = []

    def fake(image, polygons=False, emit_polygons=True, view="lateral", crop=None):
        calls.append({"image": Path(image).name, "view": view, "crop": crop})
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
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), ""


def _json(url, path):
    code, body, _ = _get(url, path)
    return code, json.loads(body)


def test_the_frontal_view_is_the_same_photograph(srv):
    url, study, _ = srv
    code, specs = _json(url, "/api/specimens?dataset=study")
    s = (specs["specimens"] if isinstance(specs, dict) else specs)[0]
    assert s["lateral"] == s["frontal"] == f"{FID}_L.JPEG"
    assert s["sizes"]["lateral"] == s["sizes"]["frontal"]
    # and asking for it under the frontal view serves that photograph
    code, body, ctype = _get(url, f"/img/frontal/{FID}_L.JPEG?dataset=study")
    assert code == 200 and ctype == "image/jpeg"
    assert body == (study / "lateral" / f"{FID}_L.JPEG").read_bytes()


def test_each_view_is_cropped_at_the_mirror_seam(srv):
    """Cut where the mirror is, not at a fixed fraction: a fixed one cannot work.

    Across the 131 photographs the seam runs from 0 to 0.35 of the width while the
    leftmost snout sits at 0.197, so any fixed choice either hands the lateral
    model a second, mirrored head or cuts a real one off.
    """
    url, _, calls = srv
    _json(url, f"/api/predict/{FID}?dataset=study")
    _json(url, f"/api/predict/{FID}?dataset=study&view=frontal")
    lat, fro = (c["crop"] for c in calls)
    margin_l, margin_f = round(450 / 6000 * 1200), round(520 / 6000 * 1200)
    # each view is cut at the seam and given the overlap the training crops had,
    # so both of them still contain the mirror's edge
    assert lat[2] == 1200 and fro[0] == 0
    assert lat[0] < 264 < fro[2]
    assert fro[2] - lat[0] == margin_l + margin_f
    # both from the one photograph
    assert {c["image"] for c in calls} == {f"{FID}_L.JPEG"}


def test_a_photograph_with_no_mirror_edge_is_not_guessed_at(srv, tmp_path):
    """No seam, no head-on view. The mouth corners would land in the flank."""
    url, study, calls = srv
    _photo(1200, 400).save(study / "lateral" / f"{FID}_L.JPEG")   # no seam in it
    code, r = _json(url, f"/api/predict/{FID}?dataset=study&view=frontal")
    assert not r["ok"] and "could not be found" in r["error"]
    assert not calls                                   # the model was never asked
    # the lateral view still runs, on the whole frame, as those photographs
    # were stored before any of this existed
    _json(url, f"/api/predict/{FID}?dataset=study")
    assert calls[-1]["crop"] is None


def test_a_study_with_crops_on_disk_is_left_alone(srv, tmp_path, monkeypatch):
    """The crop is for whole frames only: a split study still predicts its crops."""
    url, study, calls = srv
    (study / "schema.json").write_text(json.dumps({}))
    (study / "frontal").mkdir()
    Image.new("RGB", (300, 400), (90, 90, 90)).save(study / "frontal" / f"{FID}_F.JPEG")
    _json(url, f"/api/predict/{FID}?dataset=study&view=frontal")
    assert calls[-1] == {"image": f"{FID}_F.JPEG", "view": "frontal", "crop": None}


def test_an_upload_is_not_split_into_two_images(srv, tmp_path):
    url, study, _ = srv
    img = tmp_path / "new.jpg"
    Image.new("RGB", (1200, 400), (10, 10, 10)).save(img)
    req = urllib.request.Request(
        url + "/api/upload?dataset=study", data=img.read_bytes(), method="POST",
        headers={"X-Filename": "new.jpg", "X-View": "lateral", "X-Split": "mirror"})
    with urllib.request.urlopen(req) as r:
        got = json.loads(r.read())
    assert got["ok"] and got["status"] == "added_whole"
    assert (study / "lateral" / "new.jpg").is_file()
    assert not (study / "frontal").exists() and not (study / "originals").exists()


def test_the_labeler_can_ask_where_the_head_on_view_sits(srv):
    """So it opens the frontal tab on the mirror, not on a fraction of the frame."""
    url, _, _ = srv
    _, fr = _json(url, f"/api/frame/{FID}?dataset=study")
    assert fr["size"] == [1200, 400]
    assert fr["frontal"][0] == 0 and 264 < fr["frontal"][2] < 400
    assert fr["lateral"][2] == 1200 and fr["lateral"][0] < 264


def test_the_labeler_is_told_the_views_share_a_photograph(srv):
    url, _, _ = srv
    _, schema = _json(url, "/api/schema?dataset=study")
    assert schema["single_photo"] is True
