"""A photograph holding both views is split on upload -- and never destructively.

The mirror detector cannot tell when it has cut in the wrong place, so what is
tested here is mostly what protects against that: the original is always kept,
the cut is recorded, a boundary where no mirror can be is refused, and nothing
already labelled or already present is overwritten.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
pytest.importorskip("cv2")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
W, H = 1200, 800


def _load():
    spec = importlib.util.spec_from_file_location("label_server", SCRIPTS / "label_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["label_server"] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load()


def _composite(mirror_edge_frac: float) -> bytes:
    """Foam, a ruler along the top, a dark mirror frame ending at the given
    fraction of the width, and a fish to its right."""
    im = Image.new("RGB", (W, H), (228, 226, 220))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, int(W * mirror_edge_frac), H], fill=(35, 35, 40))            # mirror
    for x in range(int(W * 0.36), int(W * 0.94), 12):                               # ruler ticks
        d.line([x, 20, x, 70], fill=(15, 15, 15), width=2)
    d.ellipse([int(W * 0.45), 330, int(W * 0.9), 470], fill=(80, 70, 60))           # fish
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture
def srv(tmp_path):
    study = tmp_path / "data" / "study"
    (study / "lateral").mkdir(parents=True)
    (study / "sidecars").mkdir()
    ls.Handler.datasets = {"study": study}
    ls.Handler.default_dataset = "study"
    ls.Handler.images_dir = study
    ls.Handler.out_dir = study / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", study
    server.shutdown()


def _up(url, name, body, view="lateral", split=True):
    hdr = {"X-Filename": name, "X-View": view}
    if split:
        hdr["X-Split"] = "mirror"
    req = urllib.request.Request(f"{url}/api/upload?dataset=study", data=body,
                                 method="POST", headers=hdr)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_a_composite_is_split_and_the_original_kept(srv):
    url, study = srv
    data = _composite(0.27)
    code, body = _up(url, "IMG_0318.JPG", data)
    assert code == 200 and body["status"] == "split"
    assert 0.25 <= body["boundary_fraction"] <= 0.30
    lat = Image.open(study / "lateral" / "IMG_0318_L.JPEG")
    fro = Image.open(study / "frontal" / "IMG_0318_F.JPEG")
    assert (study / "originals" / "IMG_0318.JPG").read_bytes() == data
    rec = json.loads((study / "splits.json").read_text())["IMG_0318"]
    assert rec["split"] and rec["original"] == "originals/IMG_0318.JPG"
    assert rec["lateral_start"] == rec["boundary"] - round(450 / 6000 * W)   # guards the snout
    assert rec["frontal_end"] == rec["boundary"] + round(520 / 6000 * W)     # as the lab's crops
    assert lat.size == (W - rec["lateral_start"], H)
    assert fro.size == (rec["frontal_end"], H)


def test_a_mirror_edge_where_no_mirror_can_be_is_not_used(srv):
    """The 13 lab photos that left the frontal crop too small cut at 2-9% of the width."""
    url, study = srv
    code, body = _up(url, "IMG_0400.JPG", _composite(0.05))
    assert code == 200 and body["status"] == "stored_whole"
    assert "stored whole" in body["reason"]
    assert Image.open(study / "lateral" / "IMG_0400_L.JPEG").size == (W, H)
    assert not (study / "frontal" / "IMG_0400_F.JPEG").exists()
    assert (study / "originals" / "IMG_0400.JPG").is_file()
    assert json.loads((study / "splits.json").read_text())["IMG_0400"]["split"] is False


def test_the_same_photograph_again_is_not_split_twice(srv):
    url, study = srv
    data = _composite(0.27)
    _up(url, "IMG_1.JPG", data)
    before = (study / "lateral" / "IMG_1_L.JPEG").stat().st_mtime_ns
    code, body = _up(url, "IMG_1.JPG", data)
    assert code == 200 and body["status"] == "duplicate"
    assert (study / "lateral" / "IMG_1_L.JPEG").stat().st_mtime_ns == before


def test_a_different_photograph_under_the_same_name_is_refused(srv):
    url, study = srv
    _up(url, "IMG_2.JPG", _composite(0.27))
    code, body = _up(url, "IMG_2.JPG", _composite(0.30))
    assert code == 409 and "DIFFERENT" in body["error"]


def test_a_labelled_fish_keeps_its_images(srv):
    url, study = srv
    (study / "sidecars" / "IMG_3.json").write_text(json.dumps(
        {"fish_id": "IMG_3", "lateral": {"keypoints": {"premaxilla_tip": [5, 5]}}}))
    code, body = _up(url, "IMG_3.JPG", _composite(0.27))
    assert code == 409 and "already has labels" in body["error"]
    assert not (study / "originals" / "IMG_3.JPG").exists()
    assert not (study / "lateral" / "IMG_3_L.JPEG").exists()


def test_an_existing_crop_from_elsewhere_is_not_overwritten(srv):
    url, study = srv
    (study / "lateral" / "IMG_4_L.JPEG").write_bytes(_composite(0.27))
    code, body = _up(url, "IMG_4.JPG", _composite(0.27))
    assert code == 409 and "not overwritten" in body["error"]


def test_splitting_is_only_offered_for_lateral_uploads(srv):
    url, _ = srv
    code, body = _up(url, "IMG_5.JPG", _composite(0.27), view="frontal")
    assert code == 400


def test_an_unsplit_upload_still_behaves_as_before(srv):
    url, study = srv
    data = _composite(0.27)
    code, body = _up(url, "IMG_6.JPG", data, split=False)
    assert code == 200 and body["status"] == "added"
    assert (study / "lateral" / "IMG_6.JPG").read_bytes() == data
    assert not (study / "originals").exists()
