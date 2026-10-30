"""Renaming a specimen, and turning a rig photograph the right way up.

A batch arrives named by the camera (Img0133) or with a number read wrong, and
the name is what carries a fish's identity and strain. Tested: a rename moves
the photograph, the saved labels (fish_id included), the cached predictions and
every table naming the fish, together; it refuses what would collide, escape
the study, or break a blind round; and the head-on mirror decides which way up a
rig photograph goes, where the ruler test cannot.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")
cv2 = pytest.importorskip("cv2")
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------

def _rig(mirror: str) -> np.ndarray:
    """A pale frame with the mirror's black holder down one side."""
    img = np.full((400, 600, 3), 210, np.uint8)
    if mirror == "left":
        img[:, :80] = 20
    elif mirror == "right":
        img[:, -80:] = 20
    return img


def test_the_mirror_says_which_way_up():
    pc = _load("preprocess_cornell", SCRIPTS / "preprocess_cornell.py")
    assert pc.mirror_side(_rig("left")) == "left"
    assert pc.mirror_side(_rig("right")) == "right"
    assert pc.mirror_side(_rig("none")) is None


def test_a_photo_with_the_mirror_on_the_right_is_turned_round(tmp_path):
    pc = _load("preprocess_cornell", SCRIPTS / "preprocess_cornell.py")
    img = _rig("right")
    img[300:340, 200:260] = (0, 0, 255)                 # a mark to follow
    p = tmp_path / "Img0001.jpg"
    Image.fromarray(img[:, :, ::-1]).save(p, quality=98)
    out = pc.normalize_orientation(p)
    assert pc.mirror_side(out) == "left"
    ys, xs = np.where((out[:, :, 2] > 150) & (out[:, :, 0] < 90))
    assert ys.mean() < 200 and xs.mean() > 300          # the mark went round with it


# ---------------------------------------------------------------------------
# renaming
# ---------------------------------------------------------------------------

OLD, NEW = "Salvelinus_fontinalis_TEM_Img0151", "Salvelinus_fontinalis_TEM_5"


@pytest.fixture
def srv(tmp_path):
    ls = _load("label_server", SCRIPTS / "label_server.py")
    from fish_morpho import darwin_core
    data = tmp_path / "data"
    st = data / "trout"
    for d in ("lateral", "sidecars", "sidecars_auto", "reference"):
        (st / d).mkdir(parents=True)
    for fid in (OLD, "Salvelinus_fontinalis_ASN_1"):
        Image.new("RGB", (60, 40)).save(st / "lateral" / f"{fid}_L.JPEG")
    (st / "sidecars" / f"{OLD}.json").write_text(json.dumps(
        {"fish_id": OLD, "lateral": {"keypoints": {"premaxilla_tip": [1, 2]}}}))
    (st / "sidecars_auto" / f"{OLD}.json").write_text("{}")
    (st / "groups.csv").write_text(f"fish_id,group\n{OLD},TEM\n")
    (st / "image_fingerprints.json").write_text(json.dumps({OLD: {"lateral": {"width": 60}}}))
    (st / "reference" / "reference.json").write_text(json.dumps({"specimen": OLD}))
    darwin_core.update(st, {OLD: {"catalogNumber": "1"}})
    rnd = data / "trout.relabel-1"
    (rnd / "lateral").mkdir(parents=True)
    (rnd / "sidecars").mkdir()
    Image.new("RGB", (60, 40)).save(rnd / "lateral" / "R1-01_L.JPEG")
    (rnd / "round.json").write_text(json.dumps({"study": "trout", "round": 1,
                                                "fish": {"R1-01": OLD}}))
    ls.Handler.data_root = data
    ls.Handler.datasets = ls.discover_datasets(data)
    ls.Handler.default_dataset = "trout"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield ls, f"http://127.0.0.1:{server.server_address[1]}", st
    server.shutdown()


def _rename(url, old, new, ds="trout"):
    req = urllib.request.Request(f"{url}/api/specimen/rename?dataset={ds}", method="POST",
                                 data=json.dumps({"id": old, "new_id": new}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_a_rename_carries_everything_with_it(srv):
    _, url, st = srv
    code, r = _rename(url, OLD, NEW)
    assert code == 200 and r["ok"] and r["id"] == NEW, r
    assert (st / "lateral" / f"{NEW}_L.JPEG").is_file()
    assert not (st / "lateral" / f"{OLD}_L.JPEG").exists()
    assert json.loads((st / "sidecars" / f"{NEW}.json").read_text())["fish_id"] == NEW
    assert not (st / "sidecars" / f"{OLD}.json").exists()
    assert (st / "sidecars_auto" / f"{NEW}.json").is_file()
    assert list(csv.reader((st / "groups.csv").open()))[1] == [NEW, "TEM"]
    assert NEW in json.loads((st / "image_fingerprints.json").read_text())
    assert json.loads((st / "reference" / "reference.json").read_text())["specimen"] == NEW
    from fish_morpho import darwin_core
    assert darwin_core.load(st) == {NEW: {"catalogNumber": "1"}}
    rnd = json.loads((st.parent / "trout.relabel-1" / "round.json").read_text())
    assert rnd["fish"] == {"R1-01": NEW}
    assert set(r["updated"]) >= {"Darwin Core record", "groups.csv", "image fingerprints",
                                 "the study's example", "re-label round 1"}


def test_a_plain_named_photo_keeps_its_plain_form(srv):
    _, url, st = srv
    Image.new("RGB", (60, 40)).save(st / "lateral" / "Img0200.jpg")
    code, r = _rename(url, "Img0200", "Salvelinus_fontinalis_TEM_60")
    assert code == 200 and (st / "lateral" / "Salvelinus_fontinalis_TEM_60.jpg").is_file()


def test_what_a_rename_refuses(srv):
    ls, url, st = srv
    assert _rename(url, OLD, "Salvelinus_fontinalis_ASN_1")[0] == 409      # taken
    assert _rename(url, OLD, "../escape")[0] == 400
    assert _rename(url, OLD, "Salvelinus_fontinalis_TEM_5_L")[0] == 400   # view suffix
    assert _rename(url, OLD, OLD)[0] == 400
    assert _rename(url, "nobody", NEW)[0] == 404
    assert _rename(url, "R1-01", "R1-02", ds="trout.relabel-1")[0] == 400  # blind round
    ls.Handler.demo_mode = True
    assert _rename(url, OLD, NEW)[0] == 403
    assert (st / "lateral" / f"{OLD}_L.JPEG").is_file()                    # nothing moved
