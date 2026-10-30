"""Turning and mirroring a photograph in the labeler, with every point on it.

A photograph that came in on its side, upside down or facing right is turned
head left, back up. Tested: each turn's arithmetic, and that the eight possible
net turns compose as they should; that every saved point lands on the same
pixels after a turn (a mark painted on the fish stays under its landmark); that
the labels' image fingerprints, the cached predictions and the record of turns
follow; that turning back returns the original file exactly, however many turns
it took; that a photograph holding both views turns both views' labels; and what
is refused.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fish_morpho import image_identity as ii  # noqa: E402

OPS = ii.ORIENT_OPS
W, H = 60, 40
MARK = (10, 5, 18, 13)                        # x0, y0, x1, y1 of a red block
AT = [14.0, 9.0]                              # its centre, where the landmark sits


def _marked() -> "Image.Image":
    im = Image.new("RGB", (W, H), (40, 90, 160))
    im.paste((230, 20, 20), MARK)
    return im


def _is_mark(im, pt) -> bool:
    r, g, b = im.convert("RGB").getpixel((int(pt[0]), int(pt[1])))
    return r > 150 and g < 90 and b < 90


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", OPS)
def test_a_point_stays_on_its_pixels(op):
    im = _marked()
    turned = ii.apply_orientation(im, ii.orientation_after(None, op))
    assert turned.size == ii.oriented_size(op, W, H)
    assert _is_mark(turned, ii.orient_point(op, *AT, W, H))


def test_the_net_turn_is_the_steps_taken():
    im, state = _marked(), None
    stepwise = im
    for op in ("rotate_cw", "mirror", "rotate_cw", "flip", "rotate_ccw", "mirror", "rotate_cw"):
        state = ii.orientation_after(state, op)
        stepwise = ii.apply_orientation(stepwise, ii.orientation_after(None, op))
        assert ii.apply_orientation(im, state).tobytes() == stepwise.tobytes(), op


def test_turns_undo_and_compose():
    start = {"turns": 0, "mirrored": False}
    s = None
    for _ in range(4):
        s = ii.orientation_after(s, "rotate_cw")
    assert s == start
    assert ii.orientation_after(ii.orientation_after(None, "rotate_cw"), "rotate_ccw") == start
    assert ii.orientation_after(ii.orientation_after(None, "mirror"), "mirror") == start
    half_and_mirror = ii.orientation_after(ii.orientation_after(
        ii.orientation_after(None, "rotate_cw"), "rotate_cw"), "mirror")
    assert ii.orientation_after(None, "flip") == half_and_mirror


def test_orient_view_moves_every_kind_of_coordinate():
    doc = {"lateral": {"keypoints": {"a": [1, 2]}, "polygons": {"body": [[3, 4], [5, 6]]},
                       "calibration": {"point_a": [7, 8], "point_b": [9, 10]},
                       "midline": [[11, 12]]},
           "metadata": {"assist": {"corrected": {"a": {"from": [1, 1], "to": [2, 2]}}}}}
    assert ii.orient_view(doc, "lateral", "mirror", W, H) == 6
    lat = doc["lateral"]
    assert lat["keypoints"]["a"] == [59, 2] and lat["polygons"]["body"] == [[57, 4], [55, 6]]
    assert lat["calibration"]["point_a"] == [53, 8] and lat["midline"] == [[49, 12]]
    assert doc["metadata"]["assist"]["corrected"]["a"] == {"from": [59, 1], "to": [58, 2]}


def test_shift_view_is_unchanged():
    doc = {"lateral": {"keypoints": {"a": [1.04, 2]}}}
    assert ii.shift_view(doc, "lateral", 10, -1) == 1
    assert doc["lateral"]["keypoints"]["a"] == [11.0, 1.0]


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------

FID = "Salvelinus_fontinalis_LT_7"


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
    st = data / "trout"
    for d in ("lateral", "sidecars", "sidecars_auto"):
        (st / d).mkdir(parents=True)
    photo = st / "lateral" / f"{FID}_L.JPEG"
    _marked().save(photo, quality=95)
    (st / "sidecars" / f"{FID}.json").write_text(json.dumps({
        "fish_id": FID,
        "lateral": {"keypoints": {"premaxilla_tip": AT},
                    "calibration": {"mode": "manual", "point_a": [2, 30], "point_b": [52, 30],
                                    "known_mm": 10}},
        "metadata": {"images": {"lateral": ii.fingerprint(photo)}}}))
    (st / "sidecars_auto" / f"{FID}.json").write_text(json.dumps(
        {"lateral": {"keypoints": {"premaxilla_tip": AT}}, "metadata": {"source": "predicted"}}))
    ls.Handler.data_root = data
    ls.Handler.datasets = ls.discover_datasets(data)
    ls.Handler.default_dataset = "trout"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield ls, f"http://127.0.0.1:{server.server_address[1]}", st, photo
    server.shutdown()


def _turn(url, op, fid=FID, view="lateral"):
    req = urllib.request.Request(f"{url}/api/specimen/orient?dataset=trout", method="POST",
                                 data=json.dumps({"id": fid, "view": view, "op": op}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _doc(st):
    return json.loads((st / "sidecars" / f"{FID}.json").read_text())


def test_a_turn_carries_the_labels_with_the_photograph(srv):
    ls, url, st, photo = srv
    before = photo.read_bytes()
    code, r = _turn(url, "rotate_cw")
    assert code == 200 and r["ok"] and r["size"] == [H, W], r
    doc = _doc(st)
    pt = doc["lateral"]["keypoints"]["premaxilla_tip"]
    with Image.open(photo) as im:
        assert im.size == (H, W) and _is_mark(im, pt)
    # still recorded as placed on this image, so the labeler draws and saves them
    assert ls.alignment(st, FID, doc)["lateral"]["status"] == ii.OK
    assert doc["metadata"]["orientation"]["lateral"] == {"turns": 1, "mirrored": False}
    cal = doc["lateral"]["calibration"]
    assert abs(((cal["point_b"][0] - cal["point_a"][0]) ** 2
                + (cal["point_b"][1] - cal["point_a"][1]) ** 2) ** 0.5 - 50) < 1e-6   # same scale
    cache = json.loads((st / "sidecars_auto" / f"{FID}.json").read_text())
    assert cache["lateral"]["keypoints"]["premaxilla_tip"] == pt
    assert (st / "unturned" / "lateral" / photo.name).read_bytes() == before
    assert json.loads((st / "orientation.json").read_text()) == {
        f"lateral/{photo.name}": {"turns": 1, "mirrored": False}}


def test_turning_back_returns_the_original_exactly(srv):
    _, url, st, photo = srv
    before = photo.read_bytes()
    for _ in range(4):
        assert _turn(url, "rotate_cw")[0] == 200
    assert photo.read_bytes() == before
    assert _doc(st)["lateral"]["keypoints"]["premaxilla_tip"] == AT
    assert "orientation" not in _doc(st)["metadata"]
    assert not (st / "unturned").exists()
    assert json.loads((st / "orientation.json").read_text()) == {}
    for op in ("mirror", "mirror", "flip", "rotate_cw", "rotate_cw", "mirror"):   # nets to none
        assert _turn(url, op)[0] == 200
    assert photo.read_bytes() == before


def test_a_mirror_is_recorded_as_one(srv):
    _, url, st, photo = srv
    code, r = _turn(url, "mirror")
    assert code == 200 and r["mirrored"] is True
    doc = _doc(st)
    assert doc["metadata"]["orientation"]["lateral"]["mirrored"] is True
    with Image.open(photo) as im:
        assert _is_mark(im, doc["lateral"]["keypoints"]["premaxilla_tip"])


def test_one_photograph_holding_both_views_turns_both(srv):
    _, url, st, photo = srv
    (st / "schema.json").write_text(json.dumps({"single_photo": True}))
    doc = _doc(st)
    doc["frontal"] = {"keypoints": {"mouth_left": [50, 30]}}
    (st / "sidecars" / f"{FID}.json").write_text(json.dumps(doc))
    code, r = _turn(url, "rotate_ccw")
    assert code == 200 and set(r["views"]) == {"lateral", "frontal"}, r
    doc = _doc(st)
    assert doc["frontal"]["keypoints"]["mouth_left"] == [30, 10]       # (y, W - x)
    assert set(doc["metadata"]["orientation"]) == {"lateral", "frontal"}


def test_what_turning_refuses(srv):
    ls, url, st, photo = srv
    before = photo.read_bytes()
    assert _turn(url, "spin")[0] == 400
    assert _turn(url, "rotate_cw", fid="../x")[0] == 400
    assert _turn(url, "rotate_cw", fid="nobody")[0] == 404
    doc = _doc(st)                                  # labels that no longer fit the image
    doc["metadata"]["images"]["lateral"]["width"] = W + 100
    (st / "sidecars" / f"{FID}.json").write_text(json.dumps(doc))
    assert _turn(url, "rotate_cw")[0] == 409
    ls.Handler.demo_mode = True
    assert _turn(url, "mirror")[0] == 403
    assert photo.read_bytes() == before and not (st / "unturned").exists()
