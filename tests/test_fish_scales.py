"""Fish scales: the enhanced view, and the count along a traced row.

Synthetic flanks: a pale ground crossed by dark crescents at a known spacing (the
scale margins), with and without melanophore dots and a glare gradient laid over
them. The count has to come out exact on those, and the dots must not add ticks.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fish_morpho import fish_scales as fs  # noqa: E402


def _flank(n_scales=38, spacing=24.0, height=200, dots=0, glare=False, seed=0):
    """A flank with ``n_scales`` margins along y = height/2, from x = 40."""
    w = int(80 + n_scales * spacing)
    img = np.full((height, w, 3), 200, np.uint8)
    for i in range(n_scales):
        cx = int(40 + (i + 0.5) * spacing)
        # a crescent, open to the front, as a scale's posterior margin is
        cv2.ellipse(img, (cx - int(0.4 * spacing), height // 2), (int(0.55 * spacing), int(0.9 * spacing)),
                    0, -70, 70, (120, 120, 120), 2, cv2.LINE_AA)
        for row in (-1, 1):                  # the rows above and below, half a scale on
            cv2.ellipse(img, (cx, height // 2 + int(row * 1.4 * spacing)), (int(0.55 * spacing), int(0.9 * spacing)),
                        0, -70, 70, (120, 120, 120), 2, cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    for _ in range(dots):
        cv2.circle(img, (int(rng.uniform(0, w)), int(rng.uniform(0, height))), int(rng.integers(2, 4)),
                   (60, 60, 60), -1, cv2.LINE_AA)
    if glare:
        ramp = np.linspace(-40, 40, w)[None, :, None]
        img = np.clip(img.astype(np.float32) + ramp, 0, 255).astype(np.uint8)
    return img, [[40.0, height / 2], [40.0 + n_scales * spacing, height / 2]]


@pytest.mark.parametrize("n,spacing", [(38, 24.0), (34, 18.0), (52, 30.0)])
def test_counts_the_margins_crossed(n, spacing):
    img, path = _flank(n, spacing)
    r = fs.count_along(img, path, spacing_px=(0.6 * spacing, 1.6 * spacing))
    assert r["ok"], r
    assert abs(r["count"] - n) <= 1
    assert abs(r["spacing_px"] - spacing) < 0.1 * spacing


def test_dots_and_glare_add_no_ticks():
    img, path = _flank(38, 24.0, dots=900, glare=True)
    r = fs.count_along(img, path, spacing_px=(14, 38))
    assert r["ok"], r
    assert abs(r["count"] - 38) <= 1


def test_spacing_bounded_by_standard_length():
    # 38 scales over 0.78 SL at 24 px each: SL = 38 * 24 / 0.78
    img, path = _flank(38, 24.0)
    r = fs.count_along(img, path, sl_px=38 * 24 / 0.78)
    assert r["ok"] and abs(r["count"] - 38) <= 1


def test_ticks_lie_on_the_path_in_order():
    img, path = _flank(30, 20.0)
    r = fs.count_along(img, path, spacing_px=(12, 32))
    xs = [t[0] for t in r["ticks"]]
    assert xs == sorted(xs)
    assert all(abs(t[1] - path[0][1]) < 1 for t in r["ticks"])
    assert path[0][0] - 1 <= xs[0] and xs[-1] <= path[1][0] + 1


def test_a_plain_ground_is_not_counted():
    img = np.full((200, 1000, 3), 180, np.uint8)
    img = (img + np.random.default_rng(1).normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    r = fs.count_along(img, [[50, 100], [950, 100]], spacing_px=(15, 40))
    assert not r["ok"]
    assert "error" in r


def test_too_short_a_path_is_refused():
    img, _ = _flank(10, 20.0)
    r = fs.count_along(img, [[40, 100], [55, 100]])
    assert not r["ok"] and "short" in r["error"]


@pytest.mark.parametrize("mode", fs.MODES)
@pytest.mark.parametrize("channel", fs.CHANNELS)
def test_enhance_keeps_the_photographs_size(mode, channel):
    img, _ = _flank(20, 24.0, dots=100)
    out = fs.enhance(img, mode=mode, strength=4, scale_size_px=24, channel=channel)
    assert out.shape == img.shape[:2] and out.dtype == np.uint8


def test_margins_view_keeps_lines_and_drops_dots():
    img = np.full((120, 240, 3), 200, np.uint8)
    cv2.line(img, (60, 10), (60, 110), (110, 110, 110), 2)
    cv2.circle(img, (180, 60), 3, (110, 110, 110), -1)
    m = fs.margin_map(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # a dot's centre drops out; its rim still reads as a short curved line, so
    # the dot keeps about half a line's strength (Frangi's ratio does no better)
    assert m[60, 58:63].max() > 2 * m[57:64, 177:184].max()


def test_enhance_refuses_an_unknown_mode():
    with pytest.raises(ValueError):
        fs.enhance(np.zeros((10, 10, 3), np.uint8), mode="posterize")


# ---- the labeler's routes ---------------------------------------------------

import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture
def server(tmp_path):
    spec = importlib.util.spec_from_file_location("label_server", SCRIPTS / "label_server.py")
    ls = importlib.util.module_from_spec(spec)
    sys.modules["label_server"] = ls
    spec.loader.exec_module(ls)
    st = tmp_path / "data" / "shiners"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    img, path = _flank(38, 24.0, dots=300)
    cv2.imwrite(str(st / "lateral" / "Notropis_x_1.jpg"), img)
    ls.Handler.data_root = tmp_path / "data"
    ls.Handler.datasets = {"shiners": st}
    ls.Handler.default_dataset = "shiners"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    srv = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", path, st
    srv.shutdown()


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_count_route_proposes_the_ticks(server):
    base, path, st = server
    code, r = _post(f"{base}/api/scales/count?dataset=shiners",
                    {"id": "Notropis_x_1", "view": "lateral", "path": path, "sl": 38 * 24 / 0.78})
    assert code == 200 and r["ok"], r
    assert abs(r["count"] - 38) <= 1 and len(r["ticks"]) == r["count"]
    assert r["method"] == "every margin" and r["range"][0] <= 38 <= r["range"][1]
    assert list((st / "sidecars").iterdir()) == []            # nothing written


def test_count_route_takes_the_body_axis(server):
    base, path, _ = server
    sl = 38 * 24 / 0.78
    axis = [[path[0][0] - 0.27 * sl, path[0][1]], [path[1][0], path[1][1]]]   # snout, end of the column
    code, r = _post(f"{base}/api/scales/count?dataset=shiners",
                    {"id": "Notropis_x_1", "path": path, "sl": sl, "axis": axis})
    assert code == 200 and r["ok"] and abs(r["count"] - 38) <= 1
    assert _post(f"{base}/api/scales/count?dataset=shiners",
                 {"id": "Notropis_x_1", "path": path, "axis": [[0, 0]]})[0] == 400


def test_count_route_refuses_what_it_cannot_count(server):
    base, path, _ = server
    assert _post(f"{base}/api/scales/count?dataset=shiners",
                 {"id": "Notropis_x_1", "path": path[:1]})[0] == 400
    assert _post(f"{base}/api/scales/count?dataset=shiners",
                 {"id": "../x", "path": path})[0] == 400
    assert _post(f"{base}/api/scales/count?dataset=shiners",
                 {"id": "Notropis_x_9", "path": path})[0] == 404


@pytest.mark.parametrize("mode", fs.MODES)
def test_view_route_answers_a_jpeg_the_photographs_shape(server, mode):
    base, _, st = server
    with urllib.request.urlopen(f"{base}/api/scales/view/Notropis_x_1?dataset=shiners&view=lateral"
                                f"&mode={mode}&strength=5&channel=green&scale=24", timeout=60) as r:
        assert r.headers["Content-Type"] == "image/jpeg"
        out = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    src = cv2.imread(str(st / "lateral" / "Notropis_x_1.jpg"))
    assert out.shape[:2] == src.shape[:2]


def test_view_route_refuses_an_unknown_mode(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"{base}/api/scales/view/Notropis_x_1?dataset=shiners&mode=posterize", timeout=30)
    assert e.value.code == 400


# ---- filled in from the stretches that can be read ---------------------------

def test_every_count_carries_a_likely_range_around_it():
    img, path = _flank(38, 24.0, dots=300)
    r = fs.count_along(img, path, spacing_px=(14, 38))
    assert r["method"] == "every margin"
    assert r["range"][0] <= 38 <= r["range"][1] and r["range"][0] < r["count"] < r["range"][1]


def _obscured(n=40, spacing=24.0, keep=0.3, seed=0):
    """A flank whose margins can be read only over its first ``keep`` share: the
    rest is crossed by marks as dark as margins, placed at random."""
    img, path = _flank(n, spacing)
    x0, x1 = int(40 + keep * n * spacing), int(40 + n * spacing) + 20
    img[:, x0:x1] = 200
    rng = np.random.default_rng(seed)
    for _ in range((x1 - x0) // 4):
        x, y = int(rng.uniform(x0, x1)), int(rng.uniform(40, 160))
        cv2.line(img, (x, y - int(rng.uniform(5, 25))), (x + int(rng.uniform(-6, 6)), y + int(rng.uniform(5, 25))),
                 (110, 110, 110), 2)
    return img, path


def test_a_row_read_only_in_part_is_filled_in_from_that_part(monkeypatch):
    monkeypatch.setattr(fs, "TICKS_COVERAGE", 1.01)        # whatever is read, fill in
    img, path = _obscured(40, 24.0, keep=0.4)
    r = fs.count_along(img, path, spacing_px=(14, 38))
    assert r["ok"] and r["method"] == "filled in", r
    assert r["range"][0] <= 40 <= r["range"][1] and abs(r["count"] - 40) <= 2
    assert fs.MIN_MEASURED <= r["measured"] <= 20                 # the clear part only
    # ticks only where it measured: the clear part, and at most one window (six
    # scales) past it, since a window mostly clear still reads as regular
    assert all(t[0] <= 40 + 0.4 * 40 * 24 + 6 * 24 + 1 for t in r["ticks"])


def test_more_scales_measured_narrow_the_range():
    img, path = _flank(40, 24.0)
    rel = np.ones(int(40 * 24))
    band = fs.band_signal(img, path, spacing_px=(14, 38))
    sig, lo, hi = band[2], band[3], band[4]
    st = fs.read_stretches(sig, lo, hi, 24.0)
    few = fs.fill_in([t for t in st if t[1] <= 12 * 24], rel)
    many = fs.fill_in(st, rel)
    assert few["measured"] < many["measured"] and few["err"] > many["err"]
    assert abs(many["count"] - 40) <= 1


def test_too_few_scales_read_are_not_filled_in():
    img, path = _flank(40, 24.0)
    band = fs.band_signal(img, path, spacing_px=(14, 38))
    sig, lo, hi = band[2], band[3], band[4]
    st = [t for t in fs.read_stretches(sig, lo, hi, 24.0) if t[1] <= 7 * 24]
    assert fs.fill_in(st, np.ones(sig.size)) is None


def test_scale_length_follows_the_body():
    assert fs.relative_length(0.51) == pytest.approx(1.0)
    assert fs.relative_length(0.66) < 1 < fs.relative_length(0.96)
    pts = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    assert list(fs.body_positions(pts, [[0, 0], [100, 0]])) == [0.0, 0.5, 1.0]
    assert fs.body_positions(pts, None) is None
