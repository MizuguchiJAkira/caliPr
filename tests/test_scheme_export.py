"""The measurements export for a study on another landmark scheme.

Every caliPr trait is defined on caliPr's own landmarks, so a study on BGNN 2D
(or a scheme made here) has none, and the export used to refuse it outright. Its
measurements are its landmark coordinates: what is tested is that they come out
in the scheme's order, in millimetres where the specimen has a scale and pixels
where it does not, with centroid size, blanks as NA, and the preview's choices
(leave out, complete rows) honoured -- through the script and through the server.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("openpyxl")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exporter = _load("export_measurements", SCRIPTS / "export_measurements.py")
from fish_morpho import schemes  # noqa: E402

ORDER = [n for n, *_ in schemes.BUILTIN["bgnn_2d"]["landmarks"]]


@pytest.fixture
def study(tmp_path):
    """Two fish on BGNN 2D: one scaled and complete, one unscaled and missing two."""
    st = tmp_path / "minnows"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    (st / "schema.json").write_text(json.dumps({"scheme": "bgnn_2d"}))
    full = {n: [100 + 10 * i, 200 + (i % 5) * 7] for i, n in enumerate(ORDER)}
    part = {n: p for n, p in full.items() if n not in ("cleithrum", "opercle_medial")}
    for fid, kps, cal in (
        ("M_01", full, {"mode": "manual", "point_a": [0, 0], "point_b": [200, 0],
                        "known_mm": 10}),                         # 20 px/mm
        ("M_02", part, {"mode": "none"}),
    ):
        (st / "sidecars" / f"{fid}.json").write_text(json.dumps(
            {"fish_id": fid, "lateral": {"keypoints": kps, "calibration": cal}}))
    return st, full


def _export(st, tmp_path, *extra):
    out = tmp_path / "out"
    code = exporter.main(["--images", str(st / "lateral"), "--labels", str(st / "sidecars"),
                          "--out", str(out / "m.xlsx"), "--csv", str(out / "m.csv"),
                          "--preview-json", str(out / "p.json"), *extra])
    assert code == 0
    rows = list(csv.DictReader(open(out / "m.csv", encoding="utf-8")))
    return {r["fish_id"]: r for r in rows}, json.loads((out / "p.json").read_text()), out


def test_the_coordinates_are_the_measurements(study, tmp_path):
    st, full = study
    rows, prev, out = _export(st, tmp_path)
    head = list(rows["M_01"].keys())
    assert head[:5] == ["fish_id", "group", "units", "px_per_mm", "centroid_size"]
    assert head[5:] == [f"{n}_{a}" for n in ORDER for a in ("x", "y")]   # the scheme's order
    m1, m2 = rows["M_01"], rows["M_02"]
    assert m1["units"] == "mm" and float(m1["px_per_mm"]) == 20
    assert float(m1["dentary_anterior_x"]) == full["dentary_anterior"][0] / 20
    assert m2["units"] == "px" and float(m2["dentary_anterior_x"]) == full["dentary_anterior"][0]
    assert m2["cleithrum_x"] == "NA" and m2["centroid_size"] == "NA"
    pts = [(p[0] / 20, p[1] / 20) for p in full.values()]
    cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    cs = math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in pts))
    assert float(m1["centroid_size"]) == pytest.approx(cs, abs=1e-3)
    assert (out / "m.xlsx").is_file()


def test_the_preview_counts_missing_landmarks(study, tmp_path):
    _, prev, _ = _export(study[0], tmp_path)
    cols = {c["code"]: c["missing"] for c in prev["columns"]}
    assert prev["what"] == "landmarks" and list(cols) == ORDER
    assert cols["cleithrum"] == 1 and cols["dentary_anterior"] == 0


def test_complete_rows_and_leaving_out_are_honoured(study, tmp_path):
    rows, _, _ = _export(study[0], tmp_path, "--complete-only")
    assert list(rows) == ["M_01"]
    rows, _, _ = _export(study[0], tmp_path, "--complete-only",
                         "--leave-out", "cleithrum,opercle_medial")
    assert set(rows) == {"M_01", "M_02"} and "cleithrum_x" not in rows["M_01"]
    assert rows["M_02"]["centroid_size"] != "NA"      # every kept landmark is present


def test_the_server_no_longer_refuses_it(study, tmp_path, monkeypatch):
    st, _ = study
    ls = _load("label_server", SCRIPTS / "label_server.py")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").symlink_to(SCRIPTS)
    monkeypatch.setattr(ls, "_ROOT", repo)
    monkeypatch.setattr(ls, "show_on_this_computer", lambda path, how: (how, None))
    ls.Handler.datasets = {"minnows": st}
    ls.Handler.default_dataset = "minnows"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(url + "/api/export/preview?dataset=minnows", timeout=120) as r:
            p = json.loads(r.read())
        assert p["ok"] and p["what"] == "landmarks" and len(p["rows"]) == 2
        req = urllib.request.Request(url + "/api/export/measurements?dataset=minnows",
                                     data=json.dumps({"format": "csv"}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        assert out["ok"] and out["path"].endswith("measurements.csv")
    finally:
        server.shutdown()
