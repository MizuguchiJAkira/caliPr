"""Choosing the columns and rows of the measurements export.

An analysis that cannot take blanks -- a PCA in R drops every row with an NA --
loses most of its fish to a column most fish lack. The labeller previews the
measurements with each column's count of blanks, lets a column be left out, and
can keep only the fish with a value in every remaining column. What is tested:
the pipeline leaves the chosen columns out of every sheet and says so apart from
the study's own scope, complete-rows drops exactly the fish with a blank, the
preview counts blanks as the workbook holds them, and the server carries the
choice through.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from fish_morpho.pipeline import run  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
TESTS = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


payload = _load("_pipeline_payload", TESTS / "test_pipeline.py")._sidecar_payload
exporter = _load("export_measurements", SCRIPTS / "export_measurements.py")


@pytest.fixture
def study(tmp_path):
    """Two fish, one of them without a dorsal fin tip (so no DFh)."""
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for fid in ("BKT-0001", "BKT-0002"):
        (images / f"{fid}.jpg").write_bytes(b"\x00")
        p = payload(fid)
        if fid == "BKT-0002":
            del p["lateral"]["keypoints"]["dorsal_tip"]
        (labels / f"{fid}.json").write_text(json.dumps(p))
    return images, labels


def _sheet(path, name):
    rows = list(openpyxl.load_workbook(path)[name].iter_rows(values_only=True))
    return rows[0], rows[1:]


def _codes(header):
    return {str(h).split(" — ")[0] for h in header}


def _about(path):
    return {r[0]: r[1] for r in openpyxl.load_workbook(path)["About"].iter_rows(values_only=True)
            if r[0]}


def test_a_column_left_out_is_gone_from_every_sheet(study, tmp_path):
    images, labels = study
    out = run(images_dir=images, labels_dir=labels, output_path=tmp_path / "m.xlsx",
              mode="manual", model_config=None, left_out=("Ed",))
    hdr, _ = _sheet(out, "Measurements")
    assert "Ed" not in _codes(hdr) and "SL" in _codes(hdr)
    assert not any(str(h).startswith("Ed ") for h in _sheet(out, "Ratios")[0])
    assert "log Ed" not in _sheet(out, "Shape")[0]


def test_left_out_is_recorded_apart_from_the_studys_scope(study, tmp_path):
    images, labels = study
    out = run(images_dir=images, labels_dir=labels, output_path=tmp_path / "m.xlsx",
              mode="manual", model_config=None, drop_traits=("Mo",), left_out=("Ed",))
    about = _about(out)
    assert about["OUT OF SCOPE"] == "Mo"
    assert about["columns"] == "Ed"


def test_complete_rows_keeps_only_the_fish_with_every_value(study, tmp_path):
    images, labels = study
    out = run(images_dir=images, labels_dir=labels, output_path=tmp_path / "m.xlsx",
              mode="manual", model_config=None, complete_only=True)
    hdr, rows = _sheet(out, "Measurements")
    assert [r[hdr.index("fish_id")] for r in rows] == ["BKT-0001"]
    for sheet in ("Ratios", "Shape"):
        h, rs = _sheet(out, sheet)
        assert [r[h.index("fish_id")] for r in rs] == ["BKT-0001"]
    assert "BKT-0002" in _about(out)["specimens"]


def test_leaving_out_the_blank_column_keeps_the_fish(study, tmp_path):
    images, labels = study
    out = run(images_dir=images, labels_dir=labels, output_path=tmp_path / "m.xlsx",
              mode="manual", model_config=None, left_out=("DFh",), complete_only=True)
    hdr, rows = _sheet(out, "Measurements")
    assert len(rows) == 2


def test_the_preview_counts_blanks_as_the_workbook_holds_them(study, tmp_path):
    images, labels = study
    out = run(images_dir=images, labels_dir=labels, output_path=tmp_path / "m.xlsx",
              mode="manual", model_config=None)
    p = exporter.preview(out)
    cols = {c["code"]: c for c in p["columns"]}
    assert p["columns"][0]["code"] == "SL"           # SL leads the measurements
    assert cols["DFh"]["missing"] == 1 and cols["SL"]["missing"] == 0
    i = [c["code"] for c in p["columns"]].index("DFh")
    by_id = {r["id"]: r for r in p["rows"]}
    assert by_id["BKT-0002"]["values"][i] is None
    assert isinstance(by_id["BKT-0001"]["values"][i], float)


# ---- through the server ---------------------------------------------------

@pytest.fixture
def srv(tmp_path, monkeypatch):
    ls = _load("label_server", SCRIPTS / "label_server.py")
    Image = pytest.importorskip("PIL.Image")
    st = tmp_path / "data" / "exportstudy"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    fid = "Salvelinus_fontinalis_TXD_20"
    Image.new("RGB", (400, 200), (200, 200, 200)).save(st / "lateral" / f"{fid}_L.JPEG")
    (st / "sidecars" / f"{fid}.json").write_text(json.dumps(payload(fid)))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(ls, "_ROOT", repo)
    (repo / "scripts").symlink_to(SCRIPTS)
    monkeypatch.setattr(ls, "show_on_this_computer", lambda path, how: (how, None))
    ls.Handler.datasets = {"exportstudy": st}
    ls.Handler.default_dataset = "exportstudy"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", repo / "results" / "exportstudy"
    server.shutdown()


def test_the_server_previews_without_writing_results(srv):
    url, results = srv
    with urllib.request.urlopen(url + "/api/export/preview?dataset=exportstudy", timeout=120) as r:
        p = json.loads(r.read())
    assert p["ok"] and len(p["rows"]) == 1 and p["columns"][0]["code"] == "SL"
    assert not results.exists()


def test_the_server_exports_what_the_preview_chose(srv):
    url, results = srv
    body = json.dumps({"leave_out": ["Ed", "not a code; rm -rf"], "complete_only": True})
    req = urllib.request.Request(url + "/api/export/measurements?dataset=exportstudy",
                                 data=body.encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        assert json.loads(r.read())["ok"]
    hdr, rows = _sheet(results / "measurements.xlsx", "Measurements")
    assert "Ed" not in _codes(hdr) and len(rows) == 1
    assert _about(results / "measurements.xlsx")["columns"] == "Ed"


def test_the_csv_is_what_r_reads(study, tmp_path):
    images, labels = study
    out = run(images_dir=images, labels_dir=labels, output_path=tmp_path / "m.xlsx",
              mode="manual", model_config=None)
    import csv
    rows = list(csv.reader(open(exporter.write_csv(out, tmp_path / "m.csv"), encoding="utf-8")))
    head = rows[0]
    assert head[0] == "fish_id" and head[head.index("units") + 1] == "SL"
    assert all(" — " not in h for h in head)          # codes, not the long labels
    by_id = {r[0]: dict(zip(head, r)) for r in rows[1:]}
    assert by_id["BKT-0002"]["DFh"] == "NA"             # a blank R cannot misread
    assert float(by_id["BKT-0001"]["DFh"]) > 0


def test_the_workbook_is_handed_over_when_asked_for(srv, monkeypatch):
    url, results = srv
    body = json.dumps({"leave_out": ["Ed"], "format": "xlsx"})
    req = urllib.request.Request(url + "/api/export/measurements?dataset=exportstudy",
                                 data=body.encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    assert out["ok"] and out["path"].endswith("measurements.xlsx") and out["shown"] == "open"
    # the CSV is written beside it all the same, with the same columns
    assert "Ed" not in (results / "measurements.csv").read_text().splitlines()[0].split(",")
