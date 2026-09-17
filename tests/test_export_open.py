"""Exports open on this computer instead of downloading.

The browser pane inside the Claude app offers every download to Claude rather than
opening it, so an exported workbook could not be opened from there. The labeler
only listens on 127.0.0.1, so the server opens the file itself. What is tested is
that each export is written, the right thing is opened or shown, a machine with
nothing to open it with falls back to downloading, and the download still works.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
TESTS = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load("label_server", SCRIPTS / "label_server.py")
payload = _load("_pipeline_payload", TESTS / "test_pipeline.py")._sidecar_payload


@pytest.fixture
def srv(tmp_path, monkeypatch):
    study = tmp_path / "data" / "exportstudy"
    (study / "lateral").mkdir(parents=True)
    (study / "sidecars").mkdir()
    fid = "Salvelinus_fontinalis_TXD_20"
    Image.new("RGB", (400, 200), (200, 200, 200)).save(study / "lateral" / f"{fid}_L.JPEG")
    (study / "sidecars" / f"{fid}.json").write_text(json.dumps(payload(fid)))
    results = tmp_path / "repo"             # results/ goes here, not into the real repository
    results.mkdir()
    monkeypatch.setattr(ls, "_ROOT", results)
    (results / "scripts").symlink_to(SCRIPTS)
    shown = []
    monkeypatch.setattr(ls, "show_on_this_computer", lambda path, how: shown.append((path, how)))
    ls.Handler.datasets = {"exportstudy": study}
    ls.Handler.default_dataset = "exportstudy"
    ls.Handler.images_dir = study
    ls.Handler.out_dir = study / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", results / "results" / "exportstudy", shown
    server.shutdown()


def _post(url, path):
    req = urllib.request.Request(url + path, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_workbook_is_written_and_opened_not_downloaded(srv):
    url, results, shown = srv
    code, r = _post(url, "/api/export/measurements?dataset=exportstudy")
    assert code == 200 and r["ok"] and r["shown"] == "open", r
    assert shown == [(results / "measurements.xlsx", "open")]
    assert r["path"] == "results/exportstudy/measurements.xlsx"
    assert (results / "measurements.xlsx").stat().st_size > 1000


def test_annotations_are_zipped_and_shown_in_the_file_browser(srv):
    url, results, shown = srv
    code, r = _post(url, "/api/export/sidecars?dataset=exportstudy")
    assert code == 200 and r["shown"] == "reveal"
    zp = results / "exportstudy_sidecars.zip"
    assert shown == [(zp, "reveal")]
    assert zipfile.ZipFile(zp).namelist() == ["exportstudy/sidecars/Salvelinus_fontinalis_TXD_20.json"]


def test_landmarks_for_r_show_the_tps_file(srv):
    url, results, shown = srv
    code, r = _post(url, "/api/export/tps?dataset=exportstudy")
    assert code == 200 and shown == [(results / "tps" / "landmarks.tps", "reveal")], r
    assert (results / "exportstudy_tps.zip").is_file()


def test_with_nothing_to_open_it_the_page_is_told_to_download(srv, monkeypatch):
    url, results, shown = srv
    def no_desktop(path, how):
        raise FileNotFoundError("open")
    monkeypatch.setattr(ls, "show_on_this_computer", no_desktop)
    code, r = _post(url, "/api/export/sidecars?dataset=exportstudy")
    assert code == 200 and r["ok"] and r["shown"] is None and "could not open" in r["error"]


def test_the_download_still_works(srv):
    url, results, shown = srv
    with urllib.request.urlopen(url + "/api/export/sidecars?dataset=exportstudy") as r:
        assert r.status == 200 and "exportstudy_sidecars.zip" in r.headers["Content-Disposition"]
        assert zipfile.ZipFile(io.BytesIO(r.read())).namelist()
    assert shown == []                       # downloading opens nothing


def test_an_unknown_export_opens_nothing(srv):
    url, results, shown = srv
    code, r = _post(url, "/api/export/everything?dataset=exportstudy")
    assert code == 404 and not r["ok"] and shown == []


def test_a_failed_export_opens_nothing(srv):
    url, results, shown = srv
    for f in ls.Handler.out_dir.glob("*.json"):
        f.unlink()
    code, r = _post(url, "/api/export/sidecars?dataset=exportstudy")
    assert code == 500 and "nothing labelled" in r["error"] and shown == []
