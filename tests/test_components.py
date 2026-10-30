"""The labeler is the download; Auto-label and batch identification come apart.

Without a part, its button says so -- and whether it can be downloaded yet -- and its
routes refuse rather than fail. A part not released is never downloaded. The
reference set batch identification compares with installs from an archive that is
checked before anything is written, and never over a folder already there.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load("label_server")


@pytest.fixture
def srv(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "trout" / "lateral").mkdir(parents=True)
    (data / "trout" / "sidecars").mkdir()
    models = tmp_path / "models"
    monkeypatch.setattr(ls.paths, "MODELS", models)
    monkeypatch.setattr(ls.paths, "TRAINING_PYTHON", models / ".venv-train" / "bin" / "python")
    monkeypatch.setattr(ls, "current_model", lambda view="lateral": None)
    monkeypatch.setattr(ls.components, "released", lambda name, *a: False)
    ls.Handler.data_root = data
    ls.Handler.datasets = {"trout": data / "trout"}
    ls.Handler.default_dataset = "trout"
    ls.Handler.images_dir = data / "trout"
    ls.Handler.out_dir = data / "trout" / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", data
    server.shutdown()


def _call(url, path, method="GET"):
    req = urllib.request.Request(url + path, method=method, data=b"{}" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_labeler_alone_has_neither_part(srv):
    url, _ = srv
    code, c = _call(url, "/api/components")
    assert code == 200 and not c["autolabel"]["installed"] and not c["batch"]["installed"]
    assert not c["autolabel"]["released"] and c["batch"]["needs_autolabel"] and c["batch"]["reference_sets"] == []


def test_batch_routes_say_it_is_not_installed(srv):
    url, _ = srv
    code, r = _call(url, "/api/batch/photos?dataset=trout")
    assert code == 409 and r["not_installed"] == "batch"
    code, r = _call(url, "/api/batch/correct?dataset=trout", "POST")
    assert code == 409


def test_an_unreleased_part_is_not_downloaded(srv):
    url, _ = srv
    for route, name in (("/api/autolabel/install", "autolabel"), ("/api/install/batch", "batch")):
        code, r = _call(url, route, "POST")
        assert code == 403 and r["not_released"] == name


def test_a_reference_set_without_photographs_is_found(srv, monkeypatch):
    url, data = srv
    (data / "shiners" / "sidecars").mkdir(parents=True)
    (data / "shiners" / "schema.json").write_text(json.dumps({"reference_set": True}))
    assert ls.reference_sets() == ["shiners"]
    assert not ls.batch_installed()                                    # still no Auto-label
    monkeypatch.setattr(ls, "autolabel_installed", lambda: True)
    assert ls.batch_installed() and ls.components_state()["batch"]["installed"]


# ------------------------------------------------------------------ the installer

ic = _load("install_component")
pc = _load("package_component")


def _pack(tmp, folder="shiners", members=None, schema=None):
    z = tmp / "pack.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(f"{folder}/schema.json", json.dumps(schema if schema is not None else {"reference_set": True}))
        zf.writestr(f"{folder}/sidecars/A_b_1.json", json.dumps({"fish_id": "A_b_1", "lateral": {"keypoints": {"x": [1, 2]}}}))
        for name, body in (members or {}).items():
            zf.writestr(name, body)
    return z


def _manifest(tmp, z, released=True, folder="shiners", sha=None):
    m = tmp / "components.json"
    m.write_text(json.dumps({"batch": {"title": "Batch identification", "released": released,
                                       "file": z.name, "url": str(z), "bytes": z.stat().st_size,
                                       "sha256": sha or hashlib.sha256(z.read_bytes()).hexdigest(),
                                       "folder": folder}}))
    return m


def test_it_installs_a_checked_reference_set(tmp_path):
    z = _pack(tmp_path)
    data = tmp_path / "data"
    assert ic.main(["batch", "--data", str(data), "--manifest", str(_manifest(tmp_path, z))]) == 0
    assert (data / "shiners" / "sidecars" / "A_b_1.json").is_file() and (data / "shiners" / "lateral").is_dir()
    # never over a folder already there
    (data / "shiners" / "mine.txt").write_text("kept")
    assert ic.main(["batch", "--data", str(data), "--manifest", str(_manifest(tmp_path, z))]) == 0
    assert (data / "shiners" / "mine.txt").read_text() == "kept"


def test_it_refuses_what_it_should(tmp_path):
    data = tmp_path / "data"
    z = _pack(tmp_path)
    assert ic.main(["batch", "--data", str(data), "--manifest", str(_manifest(tmp_path, z, released=False))]) == 2
    assert ic.main(["batch", "--data", str(data), "--manifest", str(_manifest(tmp_path, z, sha="0" * 64))]) == 1
    assert not (data / "shiners").exists()
    bad = _pack(tmp_path, members={"elsewhere.txt": "x"})
    with pytest.raises(SystemExit, match="outside"):
        ic.main(["batch", "--data", str(data), "--manifest", str(_manifest(tmp_path, bad))])
    plain = _pack(tmp_path, schema={})
    with pytest.raises(SystemExit, match="not a reference set"):
        ic.main(["batch", "--data", str(data), "--manifest", str(_manifest(tmp_path, plain))])
    assert not (data / "shiners").exists() and not list(data.glob("tmp*"))


def test_a_packaged_study_installs_without_its_photographs(tmp_path):
    study = tmp_path / "shiners"
    (study / "sidecars").mkdir(parents=True)
    (study / "lateral").mkdir()
    (study / "lateral" / "A_b_1.jpg").write_bytes(b"photo")
    (study / "schema.json").write_text(json.dumps({"scheme": "bgnn_2d", "label_first": ["A_b_1"]}))
    (study / "sidecars" / "A_b_1.json").write_text(json.dumps({"fish_id": "A_b_1", "lateral": {"keypoints": {"x": [1, 2]}}}))
    (study / "sidecars" / "A_b_2.json").write_text(json.dumps({"fish_id": "A_b_2", "metadata": {"data_note": "bad"}}))
    (study / "photo_manifest.csv").write_text("fish_id,license\nA_b_1,CC0\n")
    z = pc.package(study, "shiners", tmp_path / "out" / "shiners-reference.zip")
    names = zipfile.ZipFile(z).namelist()
    assert "shiners/sidecars/A_b_1.json" in names and "shiners/sidecars/A_b_2.json" not in names
    assert "shiners/photo_manifest.csv" in names and not any(n.endswith(".jpg") for n in names)
    schema = json.loads(zipfile.ZipFile(z).read("shiners/schema.json"))
    assert schema["reference_set"] is True and "label_first" not in schema
    data = tmp_path / "data"
    assert ic.main(["batch", "--data", str(data), "--manifest", str(_manifest(tmp_path, z))]) == 0
    assert json.loads((data / "shiners" / "schema.json").read_text())["reference_set"] is True
