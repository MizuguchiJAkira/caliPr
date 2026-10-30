"""Auto-label can be installed from the page, and only from the page.

The installer itself is not run here -- it downloads 2 GB -- so a stand-in takes its
place: what is tested is that the predictor's "not installed" answer offers the
install, that the page can start it and watch it, and that another site cannot.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request
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
    models = tmp_path / "models"
    monkeypatch.setattr(ls.paths, "MODELS", models)
    monkeypatch.setattr(ls.paths, "TRAINING_PYTHON", models / ".venv-train" / "bin" / "python")
    monkeypatch.setattr(ls.Predictor, "_PYTHONS", (models / ".venv-train" / "bin" / "python",))
    ran = []

    class Installer:
        """Writes what the real one would, then exits."""
        def __init__(self, cmd, stdout=None, **kw):
            ran.append(cmd)
            stdout.write("Downloading PyTorch and DeepLabCut\nDone.\n")
            stdout.flush()
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(ls.subprocess, "Popen", Installer)
    monkeypatch.setattr(ls.AutolabelInstall, "_proc", None)
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", ran
    server.shutdown()


def _call(url, path, method="GET", headers=None):
    req = urllib.request.Request(url + path, method=method, headers=headers or {},
                                 data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_a_machine_without_auto_label_is_offered_the_install(srv):
    started = ls.Predictor._start()
    assert started.get("install") is True and "2.3 GB" in started["error"]


def test_the_page_starts_the_install_and_sees_it_finish(srv, monkeypatch):
    url, ran = srv
    monkeypatch.setattr(ls.components, "released", lambda name, *a: True)
    code, st = _call(url, "/api/autolabel/install")
    assert code == 200 and st["installed"] is False and not st["running"]
    code, r = _call(url, "/api/autolabel/install", "POST")
    assert code == 200 and r["ok"] and ran and str(ran[0][-1]).endswith("install_autolabel.py")
    for _ in range(50):
        code, st = _call(url, "/api/autolabel/install")
        if st["finished"]:
            break
        time.sleep(0.05)
    assert st["ok"] and st["line"] == "Done."


def test_another_site_cannot_start_it(srv):
    url, ran = srv
    code, r = _call(url, "/api/autolabel/install", "POST",
                    headers={"Origin": "https://example.com"})
    assert code == 403 and not ran


def test_nothing_is_downloaded_before_it_is_released(srv, monkeypatch):
    url, ran = srv
    monkeypatch.setattr(ls.components, "released", lambda name, *a: False)
    code, r = _call(url, "/api/autolabel/install", "POST")
    assert code == 403 and r["not_released"] == "autolabel" and not ran
