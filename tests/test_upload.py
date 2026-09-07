"""Tests for adding photographs to a dataset through the labeler.

The upload path writes files into a dataset directory from a request body, so the
tests that matter are the refusals: a name that is a path, contents that are not
an image, and a name that already belongs to a different photograph.
"""

from __future__ import annotations

import importlib.util
import json
import re
import struct
import sys
import threading
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("label_server",
                                                  SCRIPTS / "label_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["label_server"] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load()


def _png(w=8, h=8) -> bytes:
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    body = zlib.compress(b"".join(b"\x00" + b"\x10\x20\x30" * w for _ in range(h)))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", body) + chunk(b"IEND", b""))


JPEG = b"\xff\xd8\xff" + b"\x00" * 64


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "data"
    (root / "study" / "lateral").mkdir(parents=True)
    ls.Handler.datasets = {"study": root / "study"}
    ls.Handler.default_dataset = "study"
    ls.Handler.images_dir = root / "study"
    ls.Handler.out_dir = root / "study" / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    srv = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", root / "study" / "lateral"
    srv.shutdown()


def _post(url, name, body, view="lateral"):
    req = urllib.request.Request(f"{url}/api/upload?dataset=study", data=body,
                                 method="POST",
                                 headers={"X-Filename": name, "X-View": view})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# --------------------------------------------------------------------------
# accepted
# --------------------------------------------------------------------------

def test_a_real_image_is_added(server):
    url, lateral = server
    code, body = _post(url, "fish01.png", _png())

    assert code == 200 and body["status"] == "added"
    assert (lateral / "fish01.png").read_bytes() == _png()


def test_jpeg_is_accepted(server):
    url, lateral = server
    code, body = _post(url, "fish.jpg", JPEG)

    assert code == 200 and body["status"] == "added"


def test_reuploading_the_same_bytes_is_a_duplicate_not_an_error(server):
    """Re-dropping a folder already added is ordinary; it must not look like a
    failure and must not rewrite the file."""
    url, lateral = server
    _post(url, "fish01.png", _png())
    before = (lateral / "fish01.png").stat().st_mtime_ns

    code, body = _post(url, "fish01.png", _png())

    assert code == 200 and body["status"] == "duplicate"
    assert (lateral / "fish01.png").stat().st_mtime_ns == before


def test_frontal_view_goes_to_the_frontal_folder(server):
    url, lateral = server
    code, _ = _post(url, "f.png", _png(), view="frontal")

    assert code == 200
    assert (lateral.parent / "frontal" / "f.png").is_file()


# --------------------------------------------------------------------------
# refused
# --------------------------------------------------------------------------

def test_a_filename_that_is_a_path_cannot_escape_the_dataset(server):
    url, lateral = server
    code, body = _post(url, "../../../../tmp/evil.png", _png())

    # The traversal is stripped to a basename rather than rejected, so the file
    # lands inside the dataset. What must never happen is a write outside it.
    assert code == 200
    assert body["name"] == "evil.png"
    assert (lateral / "evil.png").is_file()
    assert not (lateral.parent.parent.parent / "evil.png").exists()


def test_odd_characters_in_a_name_are_neutralised(server):
    url, lateral = server
    code, body = _post(url, "we;ird name$(x).png", _png())

    assert code == 200
    assert re.fullmatch(r"[A-Za-z0-9._-]+", body["name"]), body["name"]
    assert (lateral / body["name"]).is_file()


def test_contents_that_are_not_an_image_are_refused(server):
    """An extension is a claim, not evidence. A .png that is not a PNG becomes a
    specimen that silently fails to load much later."""
    url, lateral = server
    code, body = _post(url, "notreally.png", b"this is plain text, not an image")

    assert code == 415
    assert "not a JPEG" in body["error"]
    assert not any(lateral.iterdir())


def test_a_non_image_extension_is_refused(server):
    url, lateral = server
    code, body = _post(url, "notes.txt", _png())

    assert code == 415
    assert not any(lateral.iterdir())


def test_a_different_file_of_the_same_name_is_refused(server):
    """Overwriting would replace a photograph that existing sidecars point at."""
    url, lateral = server
    _post(url, "fish01.png", _png(8, 8))

    code, body = _post(url, "fish01.png", _png(16, 16))

    assert code == 409
    assert "DIFFERENT" in body["error"]
    assert (lateral / "fish01.png").read_bytes() == _png(8, 8)


def test_empty_body_is_refused(server):
    url, lateral = server
    code, _ = _post(url, "empty.png", b"")

    assert code == 400
    assert not any(lateral.iterdir())


def test_demo_mode_refuses_uploads(server):
    """Demo mode must not be able to write anything, images included."""
    url, lateral = server
    ls.Handler.demo_mode = True
    try:
        code, body = _post(url, "fish01.png", _png())
    finally:
        ls.Handler.demo_mode = False

    assert code == 403
    assert body["error"].startswith("demo mode")
    assert not any(lateral.iterdir())
