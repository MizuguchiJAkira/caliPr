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


# --------------------------------------------------------------------------
# creating a study, and finding it without a restart
# --------------------------------------------------------------------------

def _post_json(url, path, payload):
    req = urllib.request.Request(f"{url}{path}",
                                 data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _datasets(url):
    with urllib.request.urlopen(f"{url}/api/datasets") as r:
        return [d["name"] for d in json.loads(r.read())["datasets"]]


@pytest.fixture
def rooted(tmp_path):
    """A server that knows its data root, so datasets can be re-discovered."""
    root = tmp_path / "data"
    (root / "study" / "lateral").mkdir(parents=True)
    ls.Handler.data_root = root
    ls.Handler.datasets = ls.discover_datasets(root)
    ls.Handler.default_dataset = "study"
    ls.Handler.images_dir = root / "study"
    ls.Handler.out_dir = root / "study" / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    srv = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", root
    srv.shutdown()
    ls.Handler.data_root = None


def test_a_new_study_is_created_and_seen_without_restart(rooted):
    url, root = rooted
    assert _datasets(url) == ["study"]

    code, body = _post_json(url, "/api/dataset/new", {"name": "lake_survey"})

    assert code == 200 and body["name"] == "lake_survey"
    assert (root / "lake_survey" / "lateral").is_dir()
    assert "lake_survey" in _datasets(url), "must appear without a restart"


def test_a_folder_created_on_disk_is_seen_without_restart(rooted):
    """The gap this closes: the dataset list used to be frozen at startup."""
    url, root = rooted
    (root / "made_by_hand" / "lateral").mkdir(parents=True)

    assert "made_by_hand" in _datasets(url)


def test_duplicate_study_name_is_refused(rooted):
    url, root = rooted
    _post_json(url, "/api/dataset/new", {"name": "dup"})

    code, body = _post_json(url, "/api/dataset/new", {"name": "dup"})

    assert code == 409 and "already exists" in body["error"]


def test_a_study_name_cannot_be_a_path(rooted):
    url, root = rooted
    code, body = _post_json(url, "/api/dataset/new", {"name": "../../escaped"})

    assert code == 200
    assert body["name"] == "escaped"
    assert (root / "escaped").is_dir()
    assert not (root.parent.parent / "escaped").exists()


def test_empty_study_name_is_refused(rooted):
    url, _ = rooted
    code, _ = _post_json(url, "/api/dataset/new", {"name": "  ../  "})

    assert code == 400


def test_demo_mode_refuses_study_creation(rooted):
    url, root = rooted
    ls.Handler.demo_mode = True
    try:
        code, _ = _post_json(url, "/api/dataset/new", {"name": "nope"})
    finally:
        ls.Handler.demo_mode = False

    assert code == 403
    assert not (root / "nope").exists()


def test_a_new_study_can_take_the_settings_of_an_existing_one(rooted):
    """More fish from the same rig want the same landmarks, strain rule and anatomy check."""
    url, root = rooted
    (root / "study" / "schema.json").write_text('{"group_from_filename": "_([A-Z]{2,4})_\\\\d+$"}')
    (root / "study" / "plausibility.json").write_text('{"landmarks": {}}')
    with urllib.request.urlopen(f"{url}/api/datasets") as r:
        listed = {d["name"]: d for d in json.loads(r.read())["datasets"]}
    assert listed["study"]["settings"] == ["schema.json", "plausibility.json"]

    code, body = _post_json(url, "/api/dataset/new", {"name": "demo", "settings_from": "study"})
    assert code == 200 and body["settings_copied"] == ["schema.json", "plausibility.json"]
    for f in ("schema.json", "plausibility.json"):
        assert (root / "demo" / f).read_text() == (root / "study" / f).read_text()

    code, body = _post_json(url, "/api/dataset/new", {"name": "blank"})
    assert code == 200 and body["settings_copied"] == []
    assert not (root / "blank" / "schema.json").exists()


def test_settings_from_a_study_that_does_not_exist_creates_nothing(rooted):
    url, root = rooted
    code, body = _post_json(url, "/api/dataset/new", {"name": "demo", "settings_from": "nope"})
    assert code == 400 and "nope" in body["error"]
    assert not (root / "demo").exists()


def test_removing_a_study_moves_it_to_the_trash_intact(rooted):
    """Nothing is deleted: a study holds hand labels, and a wrong click must be undoable."""
    url, root = rooted
    _post_json(url, "/api/dataset/new", {"name": "demo"})
    (root / "demo" / "sidecars").mkdir()
    (root / "demo" / "sidecars" / "fish.json").write_text('{"fish_id": "fish"}')
    (root / "demo" / "lateral" / "fish_L.JPEG").write_bytes(b"\xff\xd8\xffphoto")

    code, body = _post_json(url, "/api/dataset/remove", {"name": "demo"})

    assert code == 200 and body["ok"]
    assert "demo" not in _datasets(url) and not (root / "demo").exists()
    [moved] = list((root / ".trash").iterdir())
    assert moved.name.startswith("demo-")
    assert (moved / "sidecars" / "fish.json").read_text() == '{"fish_id": "fish"}'
    assert (moved / "lateral" / "fish_L.JPEG").read_bytes() == b"\xff\xd8\xffphoto"
    assert ".trash" not in _datasets(url)


def test_removing_the_default_study_picks_another(rooted):
    url, root = rooted
    _post_json(url, "/api/dataset/new", {"name": "other"})
    code, body = _post_json(url, "/api/dataset/remove", {"name": "study"})
    assert code == 200 and body["default"] == "other"


def test_only_a_listed_study_can_be_removed(rooted):
    url, root = rooted
    (root / "not_a_study").mkdir()
    for name in ("../..", "not_a_study", "", "nope"):
        code, body = _post_json(url, "/api/dataset/remove", {"name": name})
        assert code == 404, name
    assert (root / "study").is_dir() and (root / "not_a_study").is_dir()
    assert not (root / ".trash").exists()


def test_the_list_says_how_many_fish_each_study_has_labelled(rooted):
    url, root = rooted
    (root / "study" / "sidecars").mkdir()
    for i in range(3):
        (root / "study" / "sidecars" / f"f{i}.json").write_text("{}")
    with urllib.request.urlopen(f"{url}/api/datasets") as r:
        [d] = json.loads(r.read())["datasets"]
    assert d["labelled"] == 3


def test_demo_mode_refuses_removing_a_study(rooted):
    url, root = rooted
    ls.Handler.demo_mode = True
    try:
        code, _ = _post_json(url, "/api/dataset/remove", {"name": "study"})
    finally:
        ls.Handler.demo_mode = False
    assert code == 403 and (root / "study").is_dir()


# --------------------------------------------------------------------------
# a study's own landmarks, and what it calls them
# --------------------------------------------------------------------------

def _schema(url):
    with urllib.request.urlopen(f"{url}/api/schema?dataset=study") as r:
        return json.loads(r.read())


def test_a_study_can_add_its_own_landmark(rooted):
    url, root = rooted
    code, body = _post_json(url, "/api/schema/keypoint?dataset=study",
                            {"action": "add", "label": "Adipose fin base"})
    assert code == 200 and body["ok"]
    [added] = [k for k in body["schema"]["lateral"]["keypoints"] if k.get("custom")]
    assert added["name"] == "adipose_fin_base" and added["label"] == "Adipose fin base"
    assert json.loads((root / "study" / "schema.json").read_text())["extra_keypoints"][0]["name"] \
        == "adipose_fin_base"
    # it is offered on every specimen in the study, and nowhere else
    assert added["name"] in [k["name"] for k in _schema(url)["lateral"]["keypoints"]]
    assert "adipose_fin_base" not in [k.name for k in ls.KEYPOINTS]


def test_a_landmark_is_renamed_for_the_study_without_changing_its_stored_name(rooted):
    url, root = rooted
    code, body = _post_json(url, "/api/schema/keypoint?dataset=study",
                            {"action": "rename", "name": "premaxilla_tip", "label": "snout tip"})
    assert code == 200
    kp = {k["name"]: k for k in body["schema"]["lateral"]["keypoints"]}
    assert kp["premaxilla_tip"]["label"] == "snout tip"      # shown
    assert "premaxilla_tip" in kp                             # stored, and still that
    prof = json.loads((root / "study" / "schema.json").read_text())
    assert prof["labels"] == {"premaxilla_tip": "snout tip"}
    # renaming back to its own name drops the override rather than storing a no-op
    _post_json(url, "/api/schema/keypoint?dataset=study",
               {"action": "rename", "name": "premaxilla_tip", "label": "premaxilla_tip"})
    assert "labels" not in json.loads((root / "study" / "schema.json").read_text())


def test_only_a_studys_own_landmark_can_be_removed_and_placed_ones_ask_first(rooted):
    url, root = rooted
    _post_json(url, "/api/schema/keypoint?dataset=study", {"action": "add", "label": "notch"})
    code, body = _post_json(url, "/api/schema/keypoint?dataset=study",
                            {"action": "remove", "name": "premaxilla_tip"})
    assert code == 400 and "added for this study" in body["error"]

    (root / "study" / "sidecars").mkdir(exist_ok=True)
    (root / "study" / "sidecars" / "f1.json").write_text(
        json.dumps({"fish_id": "f1", "lateral": {"keypoints": {"notch": [1, 2]}}}))
    code, body = _post_json(url, "/api/schema/keypoint?dataset=study",
                            {"action": "remove", "name": "notch"})
    assert code == 409 and body["placed"] == 1
    assert "notch" in [k["name"] for k in _schema(url)["lateral"]["keypoints"]]

    code, body = _post_json(url, "/api/schema/keypoint?dataset=study",
                            {"action": "remove", "name": "notch", "force": True})
    assert code == 200
    assert "notch" not in [k["name"] for k in _schema(url)["lateral"]["keypoints"]]
    # the coordinates already saved are left alone, not edited out of the sidecar
    assert json.loads((root / "study" / "sidecars" / "f1.json").read_text())["lateral"]["keypoints"]


def test_a_nameless_landmark_is_refused(rooted):
    url, _ = rooted
    for payload in ({"action": "add", "label": "  "}, {"action": "rename", "name": "premaxilla_tip",
                                                       "label": ""}):
        assert _post_json(url, "/api/schema/keypoint?dataset=study", payload)[0] == 400
    assert _post_json(url, "/api/schema/keypoint?dataset=study",
                      {"action": "rename", "name": "nope", "label": "x"})[0] == 404


def test_a_studys_own_landmark_is_a_landmark_not_a_ruler_point(rooted):
    """It was listed among the ruler points too, and the ruler copy came first: clicking
    it started a ruler task, so the point could not be placed at all."""
    url, _ = rooted
    _post_json(url, "/api/schema/keypoint?dataset=study", {"action": "add", "label": "adipose base"})
    sch = _schema(url)["lateral"]
    assert "adipose_base" in [k["name"] for k in sch["keypoints"]]
    assert "adipose_base" not in [k["name"] for k in sch["ruler"]]
    assert [k["name"] for k in sch["ruler"]] == ["ruler_point_a", "ruler_point_b"]


# --------------------------------------------------------------------------
# switching a study to another protocol's landmark scheme
# --------------------------------------------------------------------------

def test_a_study_can_collect_another_protocols_landmarks(rooted):
    url, root = rooted
    code, body = _post_json(url, "/api/schema/scheme?dataset=study", {"scheme": "bgnn_2d"})
    assert code == 200 and body["scheme"] == "bgnn_2d"
    sch = body["schema"]
    kp = sch["lateral"]["keypoints"]
    assert len(kp) == 23 and kp[0]["name"] == "dentary_anterior"
    assert kp[0]["label"].startswith("1. ")                 # numbered as the protocol does
    assert sch["traits"] is False and sch["fin_groups"] == []
    assert sch["lateral"]["polygons"] == []                 # that scheme traces no outlines
    assert [r["name"] for r in sch["lateral"]["ruler"]] == ["ruler_point_a", "ruler_point_b"]
    assert json.loads((root / "study" / "schema.json").read_text())["scheme"] == "bgnn_2d"

    # and back again, which is what makes it a switch rather than a rewrite
    code, body = _post_json(url, "/api/schema/scheme?dataset=study", {"scheme": "calipr"})
    assert code == 200 and body["schema"]["traits"] is True
    assert "premaxilla_tip" in [k["name"] for k in body["schema"]["lateral"]["keypoints"]]
    assert "scheme" not in json.loads((root / "study" / "schema.json").read_text())


def test_an_unknown_scheme_is_refused(rooted):
    url, root = rooted
    assert _post_json(url, "/api/schema/scheme?dataset=study", {"scheme": "nope"})[0] == 404
    assert not (root / "study" / "schema.json").exists()      # and nothing written


def test_labels_already_saved_are_left_alone_when_the_scheme_changes(rooted):
    """The two schemes name different points, so nothing can be converted; what a
    person clicked stays exactly as clicked."""
    url, root = rooted
    (root / "study" / "sidecars").mkdir(exist_ok=True)
    before = json.dumps({"fish_id": "f1", "lateral": {"keypoints": {"premaxilla_tip": [10, 20]}}})
    (root / "study" / "sidecars" / "f1.json").write_text(before)
    _post_json(url, "/api/schema/scheme?dataset=study", {"scheme": "bgnn_2d"})
    assert (root / "study" / "sidecars" / "f1.json").read_text() == before
