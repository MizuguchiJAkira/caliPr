"""The labeler never shows, keeps, or saves coordinates that do not fit their image.

Three ways stale coordinates reach disk, and each is refused here: labels whose
photograph was re-cropped after labelling (HRN_4, 450 px into the ruler), labels
placed on an image that changed while the fish was open, and labels loaded from a
version of the file that has since been changed on disk -- by a repair, say.
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

Image = pytest.importorskip("PIL.Image")
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
FID = "Salvelinus_fontinalis_HRN_4"


def _load():
    spec = importlib.util.spec_from_file_location("label_server", SCRIPTS / "label_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["label_server"] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _load()


def _image(path: Path, w: int, h: int = 400, shade: int = 90):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (shade, shade, shade)).save(path, "JPEG")


LAT = {"keypoints": {"premaxilla_tip": [120, 200]},
       "polygons": {"body_plus_caudal": [[100, 150], [800, 150], [800, 250], [100, 250]]}}


@pytest.fixture
def srv(tmp_path):
    study = tmp_path / "data" / "study"
    _image(study / "lateral" / f"{FID}_L.JPEG", 1000)
    (study / "sidecars").mkdir(parents=True)
    ls.Handler.datasets = {"study": study}
    ls.Handler.default_dataset = "study"
    ls.Handler.images_dir = study
    ls.Handler.out_dir = study / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    ls._FP_CACHE.clear()
    ls._SIZE_CACHE.clear()
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", study
    server.shutdown()


def _get(url, path):
    with urllib.request.urlopen(f"{url}{path}?dataset=study") as r:
        return json.loads(r.read())


def _save(url, doc):
    req = urllib.request.Request(f"{url}/api/save?dataset=study", method="POST",
                                 data=json.dumps(doc).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _write_sidecar(study, doc):
    (study / "sidecars" / f"{FID}.json").write_text(json.dumps(doc))


def _fp(study):
    from fish_morpho import image_identity
    return image_identity.fingerprint(study / "lateral" / f"{FID}_L.JPEG")


def test_labels_on_their_own_image_load_as_aligned(srv):
    url, study = srv
    _write_sidecar(study, {"fish_id": FID, "metadata": {"images": {"lateral": _fp(study)}},
                           "lateral": LAT})
    doc = _get(url, f"/api/sidecar/{FID}")
    assert doc["_alignment"]["lateral"]["status"] == "ok"
    assert doc["_mtime"] > 0


def test_labels_from_a_recropped_image_load_as_displaced(srv):
    url, study = srv
    old = dict(_fp(study), width=550)                  # labelled when the crop was 450 px narrower
    _write_sidecar(study, {"fish_id": FID, "metadata": {"images": {"lateral": old}},
                           "lateral": LAT})
    a = _get(url, f"/api/sidecar/{FID}")["_alignment"]["lateral"]
    assert a["status"] == "size_changed"
    assert a["recorded"] == {"width": 550, "height": 400}
    assert a["current"] == {"width": 1000, "height": 400}


def test_nothing_is_saved_over_displaced_labels(srv):
    url, study = srv
    old = dict(_fp(study), width=550)
    _write_sidecar(study, {"fish_id": FID, "metadata": {"images": {"lateral": old}},
                           "lateral": LAT})
    before = (study / "sidecars" / f"{FID}.json").read_text()
    code, body = _save(url, {"fish_id": FID, "metadata": {}, "lateral": LAT})
    assert code == 409 and body["misaligned"]
    assert "realign_labels.py" in body["error"]
    assert (study / "sidecars" / f"{FID}.json").read_text() == before


def test_labels_placed_on_a_different_size_are_refused(srv):
    """The image changed under the open fish: these coordinates fit the old one."""
    url, study = srv
    code, body = _save(url, {"fish_id": FID, "metadata": {"placed_on": {"lateral": [550, 400]}},
                             "lateral": LAT})
    assert code == 409 and body["misaligned"]
    assert not (study / "sidecars" / f"{FID}.json").exists()


def test_a_good_save_records_the_image_it_was_placed_on(srv):
    url, study = srv
    code, body = _save(url, {"fish_id": FID, "_alignment": {"x": 1}, "_mtime": 1,
                             "metadata": {"placed_on": {"lateral": [1000, 400]}},
                             "lateral": LAT})
    assert code == 200 and body["mtime"] > 0
    saved = json.loads((study / "sidecars" / f"{FID}.json").read_text())
    assert saved["metadata"]["images"]["lateral"] == _fp(study)
    for transient in ("_alignment", "_mtime"):
        assert transient not in saved
    assert "placed_on" not in saved["metadata"] and "base_mtime" not in saved["metadata"]


def test_a_save_built_on_an_older_version_of_the_file_is_refused(srv):
    """HRN_4 is repaired on disk while a tab still holds its old coordinates."""
    url, study = srv
    code, first = _save(url, {"fish_id": FID, "metadata": {}, "lateral": LAT})
    assert code == 200
    time.sleep(1.2)
    code, _ = _save(url, {"fish_id": FID, "metadata": {"base_mtime": first["mtime"]},
                          "lateral": LAT})
    assert code == 200                                  # built on the latest: fine
    code, body = _save(url, {"fish_id": FID, "metadata": {"base_mtime": first["mtime"]},
                             "lateral": LAT})
    assert code == 409 and body["conflict"]             # built on the one before: refused


def test_a_new_fish_opened_empty_cannot_overwrite_one_saved_meanwhile(srv):
    url, study = srv
    assert _save(url, {"fish_id": FID, "metadata": {}, "lateral": LAT})[0] == 200
    code, body = _save(url, {"fish_id": FID, "metadata": {"base_mtime": 0}, "lateral": LAT})
    assert code == 409 and body["conflict"]


def test_specimen_list_carries_image_sizes(srv):
    url, study = srv
    [row] = _get(url, "/api/specimens")
    assert row["sizes"]["lateral"] == [1000, 400]


def test_specimen_gaps_report_what_a_saved_fish_still_lacks():
    from pathlib import Path as _P
    schema = ls.build_schema({})
    doc = {"lateral": {"keypoints": {"eye_anterior": [1, 1], "pectoral_insertion_upper": [2, 2]},
                       "polygons": {"pectoral": [[0, 0]] * 20, "dorsal": [[0, 0]] * 5},
                       "calibration": {"mode": "none"}},
           "frontal": {"keypoints": {"mouth_left": [1, 1], "mouth_right": [5, 1]}},
           "metadata": {"assist": {"unreviewed": ["eye_anterior"]}, "data_note": "fin damaged"}}
    g = ls.specimen_gaps(doc, schema, [1000, 400], {"width": 1000, "height": 400})
    assert "premaxilla_tip" in g["landmarks_missing"]
    assert "pectoral_insertion_upper" not in g["landmarks_missing"]         # fin work, reported with fins
    assert not any(n.endswith(("_base_anterior", "_base_posterior")) for n in g["landmarks_missing"])
    assert g["no_scale"] and g["no_outline"] and g["frontal_no_scale"] and g["flagged"]
    assert g["fins_thin"] == ["dorsal"] and set(g["fins_untraced"]) == {"pelvic", "anal"}
    assert "pectoral" in g["fins_no_points"] and g["unreviewed"] == 1 and not g["misaligned"]
    assert ls.specimen_gaps(doc, schema, [1450, 400], {"width": 1000, "height": 400})["misaligned"]
    assert ls.specimen_gaps({"lateral": {}}, schema, None, None)["landmarks_missing"] == []


# --- metadata a script wrote survives the next save -----------------------

def test_saving_keeps_metadata_the_labeler_does_not_manage(tmp_path, monkeypatch):
    """The page builds outgoing metadata from scratch, so anything a script put
    there was erased by the next save of that fish. coordinate_history is the
    record of a shift applied exactly; without it, coordinates moved and nothing
    says they did."""
    import importlib.util
    import json
    import sys
    import threading
    import urllib.request
    from pathlib import Path

    import pytest
    Image = pytest.importorskip("PIL.Image")
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    spec = importlib.util.spec_from_file_location("label_server", scripts / "label_server.py")
    ls = importlib.util.module_from_spec(spec)
    sys.modules["label_server"] = ls
    spec.loader.exec_module(ls)

    study = tmp_path / "study"
    (study / "lateral").mkdir(parents=True)
    (study / "sidecars").mkdir()
    fid = "fish_1"
    Image.new("RGB", (400, 200), (90, 90, 90)).save(study / "lateral" / f"{fid}.JPEG")
    history = [{"at": "2026-09-21T15:18:08", "view": "lateral", "dx": 97,
                "points": 61, "reason": "one photograph per fish"}]
    (study / "sidecars" / f"{fid}.json").write_text(json.dumps({
        "fish_id": fid,
        "metadata": {"strain": "ASN", "coordinate_history": history,
                     "collector": "someone"},
        "lateral": {"keypoints": {"premaxilla_tip": [10, 10]}}}))

    ls.Handler.datasets = {"study": study}
    ls.Handler.default_dataset = "study"
    ls.Handler.images_dir = study
    ls.Handler.out_dir = study / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    monkeypatch.setattr(ls.Handler, "_locked", lambda self: False)
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body = json.dumps({"fish_id": fid,
                           "metadata": {"strain": "ASN", "source": "hand-labeled"},
                           "lateral": {"keypoints": {"premaxilla_tip": [12, 12]}}})
        req = urllib.request.Request(url + "/api/save?dataset=study",
                                     data=body.encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["ok"]
    finally:
        server.shutdown()

    saved = json.loads((study / "sidecars" / f"{fid}.json").read_text())
    assert saved["metadata"]["coordinate_history"] == history
    assert saved["metadata"]["collector"] == "someone"
    assert saved["metadata"]["source"] == "hand-labeled"     # the save still wins
    assert saved["lateral"]["keypoints"]["premaxilla_tip"] == [12, 12]
