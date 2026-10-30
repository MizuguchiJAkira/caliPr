"""The batch page's server side, with the model stood in for.

A study of hand-labelled reference fish and one photograph of three: the fish are
found and landmarked once and kept, scored against the reference on every
request, compared at their own length once the photograph has a ruler, and each
is landmarked on a rectangle with the fish beside it painted out. A fish whose
points are put right in the labeller is scored on those, wherever it is found.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_species_match import BASE  # noqa: E402

GROUPS = {"Notropis_alpha": dict(deep=1.0, snout=1.0), "Notropis_beta": dict(deep=1.35, snout=1.3)}


def shape(rng, deep=1.0, snout=1.0, at=(0, 0), scale=4.0):
    out = {}
    for n, (x, y) in BASE.items():
        if n.startswith("caudal_crease"):
            continue
        x = x * snout if x < 14 else x
        out[n] = [at[0] + (x + rng.normal(0, 0.5)) * scale, at[1] + (y * deep + rng.normal(0, 0.5)) * scale]
    return out


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def srv(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    ls = _load("label_server", SCRIPTS / "label_server.py")
    st = tmp_path / "data" / "shiners"
    for d in ("lateral", "sidecars", "batch"):
        (st / d).mkdir(parents=True)
    (st / "schema.json").write_text(json.dumps({"scheme": "bgnn_2d",
                                                "group_from_filename": "^([A-Z][a-z]+_[a-z]+)_\\d+$"}))
    for sp, kw in GROUPS.items():
        for i in range(8):
            fid = f"{sp}_{i}"
            Image.new("RGB", (8, 8)).save(st / "lateral" / f"{fid}.jpg")
            (st / "sidecars" / f"{fid}.json").write_text(json.dumps({
                "fish_id": fid, "lateral": {"keypoints": shape(rng, **kw),
                                            "calibration": {"mode": "manual", "point_a": [0, 0],
                                                            "point_b": [80, 0], "known_mm": 10}}}))
    # a photograph of three fish on a light background: two alpha, one beta
    photo = np.full((300, 900, 3), 240, np.uint8)
    boxes = [[10, 20, 450, 140], [460, 20, 890, 140], [10, 160, 450, 290]]
    for b in boxes:
        photo[b[1] + 30:b[3] - 30, b[0] + 30:b[2] - 30] = (90, 120, 150)
    Image.fromarray(photo).save(st / "batch" / "tray.jpg")
    truth = ["Notropis_alpha", "Notropis_beta", "Notropis_alpha"]
    calls = {"detect": 0, "predict": [], "boxes": boxes}

    def detect(image):
        calls["detect"] += 1
        return {"ok": True, "width": 900, "height": 300,
                "fish": [{"box": b, "outline": [[b[0] + 30, b[1] + 30], [b[2] - 30, b[1] + 30],
                                                [b[2] - 30, b[3] - 30], [b[0] + 30, b[3] - 30]],
                          "length_px": b[2] - b[0]} for b in calls["boxes"]]}

    def predict(image, **kw):
        i = int(Path(image).stem.split("_")[1]) - 1
        calls["predict"].append((Path(image).name, kw.get("landmarks")))
        kps = shape(rng, at=(20, 95), **GROUPS[truth[i]])
        return {"ok": True, "keypoints": kps, "confidence": {n: 0.9 for n in kps},
                "low_confidence": ["anal_fin_base_anterior"]}

    monkeypatch.setattr(ls, "batch_installed", lambda: True)        # the model is a stand-in here
    monkeypatch.setattr(ls.Predictor, "detect", classmethod(lambda cls, image: detect(image)))
    monkeypatch.setattr(ls.Predictor, "predict", classmethod(lambda cls, image, **kw: predict(image, **kw)))
    monkeypatch.setattr(ls, "current_model_label", lambda view="lateral": "test-model")
    from fish_morpho import taxonomy
    monkeypatch.setattr(taxonomy, "Names", _OfflineNames)
    ls.Handler.data_root = tmp_path / "data"
    ls.Handler.datasets = {"shiners": st}
    ls.Handler.default_dataset = "shiners"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", st, calls, truth
    server.shutdown()


class _OfflineNames:
    """Names as they are given, but Notropis beta now called Alburnops beta."""
    def __init__(self, cache_path, get=None):
        pass

    def resolve(self, name):
        given = name.replace("_", " ")
        current = "Alburnops beta" if given == "Notropis beta" else given
        chain = [{"name": given, "via": "label"}] + ([{"name": current, "via": "FishBase"}] if current != given else [])
        return {"given": given, "current": current, "common": None, "gbif_key": None, "chain": chain,
                "ambiguous": False, "found": True, "offline": False}


def _get(url, path):
    try:
        with urllib.request.urlopen(url + path, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read())


def test_each_fish_is_found_landmarked_and_matched(srv):
    url, st, calls, truth = srv
    code, d = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    assert code == 200 and d["ok"] and len(d["fish"]) == 3
    for f, sp in zip(d["fish"], truth):
        assert f["match"]["ok"] and f["match"]["matches"][0]["species"] == sp
        x0, y0 = f["box"][:2]
        assert all(p[0] >= x0 and p[1] >= y0 for p in f["keypoints"].values())   # photograph pixels
    assert d["reference"] == {"study": "shiners", "fish": 16,
                              "species": {"Notropis_alpha": 8, "Notropis_beta": 8}}
    assert all(lms and "dentary_anterior" in lms for _, lms in calls["predict"])  # the study's names
    code, ph = _get(url, "/api/batch/photos?dataset=shiners")
    assert ph["photos"] == [{"name": "tray.jpg", "fish": 3}]


def test_found_once_and_scored_every_time(srv):
    url, st, calls, _ = srv
    _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    assert calls["detect"] == 1 and len(calls["predict"]) == 3
    _get(url, "/api/batch/fish/tray.jpg?dataset=shiners&refresh=1")
    assert calls["detect"] == 2


def test_asking_for_the_saved_landmarks_never_runs_the_model(srv):
    """The page asks this way first, so it says it is landmarking only when it is."""
    url, st, calls, _ = srv
    code, d = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners&cache=only")
    assert code == 200 and d == {"ok": False, "cached": False} and calls["detect"] == 0
    assert _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")[1]["cached"] is False
    code, d = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners&cache=only")
    assert d["ok"] and d["cached"] is True and len(d["fish"]) == 3 and calls["detect"] == 1


def test_a_ruler_brings_size_in(srv):
    url, st, calls, _ = srv
    code, d = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    assert all(f["sl_mm"] is None and not f["match"]["size_matched"] for f in d["fish"])
    code, d = _post(url, "/api/batch/scale?dataset=shiners",
                    {"photo": "tray.jpg", "point_a": [0, 0], "point_b": [80, 0], "known_mm": 10})
    assert d["scale"]["known_mm"] == 10
    assert all(math.isclose(f["sl_mm"], 50, rel_tol=0.1) and f["match"]["size_matched"] for f in d["fish"])
    code, d = _post(url, "/api/batch/scale?dataset=shiners", {"photo": "tray.jpg", "clear": True})
    assert d["scale"] is None


def test_the_fish_beside_it_is_painted_out(srv):
    url, st, calls, _ = srv
    _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    crop = np.asarray(Image.open(st / "batch_cache" / "tray" / "fish_1.jpg"))
    # fish 1's rectangle reaches into fish 2's; there, it is background now
    assert crop.shape[:2] == (120, 440)
    assert abs(int(crop[60, 435, 0]) - 240) < 12
    assert abs(int(crop[60, 200, 0]) - 90) < 12          # fish 1 itself is untouched


def test_an_unknown_photograph(srv):
    url, *_ = srv
    code, d = _get(url, "/api/batch/fish/nothing.jpg?dataset=shiners")
    assert code == 404 and not d["ok"]


def _correct(url, n):
    return _post(url, "/api/batch/correct?dataset=shiners", {"photo": "tray.jpg", "n": n})


def test_correcting_puts_the_photographs_fish_in_a_batch_study(srv):
    url, st, calls, _ = srv
    code, found = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    code, d = _correct(url, 2)
    assert d == {"ok": True, "dataset": "shiners_batch", "fish": "tray_fish_2"}
    bs = st.parent / "shiners_batch"
    assert sorted(p.name for p in (bs / "lateral").iterdir()) == [f"tray_fish_{i}.jpg" for i in (1, 2, 3)]
    schema = json.loads((bs / "schema.json").read_text())
    assert schema["batch_of"] == "shiners" and schema["scheme"] == "bgnn_2d"
    # the model's points wait on the fish, in its own rectangle's coordinates
    auto = json.loads((bs / "sidecars_auto" / "tray_fish_2.json").read_text())
    assert auto["metadata"]["source"] == "predicted" and auto["metadata"]["model"] == "test-model"
    # each point carries its confidence: that is what the labeller queues for review
    assert set(auto["metadata"]["keypoint_confidence"]) == set(auto["lateral"]["keypoints"])
    box = json.loads((bs / "batch_fish.json").read_text())["tray_fish_2"]["box"]
    assert box == [460, 20, 890, 140]
    on_photo = found["fish"][1]["keypoints"]
    assert all(math.isclose(x + 460, on_photo[n][0]) and math.isclose(y + 20, on_photo[n][1])
               for n, (x, y) in auto["lateral"]["keypoints"].items())
    # the labeller knows where to send you back to
    code, sc = _get(url, "/api/schema?dataset=shiners_batch")
    assert sc["batch_of"] == "shiners"
    code, specs = _get(url, "/api/specimens?dataset=shiners_batch")
    assert all(s["predicted"] for s in specs)
    # asked again, nothing already there is replaced
    before = (bs / "lateral" / "tray_fish_1.jpg").stat().st_mtime_ns
    _correct(url, 1)
    assert (bs / "lateral" / "tray_fish_1.jpg").stat().st_mtime_ns == before


def test_a_fish_is_scored_on_its_corrected_points(srv):
    url, st, calls, truth = srv
    _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    _correct(url, 2)
    # fish 2 is a beta; its corrected points say alpha
    rng = np.random.default_rng(3)
    (st.parent / "shiners_batch" / "sidecars" / "tray_fish_2.json").write_text(json.dumps({
        "fish_id": "tray_fish_2", "lateral": {"keypoints": shape(rng, at=(10, 70), **GROUPS["Notropis_alpha"])}}))
    code, d = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    f1, f2, f3 = d["fish"]
    assert f2["checked"] and f2["fid"] == "tray_fish_2" and f2["low_confidence"] == []
    assert f2["match"]["matches"][0]["species"] == "Notropis_alpha"
    assert min(p[0] for p in f2["keypoints"].values()) >= 460            # back in photograph pixels
    assert not f1["checked"] and not f3["checked"] and f1["fid"] == "tray_fish_1"
    # found again in another order, the corrected points stay with the fish they belong to
    calls["boxes"] = calls["boxes"][::-1]
    code, d = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners&refresh=1")
    assert [f["checked"] for f in d["fish"]] == [False, True, False]
    assert d["fish"][1]["box"] == [460, 20, 890, 140]


def test_correcting_needs_the_fish_found_first(srv):
    url, *_ = srv
    try:
        _correct(url, 1)
        assert False, "should have refused"
    except urllib.error.HTTPError as e:
        assert e.code == 404


class _FakeDrainageTool:
    """drainage_candidates, answering from a table instead of the web."""
    GBIF = "https://api.gbif.org/v1"
    NAS = "https://nas.er.usgs.gov/api/v2/occurrence/search"
    SHINER_GENERA = ("Notropis",)
    KEYS = {"Notropis alpha": 101, "Notropis beta": 102}

    class Web:
        def __init__(self, path, pause=0):
            pass

        def get(self, url):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if url.startswith(_FakeDrainageTool.NAS):    # USGS: what is not native there
                return {"results": [{"group": "Fishes", "scientificName": "Notropis beta", "status": "established"},
                                    {"group": "Fishes", "scientificName": "Notropis beta", "status": "stocked"},
                                    {"group": "Fishes", "scientificName": "Tinca tinca", "status": "", "commonName": "Tench"},
                                    {"group": "Plants", "scientificName": "Trapa natans", "status": "established"}]}
            if "/occurrence/search" in url:              # every ray-finned fish in the drainage
                return {"facets": [{"counts": [{"name": "101", "count": 12}, {"name": "103", "count": 4},
                                               {"name": "900", "count": 30}]}]}
            if "/species/" in url and "match" not in url:
                key = int(url.rsplit("/", 1)[1])
                return {101: {"canonicalName": "Notropis alpha", "family": "Leuciscidae"},
                        103: {"canonicalName": "Notropis gamma", "family": "Leuciscidae"},
                        900: {"canonicalName": "Lepomis gibbosus", "family": "Centrarchidae"}}[key]
            return {"usageKey": _FakeDrainageTool.KEYS.get(q.get("name", [""])[0])}

        def save(self):
            pass

    @staticmethod
    def url(base, params):
        return base + "?" + urllib.parse.urlencode(params)

    @staticmethod
    def huc8_of(web, lat, lon):
        return {"huc8": "04140201", "name": "Seneca", "states": "NY"}

    @staticmethod
    def huc8_outline(web, huc8):
        return "POLYGON((0 0,1 0,1 1,0 0))", {}, ""

    @staticmethod
    def genus_keys(web, genera):
        return {"Notropis": 1}

    @staticmethod
    def drainage_species(web, wkt, keys):
        return [{"species": "Notropis alpha", "key": 101, "records": 12},
                {"species": "Notropis gamma", "key": 103, "records": 4}]


def test_the_drainage_stands_beside_the_scores(srv, monkeypatch):
    url, st, *_ = srv
    import label_server as ls
    monkeypatch.setattr(ls, "_drainage_tool", lambda: _FakeDrainageTool)
    code, d0 = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    code, d = _post(url, "/api/batch/drainage?dataset=shiners", {"photo": "tray.jpg", "lat": 42.5, "lon": -76.9})
    assert d["drainage"]["huc8"] == "04140201" and d["drainage"]["name"] == "Seneca"
    assert [(o["species"], o["records"]) for o in d["drainage"]["others"]] == [("Notropis gamma", 4)]  # not a reference species
    for f in d["fish"]:
        rec = {x["species"]: x["drainage_records"] for x in f["match"]["matches"]}
        assert rec == {"Notropis_alpha": 12, "Notropis_beta": 0}
        assert f["match"]["matches"][0]["species"] == "Notropis_alpha"        # recorded there: first
    beta = d["fish"][1]["match"]                                               # a beta by shape
    assert beta["better_elsewhere"]["species"] == "Notropis_beta"
    # introduced there by USGS's records, under its current name; a plant is not a fish
    b = next(x for x in beta["matches"] if x["species"] == "Notropis_beta")
    assert b["introduced"]["status"] == "established" and b["introduced"]["records"] == 2
    assert next(x for x in beta["matches"] if x["species"] == "Notropis_alpha")["introduced"] is None
    tench = next(f for f in d["drainage"]["fishes"] if f["species"] == "Tinca tinca")
    assert tench["usgs_only"] and tench["introduced"]["status"] == "unknown"
    assert not any(f["species"] == "Trapa natans" for f in d["drainage"]["fishes"])
    # kept: asked again later, the photograph comes back with it
    # every fish recorded there, the reference species and the shiners marked
    fishes = {f["species"]: f for f in d["drainage"]["fishes"]}
    assert [f["species"] for f in d["drainage"]["fishes"] if not f.get("usgs_only")] == \
           ["Lepomis gibbosus", "Notropis alpha", "Notropis gamma"]
    assert sorted(f["species"] for f in d["drainage"]["fishes"] if f.get("usgs_only")) == ["Alburnops beta", "Tinca tinca"]
    assert fishes["Notropis alpha"]["reference"] and fishes["Notropis alpha"]["shiner"]
    assert not fishes["Lepomis gibbosus"]["shiner"] and fishes["Lepomis gibbosus"]["family"] == "Centrarchidae"
    code, again = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    assert again["drainage"]["huc8"] == "04140201"
    # the scores themselves are the shape's alone: the drainage orders them
    plain = {x["species"]: x["score"] for x in d0["fish"][1]["match"]["matches"]}
    assert {x["species"]: x["score"] for x in again["fish"][1]["match"]["matches"]} == plain
    code, d = _post(url, "/api/batch/drainage?dataset=shiners", {"photo": "tray.jpg", "clear": True})
    assert d["drainage"] is None


def test_each_species_comes_with_its_current_name(srv):
    url, *_ = srv
    code, d = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    assert d["taxa"]["Notropis_beta"]["current"] == "Alburnops beta"
    assert [c["name"] for c in d["taxa"]["Notropis_beta"]["chain"]] == ["Notropis beta", "Alburnops beta"]
    assert d["taxa"]["Notropis_alpha"]["current"] == "Notropis alpha"


class _FakeGallery(_FakeDrainageTool):
    """GBIF's photographed specimens of a species: two of one lot, a fresh tissue
    voucher, and one more."""
    class Web(_FakeDrainageTool.Web):
        def get(self, url):
            if "/occurrence/search" in url and "mediaType=StillImage" in url:
                def rec(key, cat, desc=""):
                    return {"key": key, "institutionCode": "FMNH", "catalogNumber": cat, "year": 1990,
                            "stateProvince": "Illinois", "scientificName": "Notropis alpha",
                            "media": [{"type": "StillImage", "identifier": f"http://x.org/{key}.jpg",
                                       "license": "CC0", "rightsHolder": "FMNH", "description": desc}]}
                return {"count": 4, "results": [rec(1, "10"), rec(2, "10"), rec(3, "11", "Tissue voucher"), rec(4, "12")]}
            return super().get(url)


def test_the_gallery_lists_photographed_specimens_without_keeping_photographs(srv, monkeypatch):
    url, st, *_ = srv
    import label_server as ls
    monkeypatch.setattr(ls, "_drainage_tool", lambda: _FakeGallery)
    code, d = _get(url, "/api/batch/specimens?dataset=shiners&taxonKey=101")
    assert d["ok"] and [it["catalog"] for it in d["items"]] == ["10", "12"]       # one a lot; no tissue voucher
    it = d["items"][0]
    assert it["thumb"].startswith("https://api.gbif.org/v1/image/cache/300x/occurrence/1/media/")
    assert it["license"] == "CC0" and it["credit"] == "FMNH" and it["place"] == "Illinois"
    kept = (st / "gallery_cache.json").read_text()
    assert "http://x.org/1.jpg" in kept and len(kept) < 5000                       # the list, not the photographs
    assert not list(st.glob("**/*.jpg")) or all("x.org" not in str(p) for p in st.glob("**/*.jpg"))
    code, d = _get(url, "/api/batch/specimens?dataset=shiners&taxonKey=nope")
    assert code == 400


def test_specimens_kept_to_compare_are_saved_with_the_study(srv):
    url, st, *_ = srv
    spec = {"occurrence": "7", "taxon": 101, "name": "Notropis rubellus", "institution": "FMNH",
            "catalog": "12", "thumb": "https://api.gbif.org/t.jpg", "large": "https://api.gbif.org/l.jpg",
            "original": "http://x.org/1.jpg", "junk": "not kept"}
    assert _get(url, "/api/batch/compare?dataset=shiners")[1] == {"ok": True, "items": []}
    code, d = _post(url, "/api/batch/compare?dataset=shiners", {"add": spec})
    assert code == 200 and d["items"][0]["occurrence"] == 7 and "junk" not in d["items"][0]
    _post(url, "/api/batch/compare?dataset=shiners", {"add": {**spec, "occurrence": 8}})
    _post(url, "/api/batch/compare?dataset=shiners", {"add": spec})                   # kept once
    assert [x["occurrence"] for x in _get(url, "/api/batch/compare?dataset=shiners")[1]["items"]] == [8, 7]
    d = _post(url, "/api/batch/compare?dataset=shiners", {"remove": 8})[1]
    assert [x["occurrence"] for x in d["items"]] == [7]
    assert "x.org" in (st / "batch_compare.json").read_text() and not list(st.glob("**/1.jpg"))
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(url, "/api/batch/compare?dataset=shiners", {"add": {**spec, "thumb": "javascript:x"}})
    assert e.value.code == 400


def test_photographs_are_added_to_the_batch_folder(srv):
    url, st, *_ = srv
    import io

    buf = io.BytesIO()
    Image.new("RGB", (40, 20), (200, 200, 200)).save(buf, "JPEG")

    def add(name, data):
        req = urllib.request.Request(url + "/api/upload?dataset=shiners", data=data, method="POST",
                                     headers={"X-Filename": urllib.parse.quote(name), "X-View": "batch"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    code, d = add("lot 7.jpg", buf.getvalue())
    assert code == 200 and d["status"] == "added" and (st / "batch" / "lot_7.jpg").is_file()
    assert add("lot 7.jpg", buf.getvalue())[1]["status"] == "duplicate"        # the same again: fine
    assert add("lot 7.jpg", buf.getvalue() + b"x")[0] == 409                   # other bytes: never overwritten
    assert not (st / "lateral" / "lot_7.jpg").exists()
    code, ph = _get(url, "/api/batch/photos?dataset=shiners")
    assert "lot_7.jpg" in [p["name"] for p in ph["photos"]]


def test_a_reference_set_needs_only_its_landmarks(srv):
    """A downloaded reference set carries landmarks and credits, no photographs."""
    url, st, *_ = srv
    for p in (st / "lateral").glob("*.jpg"):
        p.unlink()
    code, d = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    assert code == 200 and not d["fish"][0]["match"]["ok"]                 # an ordinary study: none
    schema = json.loads((st / "schema.json").read_text())
    (st / "schema.json").write_text(json.dumps({**schema, "reference_set": True}))
    code, d = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    assert d["fish"][0]["match"]["ok"] and d["reference"]["fish"] == 16
