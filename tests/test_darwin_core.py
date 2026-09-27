"""Darwin Core records: which museum specimen each photograph is.

Kept per study in darwin_core.csv under Darwin Core's term names, so a
measurement can be joined back to the collection's own record. What is tested:
the filename suggests only what it says; the table round-trips and refuses what
is not a term; an import matches rows to photographs (by filename, or a lot's
catalogue number), never erases, and reports before it writes; the exports carry
the records -- a Specimens sheet, Darwin Core's MeasurementOrFact long form, the
locality and date in the columns the workbook already had, and the identifiers in
the R specimen table -- and add nothing to a study that keeps none; and the
server does all of it without writing on a dry run or in demo mode.
"""

from __future__ import annotations

import base64
import csv
import importlib.util
import io
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from fish_morpho import darwin_core as dwc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

def test_the_filename_suggests_only_what_it_says():
    assert dwc.suggest("1947_CUMV_68133_02") == {"institutionCode": "CUMV",
                                                 "catalogNumber": "68133"}
    assert dwc.suggest("Salvelinus_fontinalis_ASN_31") == {"scientificName":
                                                           "Salvelinus fontinalis"}
    # the lot is in the study's name when the camera named the file
    assert dwc.suggest("IMG_7310", "CUMV_24306")["catalogNumber"] == "24306"
    assert dwc.suggest("IMG_7310", "cornell") == {}
    # a leading year is not read as a collection date: nothing says which date
    assert "eventDate" not in dwc.suggest("1947_CUMV_68133_02")


def test_the_table_round_trips_and_a_blank_clears(tmp_path):
    dwc.update(tmp_path, {"A_1": {"catalogNumber": "68133", "sex": "female"},
                          "A_2": {"scientificName": "Alosa pseudoharengus"}})
    rows = list(csv.DictReader((tmp_path / dwc.FILENAME).open()))
    assert list(rows[0]) == [dwc.KEY, *dwc.TERM_NAMES]
    assert dwc.load(tmp_path)["A_1"] == {"catalogNumber": "68133", "sex": "female"}
    dwc.update(tmp_path, {"A_1": {"sex": ""}, "A_2": {"scientificName": ""}})
    assert dwc.load(tmp_path) == {"A_1": {"catalogNumber": "68133"}}   # A_2 has no row now


def test_a_name_that_is_not_a_term_is_refused(tmp_path):
    with pytest.raises(ValueError, match="strain"):
        dwc.update(tmp_path, {"A_1": {"strain": "HRN"}})
    assert not (tmp_path / dwc.FILENAME).exists()


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

def test_an_idigbio_harvest_imports_by_filename():
    rows = [{"media_uuid": "u1", "institution": "NEON", "catalog_number": "NEON06X4A",
             "scientific_name": "Salvelinus fontinalis", "state": "California",
             "year": "2022", "recorded_by": "C. Halvorsen", "image_url": "https://x/1.jpg",
             "local_filename": "NEON_NEON06X4A_41a0eca5.jpg"},
            {"media_uuid": "u2", "institution": "MCZ", "catalog_number": "1",
             "local_filename": "not_here.jpg"}]
    p = dwc.plan_import(rows, ["NEON_NEON06X4A_41a0eca5", "other"])
    assert p["key"] == "local_filename" and p["by"] == "photo"
    assert p["changes"] == {"NEON_NEON06X4A_41a0eca5": {
        "institutionCode": "NEON", "catalogNumber": "NEON06X4A",
        "scientificName": "Salvelinus fontinalis", "stateProvince": "California",
        "eventDate": "2022", "recordedBy": "C. Halvorsen",
        "associatedMedia": "https://x/1.jpg"}}
    assert p["unmatched"] == ["not_here.jpg"] and "media_uuid" in p["unused"]


def test_a_lab_sheet_joins_genus_and_species_and_matches_a_view_suffix():
    rows = [{"genus": "Salvelinus", "species": "fontinalis", "field_number": "ST-HRN 2/24",
             "photo": "Salvelinus_fontinalis_HRN_5_L.JPEG", "sex": ""}]
    p = dwc.plan_import(rows, ["Salvelinus_fontinalis_HRN_5"])
    assert p["changes"] == {"Salvelinus_fontinalis_HRN_5": {
        "fieldNumber": "ST-HRN 2/24", "scientificName": "Salvelinus fontinalis"}}


def test_an_epithet_alone_is_not_a_scientific_name():
    p = dwc.plan_import([{"fish_id": "F1", "species": "fontinalis", "sex": "male"}], ["F1"])
    assert p["changes"] == {"F1": {"sex": "male"}}


def test_an_import_never_erases_and_counts_what_it_replaces():
    have = {"F1": {"sex": "female", "catalogNumber": "7"}}
    p = dwc.plan_import([{"fish_id": "F1", "sex": "male", "catalogNumber": ""}], ["F1"], have)
    assert p["changes"] == {"F1": {"sex": "male"}} and p["overwrites"] == 1


def test_a_museum_table_reaches_every_photograph_of_its_lot():
    ids = ["1947_CUMV_68133_02", "1947_CUMV_68133_03", "1950_CUMV_33063_01"]
    have = {i: {"institutionCode": "CUMV", "catalogNumber": i.split("_")[2]} for i in ids}
    rows = [{"institutionCode": "CUMV", "catalogNumber": "CUMV 68133",
             "locality": "Cayuga Lake", "year": "1947"},
            {"institutionCode": "MCZ", "catalogNumber": "33063", "locality": "elsewhere"}]
    p = dwc.plan_import(rows, ids, have)
    assert p["by"] == "lot" and p["named"] == 2
    assert set(p["changes"]) == {"1947_CUMV_68133_02", "1947_CUMV_68133_03"}
    assert p["changes"]["1947_CUMV_68133_02"] == {"locality": "Cayuga Lake", "eventDate": "1947"}
    assert "33063" in p["unmatched"]            # same number, another museum's lot


def test_excel_reads_a_catalogue_number_as_written(tmp_path):
    wb = openpyxl.Workbook()
    wb.active.append(["fish_id", "Catalog No.", "Date collected"])
    import datetime
    wb.active.append(["F1", 68133, datetime.datetime(1947, 6, 12)])
    buf = io.BytesIO()
    wb.save(buf)
    rows = dwc.read_table("lots.xlsx", buf.getvalue())
    assert rows == [{"fish_id": "F1", "Catalog No.": "68133", "Date collected": "1947-06-12"}]
    assert dwc.plan_import(rows, ["F1"])["changes"] == {
        "F1": {"catalogNumber": "68133", "eventDate": "1947-06-12"}}


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

payload = _load("_pipeline_payload", TESTS / "test_pipeline.py")._sidecar_payload
exporter = _load("export_measurements", SCRIPTS / "export_measurements.py")


@pytest.fixture
def study(tmp_path):
    st = tmp_path / "trout"
    images, labels = st / "lateral", st / "sidecars"
    images.mkdir(parents=True)
    labels.mkdir()
    for fid in ("BKT-0001", "BKT-0002"):
        (images / f"{fid}.jpg").write_bytes(b"\x00")
        p = payload(fid)
        if fid == "BKT-0001":                   # BKT-0002 keeps a locality of its own
            for k in ("locality", "collection_date"):
                p["metadata"].pop(k, None)
        (labels / f"{fid}.json").write_text(json.dumps(p))
    return st


def _export(st, tmp_path):
    out = tmp_path / "out"
    assert exporter.main(["--images", str(st / "lateral"), "--labels", str(st / "sidecars"),
                          "--out", str(out / "m.xlsx"), "--csv", str(out / "m.csv")]) == 0
    return openpyxl.load_workbook(out / "m.xlsx"), out


def test_a_study_without_records_gets_no_extra_sheets(study, tmp_path):
    wb, _ = _export(study, tmp_path)
    assert "Specimens" not in wb.sheetnames and "MeasurementOrFact" not in wb.sheetnames


def test_the_workbook_carries_the_records(study, tmp_path):
    dwc.update(study, {"BKT-0001": {"institutionCode": "CUMV", "catalogNumber": "63033",
                                    "locality": "Cayuga Inlet", "eventDate": "2025-02-24"},
                       "BKT-0002": {"locality": "somewhere else"}})
    wb, out = _export(study, tmp_path)
    sp = list(wb["Specimens"].iter_rows(values_only=True))
    assert sp[0] == ("fish_id", "institutionCode", "catalogNumber", "eventDate", "locality")
    assert ("BKT-0001", "CUMV", "63033", "2025-02-24", "Cayuga Inlet") in sp
    assert ("BKT-0002", None, None, None, "somewhere else") in sp

    meas = list(wb["Measurements"].iter_rows(values_only=True))
    hdr = list(meas[0])
    u = hdr.index("units")
    n_values = sum(1 for r in meas[1:] for v in r[u + 1:] if v not in (None, ""))
    mof = list(wb["MeasurementOrFact"].iter_rows(values_only=True))
    assert mof[0][:4] == ("fish_id", "measurementType", "measurementValue", "measurementUnit")
    assert len(mof) - 1 == n_values                              # one row per value
    sl = next(r for r in mof if r[0] == "BKT-0001" and r[1].endswith("(SL)"))
    row = next(r for r in meas if r[0] == "BKT-0001")
    sl_col = next(i for i, h in enumerate(hdr) if str(h).startswith("SL "))
    assert sl[2] == row[sl_col] and sl[3] == "mm" and "premaxilla" in sl[4]

    # where and when, in the columns the workbook already had for them
    assert row[hdr.index("locality")] == "Cayuga Inlet"
    assert row[hdr.index("collection_date")] == "2025-02-24"
    other = next(r for r in meas if r[0] == "BKT-0002")
    assert other[hdr.index("locality")] == "Hogan's Brook"      # the sidecar's own wins
    csv_rows = {r["fish_id"]: r for r in csv.DictReader(open(out / "m.csv"))}
    assert csv_rows["BKT-0001"]["locality"] == "Cayuga Inlet"
    # ...and nothing else added to the CSV, whose non-trait columns an R script names
    assert "catalogNumber" not in csv_rows["BKT-0001"]


def test_the_r_specimen_table_carries_the_identifiers(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    st = tmp_path / "study"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    for fid in ("F1", "F2"):
        Image.new("RGB", (400, 300)).save(st / "lateral" / f"{fid}_L.JPEG")
        (st / "sidecars" / f"{fid}.json").write_text(json.dumps(
            {"fish_id": fid, "lateral": {"keypoints": {"premaxilla_tip": [10, 20]}}}))
    dwc.update(st, {"F1": {"catalogNumber": "68133", "scientificName": "Alosa pseudoharengus"}})
    folder = tmp_path / "by_specimen"
    r = subprocess.run([sys.executable, str(SCRIPTS / "export_tps.py"),
                        "--sidecars", str(st / "sidecars"), "--images", str(st / "lateral"),
                        "--schema-dir", str(st), "--out", str(tmp_path / "tps"),
                        "--per-specimen", str(folder)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    for table in (folder / "specimens.csv", tmp_path / "tps" / "specimens.csv"):
        sp = {row["ID"]: row for row in csv.DictReader(table.open())}
        assert sp["F1"]["catalogNumber"] == "68133"
        assert sp["F1"]["scientificName"] == "Alosa pseudoharengus"
        assert sp["F2"]["catalogNumber"] == ""
        assert "sex" not in sp["F1"]                              # only terms someone recorded


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------

@pytest.fixture
def srv(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    ls = _load("label_server", SCRIPTS / "label_server.py")
    data = tmp_path / "data"
    st = data / "CUMV_68133"
    (st / "lateral").mkdir(parents=True)
    (st / "sidecars").mkdir()
    for fid in ("1947_CUMV_68133_02", "1947_CUMV_68133_03"):
        Image.new("RGB", (60, 40)).save(st / "lateral" / f"{fid}.jpg")
    ls.Handler.data_root = data
    ls.Handler.datasets = {"CUMV_68133": st}
    ls.Handler.default_dataset = "CUMV_68133"
    ls.Handler.images_dir = st
    ls.Handler.out_dir = st / "sidecars"
    ls.Handler.out_override = None
    ls.Handler.demo_mode = False
    server = ls.ThreadingHTTPServer(("127.0.0.1", 0), ls.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield ls, f"http://127.0.0.1:{server.server_address[1]}", st
    server.shutdown()


def _call(url, route, body=None):
    req = urllib.request.Request(url + route + "?dataset=CUMV_68133",
                                 method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_server_shows_suggestions_apart_from_records(srv):
    _, url, st = srv
    code, d = _call(url, "/api/dwc")
    assert code == 200 and d["records"] == {}
    assert d["ids"] == ["1947_CUMV_68133_02", "1947_CUMV_68133_03"]
    assert d["suggestions"]["1947_CUMV_68133_02"] == {"institutionCode": "CUMV",
                                                      "catalogNumber": "68133"}
    assert not (st / dwc.FILENAME).exists()                      # shown, not written


def test_the_server_saves_edits(srv):
    _, url, st = srv
    code, d = _call(url, "/api/dwc", {"changes": {"1947_CUMV_68133_02": {"sex": "female"}}})
    assert code == 200 and d["records"] == {"1947_CUMV_68133_02": {"sex": "female"}}
    assert dwc.load(st) == d["records"]
    assert _call(url, "/api/dwc", {"changes": {"no_such_fish": {"sex": "male"}}})[0] == 404
    assert _call(url, "/api/dwc", {"changes": {"1947_CUMV_68133_02": {"strain": "x"}}})[0] == 400


def test_an_import_is_reported_before_it_is_written(srv):
    _, url, st = srv
    table = b"fish_id,sex,notes\n1947_CUMV_68133_03,male,dissected\nZZ_9,female,\n"
    up = {"name": "dissections.csv", "data": base64.b64encode(table).decode()}
    code, d = _call(url, "/api/dwc/import", up)
    assert code == 200 and not d["applied"]
    assert d["plan"]["named"] == 1 and d["plan"]["unmatched"] == ["ZZ_9"]
    assert d["plan"]["unused"] == ["notes"]
    assert not (st / dwc.FILENAME).exists()
    code, d = _call(url, "/api/dwc/import", {**up, "apply": True})
    assert code == 200 and d["applied"]
    assert dwc.load(st) == {"1947_CUMV_68133_03": {"sex": "male"}}


def test_an_import_naming_no_photograph_is_refused(srv):
    _, url, st = srv
    up = {"name": "x.csv", "data": base64.b64encode(b"id,sex\nnobody,male\n").decode()}
    code, d = _call(url, "/api/dwc/import", {**up, "apply": True})
    assert code == 400 and not (st / dwc.FILENAME).exists()


def test_demo_mode_writes_no_records(srv):
    ls, url, st = srv
    ls.Handler.demo_mode = True
    assert _call(url, "/api/dwc", {"changes": {"1947_CUMV_68133_02": {"sex": "f"}}})[0] == 403
    up = {"name": "d.csv", "data": base64.b64encode(b"fish_id,sex\n1947_CUMV_68133_02,f\n").decode(),
          "apply": True}
    assert _call(url, "/api/dwc/import", up)[0] == 403
    assert not (st / dwc.FILENAME).exists()
