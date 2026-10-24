"""The drainage candidates for a lot identified only to genus.

Tested without the network, against a stand-in for the three services: the
polygon arithmetic GBIF's geometry search depends on (exterior rings
counter-clockwise, holes dropped, an outline generalised until it fits in a URL,
a hull or a bounding box when it will not); how each
candidate's evidence is written, the rare and the introduced marked; a whole run
from a CSV of lots, through both output tables; and a source that is down, which
is retried and, if it stays down, costs only its own lots.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import urllib.error
import urllib.parse
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
spec = importlib.util.spec_from_file_location("drainage_candidates", SCRIPTS / "drainage_candidates.py")
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

SQUARE_CW = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]          # clockwise
SQUARE_CCW = SQUARE_CW[::-1]


def test_ring_winding():
    assert dc.ring_area(SQUARE_CCW) > 0 > dc.ring_area(SQUARE_CW)


def test_exteriors_run_counter_clockwise_and_holes_are_dropped():
    hole = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6], [0.4, 0.4]]
    wkt = dc.to_wkt({"type": "Polygon", "coordinates": [SQUARE_CW, hole]})
    assert wkt.startswith("POLYGON((") and wkt.count("(") == 2       # one ring: no hole
    pts = [tuple(map(float, p.split())) for p in wkt[len("POLYGON(("):-2].split(",")]
    assert dc.ring_area([list(p) for p in pts]) > 0
    multi = dc.to_wkt({"type": "MultiPolygon", "coordinates": [[SQUARE_CW], [SQUARE_CCW]]})
    assert multi.startswith("MULTIPOLYGON(((") and multi.count("((") == 2


def test_the_hull_of_a_concave_outline():
    notch = [[0, 0], [2, 0], [2, 2], [1, 1], [0, 2], [0, 0]]        # counter-clockwise, notched
    wkt = dc.hull_wkt({"type": "Polygon", "coordinates": [notch]})
    pts = [tuple(map(float, p.split())) for p in wkt[len("POLYGON(("):-2].split(",")]
    assert pts[0] == pts[-1] and len(pts) == 5                      # the notch is filled in
    assert (1.0, 1.0) not in pts and dc.ring_area([list(p) for p in pts]) > 0


def circle(n):
    ring = [[-76.3 + 0.2 * math.cos(2 * math.pi * i / n), 42.3 + 0.2 * math.sin(2 * math.pi * i / n)]
            for i in range(n)]
    return {"type": "Polygon", "coordinates": [ring + ring[:1]]}


def test_an_outline_is_generalised_until_it_fits_a_gbif_url():
    class Outlines:
        def __init__(self, points):          # points in the outline at each generalisation
            self.points = points

        def get(self, u):
            off = urllib.parse.parse_qs(urllib.parse.urlparse(u).query)["maxAllowableOffset"][0]
            return {"features": [{"geometry": circle(self.points[off])}]}

    fine, coarse = {"0.002": 400, "0.005": 120}, {o: 400 for o in ("0.002", "0.005", "0.01", "0.02", "0.05")}
    wkt, _, stand_in = dc.huc8_outline(Outlines(fine), "01010101")
    assert stand_in == "" and wkt.count(",") == 120     # 400 points do not fit; 120 do
    assert len(urllib.parse.quote_plus(wkt)) <= dc.MAX_GEOMETRY
    wkt, _, stand_in = dc.huc8_outline(Outlines(coarse), "01010101")
    assert stand_in == "bounding box" and wkt.count(",") == 4   # a convex outline's hull is no shorter
    assert dc.approximate(circle(40)) == (dc.hull_wkt(circle(40)), "convex hull")


def test_each_candidate_carries_its_evidence():
    common = {"species": "Notropis hudsonius", "records": 40, "institutions": 6,
              "first_year": 1931, "last_year": 2016}
    assert dc.describe(common, {}) == "Notropis hudsonius (40 rec, 6 inst, 1931-2016)"
    rare = {"species": "Notropis amabilis", "records": 1, "institutions": 1,
            "first_year": 1950, "last_year": 1950}
    assert dc.describe(rare, {"Notropis amabilis": "collected"}) == (
        "Notropis amabilis (1 rec, 1 inst, 1950, introduced (collected), few records)")


class FakeWeb:
    """Answers the requests the script makes, from a small made-up drainage."""

    def __init__(self, *_a, **_k):
        self.asked = []

    def save(self):
        pass

    def get(self, u):
        self.asked.append(u)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(u).query)
        if u.startswith(dc.WBD):
            if q.get("returnGeometry") == ["false"]:
                return {"features": [{"attributes": {"huc8": "01010101", "name": "Test", "states": "NY"}}]}
            return {"features": [{"geometry": {"type": "Polygon", "coordinates": [SQUARE_CW]}}]}
        if u.startswith(dc.NAS):
            return {"results": [{"genus": "Notropis", "scientificName": "Notropis amabilis",
                                 "status": "collected"},
                                {"genus": "Scardinius", "scientificName": "Scardinius erythrophthalmus"}]}
        if "/species/match" in u:
            return {"usageKey": 7, "rank": "GENUS"}
        if "/species/" in u:
            return {"canonicalName": {"10": "Notropis hudsonius", "11": "Notropis amabilis"}[u.rsplit("/", 1)[1]]}
        assert "geometry" in q and q["basisOfRecord"] == ["PRESERVED_SPECIMEN"]
        if q.get("facet") == ["speciesKey"]:
            return {"facets": [{"counts": [{"name": "10", "count": 5}, {"name": "11", "count": 1}]}]}
        years = {"10": [1950, 2001], "11": [1932]}[q["taxonKey"][0]]
        insts = {"10": {"CUMV": 3, "UMMZ": 2}, "11": {"CUMV": 1}}[q["taxonKey"][0]]
        return {"facets": [{"field": "YEAR", "counts": [{"name": str(y)} for y in years]},
                           {"field": "INSTITUTION_CODE",
                            "counts": [{"name": k, "count": v} for k, v in insts.items()]}]}


def test_a_run_from_a_csv_of_lots(tmp_path, monkeypatch):
    lots = tmp_path / "lots.csv"
    lots.write_text("id,lat,lon,locality\nCU 1,42.4,-76.5,Fall Creek\nCU 2,42.5,-76.4,Six Mile Creek\n")
    monkeypatch.setattr(dc, "Web", FakeWeb)
    assert dc.main(["--lots", str(lots), "--out", str(tmp_path / "out")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "out" / "lot_candidates.csv")))
    assert [r["id"] for r in rows] == ["CU 1", "CU 2"] and rows[0]["locality"] == "Fall Creek"
    assert rows[0]["huc8"] == "01010101" and rows[0]["n_candidates"] == "2"
    assert rows[0]["n_more_than_2_records"] == "1"
    assert rows[0]["candidates"] == ("Notropis hudsonius (5 rec, 2 inst, 1950-2001); "
                                     "Notropis amabilis (1 rec, 1 inst, 1932, introduced (collected), "
                                     "few records)")
    table = list(csv.DictReader(open(tmp_path / "out" / "drainage_species.csv")))
    assert len(table) == 2                                            # one drainage, asked once
    assert json.loads(table[0]["records_by_institution"]) == {"CUMV": 3, "UMMZ": 2}
    assert table[1]["introduced_NAS"] == "collected"


def test_a_source_that_stays_down_leaves_its_lot_for_the_next_run(tmp_path, monkeypatch):
    class DownForOne(FakeWeb):
        def get(self, u):
            if u.startswith(dc.WBD) and "-76.40000" in u:
                raise urllib.error.HTTPError(u, 502, "Bad Gateway", {}, None)
            return super().get(u)

    lots = tmp_path / "lots.csv"
    lots.write_text("id,lat,lon\nCU 1,42.4,-76.5\nCU 2,42.5,-76.4\n")
    monkeypatch.setattr(dc, "Web", DownForOne)
    assert dc.main(["--lots", str(lots), "--out", str(tmp_path / "out")]) == 1
    rows = {r["id"]: r for r in csv.DictReader(open(tmp_path / "out" / "lot_candidates.csv"))}
    assert rows["CU 1"]["n_candidates"] == "2"
    assert rows["CU 2"]["huc8"] == "" and rows["CU 2"]["outline"].startswith("not looked up")


def test_the_cache_retries_a_server_error_but_not_a_bad_request(tmp_path, monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        code = {"http://x/flaky": 502 if len(calls) < 3 else None, "http://x/bad": 404}[req.full_url]
        if code:
            raise urllib.error.HTTPError(req.full_url, code, "", {}, None)
        return io.BytesIO(b'{"ok": 1}')

    monkeypatch.setattr(dc.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dc.Web, "BACKOFF", (0, 0, 0))
    web = dc.Web(tmp_path / "cache.json", pause=0)
    assert web.get("http://x/flaky") == {"ok": 1} and len(calls) == 3
    assert web.get("http://x/flaky") == {"ok": 1} and len(calls) == 3   # from the cache
    try:
        web.get("http://x/bad")
        raise AssertionError("a 404 should not be retried")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404 and calls.count("http://x/bad") == 1
