#!/usr/bin/env python3
"""Which species a lot identified only to genus could be, from where it was caught.

    python scripts/drainage_candidates.py --gbif-institution CUMV --gbif-genus Notropis \\
        --out results/notropis_id

A drainage holds a dozen or so shiners, not the hundred-odd of eastern North
America. For each lot this lists the candidate species that museums have recorded
from the lot's drainage -- its 8-digit hydrologic unit (HUC8, USGS Watershed
Boundary Dataset) -- with the evidence behind each, so that someone identifying the
fish can weigh it:

- how many museum specimens of the species come from the drainage (GBIF, preserved
  specimens only), from how many institutions, and over which years;
- whether USGS Nonindigenous Aquatic Species (NAS) records it as introduced there.

It is an aid, not a verdict, and it deliberately does not hide anything. Range
records are themselves past identifications: a species with one record from 1932
is shown as exactly that, beside one with forty, and a fish that looks like a
species never recorded from its drainage may be a range extension, an
introduction, or the record that corrects an old misidentification. Shape (the
landmark comparison) is a separate column, added later, never folded into this one.

Lots come from GBIF -- a collection's records identified to the genus and no
further (``--gbif-institution``, ``--gbif-genus``) -- or from a CSV with columns
``id,lat,lon`` and any others, which are carried through (``--lots``).

Every web response is cached (``<out>/web_cache.json``), so a run can be repeated
offline and reproduced exactly; delete the cache to refresh from the sources.

Writes ``<out>/lot_candidates.csv`` (one row per lot) and
``<out>/drainage_species.csv`` (one row per drainage and candidate species).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GBIF = "https://api.gbif.org/v1"
WBD = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query"   # layer 4: HUC8
NAS = "https://nas.er.usgs.gov/api/v2/occurrence/search"

#: Shiners and the genera a lot left at "Notropis sp." may belong to. GBIF's
#: backbone files Hudsonius, Alburnops, Miniellus and Paranotropis as synonyms of
#: Notropis, so its key covers them. The golden shiner (Notemigonus) is left out:
#: it was never a Notropis and is told at a glance, so it only lengthens the lists.
SHINER_GENERA = ("Notropis", "Ericymba", "Cyprinella", "Luxilus", "Lythrurus", "Hybopsis",
                 "Pteronotropis", "Opsopoeodus")
#: Longest polygon GBIF is sent, as it is spelled in the URL (every comma is three
#: characters there). GBIF's server answers a bare 400 to a request line much over
#: 3,800 characters, and the rest of a query needs room; coarser generalisation is
#: tried until the outline fits.
MAX_GEOMETRY = 3200
#: A species recorded this rarely from a drainage is marked for a second look.
FEW_RECORDS = 2
#: Requests to GBIF at once, for the per-species evidence in a drainage.
WORKERS = 4
#: Said with every request. USGS NAS refuses Python's default user agent.
USER_AGENT = "caliPr-drainage-candidates/1.0 (fish identification aid; research use)"


# --------------------------------------------------------------------------- web

class Web:
    """GET with a polite pause, retries, and a cache keyed by the URL. Safe to call
    from several threads."""

    #: Waits between attempts, in seconds: a USGS server can be down for half a minute.
    BACKOFF = (2, 4, 8, 16, 32)

    def __init__(self, cache_path: Path | None, pause: float = 0.15):
        self.path = cache_path
        self.pause = pause
        self.cache = json.loads(cache_path.read_text()) if cache_path and cache_path.is_file() else {}
        self.fresh = 0
        self.lock = threading.Lock()

    def get(self, url: str):
        key = hashlib.sha1(url.encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        for wait in (*self.BACKOFF, None):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as exc:
                if (exc.code < 500 and exc.code != 429) or wait is None:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if wait is None:
                    raise
            time.sleep(wait)
        time.sleep(self.pause)
        with self.lock:
            self.cache[key] = data
            self.fresh += 1
            if self.path and self.fresh % 50 == 0:
                self._write()
        return data

    def save(self):
        with self.lock:
            self._write()

    def _write(self):
        if self.path:
            self.path.write_text(json.dumps(self.cache))


def url(base: str, params) -> str:
    return base + "?" + urllib.parse.urlencode(params)


# ----------------------------------------------------------------------- geometry

def ring_area(ring) -> float:
    """Signed area of a closed ring: positive when it runs counter-clockwise."""
    return sum(ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
               for i in range(len(ring) - 1)) / 2


def to_wkt(geometry: dict) -> str:
    """GeoJSON (Multi)Polygon as WKT, exterior rings counter-clockwise as GBIF
    requires, holes left out: a hole in a drainage is not a place a fish is."""
    polys = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    parts = []
    for poly in polys:
        ext = poly[0] if ring_area(poly[0]) > 0 else poly[0][::-1]
        parts.append("((" + ",".join(f"{x:.4f} {y:.4f}" for x, y in ext) + "))")
    return "POLYGON" + parts[0] if len(parts) == 1 else "MULTIPOLYGON(" + ",".join(parts) + ")"


def hull_wkt(geometry: dict) -> str:
    """The convex hull of a polygon's vertices, counter-clockwise."""
    polys = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    pts = sorted({(round(x, 4), round(y, 4)) for poly in polys for x, y in poly[0]})

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    hull.append(hull[0])
    return "POLYGON((" + ",".join(f"{x} {y}" for x, y in hull) + "))"


# ------------------------------------------------------------------------ sources

def huc8_of(web: Web, lat: float, lon: float) -> dict | None:
    d = web.get(url(WBD, {"geometry": f"{lon:.5f},{lat:.5f}", "geometryType": "esriGeometryPoint",
                          "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
                          "outFields": "huc8,name,states", "returnGeometry": "false", "f": "json"}))
    feats = d.get("features") or []
    return feats[0]["attributes"] if feats else None


def huc8_outline(web: Web, huc8: str):
    """The drainage's outline as WKT short enough for a GBIF URL, the GeoJSON it was
    drawn from, and what stands in for the outline if it does not fit ("" if none)."""
    for offset in (0.002, 0.005, 0.01, 0.02, 0.05):
        fc = web.get(url(WBD, {"where": f"huc8='{huc8}'", "outFields": "huc8", "returnGeometry": "true",
                               "outSR": "4326", "maxAllowableOffset": str(offset),
                               "geometryPrecision": "4", "f": "geojson"}))
        geom = fc["features"][0]["geometry"]
        wkt = to_wkt(geom)
        if fits(wkt):
            return wkt, geom, ""
    stand_in, name = approximate(geom)
    return stand_in, geom, name


def fits(wkt: str) -> bool:
    return len(urllib.parse.quote_plus(wkt)) <= MAX_GEOMETRY


def approximate(geometry: dict) -> tuple[str, str]:
    """A stand-in for an outline GBIF will not take, and its name: the convex hull,
    or the bounding box if even the hull is too long for a URL. Both take in some
    of the neighbouring drainages, which the output says."""
    wkt = hull_wkt(geometry)
    if fits(wkt):
        return wkt, "convex hull"
    polys = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    xs = [x for poly in polys for x, _ in poly[0]]
    ys = [y for poly in polys for _, y in poly[0]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))", "bounding box"


def genus_keys(web: Web, genera) -> dict[str, int]:
    keys = {}
    for g in genera:
        m = web.get(url(f"{GBIF}/species/match", {"name": g, "rank": "GENUS", "family": "Cyprinidae"}))
        if m.get("rank") == "GENUS" and m.get("usageKey"):
            keys[g] = m.get("acceptedUsageKey") or m["usageKey"]
        else:
            print(f"  ! {g}: no GBIF genus found; left out", file=sys.stderr)
    return keys


def drainage_species(web: Web, wkt: str, keys) -> list[dict]:
    """Every candidate species with museum specimens inside the outline, with the
    evidence: records, institutions, first and last year."""
    base = [("geometry", wkt), ("basisOfRecord", "PRESERVED_SPECIMEN"), ("limit", "0")]
    d = web.get(url(f"{GBIF}/occurrence/search",
                    base + [("facet", "speciesKey"), ("facetLimit", "200")]
                    + [("taxonKey", str(k)) for k in sorted(set(keys))]))

    def evidence(c):
        sp = web.get(f"{GBIF}/species/{c['name']}")
        e = web.get(url(f"{GBIF}/occurrence/search",
                        base + [("taxonKey", c["name"]), ("facet", "year"), ("facetLimit", "400"),
                                ("facet", "institutionCode"), ("facetLimit", "400")]))
        facets = {f["field"]: f["counts"] for f in e.get("facets") or []}
        years = [int(y["name"]) for y in facets.get("YEAR", [])]
        insts = {i["name"]: i["count"] for i in facets.get("INSTITUTION_CODE", [])}
        return {"species": sp.get("canonicalName") or sp.get("scientificName"), "key": int(c["name"]),
                "records": c["count"], "institutions": len(insts),
                "first_year": min(years) if years else None,
                "last_year": max(years) if years else None,
                "by_institution": insts}

    counts = (d.get("facets") or [{}])[0].get("counts", [])
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:     # a geometry search is slow
        return list(pool.map(evidence, counts))


def introduced_in(web: Web, huc8: str, genera) -> dict[str, str]:
    """Species of the candidate genera USGS NAS records as introduced in the drainage."""
    d = web.get(url(NAS, {"huc8": huc8, "family": "Leuciscidae"}))
    out = {}
    for r in d.get("results") or []:
        if r.get("genus") in genera:
            out.setdefault(r["scientificName"], r.get("status") or "recorded")
    return out


def gbif_genus_lots(web: Web, institution: str, genus: str) -> list[dict]:
    """A collection's records identified to ``genus`` and no further, with coordinates."""
    key = web.get(url(f"{GBIF}/species/match", {"name": genus, "rank": "GENUS"}))["usageKey"]
    lots, offset = [], 0
    while True:
        d = web.get(url(f"{GBIF}/occurrence/search", {"institutionCode": institution, "taxonKey": key,
                                                     "limit": 300, "offset": offset}))
        for r in d["results"]:
            if r.get("taxonRank") == "GENUS" and r.get("decimalLatitude") is not None:
                lots.append({"id": r.get("catalogNumber") or str(r["key"]),
                             "lat": r["decimalLatitude"], "lon": r["decimalLongitude"],
                             "uncertainty_m": r.get("coordinateUncertaintyInMeters"),
                             "year": r.get("year"), "state": r.get("stateProvince"),
                             "county": r.get("county"), "locality": r.get("locality"),
                             "waterbody": r.get("waterBody"), "gbif": r.get("key")})
        offset += 300
        if d["endOfRecords"] or offset >= d["count"]:
            return lots


# ------------------------------------------------------------------------ output

def describe(sp: dict, introduced: dict) -> str:
    """One candidate, with its evidence, for the lot table."""
    yrs = (f"{sp['first_year']}" if sp["first_year"] == sp["last_year"]
           else f"{sp['first_year']}-{sp['last_year']}") if sp["first_year"] else "year unknown"
    note = f"{sp['records']} rec, {sp['institutions']} inst, {yrs}"
    if sp["species"] in introduced:
        note += f", introduced ({introduced[sp['species']]})"
    if sp["records"] <= FEW_RECORDS:
        note += ", few records"
    return f"{sp['species']} ({note})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="drainage_candidates")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lots", type=Path, help="CSV with id, lat, lon (other columns are kept)")
    src.add_argument("--gbif-institution", help="lots from GBIF: this collection's records ...")
    ap.add_argument("--gbif-genus", default="Notropis", help="... identified to this genus only")
    ap.add_argument("--genera", default=",".join(SHINER_GENERA),
                    help="candidate genera, comma-separated (default: the shiners)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N lots (a trial run)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    web = Web(args.out / "web_cache.json")
    genera = [g.strip() for g in args.genera.split(",") if g.strip()]

    if args.lots:
        with open(args.lots, newline="", encoding="utf-8-sig") as fh:
            lots = [dict(r, lat=float(r["lat"]), lon=float(r["lon"])) for r in csv.DictReader(fh)]
    else:
        lots = gbif_genus_lots(web, args.gbif_institution, args.gbif_genus)
    lots = lots[:args.limit] if args.limit else lots
    print(f"{len(lots)} lots", flush=True)
    keys = genus_keys(web, genera)

    drainages: dict[str, dict] = {}
    failed: set[str] = set()                # lots a source would not answer for: run again
    try:
        for i, lot in enumerate(lots, 1):
            lot["huc8"], lot["huc8_name"] = "", ""
            try:
                h = huc8_of(web, lot["lat"], lot["lon"])
                if h and h["huc8"] not in drainages:
                    wkt, geom, approx = huc8_outline(web, h["huc8"])
                    try:
                        species = drainage_species(web, wkt, keys.values())
                    except urllib.error.HTTPError as exc:      # an outline GBIF refuses as invalid
                        if exc.code != 400 or approx:
                            raise
                        wkt, approx = approximate(geom)
                        species = drainage_species(web, wkt, keys.values())
                    drainages[h["huc8"]] = {"name": h["name"], "species": species, "approx": approx,
                                            "introduced": introduced_in(web, h["huc8"], genera)}
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"  ! lot {lot['id']}: {exc}; left out, run again to fill it in", file=sys.stderr)
                failed.add(lot["id"])
                continue
            if h:
                lot["huc8"], lot["huc8_name"] = h["huc8"], h["name"]
            if i % 25 == 0:
                print(f"  {i}/{len(lots)} lots, {len(drainages)} drainages", flush=True)
    finally:
        web.save()

    extra = [k for k in (lots[0] if lots else {}) if k not in ("id", "lat", "lon", "huc8", "huc8_name")]
    with open(args.out / "lot_candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "lat", "lon", *extra, "huc8", "drainage", "n_candidates",
                    f"n_more_than_{FEW_RECORDS}_records", "candidates", "outline"])
        for lot in lots:
            d = drainages.get(lot["huc8"])
            sp = sorted(d["species"], key=lambda s: -s["records"]) if d else []
            w.writerow([lot["id"], lot["lat"], lot["lon"], *[lot.get(k) for k in extra],
                        lot["huc8"], lot["huc8_name"], len(sp),
                        sum(1 for s in sp if s["records"] > FEW_RECORDS),
                        "; ".join(describe(s, d["introduced"]) for s in sp) if d else "",
                        (f"approximate: {d['approx']} of the drainage" if d["approx"] else "drainage") if d else
                        "not looked up: a source did not answer, run again" if lot["id"] in failed else
                        "no drainage found (outside the USA?)"])
    with open(args.out / "drainage_species.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["huc8", "drainage", "species", "records", "institutions", "first_year",
                    "last_year", "introduced_NAS", "records_by_institution"])
        for code, d in sorted(drainages.items()):
            for s in sorted(d["species"], key=lambda s: -s["records"]):
                w.writerow([code, d["name"], s["species"], s["records"], s["institutions"],
                            s["first_year"], s["last_year"], d["introduced"].get(s["species"], ""),
                            json.dumps(s["by_institution"], sort_keys=True)])
    n = [len(d["species"]) for d in drainages.values()]
    if n:
        n.sort()
        print(f"{len(drainages)} drainages; candidates per drainage: median {n[len(n) // 2]}, "
              f"range {n[0]}-{n[-1]}")
    if failed:
        print(f"{len(failed)} lots not looked up (a source did not answer); run again to fill them in")
    print(f"wrote {args.out / 'lot_candidates.csv'} and {args.out / 'drainage_species.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
