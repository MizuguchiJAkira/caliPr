#!/usr/bin/env python3
"""Photographs of identified museum specimens, a few species at a time, from GBIF.

    python scripts/harvest_gbif_photos.py --species "Luxilus cornutus,Luxilus chrysocephalus" \\
        --per-species 80 --out data/shiners --dry-run

For a reference set of each species' shape: preserved specimens identified to
species, one photograph per lot so that no lot counts twice, spread across the
institutions that photographed them. Left out are photographs that are not of a
preserved specimen -- tissue-voucher photographs of fresh fish (the Field Museum
files these beside its lightbox photographs) and the NEON and BOLD field images --
and photographs published without a licence.

``--dry-run`` lists what would be fetched, with each file's size where the server
says, and fetches nothing. Without it the photographs are saved to
``<out>/lateral/`` as ``Genus_species_N.jpg``, the name caliPr reads the species
from, and a file already there -- or set aside in ``<out>/omitted/`` after a look
at it -- is not fetched again. ``--fetch MANIFEST`` fetches
exactly what a dry run listed: GBIF's records change, so choosing again could
choose differently.

Writes ``<out>/photo_manifest.csv``: one row per photograph, with the specimen
(institution, catalog number, where and when collected), the GBIF occurrence, the
photograph's URL, licence, rights holder and creator, its size, and its file name.
Licences vary by institution (CC0, CC BY, CC BY-NC); the manifest keeps them, and
anything published from these photographs must credit them accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

GBIF = "https://api.gbif.org/v1"
USER_AGENT = "caliPr-photo-harvest/1.0 (fish identification aid; research use)"
#: Photographs of fresh fish in the field, not of preserved specimens.
SKIP_INSTITUTIONS = {"NEON"}
SKIP_HOSTS = ("boldsystems.org", "biorepo.neonscience.org")
SKIP_DESCRIPTIONS = ("tissue voucher",)
MANIFEST = ["fish_id", "species", "institution", "catalog_number", "country", "state", "county",
            "locality", "year", "gbif_occurrence", "image_url", "license", "rights_holder",
            "creator", "bytes"]


def get(url: str, method: str = "GET", tries: int = 4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.headers if method == "HEAD" else r.read()
        except urllib.error.HTTPError as exc:
            if (exc.code < 500 and exc.code != 429) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(2 ** (attempt + 1))


def species_key(name: str) -> int:
    m = json.loads(get(f"{GBIF}/species/match?" + urllib.parse.urlencode({"name": name, "strict": "true"})))
    if m.get("rank") != "SPECIES" or "usageKey" not in m:
        raise SystemExit(f"GBIF has no species {name!r} (best match: {m.get('scientificName')})")
    return m.get("acceptedUsageKey") or m["usageKey"]


def photograph(record: dict) -> dict | None:
    """The record's photograph of the preserved specimen, if it has one: a lateral
    view when the titles say which is which."""
    if record.get("institutionCode", "").upper() in SKIP_INSTITUTIONS:
        return None
    usable = [m for m in record.get("media", [])
              if m.get("type") == "StillImage" and m.get("identifier") and m.get("license")
              and not any(h in m["identifier"] for h in SKIP_HOSTS)
              and not any(s in (m.get("description") or "").lower() for s in SKIP_DESCRIPTIONS)]
    lateral = [m for m in usable if "_lat" in (m.get("title") or "").lower()]
    return (lateral or usable or [None])[0]


def specimens(name: str, institutions: set[str] | None = None) -> list[dict]:
    """Every preserved specimen of the species with a usable photograph, one per lot,
    from the given institutions (all, if none are given)."""
    key, offset, lots = species_key(name), 0, {}
    while True:
        d = json.loads(get(f"{GBIF}/occurrence/search?" + urllib.parse.urlencode(
            {"taxonKey": key, "basisOfRecord": "PRESERVED_SPECIMEN", "mediaType": "StillImage",
             "limit": 300, "offset": offset})))
        for r in d["results"]:
            m = photograph(r)
            lot = (r.get("institutionCode", "").upper(), r.get("catalogNumber") or str(r["key"]))
            if m and lot not in lots and (not institutions or lot[0] in institutions):
                lots[lot] = {"species": name, "institution": lot[0], "catalog_number": lot[1],
                             "country": r.get("countryCode", ""), "state": r.get("stateProvince", ""),
                             "county": r.get("county", ""), "locality": r.get("locality", ""),
                             "year": r.get("year", ""), "gbif_occurrence": r["key"],
                             "image_url": m["identifier"], "license": m.get("license", ""),
                             "rights_holder": m.get("rightsHolder", ""), "creator": m.get("creator", "")}
        offset += 300
        if d["endOfRecords"] or offset >= d["count"]:
            return list(lots.values())


def choose(found: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` lots, taken in turn from each institution so that no one collection's
    preservation and photography stands for the species."""
    by = defaultdict(list)
    for s in sorted(found, key=lambda s: (s["institution"], s["catalog_number"])):
        by[s["institution"]].append(s)
    rng = random.Random(seed)
    for lots in by.values():
        rng.shuffle(lots)
    out, queues = [], [by[i] for i in sorted(by)]
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop())
    return out


def size_of(url: str) -> str:
    try:
        return get(url, method="HEAD").get("Content-Length") or ""
    except (urllib.error.URLError, TimeoutError):
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harvest_gbif_photos")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--species", help="species, comma-separated")
    src.add_argument("--fetch", type=Path, metavar="MANIFEST",
                     help="fetch exactly the photographs a dry run listed, choosing none afresh")
    ap.add_argument("--per-species", type=int, default=60)
    ap.add_argument("--institutions", default="",
                    help="only these collections, comma-separated (GBIF institution codes); "
                         "some publish only thumbnails, radiographs or label photographs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true", help="list and size the photographs, fetch none")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.fetch:
        with open(args.fetch, newline="") as fh:
            rows = list(csv.DictReader(fh))
    institutions = {i.strip().upper() for i in args.institutions.split(",") if i.strip()}
    for name in [s.strip() for s in (args.species or "").split(",") if s.strip()]:
        found = specimens(name, institutions)
        chosen = choose(found, args.per_species, args.seed)
        print(f"{name}: {len(found)} lots photographed, {len(chosen)} chosen "
              f"({', '.join(sorted({s['institution'] for s in chosen}))})", flush=True)
        for i, s in enumerate(chosen, 1):
            rows.append(dict(s, fish_id=f"{name.replace(' ', '_')}_{i}", bytes=size_of(s["image_url"])))

    if not args.dry_run:
        (args.out / "lateral").mkdir(exist_ok=True)
        for n, r in enumerate(rows, 1):
            if n % 50 == 0:
                print(f"  {n}/{len(rows)}", flush=True)
            dest = args.out / "lateral" / f"{r['fish_id']}.jpg"
            if (dest.is_file() and dest.stat().st_size > 0) or (args.out / "omitted" / dest.name).is_file():
                continue                       # fetched already, or looked at and set aside
            try:
                data = get(r["image_url"])
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"  ! {r['fish_id']}: {exc}; not fetched", file=sys.stderr)
                continue
            if not data.startswith(b"\xff\xd8"):
                print(f"  ! {r['fish_id']}: not a JPEG; not saved", file=sys.stderr)
                continue
            dest.write_bytes(data)
            r["bytes"] = str(len(data))

    with open(args.out / "photo_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in MANIFEST} for r in rows)
    known = [int(r["bytes"]) for r in rows if r["bytes"]]
    if known:
        est = sum(known) / len(known) * len(rows)
        print(f"{len(rows)} photographs; size known for {len(known)}: {sum(known) / 1e6:.0f} MB "
              f"(about {est / 1e6:.0f} MB for all)")
    print(f"wrote {args.out / 'photo_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
