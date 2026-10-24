"""Choosing museum photographs for a species' reference set, without the network.

Which photograph of a record is taken (the preserved specimen's, lateral where the
titles say, never a fresh tissue voucher, a field image or an unlicensed one); how
lots are drawn in turn from each institution, the same way every time; and that
``--fetch`` takes exactly the photographs a dry run listed.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
spec = importlib.util.spec_from_file_location("harvest_gbif_photos", SCRIPTS / "harvest_gbif_photos.py")
hg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hg)

CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"


def image(url, title=None, description=None, licence=CC0):
    return {"type": "StillImage", "identifier": url, "title": title, "description": description,
            "license": licence}


def test_the_photograph_taken_is_the_preserved_specimen_seen_from_the_side():
    fmnh = {"institutionCode": "FMNH", "media": [
        image("http://f/IMG_1196.JPG", "131534_lat_FZ", "left lateral view with scale of tissue voucher"),
        image("http://f/131534_dor.jpg", "131534_dor_FMNH"),
        image("http://f/131534_lat_FMNH_FZ.jpg", "131534_lat_FMNH_FZ")]}
    assert hg.photograph(fmnh)["identifier"] == "http://f/131534_lat_FMNH_FZ.jpg"
    assert hg.photograph({"institutionCode": "INHS", "media": [image("http://i/1.jpg")]})
    assert hg.photograph({"institutionCode": "NEON", "media": [image("http://n/1.jpg")]}) is None
    assert hg.photograph({"institutionCode": "X", "media": [image("http://www.boldsystems.org/1.jpg")]}) is None
    assert hg.photograph({"institutionCode": "X", "media": [image("http://x/1.jpg", licence="")]}) is None


def test_lots_are_drawn_in_turn_from_each_institution_and_the_same_way_every_time():
    found = ([{"institution": "FMNH", "catalog_number": str(i)} for i in range(50)]
             + [{"institution": "INHS", "catalog_number": str(i)} for i in range(5)]
             + [{"institution": "CMN", "catalog_number": "1"}])
    chosen = hg.choose(found, 12, seed=1)
    assert [c["institution"] for c in chosen].count("CMN") == 1
    assert [c["institution"] for c in chosen].count("INHS") == 5        # all it has
    assert len(chosen) == 12 and len({(c["institution"], c["catalog_number"]) for c in chosen}) == 12
    assert chosen == hg.choose(list(reversed(found)), 12, seed=1)
    assert len(hg.choose(found, 100, seed=1)) == len(found)


def test_fetch_takes_exactly_what_the_dry_run_listed(tmp_path, monkeypatch):
    manifest = tmp_path / "listed.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hg.MANIFEST)
        w.writeheader()
        w.writerow({"fish_id": "Notropis_rubellus_1", "species": "Notropis rubellus",
                    "image_url": "http://i/a.jpg", "bytes": "9"})
        w.writerow({"fish_id": "Notropis_rubellus_2", "species": "Notropis rubellus",
                    "image_url": "http://i/b.png", "bytes": "9"})
    asked = []

    def get(url, method="GET", tries=4):
        asked.append(url)
        return b"\xff\xd8 jpeg" if url.endswith(".jpg") else b"\x89PNG"

    monkeypatch.setattr(hg, "get", get)
    monkeypatch.setattr(hg, "specimens", lambda *a: (_ for _ in ()).throw(AssertionError("chose afresh")))
    out = tmp_path / "study"
    assert hg.main(["--fetch", str(manifest), "--out", str(out)]) == 0
    assert asked == ["http://i/a.jpg", "http://i/b.png"]
    assert (out / "lateral" / "Notropis_rubellus_1.jpg").read_bytes() == b"\xff\xd8 jpeg"
    assert not (out / "lateral" / "Notropis_rubellus_2.jpg").exists()       # not a JPEG
    rows = list(csv.DictReader(open(out / "photo_manifest.csv")))
    assert [r["fish_id"] for r in rows] == ["Notropis_rubellus_1", "Notropis_rubellus_2"]
    assert rows[0]["bytes"] == str(len(b"\xff\xd8 jpeg"))
    asked.clear()
    hg.main(["--fetch", str(manifest), "--out", str(out)])
    assert asked == ["http://i/b.png"]                                       # the saved one is kept
    (out / "omitted").mkdir()
    (out / "lateral" / "Notropis_rubellus_1.jpg").rename(out / "omitted" / "Notropis_rubellus_1.jpg")
    asked.clear()
    hg.main(["--fetch", str(manifest), "--out", str(out)])
    assert asked == ["http://i/b.png"]                                       # set aside stays aside
