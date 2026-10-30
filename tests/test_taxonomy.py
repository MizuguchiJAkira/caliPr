"""A name followed to the species it means now, with the names on the way.

GBIF and FishBase are stood in for by a table of answers: a label's synonym goes
to GBIF's accepted species and on to FishBase's current name; a name filed under
two species goes to neither; a subspecies is never looked up by its first two
words; nothing the web could not answer is kept; what it did answer is not asked
again.
"""

from __future__ import annotations

import json

import pytest

from fish_morpho.taxonomy import Names, spellings

MATCH = {  # name GBIF is asked -> (usageKey, acceptedUsageKey)
    "Hybognathus volucellus": (11, 2362211), "Notropis volucellus": (2362211, None),
    "Notropis cornutus chrysocephalus": (12, 2366993),
}
SPECIES = {2362211: "Notropis volucellus", 2366993: "Luxilus chrysocephalus"}
FAMILY = "<a href='../Summary/FamilySummary.php?ID=756' class='sciname' title='Family'>Leuciscidae</a>"
FISHBASE = {"Notropis-volucellus": "Paranotropis volucellus,  Mimic shiner",
            "Luxilus-chrysocephalus": "Luxilus chrysocephalus,  Striped shiner : fisheries"}


def fake_web(asked):
    def get(url):
        asked.append(url)
        if "/species/match?" in url:
            name = url.split("name=")[1].split("&")[0].replace("+", " ")
            k = MATCH.get(name)
            return json.dumps({"matchType": "EXACT", "usageKey": k[0], "acceptedUsageKey": k[1]} if k
                              else {"matchType": "NONE"})
        if "/species/" in url:
            return json.dumps({"canonicalName": SPECIES[int(url.rsplit("/", 1)[1])]})
        page = url.rsplit("/", 1)[1][:-5]
        return f"<html><title>{FISHBASE[page]}</title>{FAMILY}</html>" if page in FISHBASE else "<title>FishBase</title>"
    return get


def test_a_synonym_goes_to_the_species_and_on_to_its_current_name():
    r = Names(None, get=fake_web([])).resolve("Hybognathus_volucellus")
    assert r["current"] == "Paranotropis volucellus" and r["common"] == "Mimic shiner"
    assert [c["name"] for c in r["chain"]] == ["Hybognathus volucellus", "Notropis volucellus", "Paranotropis volucellus"]
    assert r["gbif_key"] == 2362211 and r["found"] and r["family"] == "Leuciscidae"


def test_a_subspecies_is_not_taken_for_its_first_two_words():
    asked = []
    r = Names(None, get=fake_web(asked)).resolve("Notropis cornutus chrysocephalus")
    assert r["current"] == "Luxilus chrysocephalus"
    assert not any("Notropis-cornutus" in u for u in asked)


def test_a_name_used_for_two_species_means_neither():
    r = Names(None, get=fake_web([])).resolve("Notropis cayuga")
    assert r["ambiguous"] and r["current"] is None


def test_an_unknown_name_is_said_to_be_unknown():
    r = Names(None, get=fake_web([])).resolve("Notropis nonesuch")
    assert not r["found"] and r["current"] is None


def test_answers_are_kept_and_failures_are_not(tmp_path):
    asked = []
    cache = tmp_path / "names.json"
    Names(cache, get=fake_web(asked)).resolve("Notropis volucellus")
    n = len(asked)
    assert Names(cache, get=fake_web(asked)).resolve("Notropis volucellus")["current"] == "Paranotropis volucellus"
    assert len(asked) == n                                    # from the cache

    def down(url):
        raise OSError("no network")
    r = Names(cache, get=down).resolve("Notropis atherinoides")
    assert r["offline"] and "notropis atherinoides" not in json.loads(cache.read_text())


@pytest.mark.parametrize("name, also", [("Notropis cornuta", "Notropis cornutus"),
                                        ("Hybopsis volucella", "Hybopsis volucellus")])
def test_other_genders_of_an_epithet_are_tried(name, also):
    assert also in spellings(name)
