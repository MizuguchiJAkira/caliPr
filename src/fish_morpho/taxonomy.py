"""Which species a name means now, and the names it has carried on the way.

A museum label, a GBIF record and the newest checklist can name one fish three
ways: a lot labelled *Hybopsis volucella*, GBIF's records under *Notropis
volucellus*, and the species moved to *Paranotropis* by Stout et al. (2022) and
the AFS list (Page et al. 2023). Every name is followed along that chain to one
current name, so that whatever a photograph, a reference fish or a drainage
record calls a species, it is counted as the same one -- and the names it went
by are kept, to show.

- **GBIF's backbone** takes a name to the species it is a synonym of, and gives
  the key GBIF files that species' specimen records under (its maps and
  occurrence searches take that key, synonyms included).
- **FishBase** gives the current valid name. It follows Eschmeyer's Catalog of
  Fishes and keeps up with revisions the backbone has not taken in yet; its page
  for an old name is the page of the species as it is now called.

A name museums have used for two species (``ambiguous_names`` in
``species_sizes.json``: *Notropis cayuga*, say) is resolved to neither, and said
to be ambiguous.

Every answer is kept in a JSON cache, so a name is looked up once and the chain
can be followed offline after; an answer the web could not give is not kept, so
it is asked again next time.
"""

from __future__ import annotations

import datetime
import json
import re
import threading
import urllib.parse
import urllib.request
from pathlib import Path

GBIF = "https://api.gbif.org/v1"
FISHBASE = "https://www.fishbase.se/summary/{}.html"
USER_AGENT = "caliPr-taxonomy/1.0 (fish identification aid; research use)"
SIZES = Path(__file__).with_name("species_sizes.json")

_FAMILY = re.compile(r"title='Family'>\s*([A-Z][a-z]+idae)")
_TITLE = re.compile(r"<title>\s*([A-Z][a-z]+ [a-z]+)\s*,\s*([^:<]*?)\s*(?::|</title>)", re.S)


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def binomial(name: str) -> str:
    """"Notropis_cornutus chrysocephalus" -> "Notropis cornutus": genus and species."""
    words = re.sub(r"[_\s]+", " ", str(name)).strip().split(" ")
    return " ".join(words[:2]) if len(words) >= 2 else " ".join(words)


def spellings(name: str) -> list[str]:
    """The name, then with its epithet in the other genders: labels carry
    "Notropis cornuta" and "Hybopsis volucella" for names now spelled -us."""
    words = name.split(" ")
    if len(words) < 2:
        return [name]
    out = [name]
    for a, b in (("a", "us"), ("us", "a"), ("um", "us"), ("us", "um"), ("is", "e"), ("e", "is")):
        if words[-1].endswith(a):
            out.append(" ".join(words[:-1] + [words[-1][: -len(a)] + b]))
    return list(dict.fromkeys(out))


def _table() -> dict:
    try:
        return json.loads(SIZES.read_text())
    except (OSError, ValueError):
        return {}


def _ambiguous() -> set[str]:
    return {n.lower() for n in _table().get("ambiguous_names") or ()}


def _from_table(name: str) -> dict | None:
    """The species in ``species_sizes.json`` that lists ``name`` among the names
    museum labels carry for it -- old spellings neither GBIF nor FishBase keeps."""
    wanted = {s.lower() for s in spellings(name)}
    for e in _table().get("species") or ():
        if wanted & {n.lower() for n in e.get("names") or ()}:
            return e
    return None


class Names:
    """Names resolved to current species, kept in ``cache_path``. ``get``: fetch a
    URL's text (the web by default; a table in tests)."""

    def __init__(self, cache_path: Path | None, get=_get):
        self.path = Path(cache_path) if cache_path else None
        self.get = get
        self.lock = threading.Lock()
        try:
            self.cache = json.loads(self.path.read_text()) if self.path and self.path.is_file() else {}
        except ValueError:
            self.cache = {}

    def resolve(self, name: str) -> dict:
        """``{"given", "current", "common", "family", "gbif_key", "gbif_name", "chain":
        [{"name", "via"}], "ambiguous", "found", "offline"}`` -- ``chain`` from the
        name given to the current one, each step once; ``current`` is None for a name
        neither GBIF nor FishBase knows (``found`` false) or one that is ambiguous."""
        given = re.sub(r"[_\s]+", " ", str(name)).strip()
        key = given.lower()
        with self.lock:
            if key in self.cache and "family" in self.cache[key]:       # older entries lack the family
                return self.cache[key]
        if key in _ambiguous():
            out = {"given": given, "current": None, "common": None, "family": None, "gbif_key": None, "gbif_name": None,
                   "chain": [{"name": given, "via": "label"}], "ambiguous": True, "found": False,
                   "offline": False}
            return self._keep(key, out)
        offline = False
        gbif_key = gbif_name = None
        try:
            for spelled in spellings(given):
                m = json.loads(self.get(f"{GBIF}/species/match?" + urllib.parse.urlencode(
                    {"name": spelled, "kingdom": "Animalia", "class": "Actinopterygii"})))
                if m.get("matchType") not in (None, "NONE", "HIGHERRANK") and m.get("usageKey"):
                    gbif_key = m.get("acceptedUsageKey") or m["usageKey"]
                    acc = json.loads(self.get(f"{GBIF}/species/{gbif_key}"))
                    gbif_name = acc.get("canonicalName") or acc.get("species")
                    break
        except Exception:
            offline = True
        current = common = family = None
        # the name given is asked of FishBase only as a binomial: the first two words of
        # "Notropis cornutus chrysocephalus" are another species
        asked = (gbif_name,) + (tuple(spellings(given)) if len(given.split()) == 2 else ())
        for candidate in dict.fromkeys(n for n in asked if n):
            try:
                page = self.get(FISHBASE.format(binomial(candidate).replace(" ", "-")))
            except Exception:
                offline = True
                continue
            t, f = _TITLE.search(page), _FAMILY.search(page)
            if t:
                current, common = t.group(1), t.group(2).strip() or None
                family = f.group(1) if f else None
                break
        table_name = None
        if not (current or gbif_name):
            e = _from_table(given)
            if e:
                gbif_key, gbif_name, current, table_name = e.get("gbif_key"), e.get("gbif_name"), e["name"], e["name"]
        found = bool(current or gbif_name)
        current = current or gbif_name
        chain = []
        steps = ((given, "label"), (current, "caliPr's table of label names (Gilbert 1964; Scott & Crossman 1973)")) \
            if table_name else ((given, "label"), (gbif_name, "GBIF backbone"), (current, "FishBase (Catalog of Fishes)"))
        for n, via in steps:
            if n and all(binomial(n) != binomial(c["name"]) for c in chain):
                chain.append({"name": n, "via": via})
        out = {"given": given, "current": current, "common": common, "family": family, "gbif_key": gbif_key,
               "gbif_name": gbif_name, "chain": chain, "ambiguous": False, "found": found, "offline": offline}
        return out if offline else self._keep(key, out)

    def _keep(self, key: str, out: dict) -> dict:
        out = {**out, "checked": datetime.date.today().isoformat()}
        with self.lock:
            self.cache[key] = out
            if self.path:
                self.path.write_text(json.dumps(self.cache, indent=1))
        return out
