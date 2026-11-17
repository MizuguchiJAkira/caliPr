"""The fish of a photograph of several, scored against a reference set.

What the labeller's batch page shows, worked out by one set of functions that both
the page (``scripts/label_server.py``) and the command line
(``scripts/batch_identify.py``) call, so a score read off the page can be had again
by script and the two cannot come to disagree.

A study keeps its photographs of several fish in ``batch/``. Finding the fish and
placing their landmarks is the model's work, slow, and kept beside the photograph
in ``batch_cache/<photo>.json`` (``landmark_photo``). Everything else is worked
out again each time (``score_photo``): the reference fish, a scale set by hand
(``batch_scale.json``) or read from the ruler's numbers, the longest each species
is recorded to reach, the characters read from the photograph, the points a
person has put right in the batch study beside it (``<study>_batch``), and the
species recorded from the lot's drainage (``batch_drainage.json``).

The scoring itself is ``species_match``'s; this gathers what it is given.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fish_morpho import paths

#: What a batch photograph's kept fish hold; older ones are found and landmarked
#: again. 2: each point's confidence, which the labeller reviews by.
CACHE_VERSION = 2
#: Two boxes overlapping this much (intersection over union) are the same fish.
SAME_FISH = 0.7
#: Photographs, as the labeller lists them (``label_server.IMAGE_SUFFIXES``).
_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
#: Reads the characters from a study's own photographs.
CHARACTERS_SCRIPT = paths.CODE / "scripts" / "photo_characters.py"


def _read_json(path: Path):
    """A JSON file's contents, or None if it is missing or unreadable."""
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except Exception:
        return None


def photos(directory: Path) -> dict[str, Path]:
    """The photographs in ``directory``, by filename; empty if it is missing."""
    if not directory.is_dir():
        return {}
    return {p.name: p for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in _PHOTO_SUFFIXES}


def _settings(study: Path) -> dict:
    """What batch identification reads from a study's ``schema.json``, as the
    labeller's ``load_profile`` reads it."""
    prof = _read_json(Path(study) / "schema.json")
    prof = prof if isinstance(prof, dict) else {}
    return {"reference_study": str(prof.get("reference_study") or ""),
            "reference_set": bool(prof.get("reference_set")),
            "terminal_mouth": [g for g in (prof.get("terminal_mouth") or []) if isinstance(g, str)]}


# ----------------------------------------------------------------- where things are

def reference_dir(study: Path, data_root: Path | None = None, datasets: dict | None = None) -> Path:
    """The study whose hand-labelled fish the batch is compared with: the one
    schema.json names as ``reference_study`` -- a study the labeller knows by that
    name (``datasets``), or a folder of that name in ``data_root`` -- or this one."""
    name = _settings(study)["reference_study"]
    if not name:
        return Path(study)
    beside = Path(data_root) / name if data_root is not None and "/" not in name else None
    return (datasets or {}).get(name) or (beside if beside and beside.is_dir() else Path(study))


def batch_study_dir(study: Path) -> Path:
    """Where a study's batch fish go to have their points put right: a study of
    their own beside it. What is corrected there scores the fish and trains the
    landmark model, and never becomes a reference fish."""
    study = Path(study)
    return study.parent / f"{study.name}_batch"


def taxonomy_cache(study: Path, data_root: Path | None = None) -> Path:
    """One cache of resolved names for every study: a name means the same fish in each."""
    return Path(data_root or Path(study).parent) / "taxonomy_cache.json"


def cache_path(study: Path, name: str) -> Path:
    """Where batch photograph ``name``'s fish and their landmarks are kept."""
    return Path(study) / "batch_cache" / f"{Path(name).stem}.json"


def read_cache(study: Path, name: str) -> dict | None:
    """Batch photograph ``name``'s kept fish, or None while it has none."""
    return _read_json(cache_path(study, name))


def cache_key(photo: Path, model: str | None) -> dict:
    """What the kept fish of ``photo`` were found from: the photograph as it was,
    the landmark model, and what a kept fish holds. Any of them changed, the
    labeller finds and landmarks the fish again."""
    st = Path(photo).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "model": model, "version": CACHE_VERSION}


# --------------------------------------------------------- finding and landmarking

def overlap(a, b) -> float:
    """Intersection over union of two [x0, y0, x1, y1] boxes; 0 for anything else."""
    try:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0
    except (TypeError, IndexError):
        return 0.0


def isolate_fish(photo: Path, fish: list[dict], i: int, out: Path) -> None:
    """Fish ``i``'s rectangle of a batch photograph, with every other fish in it
    painted over in the background's colour, so the landmark model -- which looks
    for one fish -- cannot take a point from the fish beside it."""
    import numpy as np
    from PIL import Image, ImageDraw

    with Image.open(photo) as im:
        im = im.convert("RGB")
        x0, y0, x1, y1 = fish[i]["box"]
        crop = im.crop((x0, y0, x1, y1))
    w, h = crop.size
    shift = lambda poly: [(x - x0, y - y0) for x, y in poly]  # noqa: E731
    pad = max(2, round(0.01 * max(w, h)))                     # fins reach past the outline
    every = Image.new("L", (w, h), 0)
    draw_every = ImageDraw.Draw(every)
    for f in fish:
        draw_every.polygon(shift(f["outline"]), fill=255, outline=255, width=pad)
    px = np.asarray(crop)
    bg = tuple(int(v) for v in np.median(px[np.asarray(every) == 0], axis=0)) \
        if (np.asarray(every) == 0).any() else (255, 255, 255)
    others = Image.new("L", (w, h), 0)
    draw_others = ImageDraw.Draw(others)
    for j, f in enumerate(fish):
        if j != i:
            draw_others.polygon(shift(f["outline"]), fill=255, outline=255, width=pad)
    mine = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mine).polygon(shift(fish[i]["outline"]), fill=255, outline=255, width=pad)
    hide = np.logical_and(np.asarray(others) > 0, np.asarray(mine) == 0)
    px = px.copy()
    px[hide] = bg
    Image.fromarray(px).save(out, quality=92)


def read_ruler(photo: Path) -> dict:
    """The photograph's scale, from the numbers along its ruler: ``{"px_per_mm",
    "numbers", "notes"}``, or ``{"error"}`` where none is found."""
    try:
        import cv2

        from fish_morpho.ruler_calibration import detect_numbered_scale

        r = detect_numbered_scale(cv2.imread(str(photo)))
        return {"px_per_mm": r["px_per_mm"], "numbers": r["numbers"], "notes": r["notes"]}
    except Exception as exc:                       # no numbered ruler: the scale is set by hand
        return {"error": str(exc)}


def landmark_photo(study: Path, name: str, ref_dir: Path, detect, predict, key: dict) -> tuple[dict | None, dict | None]:
    """Find every fish in batch photograph ``name`` and place its landmarks --
    the ones the reference study collects -- each on its own rectangle with the
    fish beside it painted out. ``detect`` and ``predict`` are the landmark
    model's (the labeller's ``Predictor``); ``key`` is ``cache_key``'s.

    Kept in ``cache_path``, and answered as ``(doc, None)``; or ``(None, answer)``
    with the model's answer when it could not look for fish."""
    from fish_morpho import schemes

    path = Path(study) / "batch" / name
    cache = cache_path(study, name)
    found = detect(path)
    if not found.get("ok"):
        return None, found
    wanted = list(schemes.study_landmarks(ref_dir)[0])
    crops = cache.parent / path.stem
    crops.mkdir(parents=True, exist_ok=True)
    fish = []
    for i, f in enumerate(found["fish"]):
        crop_path = crops / f"fish_{i + 1}.jpg"
        isolate_fish(path, found["fish"], i, crop_path)
        res = predict(crop_path, polygons=False, emit_polygons=False, landmarks=wanted)
        x0, y0 = f["box"][:2]
        kps = {n: [p[0] + x0, p[1] + y0] for n, p in (res.get("keypoints") or {}).items() if p}
        fish.append({**f, "keypoints": kps,
                     "confidence": res.get("confidence") or {},
                     "low_confidence": res.get("low_confidence") or [],
                     "error": None if res.get("ok") else res.get("error")})
    doc = {"key": key, "width": found["width"], "height": found["height"], "fish": fish}
    cache.write_text(json.dumps(doc))
    return doc, None


def with_ruler(study: Path, name: str, doc: dict) -> dict:
    """``doc`` with the photograph's ruler read, once, and kept with it."""
    if "ruler" not in doc:
        doc["ruler"] = read_ruler(Path(study) / "batch" / name)
        cache_path(study, name).write_text(json.dumps(doc))
    return doc


# ------------------------------------------------------------- what is compared

def corrections(study: Path, name: str, doc: dict) -> list[tuple]:
    """For each fish found in batch photograph ``name``: its id in the batch
    study and its saved points there, in the photograph's coordinates -- or
    ``(None, None)``, ``(id, None)`` while it has none. A fish is known by where
    it lies, not by its number, so finding the fish again cannot hand one
    fish's corrected points to another."""
    bdir = batch_study_dir(study)
    mine = {fid: e for fid, e in (_read_json(bdir / "batch_fish.json") or {}).items()
            if isinstance(e, dict) and e.get("photo") == name}
    out = []
    for f in doc["fish"]:
        fid = next((k for k, e in mine.items() if overlap(e.get("box"), f["box"]) >= SAME_FISH), None)
        fixed = None
        if fid:
            kp = ((_read_json(bdir / "sidecars" / f"{fid}.json") or {}).get("lateral") or {}).get("keypoints")
            if kp:
                x0, y0 = mine[fid]["box"][:2]
                fixed = {n: [p[0] + x0, p[1] + y0] for n, p in kp.items() if p}
        out.append((fid, fixed))
    return out


def refresh_characters(study: Path, script: Path = CHARACTERS_SCRIPT) -> None:
    """Read the characters again (``scripts/photo_characters.py``) for the fish
    whose labels changed since they were last read, if any did."""
    from fish_morpho.species_match import CHARACTERS_FILE
    out = Path(study) / CHARACTERS_FILE
    try:
        newest = max((f.stat().st_mtime for f in (Path(study) / "sidecars").glob("*.json")), default=0)
    except OSError:
        return
    if out.is_file() and out.stat().st_mtime >= newest:
        return
    subprocess.run([sys.executable, str(script), "--study", str(study)],
                   capture_output=True, text=True, timeout=1800, cwd=paths.CODE)


def fish_characters(study: Path, name: str, doc: dict, landmarks: list) -> list[dict]:
    """The characters read from a batch photograph for each of its fish, at the
    places its landmarks (``landmarks``, one dict per fish) set. Kept in the
    photograph's cache under a fingerprint of those landmarks, so a fish is read
    once until its points change."""
    from fish_morpho import photo_characters as pc
    path = Path(study) / "batch" / name
    img, changed, out = None, False, []
    for f, kps in zip(doc["fish"], landmarks):
        key = pc.landmarks_key(kps)
        if f.get("characters_for") != key:
            if img is None:
                import cv2
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    return [{} for _ in landmarks]
            f["characters"] = pc.characters(img, kps)
            f["characters_for"] = key
            changed = True
        out.append(f.get("characters") or {})
    if changed:
        cache_path(study, name).write_text(json.dumps(doc))
    return out


def reference_fish(ref_dir: Path) -> list:
    """The reference study's hand-labelled fish (``species_match.RefFish``), each
    with its species from its filename or ``groups.csv``. A fish whose photograph
    has left the study is not compared -- except in a downloaded reference set,
    which carries landmarks and credits, no photographs. The characters of fish
    whose labels changed are read again first."""
    from fish_morpho import grouping, species_match

    ref_dir = Path(ref_dir)
    settings = _settings(ref_dir)
    pattern = grouping.filename_pattern(ref_dir)
    table = grouping.load_group_table(ref_dir)
    present = {Path(n).stem for n in photos(ref_dir / "lateral")}
    landmarks_only = settings["reference_set"] and not present
    if present:
        refresh_characters(ref_dir)
    return [r for r in species_match.load_reference(
                ref_dir, lambda fid: grouping.resolve(fid, table=table, pattern=pattern),
                settings["terminal_mouth"])
            if landmarks_only or r.fid in present]


def species_sizes(species, cache: Path) -> tuple[dict, dict]:
    """Each species by its current name and the names it went by (``taxa``), and
    the longest it is recorded to reach (``sizes``, ``species_match.sizes_for``):
    filed under the name the reference uses, or else under its current name."""
    from fish_morpho import species_match, taxonomy

    names = taxonomy.Names(cache)
    taxa = {sp: names.resolve(sp) for sp in sorted(set(species))}
    sizes = species_match.sizes_for(taxa)
    for sp, t in taxa.items():                     # a table entry under the current name will do
        if sp not in sizes and t.get("current"):
            hit = species_match.sizes_for([t["current"]]).get(t["current"])
            if hit:
                sizes[sp] = hit
    return taxa, sizes


# ------------------------------------------------------------------- the scores

def score_photo(study: Path, name: str, doc: dict, ref_dir: Path, taxonomy_cache_path: Path) -> dict:
    """Every fish of batch photograph ``name`` (``doc``: its kept fish), and how
    typical its shape is of each reference species, best first -- the batch page's
    answer, less whether the fish were found just now."""
    from fish_morpho import batch_fish, species_match

    study, ref_dir = Path(study), Path(ref_dir)
    ref = reference_fish(ref_dir)
    # a scale set by hand wins; else the one read from the ruler's numbers
    scale = (_read_json(study / "batch_scale.json") or {}).get(name)
    ruler = doc.get("ruler") or {}
    ppm = species_match.px_per_mm(scale) or ruler.get("px_per_mm")
    scale_from = "hand" if species_match.px_per_mm(scale) else ("ruler" if ruler.get("px_per_mm") else None)
    # each reference species by its current name, and the names it went by
    taxa, sizes = species_sizes((r.species for r in ref), taxonomy_cache_path)
    fish = []
    pairs = list(zip(doc["fish"], corrections(study, name, doc)))
    chars = fish_characters(study, name, doc, [fixed or f["keypoints"] for f, (_, fixed) in pairs])
    for i, (f, (fid, fixed)) in enumerate(pairs):
        kps = fixed or f["keypoints"]
        m = (species_match.identify(kps, ref, ppm, sizes=sizes, query_chars=chars[i])
             if fixed or not f.get("error") else {"ok": False, "error": f["error"]})
        sl_px = species_match.standard_length_px(kps)
        fish.append({"n": i + 1, "box": f["box"], "outline": f["outline"],
                     "keypoints": kps, "low_confidence": [] if fixed else f.get("low_confidence", []),
                     "sl_mm": m.get("sl_mm"),
                     "sl_px": round(sl_px) if sl_px else None,
                     "too_small": bool(sl_px and sl_px < batch_fish.SMALLEST_SL_PX),
                     "fid": fid, "checked": bool(fixed), "match": m, "characters": chars[i]})
    counts: dict[str, int] = {}
    for r in ref:
        counts[r.species] = counts.get(r.species, 0) + 1
    drainage = (_read_json(study / "batch_drainage.json") or {}).get(name)
    if drainage:
        by_key = {s["key"]: s for s in drainage["species"]}
        keys = drainage.get("reference_keys") or {}
        # the species recorded from the drainage first: see species_match.with_drainage
        records = {sp: (by_key.get(keys.get(sp)) or {}).get("records", 0) for sp in counts}
        for f in fish:
            species_match.with_drainage(f["match"], records)
        mine = {keys.get(sp) for sp in counts}
        shiner_keys = {s["key"] for s in drainage["species"]}
        intro = drainage.get("introduced") or {}
        fishes = [{**f, "reference": f["key"] in mine, "shiner": f["key"] in shiner_keys,
                   "introduced": intro.get(f["species"]) or intro.get(f["gbif_name"])}
                  for f in drainage.get("fishes") or []]
        # introduced fish USGS records there that no museum specimen on GBIF does
        listed = {f["species"] for f in fishes} | {f["gbif_name"] for f in fishes}
        fishes += [{"key": e.get("gbif_key"), "species": cur, "gbif_name": e["nas_name"], "common": e.get("common"),
                    "family": None, "chain": e.get("chain") or [cur], "records": 0, "reference": False,
                    "shiner": False, "introduced": e, "usgs_only": True}
                   for cur, e in intro.items() if cur not in listed and e["nas_name"] not in listed]
        for f in fish:                             # the reference species' own status there
            for x in f["match"].get("matches") or []:
                t = taxa.get(x["species"]) or {}
                x["introduced"] = intro.get(t.get("current")) or intro.get(x["species"].replace("_", " "))
        drainage = {**{k: v for k, v in drainage.items() if k not in ("species", "reference_keys", "fishes")},
                    "fishes": fishes,
                    "others": sorted(({"species": s.get("current") or s["species"], "gbif_name": s["species"],
                                       "common": s.get("common"), "chain": s.get("chain") or [s["species"]],
                                       "key": s["key"], "records": s["records"]}
                                      for s in drainage["species"] if s["key"] not in mine),
                                     key=lambda s: -s["records"])}
    return {"ok": True, "photo": name, "width": doc["width"], "height": doc["height"],
            "scale": scale, "ruler": ruler, "scale_from": scale_from, "px_per_mm": ppm,
            "drainage": drainage, "taxa": taxa,
            "fish": fish, "smallest_sl_px": batch_fish.SMALLEST_SL_PX,
            "reference": {"study": ref_dir.name, "fish": len(ref), "species": counts}}
