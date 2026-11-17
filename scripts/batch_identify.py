#!/usr/bin/env python3
"""Batch identification by script: the scores the labeler's batch page shows.

Batch identification compares each fish in a photograph of several -- a museum
lot on a tray -- with a reference set of identified specimens landmarked by hand,
and scores how typical the fish's shape is of each reference species
(``fish_morpho.species_match``). The batch page in the labeler shows these
scores. This gives the same ones from the same code (``fish_morpho.batch_score``),
so a result read off the page can be reproduced without it.

Scoring a study's batch photographs::

    python scripts/batch_identify.py --study data/shiners --out results/shiners/batch

Every photograph in ``<study>/batch/`` whose fish have been found and landmarked
(kept in ``<study>/batch_cache/``) is scored against the reference study: the one
``schema.json`` names as ``reference_study``, else the study itself, or
``--reference``. As on the page, points a person has put right in the batch study
(``<study>_batch``) replace the model's; the scale is the one set by hand
(``batch_scale.json``), else the one read from the ruler's numbers; the longest
each species is recorded to reach weighs against a fish longer than that; the
characters read from the photograph enter beside shape; and the species museums
have recorded from the lot's drainage (``batch_drainage.json``) are ranked first.
Writes into ``--out``:

    batch_scores.csv   one row per fish: its photograph and number, standard
                       length, whether its points were checked by hand, the best
                       and second species and their scores, the species tied with
                       the best, whether it is like no species, the landmarks left
                       out as misplaced, and whether the drainage ordered the list
    batch_scores.json  everything the page is given, per photograph
    RERUN.txt          the command that wrote them

Finding and landmarking the fish is Auto-label's work. A photograph not yet
landmarked is skipped and listed; with ``--detect`` it is landmarked first, as the
page would, where Auto-label is installed. Landmarks once kept are scored as kept,
even if the landmark model has changed since (the page would landmark the fish
again), so a rerun gives the same scores.

Leave-one-out on the reference set::

    python scripts/batch_identify.py loo --reference data/shiners --out results/shiners/loo

Each reference fish is identified from its own hand-placed landmarks, its ruler
and the characters read from its photograph, against all the other reference
fish -- as a batch fish whose points were put right is. A fish is tested when its
species keeps ``species_match.MIN_FISH`` reference fish without it. Reports how
often its own species comes first and in the first two, overall and per species,
and the share of fish whose own species scores under 0.25: a quarter, if the
scores are the typicalities they claim to be. ``--axes 4 8 12`` repeats it with
that many within-species shape axes in place of ``species_match.AXES``. Writes
``loo.csv`` (one row per fish), ``loo.json`` (with the summaries) and
``RERUN.txt`` into ``--out``.

Both write what the page writes: the characters read from a batch photograph are
kept in its cache, and those of reference fish whose landmarks changed are read
again into ``photo_characters.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho import batch_score, species_match  # noqa: E402
from fish_morpho.provenance import write_rerun  # noqa: E402

SCRIPT = "scripts/batch_identify.py"
#: An own-species score under this is in the bottom quarter of a typicality.
LOW = 0.25

SCORE_COLUMNS = ["photo", "fish", "sl_mm", "checked", "best_species", "best_score",
                 "second_species", "second_score", "tied_with", "no_match", "left_out",
                 "drainage_ranked", "note"]
LOO_COLUMNS = ["axes", "fish_id", "species", "tested", "rank", "own_score", "best_species",
               "best_score", "second_species", "second_score", "no_match", "left_out", "note"]


# --------------------------------------------------------------------- scoring

def _autolabel():
    """The labeler, whose model finds and landmarks fish, or None where Auto-label
    is not installed."""
    sys.path.insert(0, str(_ROOT / "scripts"))
    import label_server

    return label_server if label_server.autolabel_installed() else None


def score_row(photo: str, f: dict) -> dict:
    """One fish of ``score_photo``'s answer, as a row of ``batch_scores.csv``."""
    m = f["match"]
    ranked = m.get("matches") or []
    best = ranked[0] if ranked else {}
    second = ranked[1] if len(ranked) > 1 else {}
    return {"photo": photo, "fish": f["n"], "sl_mm": f.get("sl_mm"), "checked": f["checked"],
            "best_species": best.get("species"), "best_score": best.get("score"),
            "second_species": second.get("species"), "second_score": second.get("score"),
            "tied_with": ";".join(s for s in m.get("tied") or [] if s != best.get("species")),
            "no_match": bool(m.get("no_match")),
            "left_out": ";".join(x["landmark"] for x in m.get("left_out") or []),
            "drainage_ranked": bool(m.get("drainage_ranked")),
            "note": None if m.get("ok") else m.get("error")}


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r[k] for k in columns})


def score(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="batch_identify",
        description="Score the fish of a study's batch photographs against its reference set, "
                    "as the labeler's batch page does. 'batch_identify.py loo --help' for the "
                    "leave-one-out check of a reference set.")
    ap.add_argument("--study", type=Path, required=True,
                    help="The study whose batch/ photographs are scored.")
    ap.add_argument("--reference", type=Path, default=None,
                    help="The reference study (default: the one the study's schema.json names "
                         "as reference_study, else the study itself).")
    ap.add_argument("--photo", action="append", default=None, metavar="NAME",
                    help="Only this photograph of batch/ (repeat for more).")
    ap.add_argument("--detect", action="store_true",
                    help="Find and landmark the fish of photographs not yet landmarked, as the "
                         "batch page does. Needs Auto-label.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    study = args.study
    if not (study / "batch").is_dir():
        print(f"{study} has no batch/ folder of photographs", file=sys.stderr)
        return 1
    ref_dir = args.reference or batch_score.reference_dir(study, study.parent)
    taxonomy = batch_score.taxonomy_cache(study)
    photos = batch_score.photos(study / "batch")
    if args.photo:
        missing = [n for n in args.photo if n not in photos]
        if missing:
            print(f"not in {study / 'batch'}: {', '.join(missing)}", file=sys.stderr)
            return 1
        photos = {n: photos[n] for n in args.photo}

    model, results, rows, skipped = None, [], [], []
    for name, path in photos.items():
        doc = batch_score.read_cache(study, name)
        if not doc and args.detect:
            if model is None:
                model = _autolabel() or False
            if model:
                print(f"{name}: finding and landmarking the fish ...")
                key = batch_score.cache_key(path, model.current_model_label("lateral"))
                doc, failed = batch_score.landmark_photo(study, name, ref_dir, model.Predictor.detect,
                                                         model.Predictor.predict, key)
                if failed:
                    print(f"  ! {failed.get('error') or 'the model could not look for fish'}")
        if not doc:
            skipped.append(name)
            continue
        recorded = (doc.get("key") or {}).get("size")
        if recorded is not None and recorded != path.stat().st_size:
            print(f"  ! {name} has changed since its fish were landmarked; scored on the landmarks kept")
        batch_score.with_ruler(study, name, doc)
        res = batch_score.score_photo(study, name, doc, ref_dir, taxonomy)
        res["landmarked_with"] = doc.get("key")
        results.append(res)
        scale = (f"{res['px_per_mm']:.2f} px/mm, "
                 f"{'set by hand' if res['scale_from'] == 'hand' else 'read from the ruler'}"
                 if res["px_per_mm"] else "no scale")
        print(f"{name}: {len(res['fish'])} fish, {scale}")
        for f in res["fish"]:
            row = score_row(name, f)
            rows.append(row)
            shown = (f"{row['best_species']} {row['best_score']:.2f}" if row["best_species"]
                     else f"no score: {row['note']}")
            if row["second_species"]:
                shown += f", {row['second_species']} {row['second_score']:.2f}"
            flags = [w for w, on in (("checked", row["checked"]), ("like none", row["no_match"]),
                                     ("drainage", row["drainage_ranked"])) if on]
            sl = f"{row['sl_mm']:.1f} mm" if row["sl_mm"] else "-"
            print(f"  {f['n']:>2}  {sl:>9}  {shown}" + (f"  [{', '.join(flags)}]" if flags else ""))

    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "batch_scores.csv", SCORE_COLUMNS, rows)
    (args.out / "batch_scores.json").write_text(json.dumps(
        {"study": study.name, "reference": Path(ref_dir).name, "photos": results,
         "not_landmarked": skipped}, indent=1))
    if skipped:
        why = ("Auto-label is not installed (scripts/install_autolabel.py)" if args.detect and model is False
               else "--detect finds and landmarks them, with Auto-label" if not args.detect
               else "the model could not landmark them")
        print(f"\n{len(skipped)} photograph(s) not landmarked yet, skipped: {why}")
        for n in skipped:
            print(f"  {n}")
    print(f"\nwrote {args.out / 'batch_scores.csv'}  ({len(rows)} fish in {len(results)} photographs, "
          f"against {Path(ref_dir).name})")
    print(f"      {args.out / 'batch_scores.json'}")
    print(f"to reproduce: {write_rerun(args.out, SCRIPT, argv)}")
    return 0


# --------------------------------------------------------------- leave-one-out

@contextmanager
def _axes(k: int):
    """``species_match`` measuring distance in ``k`` within-species axes, for a while."""
    was = species_match.AXES
    species_match.AXES = k
    try:
        yield
    finally:
        species_match.AXES = was


def leave_one_out(ref: list, sizes: dict | None = None) -> list[dict]:
    """Each reference fish identified against all the others, from its own
    landmarks, ruler and photograph characters: one row per fish."""
    counts: dict[str, int] = {}
    for r in ref:
        counts[r.species] = counts.get(r.species, 0) + 1
    rows = []
    for r in ref:
        row = {"fish_id": r.fid, "species": r.species, "tested": counts[r.species] - 1 >= species_match.MIN_FISH,
               "rank": None, "own_score": None, "best_species": None, "best_score": None,
               "second_species": None, "second_score": None, "no_match": False, "left_out": "", "note": None}
        if not row["tested"]:
            row["note"] = f"too few other {r.species} ({counts[r.species] - 1})"
            rows.append(row)
            continue
        sl_px = species_match.standard_length_px(r.kp)
        ppm = sl_px / r.sl_mm if sl_px and r.sl_mm else None
        m = species_match.identify(r.kp, [o for o in ref if o is not r], ppm,
                                   sizes=sizes, query_chars=r.chars)
        if not m.get("ok"):
            row["note"] = m.get("error")
            rows.append(row)
            continue
        ranked = m["matches"]
        order = [x["species"] for x in ranked]
        if r.species in order:
            row["rank"] = order.index(r.species) + 1
            row["own_score"] = ranked[row["rank"] - 1]["score"]
        else:
            row["note"] = "its own species was not scored"
        row.update(best_species=ranked[0]["species"], best_score=ranked[0]["score"],
                   second_species=ranked[1]["species"] if len(ranked) > 1 else None,
                   second_score=ranked[1]["score"] if len(ranked) > 1 else None,
                   no_match=bool(m.get("no_match")),
                   left_out=";".join(x["landmark"] for x in m.get("left_out") or []))
        rows.append(row)
    return rows


def _rates(rows: list[dict]) -> dict:
    n = len(rows)
    top1 = sum(1 for r in rows if r["rank"] == 1)
    top2 = sum(1 for r in rows if r["rank"] in (1, 2))
    scored = [r for r in rows if r["own_score"] is not None]
    low = sum(1 for r in scored if r["own_score"] < LOW)
    return {"fish": n, "top1": top1, "top2": top2,
            "top1_rate": round(top1 / n, 3) if n else None,
            "top2_rate": round(top2 / n, 3) if n else None,
            "own_scored": len(scored), "own_under_0.25": low,
            "own_under_0.25_share": round(low / len(scored), 3) if scored else None}


def summarise(rows: list[dict]) -> dict:
    """Top-1 and top-2 rates and the share of low own-species scores, over the
    tested fish and per species."""
    tested = [r for r in rows if r["tested"]]
    species = sorted({r["species"] for r in tested})
    return {"overall": _rates(tested),
            "species": {sp: _rates([r for r in tested if r["species"] == sp]) for sp in species},
            "not_tested": sorted({r["species"] for r in rows if not r["tested"]})}


def _pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}% ({k} of {n})" if n else "-"


def loo(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="batch_identify loo",
        description="Leave-one-out on a reference set: each reference fish identified from its "
                    "own landmarks against all the others.")
    ap.add_argument("--reference", type=Path, required=True, help="The reference study.")
    ap.add_argument("--axes", type=int, nargs="+", default=None, metavar="K",
                    help=f"Within-species shape axes to measure distance in (default "
                         f"{species_match.AXES}, species_match.AXES); several run it once each.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    ref = batch_score.reference_fish(args.reference)
    if not ref:
        print(f"no labelled reference fish in {args.reference}", file=sys.stderr)
        return 1
    _, sizes = batch_score.species_sizes((r.species for r in ref), batch_score.taxonomy_cache(args.reference))
    runs, all_rows = [], []
    for k in args.axes or [species_match.AXES]:
        with _axes(k):
            rows = leave_one_out(ref, sizes)
        summary = summarise(rows)
        runs.append({"axes": k, "summary": summary, "fish": rows})
        all_rows += [{"axes": k, **r} for r in rows]
        o = summary["overall"]
        print(f"{k} axes, {o['fish']} fish of {len(summary['species'])} species "
              f"(of {len(ref)} reference fish):")
        print(f"  own species first        {_pct(o['top1'], o['fish'])}")
        print(f"  own species in first two {_pct(o['top2'], o['fish'])}")
        print(f"  own-species score < {LOW}  {_pct(o['own_under_0.25'], o['own_scored'])}")
        width = max(len(sp) for sp in summary["species"]) if summary["species"] else 10
        print(f"  {'species':<{width}}  fish  first  first two  own < {LOW}")
        for sp, s in summary["species"].items():
            print(f"  {sp:<{width}}  {s['fish']:>4}  {s['top1']:>5}  {s['top2']:>9}  "
                  f"{s['own_under_0.25']:>4} of {s['own_scored']}")
        if summary["not_tested"]:
            print(f"  not tested (too few other fish of the species): {', '.join(summary['not_tested'])}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "loo.csv", LOO_COLUMNS, all_rows)
    (args.out / "loo.json").write_text(json.dumps(
        {"reference": args.reference.name, "reference_fish": len(ref), "min_fish": species_match.MIN_FISH,
         "low": LOW, "runs": runs}, indent=1))
    print(f"wrote {args.out / 'loo.csv'}")
    print(f"      {args.out / 'loo.json'}")
    print(f"to reproduce: {write_rerun(args.out, SCRIPT, ['loo', *argv])}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["loo"]:
        return loo(argv[1:])
    return score(argv)


if __name__ == "__main__":
    raise SystemExit(main())
