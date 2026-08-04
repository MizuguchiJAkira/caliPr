"""Batch-preprocess Jonah Cheng's pre-named hatchery trout photos.

Unlike ``preprocess_cornell.py`` (which expects raw ``Img####.JPG`` files
plus a ``specimen_map.csv`` and a single ``--strain``), this batch already
arrives catalog-named across multiple strains, e.g.::

    Salvelinus_fontinalus_ASN_1_L.JPG   (note: "fontinalus" is a typo in
    Salvelinus_fontinalus_HRN_2_L.JPG    the source filenames; corrected to
    Salvelinus_fontinalus_TXD_50_L.JPG   "fontinalis" on output)

So this driver derives ``(strain, specimen_number)`` straight from each
filename and delegates the actual orientation-normalization + mirror-split
to :func:`preprocess_cornell.process_one`, keeping that logic the single
source of truth. Files that don't match the catalog pattern (a handful of
leftover ``Img####.JPG``) are looked up in the photos-info spreadsheet's
``lateral_photos`` column when ``--info-csv`` is supplied, and otherwise
reported and skipped.

Usage::

    python scripts/preprocess_jonah.py \\
        --raw-dir data/cornell_raw/jonah \\
        --out-dir data/cornell \\
        --info-csv data/cornell_raw/jonah_photos_info_v2.csv   # optional
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

# Allow running as a script without installing the package.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import preprocess_cornell as pc  # noqa: E402

log = logging.getLogger("preprocess_jonah")

# Salvelinus_fontinal{us,is}_{STRAIN}_{NUM}_L.JPG (leading/trailing space tolerated)
CATALOG_RE = re.compile(
    r"^\s*[A-Za-z]+_[A-Za-z]+_([A-Za-z]{2,4})_(\d+)_[LlFf]\s*$"
)
IMG_RE = re.compile(r"^\s*(Img\d+)\s*$", re.IGNORECASE)


def load_img_fallback(info_csv: Path) -> dict[str, tuple[str, str]]:
    """Map ``Img#### -> (strain, specimen_number)`` from the photos-info CSV.

    Expects columns ``field_number`` (e.g. ``"ST-HRN LMH 2/24/2025"``),
    ``specimen``, and ``lateral_photos`` (e.g. ``"Img0318"``).
    """
    mapping: dict[str, tuple[str, str]] = {}
    with info_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            img = (row.get("lateral_photos") or "").strip()
            if not img:
                continue
            m = re.search(r"ST-([A-Za-z]{2,4})", row.get("field_number") or "")
            strain = m.group(1).upper() if m else "UNK"
            spec = (row.get("specimen") or "").strip()
            if spec:
                mapping[img] = (strain, spec)
    return mapping


def parse_specimen(stem: str, img_fallback: dict[str, tuple[str, str]]):
    """Return ``(strain, specimen_number)`` for a filename stem, or ``None``."""
    m = CATALOG_RE.match(stem)
    if m:
        return m.group(1).upper(), m.group(2)
    m = IMG_RE.match(stem)
    if m:
        return img_fallback.get(m.group(1))
    return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(prog="preprocess_jonah")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/cornell"))
    p.add_argument("--info-csv", type=Path, help="Optional Img#### fallback map.")
    p.add_argument("--overrides", type=Path,
                   default=Path("data/cornell_raw/boundary_overrides.json"),
                   help="JSON map of catalog stem -> forced mirror-split x.")
    p.add_argument("--lateral-margin", type=int, default=0,
                   help="Extend the lateral crop this many px left of the split "
                        "(crops overlap; guards against clipping the snout).")
    p.add_argument("--genus", default="Salvelinus")
    p.add_argument("--species", default="fontinalis")  # corrects the source typo
    args = p.parse_args(argv)

    img_fallback: dict[str, tuple[str, str]] = {}
    if args.info_csv and args.info_csv.is_file():
        img_fallback = load_img_fallback(args.info_csv)
        log.info("Loaded %d Img#### fallback mappings", len(img_fallback))

    overrides = {}
    if args.overrides and args.overrides.is_file():
        overrides = {k: v for k, v in json.loads(args.overrides.read_text()).items()
                     if not k.startswith("_")}
        log.info("Loaded %d mirror-boundary override(s)", len(overrides))

    raw_files = sorted(
        p for p in args.raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )
    log.info("Found %d candidate images in %s", len(raw_files), args.raw_dir)

    per_strain: dict[str, int] = {}
    seen: dict[tuple[str, str], str] = {}
    duplicates: list[str] = []
    skipped: list[str] = []
    ok = 0

    for raw in raw_files:
        parsed = parse_specimen(raw.stem, img_fallback)
        if parsed is None:
            skipped.append(raw.name)
            log.warning("SKIP %s — could not parse strain/specimen", raw.name)
            continue
        strain, spec = parsed
        key = (strain, spec)
        if key in seen:
            duplicates.append(f"  {strain}-{spec}: {seen[key]} AND {raw.name}")
        seen[key] = raw.name
        stem = f"{args.genus}_{args.species}_{strain}_{spec}"
        try:
            pc.process_one(
                raw_path=raw,
                out_dir=args.out_dir,
                strain=strain,
                specimen_number=spec,
                genus=args.genus,
                species=args.species,
                boundary_override=overrides.get(stem),
                lateral_margin=args.lateral_margin,
            )
            ok += 1
            per_strain[strain] = per_strain.get(strain, 0) + 1
        except Exception:
            log.exception("FAILED on %s", raw.name)
            skipped.append(raw.name)

    log.info("\n=== Done: %d processed ===", ok)
    for s in sorted(per_strain):
        log.info("  %s: %d", s, per_strain[s])
    if duplicates:
        log.warning("\nDuplicate (strain, specimen) — later overwrites:\n%s",
                    "\n".join(duplicates))
    if skipped:
        log.warning("\nSkipped %d file(s):\n  %s", len(skipped), "\n  ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
