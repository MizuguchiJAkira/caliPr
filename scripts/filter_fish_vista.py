"""Filter the Hugging Face ``imageomics/fish-vista`` dataset for brook trout.

Fish-Vista ships 11 split CSVs at the root of the repo (classification,
identification, and segmentation × train/val/test) plus the image files
under ``Images/chunk_N/``. Each row's ``standardized_species`` column holds
the cleaned species name we want to filter on, and ``file_name`` points
into the image tree.

Running this script will:

  1. Download all 11 split CSVs from HF to ``data/fish_vista/csvs/``.
  2. Scan each CSV for rows whose ``standardized_species`` equals
     ``salvelinus fontinalis`` (case-insensitive).
  3. Deduplicate by ``file_name``.
  4. Download each matched image — preferring the row's ``original_url``
     (fishair.org canonical archive), falling back to HF resolve URLs.
  5. Write ``data/fish_vista/metadata.csv`` with the full per-row provenance.

Realistic expectations: despite Fish-Vista's size (~60k fish photos),
brook trout representation is sparse. At the time this script was written
there were only 2 rows matching ``salvelinus fontinalis``, both from the
``identification_train.csv`` split. The script is still worth running
because (a) new splits may be pushed upstream, and (b) we want the
matched rows under version-controlled provenance.

Usage
-----
    python scripts/filter_fish_vista.py                 # full download
    python scripts/filter_fish_vista.py --dry-run       # list only
    python scripts/filter_fish_vista.py --skip-csv-download  # use cached CSVs
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

HF_REPO = "imageomics/fish-vista"
HF_CSV_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
HF_IMAGE_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

CSV_FILES = (
    "classification_train.csv",
    "classification_val.csv",
    "classification_test.csv",
    "identification_train.csv",
    "identification_val.csv",
    "identification_test_insp.csv",
    "identification_test_lvsp.csv",
    "identification_test_manual_annot.csv",
    "segmentation_train.csv",
    "segmentation_val.csv",
    "segmentation_test.csv",
)

TARGET_SPECIES = "salvelinus fontinalis"
USER_AGENT = (
    "fish-morpho-harvester/0.1 (Cornell Museum of Vertebrates; "
    "research use; contact the repo owner)"
)
REQUEST_TIMEOUT = 120
RETRY_COUNT = 3
RETRY_SLEEP = 2.0

log = logging.getLogger("filter_fish_vista")


@dataclass
class FishVistaRecord:
    split: str
    filename: str
    source_filename: str
    arkid: str
    family: str
    source: str
    owner: str
    standardized_species: str
    original_url: str
    license: str
    file_name: str  # HF path e.g. Images/chunk_6/INHS_FISH_62305.jpg
    local_filename: str = ""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _open(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)


def download_with_retry(url: str, dest: Path) -> int:
    last_err: Exception | None = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            with _open(url) as resp:
                content = resp.read()
            if len(content) < 512:
                raise RuntimeError(
                    f"suspiciously small response ({len(content)} bytes)"
                )
            dest.write_bytes(content)
            return len(content)
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
            last_err = exc
            log.warning(
                "attempt %d/%d failed for %s: %s", attempt, RETRY_COUNT, url, exc
            )
            time.sleep(RETRY_SLEEP * attempt)
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# CSVs
# ---------------------------------------------------------------------------


def download_split_csvs(csv_dir: Path) -> list[Path]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in CSV_FILES:
        dest = csv_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            log.info("csv exists  %s (%d bytes)", name, dest.stat().st_size)
        else:
            url = f"{HF_CSV_BASE}/{name}"
            nbytes = download_with_retry(url, dest)
            log.info("csv ok      %s (%d bytes)", name, nbytes)
        paths.append(dest)
    return paths


def scan_csv_for_target(csv_path: Path) -> list[FishVistaRecord]:
    split = csv_path.stem
    out: list[FishVistaRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = (row.get("standardized_species") or "").strip().lower()
            if sp != TARGET_SPECIES:
                continue
            out.append(
                FishVistaRecord(
                    split=split,
                    filename=row.get("filename", ""),
                    source_filename=row.get("source_filename", ""),
                    arkid=row.get("arkid", ""),
                    family=row.get("family", ""),
                    source=row.get("source", ""),
                    owner=row.get("owner", ""),
                    standardized_species=sp,
                    original_url=row.get("original_url", ""),
                    license=row.get("license", ""),
                    file_name=row.get("file_name", ""),
                )
            )
    log.info("%s: %d brook trout rows", csv_path.name, len(out))
    return out


# ---------------------------------------------------------------------------
# Downloading matched images
# ---------------------------------------------------------------------------


def download_record(record: FishVistaRecord, images_dir: Path) -> bool:
    """Download ``record`` into ``images_dir``. Returns True on success.

    Tries the fishair.org canonical URL first (that's the archival copy
    Fish-Vista itself points at), then falls back to the HF ``file_name``
    path if that fails or is missing.
    """
    safe_name = record.file_name.split("/")[-1] or record.filename or "unknown.jpg"
    dest = images_dir / safe_name

    if dest.exists() and dest.stat().st_size > 1024:
        log.info("exists      %s (%d bytes)", safe_name, dest.stat().st_size)
        record.local_filename = safe_name
        return True

    candidates: list[str] = []
    if record.original_url:
        candidates.append(record.original_url)
    if record.file_name:
        candidates.append(f"{HF_IMAGE_BASE}/{record.file_name}")

    for url in candidates:
        try:
            nbytes = download_with_retry(url, dest)
            log.info(
                "ok          %s (%d bytes) from %s",
                safe_name,
                nbytes,
                "original" if url == record.original_url else "hf",
            )
            record.local_filename = safe_name
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("source failed %s: %s", url, exc)

    log.error("FAIL        %s (all sources exhausted)", safe_name)
    return False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(out_dir: Path, dry_run: bool, skip_csv_download: bool) -> int:
    csv_dir = out_dir / "csvs"
    images_dir = out_dir / "images"
    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)

    if skip_csv_download:
        csv_paths = [csv_dir / name for name in CSV_FILES]
        missing = [p for p in csv_paths if not p.exists()]
        if missing:
            log.error("--skip-csv-download but missing: %s", [p.name for p in missing])
            return 2
    else:
        csv_paths = download_split_csvs(csv_dir)

    matches: dict[str, FishVistaRecord] = {}
    for csv_path in csv_paths:
        for rec in scan_csv_for_target(csv_path):
            # Dedupe by filename; keep first occurrence's split label.
            key = rec.file_name or rec.filename
            matches.setdefault(key, rec)

    log.info("unique brook trout rows across all splits: %d", len(matches))

    if dry_run:
        for rec in matches.values():
            log.info(
                "[dry-run] %s  split=%s  owner=%s  license=%s",
                rec.filename,
                rec.split,
                rec.owner,
                rec.license,
            )
    else:
        downloaded = 0
        failed = 0
        for rec in matches.values():
            if download_record(rec, images_dir):
                downloaded += 1
            else:
                failed += 1
            time.sleep(0.1)
        log.info(
            "fish-vista harvest complete: %d downloaded, %d failed",
            downloaded,
            failed,
        )

        # Write metadata CSV even for small result sets — we want
        # provenance to be under version control.
        write_metadata_csv(list(matches.values()), out_dir / "metadata.csv")
    return 0


def write_metadata_csv(records: list[FishVistaRecord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FishVistaRecord.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))
    log.info("wrote metadata CSV → %s (%d rows)", csv_path, len(records))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/fish_vista"),
        help="Output directory (csvs/ + images/ + metadata.csv written here).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List matches without downloading any images.",
    )
    p.add_argument(
        "--skip-csv-download",
        action="store_true",
        help="Assume CSVs already in out/csvs/ (useful for offline re-runs).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run(args.out, dry_run=args.dry_run, skip_csv_download=args.skip_csv_download)


if __name__ == "__main__":
    raise SystemExit(main())
