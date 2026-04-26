"""Harvest Salvelinus fontinalis specimen photos from the iDigBio media API.

As of this writing the iDigBio portal has ~318 media records for brook trout
across MCZ (Harvard), USNM (Smithsonian), NEON, Yale Peabody, Canadian Museum
of Nature, NHM London, Field Museum, UCMP, and INHS. This script:

  1. Walks the media search API with pagination.
  2. Pulls every record that has a retrievable access URL.
  3. Downloads the image to ``data/idigbio/images/``.
  4. Writes a metadata CSV with full provenance (institution, catalog number,
     specimen UUID, license, original URL) so you can trace any image back
     to its source and filter out specimens that turn out to be unusable
     (juveniles, damaged fish, non-lateral views, etc.) during labeling.

The script is idempotent: if an image has already been downloaded and its
size on disk looks plausible, it is skipped on re-run. You can safely
interrupt and restart.

No deps beyond the Python stdlib so you don't have to bring another package
into the environment just for harvesting.

Usage
-----
    python scripts/harvest_idigbio.py                 # full harvest
    python scripts/harvest_idigbio.py --dry-run       # list only, don't download
    python scripts/harvest_idigbio.py --limit 20      # cap for smoke tests
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

API_MEDIA = "https://search.idigbio.org/v2/search/media/"
API_RECORDS = "https://search.idigbio.org/v2/search/records/"
SPECIES = "Salvelinus fontinalis"
USER_AGENT = (
    "fish-morpho-harvester/0.1 (Cornell Museum of Vertebrates; "
    "research use; contact the repo owner)"
)
PAGE_SIZE = 100
REQUEST_TIMEOUT = 120  # seconds
RETRY_COUNT = 3
RETRY_SLEEP = 2.0

log = logging.getLogger("harvest_idigbio")


@dataclass
class SpecimenMeta:
    """Metadata that lives on the specimen record, not the media record."""

    specimen_uuid: str
    institution: str
    collection: str
    catalog_number: str
    scientific_name: str
    country: str
    state: str
    county: str
    locality: str
    year: str
    recorded_by: str


@dataclass
class MediaRecord:
    media_uuid: str
    specimen_uuid: str
    institution: str
    collection: str
    catalog_number: str
    scientific_name: str
    country: str
    state: str
    county: str
    locality: str
    year: str
    recorded_by: str
    creator: str
    license: str
    rights: str
    image_url: str
    media_format: str
    local_filename: str = ""


# ---------------------------------------------------------------------------
# iDigBio API
# ---------------------------------------------------------------------------


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.load(resp)


def _paginated_search(endpoint: str, rq: dict, label: str) -> Iterator[dict]:
    """Generic paginated search over an iDigBio endpoint.

    Yields raw records. ``label`` is used only in the logging header.
    """
    offset = 0
    total: int | None = None
    rq_json = json.dumps(rq)

    while True:
        params = urllib.parse.urlencode(
            {"rq": rq_json, "limit": PAGE_SIZE, "offset": offset}
        )
        url = f"{endpoint}?{params}"
        log.debug("GET %s", url)
        page = _http_get_json(url)
        items = page.get("items", [])
        if total is None:
            total = page.get("itemCount", 0)
            log.info("iDigBio %s for %r: %d total", label, rq.get("scientificname"), total)

        if not items:
            break
        for item in items:
            yield item

        offset += len(items)
        if offset >= (total or 0):
            break
        time.sleep(0.2)  # be polite to the API


def search_media_records(species: str) -> Iterator[dict]:
    """Yield every raw media record matching ``species`` via paginated search."""
    yield from _paginated_search(
        API_MEDIA, {"scientificname": species}, "media records"
    )


def search_specimen_records(species: str) -> Iterator[dict]:
    """Yield every specimen record (with image) matching ``species``."""
    yield from _paginated_search(
        API_RECORDS,
        {"scientificname": species, "hasImage": True},
        "specimen records",
    )


def parse_specimen(raw: dict) -> SpecimenMeta:
    """Extract DwC fields from a /search/records/ response item."""
    data = raw.get("data", {}) or {}
    idx = raw.get("indexTerms", {}) or {}
    return SpecimenMeta(
        specimen_uuid=raw.get("uuid", "") or idx.get("uuid", ""),
        institution=_first(data, "dwc:institutionCode", "dwc:institutionID"),
        collection=_first(data, "dwc:collectionCode"),
        catalog_number=_first(data, "dwc:catalogNumber"),
        scientific_name=_first(data, "dwc:scientificName") or SPECIES,
        country=_first(data, "dwc:country"),
        state=_first(data, "dwc:stateProvince"),
        county=_first(data, "dwc:county"),
        locality=_first(data, "dwc:locality", "dwc:verbatimLocality"),
        year=_first(data, "dwc:year", "dwc:eventDate"),
        recorded_by=_first(data, "dwc:recordedBy"),
    )


def fetch_specimen_index(species: str) -> dict[str, SpecimenMeta]:
    """Build a ``specimen_uuid -> SpecimenMeta`` lookup table.

    A single iDigBio specimen can have multiple media records, so doing one
    paginated query here and joining in memory is much cheaper than fetching
    specimen metadata for every media record individually.
    """
    index: dict[str, SpecimenMeta] = {}
    for raw in search_specimen_records(species):
        meta = parse_specimen(raw)
        if meta.specimen_uuid:
            index[meta.specimen_uuid] = meta
    log.info("indexed %d specimens with metadata", len(index))
    return index


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


def _first(mapping: dict, *keys: str) -> str:
    for k in keys:
        v = mapping.get(k)
        if v:
            return str(v)
    return ""


def parse_record(raw: dict, specimens: dict[str, SpecimenMeta]) -> MediaRecord | None:
    """Turn an iDigBio media record JSON blob into a ``MediaRecord``.

    Joins on-the-fly against the specimen index so institution / catalog /
    locality come through from the owning specimen.

    Returns ``None`` if the record has no retrievable image URL or points
    to non-image media.
    """
    idx = raw.get("indexTerms", {}) or {}
    data = raw.get("data", {}) or {}

    # Prefer the highest-quality access URL we can find. ac:goodQualityAccessURI
    # is the full-resolution original when present; ac:accessURI is usually a
    # web-sized derivative; dcterms:identifier is the fallback.
    image_url = (
        _first(data, "ac:goodQualityAccessURI")
        or _first(idx, "accessuri")
        or _first(data, "ac:accessURI", "dcterms:identifier")
    )
    if not image_url:
        return None

    fmt = _first(idx, "format") or _first(data, "dcterms:format", "dc:format")
    if fmt and "image" not in fmt.lower():
        # Skip non-image media (video, audio, 3D scans, etc.)
        return None

    records = idx.get("records") or []
    specimen_uuid = records[0] if records else ""
    meta = specimens.get(specimen_uuid) or SpecimenMeta(
        specimen_uuid=specimen_uuid,
        institution="",
        collection="",
        catalog_number="",
        scientific_name=SPECIES,
        country="",
        state="",
        county="",
        locality="",
        year="",
        recorded_by="",
    )

    return MediaRecord(
        media_uuid=raw.get("uuid", ""),
        specimen_uuid=specimen_uuid,
        institution=meta.institution,
        collection=meta.collection,
        catalog_number=meta.catalog_number,
        scientific_name=meta.scientific_name,
        country=meta.country,
        state=meta.state,
        county=meta.county,
        locality=meta.locality,
        year=meta.year,
        recorded_by=meta.recorded_by,
        creator=_first(data, "dc:creator"),
        license=_first(data, "xmpRights:UsageTerms", "dcterms:license"),
        rights=_first(data, "dcterms:rights", "dc:rights"),
        image_url=image_url,
        media_format=fmt or "",
    )


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_token(value: str, fallback: str = "unknown") -> str:
    cleaned = _SAFE_CHARS.sub("_", value.strip())
    return cleaned.strip("_") or fallback


def build_filename(record: MediaRecord) -> str:
    """Stable, human-scannable filename: ``INST_CAT_UUIDPREFIX.jpg``."""
    inst = _safe_token(record.institution or "UNK")
    cat = _safe_token(record.catalog_number or "nocat")
    short_uuid = (record.media_uuid or "")[:8] or "nouuid"

    suffix = Path(urllib.parse.urlparse(record.image_url).path).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        suffix = ".jpg"
    return f"{inst}_{cat}_{short_uuid}{suffix}"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_with_retry(url: str, dest: Path) -> int:
    """Download ``url`` to ``dest``. Returns bytes written.

    Retries transient failures up to ``RETRY_COUNT`` times with backoff.
    Raises the last error if all retries fail.
    """
    last_err: Exception | None = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                content = resp.read()
            if len(content) < 1024:
                # Anything under 1 KB is almost certainly an error page or
                # placeholder, not a real specimen photo.
                raise RuntimeError(
                    f"suspiciously small response ({len(content)} bytes)"
                )
            dest.write_bytes(content)
            return len(content)
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
            last_err = exc
            log.warning("attempt %d/%d failed for %s: %s", attempt, RETRY_COUNT, url, exc)
            time.sleep(RETRY_SLEEP * attempt)
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def harvest(
    out_dir: Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> list[MediaRecord]:
    images_dir = out_dir / "images"
    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)

    # Build specimen metadata lookup once so we can join each media record
    # with its owning specimen's institution / catalog / locality.
    specimens = fetch_specimen_index(SPECIES)

    collected: list[MediaRecord] = []
    skipped = 0
    downloaded = 0
    failed = 0

    for raw in search_media_records(SPECIES):
        rec = parse_record(raw, specimens)
        if rec is None:
            skipped += 1
            continue

        rec.local_filename = build_filename(rec)
        dest = images_dir / rec.local_filename

        if dry_run:
            log.info("[dry-run] %s  <-  %s", rec.local_filename, rec.image_url)
            collected.append(rec)
        elif dest.exists() and dest.stat().st_size > 1024:
            log.info("exists  %s  (%d bytes)", rec.local_filename, dest.stat().st_size)
            collected.append(rec)
        else:
            try:
                nbytes = download_with_retry(rec.image_url, dest)
                log.info(
                    "ok      %s  (%d bytes)  inst=%s cat=%s",
                    rec.local_filename,
                    nbytes,
                    rec.institution,
                    rec.catalog_number,
                )
                collected.append(rec)
                downloaded += 1
                time.sleep(0.1)  # spread the load across source servers
            except Exception as exc:  # noqa: BLE001 — we log & continue
                log.error("FAIL    %s: %s", rec.image_url, exc)
                failed += 1

        if limit is not None and len(collected) >= limit:
            log.info("hit --limit cap at %d records", limit)
            break

    log.info(
        "harvest complete: %d collected, %d new, %d failed, %d skipped (non-image)",
        len(collected),
        downloaded,
        failed,
        skipped,
    )
    return collected


def write_metadata_csv(records: list[MediaRecord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(MediaRecord.__dataclass_fields__.keys())
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
        default=Path("data/idigbio"),
        help="Output directory (images/ + metadata.csv written here).",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Cap the number of records processed (useful for smoke tests).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without writing anything.",
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

    records = harvest(args.out, limit=args.limit, dry_run=args.dry_run)
    if not args.dry_run:
        write_metadata_csv(records, args.out / "metadata.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
