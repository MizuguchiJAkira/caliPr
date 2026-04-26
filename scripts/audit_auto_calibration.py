"""Run ``detect_ruler_scale`` on every iDigBio image and report results.

This script answers the question "is the auto-ruler detector good enough
to trust in production, or should we default to manual calibration?"
It walks ``data/idigbio/images/``, runs the auto-detector on each image
without an ROI hint (the hardest case — no prior information about where
on the photo the ruler sits), and writes a per-image CSV report plus a
Markdown summary grouped by institution.

For each image we record:
  * ``filename`` — basename
  * ``institution`` — parsed from the metadata CSV
  * ``outcome``    — ``ok`` / ``low_confidence`` / ``detect_failed`` / ``image_read_failed``
  * ``px_per_mm``  — the returned scale, or blank on failure
  * ``confidence`` — the autocorrelation prominence-derived score, or blank
  * ``method``     — always ``auto`` when the detector returned anything
  * ``notes``      — the detector's own notes, or the error message on failure

"low_confidence" here means the detector returned a number but it fell
below the same 0.3 threshold the pipeline's ``calibrate()`` helper uses
to decide whether to fall back to manual — i.e. the pipeline would
silently reject this auto result.

Usage::

    python scripts/audit_auto_calibration.py \\
        --images data/idigbio/images/ \\
        --metadata data/idigbio/metadata.csv \\
        --out data/validation/auto_calibration_audit.csv \\
        --summary data/validation/auto_calibration_audit.md

Both outputs are optional — omit ``--summary`` for CSV-only.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Allow running as a script without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fish_morpho.ruler_calibration import detect_ruler_scale  # noqa: E402

log = logging.getLogger("fish_morpho.audit_auto_calibration")

MIN_CONFIDENCE = 0.3  # same threshold as ruler_calibration.calibrate()
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Report row
# ---------------------------------------------------------------------------


@dataclass
class AuditRow:
    filename: str
    institution: str
    outcome: str
    px_per_mm: float | None
    confidence: float | None
    method: str
    notes: str

    def as_csv_row(self) -> list[str]:
        return [
            self.filename,
            self.institution,
            self.outcome,
            "" if self.px_per_mm is None else f"{self.px_per_mm:.4f}",
            "" if self.confidence is None else f"{self.confidence:.3f}",
            self.method,
            self.notes,
        ]


CSV_HEADER = [
    "filename",
    "institution",
    "outcome",
    "px_per_mm",
    "confidence",
    "method",
    "notes",
]


# ---------------------------------------------------------------------------
# Metadata loader
# ---------------------------------------------------------------------------


def _load_institution_map(metadata_path: Path | None) -> dict[str, str]:
    """Return ``{local_filename: institution}`` from the harvest CSV.

    Missing or absent CSV → empty dict, callers default to ``"?"``.
    """
    if metadata_path is None or not metadata_path.is_file():
        return {}
    out: dict[str, str] = {}
    with metadata_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("local_filename") or "").strip()
            inst = (row.get("institution") or "").strip() or "?"
            if fname:
                out[fname] = inst
    return out


# ---------------------------------------------------------------------------
# Single-image audit
# ---------------------------------------------------------------------------


def audit_image(image_path: Path, institution: str) -> AuditRow:
    import cv2  # lazy

    img = cv2.imread(str(image_path))
    if img is None:
        return AuditRow(
            filename=image_path.name,
            institution=institution,
            outcome="image_read_failed",
            px_per_mm=None,
            confidence=None,
            method="",
            notes=f"cv2.imread returned None for {image_path}",
        )

    try:
        result = detect_ruler_scale(img)
    except RuntimeError as exc:
        return AuditRow(
            filename=image_path.name,
            institution=institution,
            outcome="detect_failed",
            px_per_mm=None,
            confidence=None,
            method="auto",
            notes=str(exc),
        )
    except Exception as exc:  # pragma: no cover - unexpected crash
        return AuditRow(
            filename=image_path.name,
            institution=institution,
            outcome="detect_failed",
            px_per_mm=None,
            confidence=None,
            method="auto",
            notes=f"{type(exc).__name__}: {exc}",
        )

    outcome = "ok" if result.confidence >= MIN_CONFIDENCE else "low_confidence"
    return AuditRow(
        filename=image_path.name,
        institution=institution,
        outcome=outcome,
        px_per_mm=result.px_per_mm,
        confidence=result.confidence,
        method=result.method,
        notes=result.notes,
    )


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def audit_directory(
    images_dir: Path,
    metadata_path: Path | None,
) -> list[AuditRow]:
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images dir not found: {images_dir}")

    institution_map = _load_institution_map(metadata_path)

    rows: list[AuditRow] = []
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    for i, path in enumerate(image_files, start=1):
        institution = institution_map.get(path.name, "?")
        log.info("[%d/%d] %s (%s)", i, len(image_files), path.name, institution)
        rows.append(audit_image(path, institution))
    return rows


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_csv(rows: list[AuditRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow(row.as_csv_row())


def _counts_by(
    rows: list[AuditRow], key: str
) -> dict[str, dict[str, int]]:
    """Return ``{group: {outcome: count}}`` for a group key like "institution"."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        group = getattr(row, key)
        out[group][row.outcome] += 1
    return out


def write_summary(rows: list[AuditRow], out_path: Path) -> None:
    total = len(rows)
    by_outcome: dict[str, int] = defaultdict(int)
    for row in rows:
        by_outcome[row.outcome] += 1

    lines: list[str] = []
    lines.append("# Auto-calibration audit on iDigBio brook trout pool\n")
    lines.append(
        "Every image was run through `detect_ruler_scale` with no ROI hint. "
        "`ok` means the detector returned a result at or above the 0.3 "
        "confidence threshold the pipeline's `calibrate()` helper uses to "
        "decide whether to fall back to manual. `low_confidence` means a "
        "number came back but the pipeline would reject it.\n"
    )
    lines.append("## Overall\n")
    lines.append(f"- Total images: **{total}**")
    for outcome in ("ok", "low_confidence", "detect_failed", "image_read_failed"):
        count = by_outcome.get(outcome, 0)
        pct = (count / total * 100) if total else 0.0
        lines.append(f"- `{outcome}`: **{count}** ({pct:.0f}%)")
    lines.append("")

    lines.append("## By institution\n")
    lines.append("| Institution | n | ok | low_conf | detect_fail | read_fail |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    inst_counts = _counts_by(rows, "institution")
    for inst in sorted(inst_counts.keys()):
        counts = inst_counts[inst]
        n = sum(counts.values())
        lines.append(
            f"| {inst} | {n} | "
            f"{counts.get('ok', 0)} | "
            f"{counts.get('low_confidence', 0)} | "
            f"{counts.get('detect_failed', 0)} | "
            f"{counts.get('image_read_failed', 0)} |"
        )
    lines.append("")

    # Surface the actual numeric distribution for `ok` cases so we can eyeball
    # whether they cluster around a sensible range.
    ok_rows = [r for r in rows if r.outcome == "ok"]
    if ok_rows:
        px_per_mm = sorted(r.px_per_mm for r in ok_rows if r.px_per_mm is not None)
        if px_per_mm:
            mid = px_per_mm[len(px_per_mm) // 2]
            lines.append("## `ok` px/mm distribution\n")
            lines.append(f"- min: {px_per_mm[0]:.3f}")
            lines.append(f"- median: {mid:.3f}")
            lines.append(f"- max: {px_per_mm[-1]:.3f}")
            lines.append("")

    # If failures outnumber successes, list them so a human can eyeball why.
    failing = [r for r in rows if r.outcome != "ok"]
    if failing:
        lines.append("## Per-image failures (not `ok`)\n")
        lines.append("| File | Institution | Outcome | Conf | Notes |")
        lines.append("|---|---|---|---:|---|")
        for r in failing:
            conf = f"{r.confidence:.2f}" if r.confidence is not None else ""
            notes = (r.notes or "").replace("|", "\\|")
            if len(notes) > 80:
                notes = notes[:77] + "..."
            lines.append(
                f"| {r.filename} | {r.institution} | `{r.outcome}` | {conf} | {notes} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audit_auto_calibration",
        description="Run the auto-ruler detector on a pool of images and "
        "report per-image + aggregate results.",
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("data/idigbio/images"),
        help="Directory of images to audit",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/idigbio/metadata.csv"),
        help="Harvest metadata CSV (for institution grouping)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/validation/auto_calibration_audit.csv"),
        help="CSV output path",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("data/validation/auto_calibration_audit.md"),
        help="Markdown summary output path (omit with empty string to skip)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        rows = audit_directory(args.images, args.metadata)
    except (FileNotFoundError, NotADirectoryError) as exc:
        log.error("%s", exc)
        return 2

    write_csv(rows, args.out)
    log.info("Wrote per-image CSV to %s", args.out)

    if args.summary and str(args.summary):
        write_summary(rows, args.summary)
        log.info("Wrote Markdown summary to %s", args.summary)

    total = len(rows)
    ok = sum(1 for r in rows if r.outcome == "ok")
    log.info("%d/%d images passed (≥%.2f confidence)", ok, total, MIN_CONFIDENCE)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
