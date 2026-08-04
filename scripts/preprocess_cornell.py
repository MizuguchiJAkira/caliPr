"""Preprocess raw Cornell lab photos into CVAT-ready lateral + frontal crops.

Raw photos from the lab have a consistent layout:
  * Fish (lateral view) in the center of a white styrofoam tray
  * Metric cm ruler along one edge
  * Handwritten label card (ST-HRN + specimen number + date)
  * Mirror on one side showing the frontal head view

Photos may have varying EXIF orientations (upside-down, 90° rotated,
etc.). The script applies EXIF orientation metadata first, then detects
whether the result is portrait (camera was rotated 90°) and rotates to
landscape. The final output always has the fish facing left per the
annotation schema convention.

In the correctly-oriented image the mirror is on the LEFT side:
  * Lateral crop = right portion (fish + ruler)
  * Frontal crop = left portion (mirror reflection of the head)

Output naming follows the catalog convention::

    Salvelinus_fontinalis_{strain}_{specimen#}_L.JPEG   (lateral)
    Salvelinus_fontinalis_{strain}_{specimen#}_F.JPEG   (frontal)

Usage::

    python scripts/preprocess_cornell.py \\
        --raw-dir data/cornell_raw \\
        --map-csv data/cornell_raw/specimen_map.csv \\
        --out-dir data/cornell \\
        --strain HRN

The specimen_map.csv has columns: raw_filename, specimen_number, notes.
Edit it to fix any misread label-card numbers before running.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orientation normalization
# ---------------------------------------------------------------------------


def _ruler_band_energy(gray: np.ndarray) -> tuple[float, float]:
    """Return (top_energy, bottom_energy) of ruler-like tick texture.

    The metric ruler is the densest field of regularly spaced vertical
    edges in the frame, so a horizontal Sobel summed over a band picks it
    out clearly against styrofoam, fish, and handwritten cards.
    """
    h, w = gray.shape[:2]
    sob = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    # Ignore the outer margins where the mirror rig and tray edges sit.
    x0, x1 = int(w * 0.35), int(w * 0.95)
    top = sob[: int(h * 0.30), x0:x1]
    bottom = sob[int(h * 0.70):, x0:x1]
    return float(top.mean()), float(bottom.mean())


def is_upside_down(bgr: np.ndarray) -> bool:
    """True if the frame is 180° from canonical (ruler at the bottom).

    In the lab's canonical layout the ruler lies along the TOP of the frame
    with the fish below it. A 180° rotation moves the ruler to the bottom,
    which is exactly the state that leaves the fish facing right with the
    mirror on the right.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    top, bottom = _ruler_band_energy(gray)
    return bottom > top


def normalize_orientation(raw_path: Path) -> np.ndarray:
    """Load a JPEG and return a BGR OpenCV array in canonical orientation.

    Canonical = landscape, fish facing LEFT, mirror on the LEFT.

    Steps:
      1. Apply EXIF orientation metadata.
      2. If the result is portrait (height > width), rotate 90° CW.
      3. Detect whether the frame is upside-down (ruler at the bottom) and
         rotate 180° if so.

    Step 3 exists because the EXIF tag is NOT consistent across this
    collection: most lab photos carry orientation 6, but a substantial
    minority carry orientation 1 and are already landscape, so steps 1-2
    leave them 180° from canonical — fish facing right with the mirror on
    the right. The mirror split then cut the left side of the frame, which
    is where those specimens' caudal peduncle and fin were, silently
    truncating the fish. Detecting the layout from pixels instead of
    trusting EXIF fixes that whole class of specimens.
    """
    pil = Image.open(raw_path)
    pil = ImageOps.exif_transpose(pil)

    w, h = pil.size
    rotated = False
    if h > w:
        # Portrait → rotate 90° CW to landscape.
        pil = pil.rotate(-90, expand=True)
        rotated = True

    # Convert PIL RGB → OpenCV BGR.
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    flipped = False
    if is_upside_down(bgr):
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        flipped = True

    log.info(
        "%s: EXIF applied%s%s → %dx%d",
        raw_path.name,
        " + 90° CW (portrait→landscape)" if rotated else "",
        " + 180° (was upside-down: ruler at bottom)" if flipped else "",
        bgr.shape[1],
        bgr.shape[0],
    )
    return bgr


# ---------------------------------------------------------------------------
# Mirror boundary detection
# ---------------------------------------------------------------------------


def detect_mirror_boundary(gray: np.ndarray, search_frac: float = 0.35) -> int:
    """Find the x-coordinate of the mirror frame's inner edge.

    In the canonical orientation the mirror is on the LEFT side. The
    mirror frame is a dark vertical stripe — the strongest vertical edge
    in the left portion of the image. We search the left ``search_frac``
    of the image width for the column with the highest total vertical
    gradient (horizontal Sobel) and treat that as the boundary.

    Returns the x-coordinate (column index) of the mirror boundary. The
    frontal crop is everything to the LEFT of this boundary; the lateral
    crop is everything to the RIGHT.
    """
    h, w = gray.shape[:2]
    search_w = int(w * search_frac)
    roi = gray[:, :search_w]

    # Vertical edges → horizontal Sobel
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    col_energy = np.sum(np.abs(sobel_x), axis=0)

    # The mirror frame is the dominant peak. Take the column with the max
    # energy, then refine to the rightmost column within 90% of peak
    # (the frame has width, and we want the inner edge = rightmost).
    peak = float(np.max(col_energy))
    threshold = peak * 0.9
    candidates = np.where(col_energy >= threshold)[0]
    boundary = int(candidates[-1])

    # Add a small margin (10 px) to clear the frame edge.
    boundary = min(boundary + 10, search_w - 1)

    log.debug("mirror boundary detected at x=%d (image width=%d)", boundary, w)
    return boundary


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_one(
    raw_path: Path,
    out_dir: Path,
    strain: str,
    specimen_number: int | str,
    genus: str = "Salvelinus",
    species: str = "fontinalis",
    boundary_override: int | None = None,
    lateral_margin: int = 0,
) -> tuple[Path, Path]:
    """Split one raw photo into lateral + frontal crops.

    ``boundary_override`` forces the mirror-split column instead of running
    :func:`detect_mirror_boundary`. The detector keys on the strongest vertical
    edge in the left 35%, and on a few photos the ruler's dense tick marks
    out-shout the mirror frame, pushing the split to the search ceiling and
    slicing through the fish's head. Those are recorded in
    ``data/cornell_raw/boundary_overrides.json`` after visual verification.

    Returns (lateral_path, frontal_path).
    """
    image = normalize_orientation(raw_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boundary = (
        int(boundary_override)
        if boundary_override is not None
        else detect_mirror_boundary(gray)
    )

    # Mirror on left, fish on right. The two crops deliberately OVERLAP by
    # ``lateral_margin``: detect_mirror_boundary keys on the strongest vertical
    # edge in the left 35%, and the ruler's tick marks can out-shout the mirror
    # frame and push the split right, through the fish's snout. Extending the
    # lateral crop leftward costs only some mirror background in frame, while
    # cutting a snout destroys data that cannot be recovered — so the margin is
    # applied asymmetrically. The frontal crop keeps its full extent.
    lat_start = max(0, boundary - lateral_margin)
    frontal_crop = image[:, :boundary]
    lateral_crop = image[:, lat_start:]

    base = f"{genus}_{species}_{strain}_{specimen_number}"
    lateral_name = f"{base}_L.JPEG"
    frontal_name = f"{base}_F.JPEG"

    lat_dir = out_dir / "lateral"
    fro_dir = out_dir / "frontal"
    lat_dir.mkdir(parents=True, exist_ok=True)
    fro_dir.mkdir(parents=True, exist_ok=True)

    lat_path = lat_dir / lateral_name
    fro_path = fro_dir / frontal_name

    cv2.imwrite(str(lat_path), lateral_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(fro_path), frontal_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    log.info(
        "  → %s (%dx%d) + %s (%dx%d)",
        lateral_name,
        lateral_crop.shape[1],
        lateral_crop.shape[0],
        frontal_name,
        frontal_crop.shape[1],
        frontal_crop.shape[0],
    )
    return lat_path, fro_path


def load_specimen_map(csv_path: Path) -> dict[str, str]:
    """Read specimen_map.csv → {raw_filename: specimen_number}."""
    mapping: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["raw_filename"].strip()] = str(
                row["specimen_number"]
            ).strip()
    return mapping


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess_cornell",
        description="Split raw Cornell lab photos into lateral + frontal "
        "crops with EXIF-aware orientation and catalog naming.",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing raw Img####.JPG files.",
    )
    p.add_argument(
        "--map-csv",
        type=Path,
        required=True,
        help="CSV with raw_filename,specimen_number columns.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/cornell"),
        help="Output directory (lateral/ and frontal/ subdirs created).",
    )
    p.add_argument(
        "--strain",
        default="HRN",
        help="Strain code for the catalog name (default: HRN).",
    )
    p.add_argument(
        "--genus",
        default="Salvelinus",
        help="Genus for catalog name.",
    )
    p.add_argument(
        "--species",
        default="fontinalis",
        help="Species for catalog name.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)

    specimen_map = load_specimen_map(args.map_csv)
    if not specimen_map:
        log.error("Empty specimen map — nothing to process.")
        return 2

    raw_dir: Path = args.raw_dir
    seen_specimens: dict[str, str] = {}  # specimen_number → raw_filename
    duplicates: list[str] = []

    for raw_name, spec_num in sorted(specimen_map.items()):
        raw_path = raw_dir / raw_name
        if not raw_path.exists():
            log.warning("SKIP %s — file not found in %s", raw_name, raw_dir)
            continue

        if spec_num in seen_specimens:
            duplicates.append(
                f"  specimen {spec_num}: {seen_specimens[spec_num]} AND {raw_name}"
            )
        seen_specimens[spec_num] = raw_name

        try:
            process_one(
                raw_path=raw_path,
                out_dir=args.out_dir,
                strain=args.strain,
                specimen_number=spec_num,
                genus=args.genus,
                species=args.species,
            )
        except Exception:
            log.exception("FAILED on %s", raw_name)

    if duplicates:
        log.warning(
            "\nDuplicate specimen numbers detected (later file overwrites):\n%s",
            "\n".join(duplicates),
        )

    log.info(
        "\nDone. Lateral crops in %s/lateral/, frontal crops in %s/frontal/",
        args.out_dir,
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
