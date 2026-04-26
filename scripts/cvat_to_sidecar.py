"""Convert a CVAT "CVAT for images 1.1" XML export into sidecar JSONs.

CVAT's native export formats don't match ``examples/sample_sidecar.json``
directly, so labelers who finish a task in CVAT can't feed their work
into ``fish_morpho.pipeline`` without this bridge. This script walks the
XML, groups annotations by image, validates labels against the
:mod:`fish_morpho.landmark_config` schema, and writes one sidecar JSON
per image into ``--out-dir``.

The lateral and frontal views live in two separate CVAT projects (see
``data/README.md``) so each project exports its own XML and this script
runs twice:

    # Lateral project (polygons + 19 keypoints)
    python scripts/cvat_to_sidecar.py \\
        --cvat-xml exports/lateral.xml \\
        --view lateral \\
        --out-dir data/sidecars/

    # Frontal project (2 mouth keypoints) — merged into existing sidecars
    python scripts/cvat_to_sidecar.py \\
        --cvat-xml exports/frontal.xml \\
        --view frontal \\
        --out-dir data/sidecars/ \\
        --merge

Calibration is stored in CVAT itself via two dedicated "calibration"
keypoints (``ruler_point_a``, ``ruler_point_b``) that the labeler drops
on a known-length ruler span as part of the lateral labeling task. The
converter recognizes those points, reads a shared ``--known-mm``
span length from the CLI, and writes a manual calibration block into
each sidecar automatically — no separate calibration file needed for
the common case. The ruler keypoints live in ``CALIBRATION_KEYPOINTS``,
not ``KEYPOINTS``, so they never pollute the measurement schema.

Three precedence levels for the calibration block (highest first):

  1. ``--calibration-json path.json`` — reads a
     ``{fish_id: {lateral: {...}, frontal: {...}}}`` map and splices
     each matching entry into the sidecar. Use this for per-fish
     overrides when a batch mixes ruler types.
  2. CVAT ruler keypoints + ``--known-mm N`` — if both
     ``ruler_point_a`` and ``ruler_point_b`` are clicked for an image
     and ``--known-mm`` is provided, the converter synthesizes
     ``{"mode": "manual", "point_a": [...], "point_b": [...], "known_mm": N}``.
  3. ``--calibration-mode`` — fallback for images with no CVAT ruler
     clicks. ``auto`` writes ``{"mode": "auto"}`` so the pipeline runs
     the ruler detector at process time; ``none`` leaves calibration
     absent and the pipeline will error clearly when the sidecar loads.

Unknown labels (ones not in the schema) are warned about and skipped,
not silently included — drift in the CVAT project should surface loudly
rather than corrupting the morphometrics downstream.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# Allow running as a script without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fish_morpho.landmark_config import (  # noqa: E402
    View,
    calibration_keypoint_names,
    keypoint_names,
    polygon_names,
)

log = logging.getLogger("fish_morpho.cvat_to_sidecar")


# ---------------------------------------------------------------------------
# Parsed intermediate form
# ---------------------------------------------------------------------------


@dataclass
class ParsedImage:
    """One <image> element's annotations after whitelist filtering."""

    name: str
    polygons: dict[str, list[list[float]]] = field(default_factory=dict)
    keypoints: dict[str, list[float]] = field(default_factory=dict)
    # Ruler clicks for manual calibration (ruler_point_a/b) — kept
    # separate from anatomical keypoints so the measurement schema never
    # sees them and missing-landmark tracking ignores them entirely.
    calibration_points: dict[str, list[float]] = field(default_factory=dict)
    unknown_labels: list[str] = field(default_factory=list)
    skipped_wrong_view: list[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return Path(self.name).stem


# ---------------------------------------------------------------------------
# CVAT XML parsing
# ---------------------------------------------------------------------------


def _parse_points(points_attr: str) -> list[list[float]]:
    """CVAT point strings: ``"x1,y1;x2,y2;..."`` → ``[[x1, y1], [x2, y2], ...]``."""
    out: list[list[float]] = []
    for pair in points_attr.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        try:
            x_str, y_str = pair.split(",")
            out.append([float(x_str), float(y_str)])
        except ValueError as exc:
            raise ValueError(f"Malformed CVAT point pair {pair!r}") from exc
    return out


def parse_cvat_xml(xml_path: Path, view: View) -> list[ParsedImage]:
    """Parse a CVAT for-images 1.1 XML export, filtering by schema.

    Only labels that appear in the schema *for this view* are kept.
    Polygons on a frontal export (or frontal keypoints on a lateral
    export) are recorded in ``skipped_wrong_view`` for the CLI to
    report, never silently dropped.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    valid_polygons = set(polygon_names(view))
    valid_keypoints = set(keypoint_names(view))
    valid_calibration_points = set(calibration_keypoint_names(view))
    known_polygons_anyview = set(polygon_names())
    known_keypoints_anyview = set(keypoint_names()) | set(
        calibration_keypoint_names()
    )

    images: list[ParsedImage] = []
    for image_el in root.findall("image"):
        name = image_el.get("name")
        if not name:
            continue
        parsed = ParsedImage(name=name)

        for poly_el in image_el.findall("polygon"):
            label = poly_el.get("label", "")
            raw = poly_el.get("points", "")
            if label in valid_polygons:
                pts = _parse_points(raw)
                if len(pts) < 3:
                    log.warning(
                        "%s: polygon %r has only %d vertices — skipped",
                        name,
                        label,
                        len(pts),
                    )
                    continue
                if label in parsed.polygons:
                    log.warning(
                        "%s: polygon %r appears twice — using last occurrence",
                        name,
                        label,
                    )
                parsed.polygons[label] = pts
            elif label in known_polygons_anyview or label in known_keypoints_anyview:
                parsed.skipped_wrong_view.append(label)
            else:
                parsed.unknown_labels.append(label)

        for pts_el in image_el.findall("points"):
            label = pts_el.get("label", "")
            raw = pts_el.get("points", "")
            if label in valid_keypoints or label in valid_calibration_points:
                pts = _parse_points(raw)
                if len(pts) != 1:
                    log.warning(
                        "%s: keypoint %r has %d points (expected 1) — using first",
                        name,
                        label,
                        len(pts),
                    )
                if not pts:
                    continue
                if label in valid_calibration_points:
                    parsed.calibration_points[label] = pts[0]
                else:
                    parsed.keypoints[label] = pts[0]
            elif label in known_polygons_anyview or label in known_keypoints_anyview:
                parsed.skipped_wrong_view.append(label)
            else:
                parsed.unknown_labels.append(label)

        images.append(parsed)
    return images


# ---------------------------------------------------------------------------
# Sidecar assembly
# ---------------------------------------------------------------------------


def _synthesize_calibration_from_ruler_points(
    parsed: ParsedImage,
    known_mm: float | None,
) -> dict | None:
    """Build a manual calibration block from ruler_point_a/b + known_mm.

    Returns ``None`` if the labeler didn't drop both ruler points, if
    ``--known-mm`` wasn't supplied, or if the two clicks landed on top of
    each other (degenerate zero-length span). Otherwise returns a block
    the pipeline's ``calibrate()`` helper consumes directly.
    """
    a = parsed.calibration_points.get("ruler_point_a")
    b = parsed.calibration_points.get("ruler_point_b")
    if a is None or b is None:
        # Warn if exactly one is present — that's a labeler mistake worth surfacing.
        if (a is None) != (b is None):
            log.warning(
                "%s: only one ruler point found (need both ruler_point_a and "
                "ruler_point_b for manual calibration)",
                parsed.name,
            )
        return None
    if known_mm is None:
        log.warning(
            "%s: ruler keypoints present but --known-mm not supplied — falling "
            "back to --calibration-mode",
            parsed.name,
        )
        return None
    if known_mm <= 0:
        raise ValueError(f"--known-mm must be positive, got {known_mm}")
    if a == b:
        log.warning(
            "%s: ruler_point_a and ruler_point_b are identical — skipping "
            "calibration synthesis",
            parsed.name,
        )
        return None
    return {
        "mode": "manual",
        "point_a": list(a),
        "point_b": list(b),
        "known_mm": float(known_mm),
    }


def _build_view_block(
    parsed: ParsedImage,
    view: View,
    calibration_mode: str | None,
    calibration_override: dict | None,
    known_mm: float | None,
) -> dict:
    """Produce one view's nested block in the pipeline sidecar schema."""
    block: dict = {}
    if view is View.LATERAL:
        block["polygons"] = {k: list(v) for k, v in parsed.polygons.items()}
        block["keypoints"] = {k: list(v) for k, v in parsed.keypoints.items()}
    else:
        block["keypoints"] = {k: list(v) for k, v in parsed.keypoints.items()}

    # Precedence: explicit JSON override > CVAT ruler clicks > mode flag.
    if calibration_override is not None:
        block["calibration"] = calibration_override
    else:
        synthesized = _synthesize_calibration_from_ruler_points(parsed, known_mm)
        if synthesized is not None:
            block["calibration"] = synthesized
        elif calibration_mode == "auto":
            block["calibration"] = {"mode": "auto"}
        # "none" → leave the calibration block absent; pipeline will error loudly.
    return block


def _load_calibration_map(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of fish_id → calibration")
    return data


def _load_existing_sidecar(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: sidecar root must be an object")
    return data


def write_sidecars(
    images: list[ParsedImage],
    out_dir: Path,
    view: View,
    calibration_mode: str | None,
    calibration_map: dict[str, dict],
    merge: bool,
    known_mm: float | None = None,
) -> list[Path]:
    """Write one sidecar JSON per parsed image. Returns the list of paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for parsed in images:
        stem = parsed.stem
        sidecar_path = out_dir / f"{stem}.json"

        calibration_override = None
        if stem in calibration_map:
            view_key = view.value
            fish_block = calibration_map[stem]
            if view_key in fish_block:
                calibration_override = fish_block[view_key]

        view_block = _build_view_block(
            parsed=parsed,
            view=view,
            calibration_mode=calibration_mode,
            calibration_override=calibration_override,
            known_mm=known_mm,
        )

        if merge and sidecar_path.exists():
            sidecar = _load_existing_sidecar(sidecar_path)
            if view.value in sidecar:
                log.warning(
                    "%s: existing sidecar already has a %s block — overwriting",
                    sidecar_path,
                    view.value,
                )
        else:
            if sidecar_path.exists() and not merge:
                log.warning(
                    "%s: overwriting existing sidecar (pass --merge to keep the other view)",
                    sidecar_path,
                )
            sidecar = {"fish_id": stem, "metadata": {}}

        sidecar[view.value] = view_block

        with sidecar_path.open("w") as f:
            json.dump(sidecar, f, indent=2)
            f.write("\n")
        written.append(sidecar_path)
    return written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(images: list[ParsedImage], view: View) -> None:
    expected_polygons = set(polygon_names(view))
    expected_keypoints = set(keypoint_names(view))
    expected_calibration = set(calibration_keypoint_names(view))

    total_ok = 0
    total_calibrated = 0
    for p in images:
        missing_polys = sorted(expected_polygons - p.polygons.keys())
        missing_kps = sorted(expected_keypoints - p.keypoints.keys())
        missing_cal = sorted(expected_calibration - p.calibration_points.keys())
        if not missing_polys and not missing_kps:
            total_ok += 1
        else:
            log.info(
                "%s: missing polygons=%s, missing keypoints=%s",
                p.name,
                missing_polys or "none",
                missing_kps or "none",
            )
        if expected_calibration and not missing_cal:
            total_calibrated += 1
        elif expected_calibration and missing_cal:
            log.info(
                "%s: missing ruler calibration keypoints=%s", p.name, missing_cal
            )
        if p.unknown_labels:
            log.warning(
                "%s: unknown labels (not in schema, skipped): %s",
                p.name,
                sorted(set(p.unknown_labels)),
            )
        if p.skipped_wrong_view:
            log.warning(
                "%s: labels from the other view skipped: %s",
                p.name,
                sorted(set(p.skipped_wrong_view)),
            )
    log.info(
        "%d/%d images fully cover the %s schema",
        total_ok,
        len(images),
        view.value,
    )
    if expected_calibration:
        log.info(
            "%d/%d images carry both ruler calibration keypoints",
            total_calibrated,
            len(images),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cvat_to_sidecar",
        description="Convert a CVAT for-images 1.1 XML export into "
        "fish_morpho pipeline sidecar JSONs.",
    )
    p.add_argument("--cvat-xml", type=Path, required=True, help="CVAT XML export file")
    p.add_argument(
        "--view",
        choices=("lateral", "frontal"),
        required=True,
        help="Which view this CVAT project covers",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write sidecar JSONs into (one per image)",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Merge into existing sidecars instead of overwriting them "
        "(use this when adding the frontal block to existing lateral "
        "sidecars, or vice versa).",
    )
    p.add_argument(
        "--calibration-mode",
        choices=("auto", "none"),
        default="none",
        help='Fallback for images with no CVAT ruler clicks. "auto" inserts '
        '{"mode": "auto"} so the pipeline runs the ruler detector at '
        'process time; "none" (default) leaves calibration absent and the '
        "pipeline will error clearly when the sidecar loads.",
    )
    p.add_argument(
        "--known-mm",
        type=float,
        default=None,
        help="Real-world length (mm) of the span between ruler_point_a "
        "and ruler_point_b in every image. When supplied, images with "
        "both ruler clicks get a synthesized manual calibration block. "
        "Leave unset if ruler spans differ across the batch and use "
        "--calibration-json instead.",
    )
    p.add_argument(
        "--calibration-json",
        type=Path,
        help="JSON file mapping fish_id → {lateral: {...}, frontal: {...}} "
        "calibration blocks, spliced into the matching sidecars. Overrides "
        "both the ruler-keypoint path and --calibration-mode for any fish "
        "present in the file.",
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
    view = View(args.view)
    try:
        images = parse_cvat_xml(args.cvat_xml, view)
    except (FileNotFoundError, ET.ParseError, ValueError) as exc:
        log.error("Failed to parse %s: %s", args.cvat_xml, exc)
        return 2
    try:
        calibration_map = _load_calibration_map(args.calibration_json)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Failed to read calibration JSON: %s", exc)
        return 2

    _report(images, view)
    written = write_sidecars(
        images=images,
        out_dir=args.out_dir,
        view=view,
        calibration_mode=args.calibration_mode,
        calibration_map=calibration_map,
        merge=args.merge,
        known_mm=args.known_mm,
    )
    log.info("Wrote %d sidecar(s) to %s", len(written), args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
