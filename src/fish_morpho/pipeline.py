"""End-to-end orchestrator for the fish morphometrics pipeline.

Usage (CLI)::

    fish-morpho --images ./photos/ --labels ./labels/ --out results.xlsx \\
                --mode manual

    fish-morpho --images ./photos/ --model-config ./models/config.yaml \\
                --out results.xlsx --mode auto

In manual mode, annotations are loaded from a JSON sidecar file per
image — this is what we use right now, before the DLC + SAM model stack
is trained, so the geometry pipeline can be exercised and validated
against hand measurements emitted from CVAT.

In auto mode, the trained model is invoked to predict polygons and
keypoints. This is stubbed out — the integration point is clearly marked
so that when the model is ready we only touch one function.

JSON sidecar format
-------------------
The sidecar mirrors the hybrid polygon + keypoint schema from
:mod:`fish_morpho.landmark_config`. Both views group their polygons,
keypoints, and calibration into one nested block::

    {
      "fish_id": "BKT-2025-0142",
      "metadata": {
        "locality": "Hogan's Brook",
        "collection_date": "2025-07-14"
      },
      "lateral": {
        "polygons": {
          "body_plus_caudal": [[120, 345], [180, 310], ...],
          "pectoral":         [[395, 370], ...],
          "dorsal":           [[620, 285], ...],
          "pelvic":           [[610, 440], ...],
          "anal":             [[870, 430], ...]
        },
        "keypoints": {
          "eye_anterior":   [190, 335],
          "eye_posterior":  [230, 335],
          "eye_dorsal":     [210, 320],
          "eye_ventral":    [210, 350],
          "premaxilla_tip": [120, 345],
          ...
        },
        "calibration": {
          "mode": "manual",
          "point_a": [100, 1200],
          "point_b": [1100, 1200],
          "known_mm": 150.0
        }
      },
      "frontal": {
        "keypoints": {
          "mouth_left":  [1420, 210],
          "mouth_right": [1478, 208]
        },
        "calibration": {
          "mode": "manual",
          "point_a": [1400, 1150],
          "point_b": [1460, 1150],
          "known_mm": 10.0
        }
      }
    }

The frontal block is optional (omit it if mouth width isn't being
collected for a specimen). Either calibration block may instead be
``{"mode": "auto"}`` to trigger automatic ruler detection on the paired
image.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .export import ExportRecord, export_to_xlsx
from .landmark_config import View
from .measurement_engine import (
    Annotation,
    MeasurementSet,
    compute_all,
)
from .ruler_calibration import (
    CalibrationResult,
    calibrate,
    scale_from_known_span,
)

log = logging.getLogger("fish_morpho.pipeline")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class SpecimenInput:
    """One specimen's raw inputs: paired image file and sidecar JSON."""

    fish_id: str
    image_path: Path
    sidecar_path: Path
    sidecar: dict[str, Any]


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def discover_specimens(images_dir: Path, labels_dir: Path) -> list[SpecimenInput]:
    """Pair image files in ``images_dir`` with JSON sidecars in ``labels_dir``.

    Files are paired by stem: ``foo.jpg`` ↔ ``foo.json``. Images without a
    sidecar are skipped with a warning; sidecars without an image are an
    error (it almost always means the image is misnamed).
    """
    images: dict[str, Path] = {}
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images[p.stem] = p

    specimens: list[SpecimenInput] = []
    seen_sidecars: set[str] = set()
    for sidecar in sorted(labels_dir.glob("*.json")):
        stem = sidecar.stem
        seen_sidecars.add(stem)
        if stem not in images:
            raise FileNotFoundError(
                f"Sidecar {sidecar} has no matching image in {images_dir}"
            )
        with sidecar.open() as f:
            data = json.load(f)
        fish_id = data.get("fish_id", stem)
        specimens.append(
            SpecimenInput(
                fish_id=fish_id,
                image_path=images[stem],
                sidecar_path=sidecar,
                sidecar=data,
            )
        )

    for stem, path in images.items():
        if stem not in seen_sidecars:
            log.warning("No sidecar for image %s — skipping", path.name)

    return specimens


# ---------------------------------------------------------------------------
# Per-specimen processing
# ---------------------------------------------------------------------------


def _coerce_point(raw: Any) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"Expected [x, y], got {raw!r}")
    return float(raw[0]), float(raw[1])


def _coerce_polygon(raw: Any, name: str) -> list[tuple[float, float]]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(
            f"Polygon {name!r} must be a list of >= 3 [x, y] points, got {raw!r}"
        )
    return [_coerce_point(v) for v in raw]


def _load_view_annotation(
    block: dict[str, Any] | None,
    annotation: Annotation,
    view_label: str,
) -> None:
    """Merge one view's polygons and keypoints into ``annotation``.

    Polygons and keypoints live in a flat per-view block (see the module
    docstring for the schema). Silently accepts an empty or missing
    block — callers that need a specific shape should surface the
    missing-input error downstream via the trait's declared
    requirements.
    """
    if not block:
        return
    polys = block.get("polygons") or {}
    for name, verts in polys.items():
        annotation.polygons[name] = _coerce_polygon(verts, name)
    kps = block.get("keypoints") or {}
    for name, xy in kps.items():
        annotation.keypoints[name] = _coerce_point(xy)
    log.debug(
        "[%s] loaded %d polygons, %d keypoints",
        view_label,
        len(polys),
        len(kps),
    )


def _calibration_from_block(
    block: dict[str, Any] | None,
    image_path: Path,
    view_label: str,
) -> CalibrationResult | None:
    """Build a CalibrationResult from a view block's ``calibration`` section."""
    if not block:
        return None
    mode = block.get("mode", "manual")
    if mode == "manual":
        a = _coerce_point(block["point_a"])
        b = _coerce_point(block["point_b"])
        known_mm = float(block["known_mm"])
        return scale_from_known_span(a, b, known_mm)

    if mode == "auto":
        import cv2  # local import

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image {image_path}")
        roi = block.get("roi")
        if roi is not None:
            roi = tuple(int(v) for v in roi)  # type: ignore[assignment]
        fallback_span = block.get("fallback")
        manual_span = None
        if fallback_span:
            manual_span = (
                _coerce_point(fallback_span["point_a"]),
                _coerce_point(fallback_span["point_b"]),
                float(fallback_span["known_mm"]),
            )
        result = calibrate(
            image=img,
            roi=roi,  # type: ignore[arg-type]
            manual_span=manual_span,
        )
        log.info(
            "[%s] %s calibration: %.3f px/mm (%s, conf=%.2f)",
            image_path.name,
            view_label,
            result.px_per_mm,
            result.method,
            result.confidence,
        )
        return result

    raise ValueError(f"Unknown calibration mode {mode!r} in sidecar")


def process_specimen(spec: SpecimenInput) -> ExportRecord:
    """Turn one SpecimenInput into a fully computed ExportRecord."""
    annotation = Annotation()

    lateral_block = spec.sidecar.get("lateral")
    if not lateral_block:
        raise ValueError(
            f"{spec.sidecar_path}: sidecar must contain a 'lateral' block"
        )
    _load_view_annotation(lateral_block, annotation, "lateral")

    frontal_block = spec.sidecar.get("frontal")
    if frontal_block:
        _load_view_annotation(frontal_block, annotation, "frontal")

    lateral_calib = _calibration_from_block(
        lateral_block.get("calibration"), spec.image_path, "lateral"
    )
    if lateral_calib is None:
        raise ValueError(
            f"{spec.sidecar_path}: lateral.calibration is required"
        )

    frontal_calib: CalibrationResult | None = None
    if frontal_block:
        frontal_calib = _calibration_from_block(
            frontal_block.get("calibration"), spec.image_path, "frontal"
        )

    calibrations = {View.LATERAL: lateral_calib}
    if frontal_calib is not None:
        calibrations[View.FRONTAL] = frontal_calib

    metadata = dict(spec.sidecar.get("metadata", {}))
    metadata.setdefault("image_filename", spec.image_path.name)

    ms: MeasurementSet = compute_all(
        fish_id=spec.fish_id,
        annotation=annotation,
        calibrations=calibrations,
        metadata=metadata,
    )

    calibs_for_export: dict[str, CalibrationResult] = {
        View.LATERAL.value: lateral_calib,
    }
    if frontal_calib is not None:
        calibs_for_export[View.FRONTAL.value] = frontal_calib

    return ExportRecord(
        measurements=ms,
        calibrations=calibs_for_export,
        image_filename=spec.image_path.name,
    )


# ---------------------------------------------------------------------------
# Auto-mode annotation (DLC + SAM integration point)
# ---------------------------------------------------------------------------


def predict_annotation(image_path: Path, model_config: Path) -> Annotation:
    """Run the trained DLC + SAM stack on ``image_path``.

    Stub. The final implementation will:

    1. Run the DLC keypoint model to produce the 21 landmarks (19 lateral
       + 2 frontal) from :mod:`fish_morpho.landmark_config`.
    2. Feed the keypoints near each fin as prompts to SAM (Segment
       Anything) to produce the 5 polygons (``body_plus_caudal``,
       ``pectoral``, ``dorsal``, ``pelvic``, ``anal``).
    3. Assemble a single :class:`~fish_morpho.measurement_engine.Annotation`
       containing both and return it.

    Raising NotImplementedError here makes ``--mode auto`` fail fast with
    a clear message rather than silently producing wrong numbers.
    """
    raise NotImplementedError(
        "auto mode: the DLC + SAM annotation stack is not wired up yet. "
        "Once the models are trained, implement predict_annotation() to "
        "run DLC inference for keypoints and SAM inference (prompted by "
        "those keypoints) for polygons, then assemble an Annotation."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    images_dir: Path,
    labels_dir: Path | None,
    output_path: Path,
    mode: str,
    model_config: Path | None,
) -> Path:
    if mode == "manual":
        if labels_dir is None:
            raise ValueError("--labels is required when --mode manual")
        specimens = discover_specimens(images_dir, labels_dir)
        if not specimens:
            raise RuntimeError(
                f"No paired image/sidecar specimens found in {images_dir} / {labels_dir}"
            )
        records = [process_specimen(s) for s in specimens]

    elif mode == "auto":
        if model_config is None:
            raise ValueError("--model-config is required when --mode auto")
        # For each image, predict an Annotation via DLC + SAM, then run
        # the same measurement machinery. Currently stubbed.
        raise NotImplementedError(
            "auto mode not yet wired up; see predict_annotation()"
        )

    else:
        raise ValueError(f"Unknown mode {mode!r}")

    return export_to_xlsx(records, output_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fish-morpho",
        description="Automated morphometrics for brook trout specimens",
    )
    p.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory of fish photos (one image per specimen).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        help="Directory of JSON sidecar files (required for --mode manual).",
    )
    p.add_argument(
        "--model-config",
        type=Path,
        help="Path to the trained DLC + SAM stack config (required for "
        "--mode auto).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .xlsx file.",
    )
    p.add_argument(
        "--mode",
        choices=("manual", "auto"),
        default="manual",
        help="manual = read polygons + keypoints from JSON sidecars; "
        "auto = run the trained DLC + SAM stack on each image.",
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
        out = run(
            images_dir=args.images,
            labels_dir=args.labels,
            output_path=args.out,
            mode=args.mode,
            model_config=args.model_config,
        )
    except Exception as exc:  # pragma: no cover - CLI error surface
        log.error("%s", exc)
        return 1
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
