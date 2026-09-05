"""End-to-end orchestrator for the fish morphometrics pipeline.

Usage (CLI)::

    fish-morpho --images ./photos/ --labels ./labels/ --out results.xlsx \\
                --mode manual

    fish-morpho --images ./photos/ --model-config ./models/config.yaml \\
                --out results.xlsx --mode auto

In manual mode, annotations are loaded from a JSON sidecar file per
image, normally written by the browser labeler in ``scripts/``. This is
the released path, and the one every published number comes from.

In auto mode, a trained model is invoked to predict polygons and
keypoints. This is stubbed out — the integration point is clearly marked
so that when the models are ready we only touch one function. The
keypoint model is trained but not yet wired in here; SAM is usable for
``body_plus_caudal`` only. See the README.

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
import math
import re
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .export import ExportRecord, export_to_xlsx
from .validation import summarise, validate
from .landmark_config import (
    FIN_POLYGON_TARGET_VERTICES,
    FIN_POLYGONS,
    View,
)
from .measurement_engine import (
    Annotation,
    MeasurementSet,
    MeasurementValue,
    compute_all,
)
from .ruler_calibration import (
    CalibrationResult,
    calibrate,
    scale_from_known_span,
)

log = logging.getLogger("fish_morpho.pipeline")

_LOT_RE = re.compile(r"(?:CUMV[A-Za-z]*_(\d+))|_([A-Z]{2,4})_\d+$", re.IGNORECASE)


def _lot_of(fish_id: str) -> str:
    """Collection lot, used to compare a specimen's scale against its peers."""
    m = _LOT_RE.search(fish_id)
    return (m.group(1) or m.group(2) or "") if m else ""


def _git_commit() -> str:
    """Which version of caliPr produced this file. Provenance beats memory."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=Path(__file__).resolve().parent.parent.parent,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


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

    Paired by stem, with the rig's view suffix tolerated: ``foo.json`` matches
    ``foo.jpg`` or ``foo_L.JPEG``. The CUMV preprocessing writes ``<id>_L`` for
    the lateral crop and ``<id>_F`` for the mirror, while the sidecar is named for
    the specimen, so a strict stem match found nothing on the entire trout series.
    Images without a sidecar are skipped with a warning; a sidecar with no image
    is an error, since it almost always means a file was renamed.
    """
    images: dict[str, Path] = {}
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images[p.stem] = p
            if p.stem.endswith(("_L", "_F")):
                images.setdefault(p.stem[:-2], p)

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

    unlabelled = sorted({p.name for stem, p in images.items()
                         if stem not in seen_sidecars})
    if unlabelled:
        # One line, not one per file: a dataset of 181 photographs with 5 labelled
        # produced 176 warnings that buried everything worth reading.
        log.info("%d image(s) have no sidecar yet and were skipped", len(unlabelled))

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


def _sparse_fin_polygons(annotation: Annotation) -> dict[str, int]:
    """Fin outlines traced with too few vertices, name -> vertex count."""
    return {
        name: len(annotation.polygons[name])
        for name in FIN_POLYGONS
        if name in annotation.polygons
        and len(annotation.polygons[name]) < FIN_POLYGON_TARGET_VERTICES
    }


def _sparse_fin_note(sparse: dict[str, int]) -> str:
    detail = ", ".join(f"{n}={c}" for n, c in sorted(sparse.items()))
    return (
        f"fin area biased low: {detail} vertices, under the "
        f"{FIN_POLYGON_TARGET_VERTICES} needed for a reliable outline"
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

    if mode == "ticks":
        # Scale measured from the ruler's millimetre ticks (see
        # ruler_calibration.detect_tick_scale). Removes the hand-typed
        # known_mm, which is the one calibration input nothing downstream can
        # sanity-check — a mistyped span scales every trait silently.
        return CalibrationResult(
            px_per_mm=float(block["px_per_mm"]),
            method="ticks",
            confidence=float(block.get("confidence", 0.9)),
            notes=str(block.get("notes", "mm-tick autocalibration")),
        )

    if mode == "none":
        # Scale-free. Some studies compare *proportions* between populations and
        # never need millimetres -- and some rigs cannot supply them honestly. The
        # alewife series photographs a specimen suspended mid-tank with the ruler
        # taped to the near glass, so the ruler sits in a different focal plane
        # than the fish: parallax and refraction through the fluid make any
        # absolute scale read off it wrong by an unknown factor.
        #
        # Ratios are unaffected. Every landmark on a planar specimen shares one
        # magnification, so trait/SL is exact whatever that magnification is.
        # px_per_mm = 1.0 leaves the Measurements sheet in PIXELS -- deliberately
        # not millimetres, because calling them millimetres would be a lie -- and
        # the Ratios sheet is the one to actually use.
        return CalibrationResult(
            px_per_mm=1.0,
            method="none",
            confidence=1.0,
            notes=str(block.get("notes", "scale-free: values are PIXELS, use the "
                                         "Ratios sheet")),
        )

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
    frontal_block = spec.sidecar.get("frontal")
    if not lateral_block and not frontal_block:
        raise ValueError(
            f"{spec.sidecar_path}: sidecar has neither a 'lateral' nor a "
            "'frontal' block"
        )
    if lateral_block:
        _load_view_annotation(lateral_block, annotation, "lateral")
    if frontal_block:
        _load_view_annotation(frontal_block, annotation, "frontal")

    # A frontal-only sidecar is legitimate — mouth width is collected from the
    # mirror view and needs no lateral data. Refusing to process it would
    # discard a real measurement to satisfy a shape requirement; the lateral
    # traits simply surface as missing-input NaNs, which is what the QC sheet
    # is for. (The reverse, lateral-only, has always been supported.)
    lateral_calib = (
        _calibration_from_block(
            lateral_block.get("calibration"), spec.image_path, "lateral"
        )
        if lateral_block
        else None
    )
    if lateral_block and lateral_calib is None:
        raise ValueError(
            f"{spec.sidecar_path}: lateral.calibration is required"
        )

    frontal_calib: CalibrationResult | None = None
    if frontal_block:
        frontal_calib = _calibration_from_block(
            frontal_block.get("calibration"), spec.image_path, "frontal"
        )

    calibrations: dict[View, CalibrationResult] = {}
    if lateral_calib is not None:
        calibrations[View.LATERAL] = lateral_calib
    if frontal_calib is not None:
        calibrations[View.FRONTAL] = frontal_calib

    metadata = dict(spec.sidecar.get("metadata", {}))
    metadata.setdefault("image_filename", spec.image_path.name)

    # Under-traced fins read small (see FIN_POLYGON_TARGET_VERTICES). The areas
    # are still computed — the bias is systematic, not random, so the numbers
    # stay comparable within a density band — but the QC sheet has to say which
    # rows are affected, or a low fin area is indistinguishable from a small fin.
    sparse = _sparse_fin_polygons(annotation)
    if sparse:
        metadata["data_note"] = "; ".join(
            filter(None, [metadata.get("data_note"), _sparse_fin_note(sparse)])
        )

    ms: MeasurementSet = compute_all(
        fish_id=spec.fish_id,
        annotation=annotation,
        calibrations=calibrations,
        metadata=metadata,
    )

    # Data-compromise handling: if the labeler flagged a photo problem (e.g. a
    # fin clipped by the frame), salvage the usable traits but force-NaN the
    # traits that can't be trusted — tagged 'data_compromise' so the QC sheet
    # shows *why* they're blank — and log the compromise loudly when we run.
    sidecar_meta = spec.sidecar.get("metadata", {}) or {}
    exclude = sidecar_meta.get("exclude_traits") or []
    data_note = sidecar_meta.get("data_note")
    for code in exclude:
        mv = ms.values.get(code)
        if mv is not None:
            ms.values[code] = MeasurementValue(
                key=mv.key, label=mv.label, value=math.nan, unit=mv.unit,
                view=mv.view, missing_landmarks=("data_compromise",),
            )
    if exclude:
        log.warning(
            "%s: DATA COMPROMISE — %s | excluded traits: %s",
            spec.fish_id, data_note or "(no note provided)", ", ".join(exclude),
        )
    elif data_note:
        log.warning("%s: data note — %s", spec.fish_id, data_note)

    calibs_for_export: dict[str, CalibrationResult] = {}
    if lateral_calib is not None:
        calibs_for_export[View.LATERAL.value] = lateral_calib
    if frontal_calib is not None:
        calibs_for_export[View.FRONTAL.value] = frontal_calib

    size = None
    try:                                    # cheap: reads the header, not the pixels
        from PIL import Image
        with Image.open(spec.image_path) as im:
            size = im.size
    except Exception:
        pass

    return ExportRecord(
        measurements=ms,
        calibrations=calibs_for_export,
        image_filename=spec.image_path.name,
        keypoints=dict(annotation.keypoints),
        polygons={k: list(v) for k, v in annotation.polygons.items()},
        image_size=size,
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
    drop_traits: tuple[str, ...] = (),
) -> Path:
    if mode == "manual":
        if labels_dir is None:
            raise ValueError("--labels is required when --mode manual")
        specimens = discover_specimens(images_dir, labels_dir)
        if not specimens:
            raise RuntimeError(
                f"No paired image/sidecar specimens found in {images_dir} / {labels_dir}"
            )
        # One unusable sidecar must not cost the whole batch. A specimen that
        # cannot be processed is reported by name and skipped, so an export of 45
        # good fish still happens instead of aborting on the 46th.
        records = []
        failed: list[tuple[str, str]] = []
        for spec in specimens:
            try:
                records.append(process_specimen(spec))
            except Exception as exc:
                failed.append((spec.fish_id, str(exc)))
                log.warning("skipped %s: %s", spec.fish_id, exc)
        if failed:
            log.warning("%d of %d specimens skipped: %s",
                        len(failed), len(specimens),
                        ", ".join(f for f, _ in failed))
        if not records:
            raise RuntimeError(
                f"Every specimen failed to process ({len(failed)} of them). "
                f"First error: {failed[0][1] if failed else 'unknown'}"
            )

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

    # Say it out loud as well as in the sheet: a mixed workbook is easy to
    # misread as uniform, and the mixture is invisible once it is in Excel.
    free = [r.measurements.fish_id for r in records
            if r.calibrations.get("lateral") is not None
            and r.calibrations["lateral"].method == "none"]
    scaled = [r for r in records
              if r.calibrations.get("lateral") is not None
              and r.calibrations["lateral"].method != "none"]
    if free:
        log.warning(
            "%d of %d specimens have no scale reference; their values are PIXELS, "
            "not mm (see the 'units' column): %s",
            len(free), len(records), ", ".join(free[:6]) + (" ..." if len(free) > 6 else ""))
        if scaled:
            log.warning(
                "This workbook MIXES mm and px rows. Do not compare a length "
                "across rows without checking 'units'.")

    if drop_traits:
        log.info("omitting %d trait column(s) the study does not collect: %s",
                 len(drop_traits), ", ".join(sorted(drop_traits)))

    issues = validate(records, lot_of=_lot_of)
    counts = summarise(issues)
    if counts["error"]:
        log.warning("%d validation ERROR(s) — see the Validation sheet", counts["error"])
        for i in [x for x in issues if x.level == "error"][:5]:
            log.warning("  %s %s: %s", i.check, i.fish_id, i.message)
    if counts["warning"]:
        log.warning("%d validation warning(s) — see the Validation sheet",
                    counts["warning"])

    return export_to_xlsx(
        records, output_path, drop_traits=drop_traits, issues=issues,
        provenance={
            "dataset": images_dir.parent.name,
            "generated": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
            "commit": _git_commit(),
        })


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fish-morpho",
        description="Reproducible morphometrics for museum fish collections",
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
