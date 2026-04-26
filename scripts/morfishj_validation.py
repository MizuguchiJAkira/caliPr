"""Validate the fish_morpho port against MorFishJ GUI output.

Running MorFishJ (the ImageJ plugin) by hand on a reference set produces
a CSV of the 22 canonical traits per specimen. This script loads that
CSV alongside the same specimens' sidecar JSONs and runs them through
the Python engine, then prints a per-trait diff report with tolerance
and exits nonzero if anything is out of spec.

Usage::

    python scripts/morfishj_validation.py \\
        --reference data/validation/morfishj_reference.csv \\
        --labels    data/validation/labels/ \\
        --tolerance-mm 0.2 \\
        --tolerance-mm2 2.0 \\
        --tolerance-deg 1.0

Reference CSV format
--------------------
One row per fish. Required columns:

  * ``fish_id`` — matches the sidecar filename stem.
  * One column per MorFishJ trait code (TL, SL, MBd, Hl, Hd, Ed, Eh,
    Snl, POC, AO, EMd, EMa, Mo, Jl, Bs, CPd, CFd, CFs, PFs, PFl, PFi,
    PFb). Blank cells mean "MorFishJ didn't report this trait for this
    specimen" and are skipped.

Cornell-extras trait codes (LJl, DFh, DFs, PlFl, PlFs, AFh, AFs, MW) are
ignored — MorFishJ doesn't compute them, so there's no oracle to
compare against.

Tolerance semantics
-------------------
Each trait is compared using the tolerance appropriate to its unit
(``--tolerance-mm``, ``--tolerance-mm2``, or ``--tolerance-deg``). The
defaults (0.2 mm, 2.0 mm^2, 1.0 deg) are the hand-measurement noise
floor we're willing to accept for the port to be "good enough" — they
should be revisited once the real reference set lands.

Exit codes
----------
  0 — every compared trait is within tolerance.
  1 — at least one trait deviated (printed with Python vs MorFishJ values).
  2 — input error (missing files, malformed CSV, etc.).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as a script without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fish_morpho.landmark_config import (  # noqa: E402
    TraitSource,
    Unit,
    View,
    trait_by_code,
    traits_by_source,
)
from fish_morpho.measurement_engine import (  # noqa: E402
    Annotation,
    MeasurementSet,
    compute_all,
)
from fish_morpho.ruler_calibration import (  # noqa: E402
    CalibrationResult,
    scale_from_known_span,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Deviation:
    fish_id: str
    trait_code: str
    unit: str
    python_value: float
    morfishj_value: float
    delta: float
    tolerance: float

    def format(self) -> str:
        return (
            f"  {self.fish_id:<20} {self.trait_code:<6} "
            f"python={self.python_value:>10.4f} {self.unit:<4} "
            f"morfishj={self.morfishj_value:>10.4f} {self.unit:<4} "
            f"Δ={self.delta:+.4f} (tol ±{self.tolerance:.4f})"
        )


# ---------------------------------------------------------------------------
# Sidecar -> Annotation (same shape the pipeline produces, minimal copy)
# ---------------------------------------------------------------------------


def _coerce_point(raw) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"Expected [x, y], got {raw!r}")
    return float(raw[0]), float(raw[1])


def _coerce_polygon(raw, name: str) -> list[tuple[float, float]]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(
            f"Polygon {name!r} must be a list of >= 3 [x, y] points, got {raw!r}"
        )
    return [_coerce_point(v) for v in raw]


def _load_sidecar(path: Path) -> tuple[Annotation, dict]:
    """Return an Annotation plus the raw sidecar dict (for calibration)."""
    with path.open() as f:
        data = json.load(f)

    ann = Annotation()
    lateral = data.get("lateral") or {}
    for pname, verts in (lateral.get("polygons") or {}).items():
        ann.polygons[pname] = _coerce_polygon(verts, pname)
    for kname, xy in (lateral.get("keypoints") or {}).items():
        ann.keypoints[kname] = _coerce_point(xy)
    frontal = data.get("frontal") or {}
    for kname, xy in (frontal.get("keypoints") or {}).items():
        ann.keypoints[kname] = _coerce_point(xy)
    return ann, data


def _calibration_from_block(block: dict | None) -> CalibrationResult | None:
    """Only the manual-mode branch is supported for validation runs."""
    if not block:
        return None
    mode = block.get("mode", "manual")
    if mode != "manual":
        raise ValueError(
            "Validation oracle only supports manual-mode calibration blocks "
            "(auto mode would couple the oracle to the ruler detector)."
        )
    a = _coerce_point(block["point_a"])
    b = _coerce_point(block["point_b"])
    return scale_from_known_span(a, b, float(block["known_mm"]))


def _compute_for_fish(sidecar_path: Path, fish_id: str) -> MeasurementSet:
    ann, data = _load_sidecar(sidecar_path)
    lateral_calib = _calibration_from_block((data.get("lateral") or {}).get("calibration"))
    if lateral_calib is None:
        raise ValueError(f"{sidecar_path}: lateral calibration is required")
    calibrations = {View.LATERAL: lateral_calib}
    frontal_block = data.get("frontal") or {}
    frontal_calib = _calibration_from_block(frontal_block.get("calibration"))
    if frontal_calib is not None:
        calibrations[View.FRONTAL] = frontal_calib
    return compute_all(fish_id=fish_id, annotation=ann, calibrations=calibrations)


# ---------------------------------------------------------------------------
# Reference CSV loader
# ---------------------------------------------------------------------------


def _load_reference(csv_path: Path) -> dict[str, dict[str, float]]:
    """Return ``{fish_id: {trait_code: value}}`` from MorFishJ GUI output."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Reference CSV not found: {csv_path}")
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "fish_id" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing 'fish_id' column")
        morfishj_codes = {
            t.code for t in traits_by_source(TraitSource.MORFISHJ)
        }
        out: dict[str, dict[str, float]] = {}
        for row in reader:
            fid = row["fish_id"].strip()
            if not fid:
                continue
            values: dict[str, float] = {}
            for code in morfishj_codes:
                raw = (row.get(code) or "").strip()
                if raw == "":
                    continue
                try:
                    values[code] = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{csv_path}: fish {fid!r} has non-numeric {code}={raw!r}"
                    ) from exc
            out[fid] = values
    return out


# ---------------------------------------------------------------------------
# Comparison core
# ---------------------------------------------------------------------------


def _tolerance_for(trait_code: str, tol_mm: float, tol_mm2: float, tol_deg: float) -> float:
    unit = trait_by_code(trait_code).unit
    if unit is Unit.MM:
        return tol_mm
    if unit is Unit.MM2:
        return tol_mm2
    if unit is Unit.DEG:
        return tol_deg
    raise AssertionError(f"Unknown unit for {trait_code}: {unit}")


def compare(
    reference: dict[str, dict[str, float]],
    sidecars: dict[str, Path],
    tol_mm: float,
    tol_mm2: float,
    tol_deg: float,
) -> list[Deviation]:
    """Run every reference fish through the Python port and diff against MorFishJ."""
    deviations: list[Deviation] = []
    for fish_id, ref_values in sorted(reference.items()):
        sidecar = sidecars.get(fish_id)
        if sidecar is None:
            raise FileNotFoundError(
                f"Reference row {fish_id!r} has no matching sidecar JSON"
            )
        ms = _compute_for_fish(sidecar, fish_id)
        for code, ref in ref_values.items():
            mv = ms.values.get(code)
            if mv is None:
                raise KeyError(
                    f"Trait {code!r} missing from Python output for {fish_id!r}"
                )
            if math.isnan(mv.value):
                deviations.append(
                    Deviation(
                        fish_id=fish_id,
                        trait_code=code,
                        unit=mv.unit,
                        python_value=math.nan,
                        morfishj_value=ref,
                        delta=math.nan,
                        tolerance=_tolerance_for(code, tol_mm, tol_mm2, tol_deg),
                    )
                )
                continue
            tol = _tolerance_for(code, tol_mm, tol_mm2, tol_deg)
            delta = mv.value - ref
            if abs(delta) > tol:
                deviations.append(
                    Deviation(
                        fish_id=fish_id,
                        trait_code=code,
                        unit=mv.unit,
                        python_value=mv.value,
                        morfishj_value=ref,
                        delta=delta,
                        tolerance=tol,
                    )
                )
    return deviations


def discover_sidecars(labels_dir: Path) -> dict[str, Path]:
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels dir not found: {labels_dir}")
    return {p.stem: p for p in sorted(labels_dir.glob("*.json"))}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="morfishj_validation",
        description="Diff the fish_morpho Python port against MorFishJ GUI output.",
    )
    p.add_argument("--reference", type=Path, required=True, help="CSV of MorFishJ output")
    p.add_argument("--labels", type=Path, required=True, help="Sidecar JSON directory")
    p.add_argument("--tolerance-mm", type=float, default=0.2)
    p.add_argument("--tolerance-mm2", type=float, default=2.0)
    p.add_argument("--tolerance-deg", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        reference = _load_reference(args.reference)
        sidecars = discover_sidecars(args.labels)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        deviations = compare(
            reference,
            sidecars,
            tol_mm=args.tolerance_mm,
            tol_mm2=args.tolerance_mm2,
            tol_deg=args.tolerance_deg,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    total_compared = sum(len(v) for v in reference.values())
    print(
        f"Compared {len(reference)} fish × ~{total_compared // max(len(reference), 1)} "
        f"traits = {total_compared} values"
    )
    if not deviations:
        print("OK — every trait within tolerance")
        return 0

    print(f"FAIL — {len(deviations)} trait(s) out of tolerance:")
    for d in deviations:
        print(d.format())
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
