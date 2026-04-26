"""Emit a CVAT project labels config from the fish_morpho schema.

CVAT projects can be created from a JSON labels file via the REST API or
the web UI's "Project > Constructor > Raw" tab. This script renders one
such file directly from :mod:`fish_morpho.landmark_config`, so the
labelers always see the exact set of polygons and keypoints the
measurement engine expects — no hand-copied names, no drift.

Two outputs (one per view):

  * ``cvat_labels_lateral.json`` — 5 polygon labels + 19 anatomical point
    labels + 2 calibration point labels (``ruler_point_a``,
    ``ruler_point_b``). The calibration points are part of the CVAT task
    so the labeler establishes the pixel-to-mm scale with two clicks per
    fish — the auto-ruler detector is unreliable on real museum photos.

  * ``cvat_labels_frontal.json`` — 2 point labels for ``mouth_left`` and
    ``mouth_right`` on the mirror head shot.

Usage::

    python scripts/export_cvat_config.py --out-dir cvat/

The resulting files can be uploaded via::

    curl -X POST https://<cvat-host>/api/projects \\
         -H "Authorization: Token ..." \\
         -H "Content-Type: application/json" \\
         -d @cvat/cvat_labels_lateral.json

or pasted into the Raw tab when creating a project in the CVAT UI.

Color palette
-------------
Polygons and keypoints get stable, distinguishable hex colors so the
CVAT canvas stays readable when many shapes overlap. Colors are picked
deterministically from the label name so re-running the script never
churns the palette.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import sys
from pathlib import Path

# Allow running as a script without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fish_morpho.landmark_config import (  # noqa: E402
    CALIBRATION_KEYPOINTS,
    KEYPOINTS,
    POLYGONS,
    View,
    calibration_keypoint_by_name,
    calibration_keypoint_names,
    keypoint_by_name,
    keypoint_names,
    polygon_by_name,
    polygon_names,
)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _stable_color(name: str) -> str:
    """Deterministic pastel-ish hex color derived from the label name."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.92)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# Label builders
# ---------------------------------------------------------------------------


def _polygon_label(name: str) -> dict:
    p = polygon_by_name(name)
    return {
        "name": name,
        "color": _stable_color(f"polygon:{name}"),
        "type": "polygon",
        "attributes": [
            {
                "name": "description",
                "mutable": False,
                "input_type": "text",
                "default_value": p.description,
                "values": [p.description],
            },
            {
                "name": "labeling_hint",
                "mutable": False,
                "input_type": "text",
                "default_value": p.labeling_hint,
                "values": [p.labeling_hint],
            },
        ],
    }


def _keypoint_label(name: str, *, calibration: bool = False) -> dict:
    """Build a CVAT `points` label for an anatomical or calibration keypoint.

    ``calibration=True`` pulls from ``CALIBRATION_KEYPOINTS`` instead of
    the anatomical ``KEYPOINTS`` tuple and uses a distinct color-namespace
    prefix so the labeler can tell ruler clicks from landmark clicks at
    a glance.
    """
    k = calibration_keypoint_by_name(name) if calibration else keypoint_by_name(name)
    color_prefix = "calibration" if calibration else "keypoint"
    return {
        "name": name,
        "color": _stable_color(f"{color_prefix}:{name}"),
        "type": "points",
        "attributes": [
            {
                "name": "description",
                "mutable": False,
                "input_type": "text",
                "default_value": k.description,
                "values": [k.description],
            },
            {
                "name": "labeling_hint",
                "mutable": False,
                "input_type": "text",
                "default_value": k.labeling_hint,
                "values": [k.labeling_hint],
            },
        ],
    }


def build_lateral_config(project_name: str = "fish_morpho_lateral") -> dict:
    """Render the lateral-view CVAT project config."""
    labels: list[dict] = []
    for pname in polygon_names():  # all polygons are lateral
        labels.append(_polygon_label(pname))
    for kname in keypoint_names(View.LATERAL):
        labels.append(_keypoint_label(kname))
    for kname in calibration_keypoint_names(View.LATERAL):
        labels.append(_keypoint_label(kname, calibration=True))
    return {
        "name": project_name,
        "labels": labels,
    }


def build_frontal_config(project_name: str = "fish_morpho_frontal") -> dict:
    """Render the frontal (mirror mouth-width) CVAT project config."""
    labels = [_keypoint_label(k) for k in keypoint_names(View.FRONTAL)]
    return {
        "name": project_name,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="export_cvat_config",
        description="Emit CVAT project labels JSON files from the "
        "fish_morpho schema (one per view).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cvat"),
        help="Directory to write cvat_labels_{lateral,frontal}.json into.",
    )
    p.add_argument(
        "--lateral-project-name",
        default="fish_morpho_lateral",
        help="CVAT project name for the lateral config.",
    )
    p.add_argument(
        "--frontal-project-name",
        default="fish_morpho_frontal",
        help="CVAT project name for the frontal config.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lateral = build_lateral_config(args.lateral_project_name)
    frontal = build_frontal_config(args.frontal_project_name)

    lateral_path = out_dir / "cvat_labels_lateral.json"
    frontal_path = out_dir / "cvat_labels_frontal.json"
    lateral_path.write_text(json.dumps(lateral, indent=2) + "\n")
    frontal_path.write_text(json.dumps(frontal, indent=2) + "\n")

    print(f"Wrote {lateral_path} ({len(lateral['labels'])} labels)")
    print(f"Wrote {frontal_path} ({len(frontal['labels'])} labels)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
