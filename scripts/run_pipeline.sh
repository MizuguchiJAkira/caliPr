#!/usr/bin/env bash
# run_pipeline.sh — One-shot: CVAT XML export → sidecar JSONs → Excel
#
# Usage:
#   bash scripts/run_pipeline.sh /path/to/lateral_export.xml [KNOWN_MM]
#
# Arguments:
#   $1  Path to the CVAT XML export (lateral project, "CVAT for images 1.1")
#   $2  Known ruler span in mm (default: 100 — two 10-cm tick marks)
#
# Output:
#   data/cornell/labels/*.json   (one sidecar per specimen)
#   results/cornell_measurements.xlsx

set -euo pipefail
cd "$(dirname "$0")/.."

CVAT_XML="${1:?Usage: bash scripts/run_pipeline.sh <lateral.xml> [known_mm]}"
KNOWN_MM="${2:-100}"

LABEL_DIR="data/cornell/labels"
IMAGE_DIR="data/cornell/lateral"
OUT_XLSX="results/cornell_measurements.xlsx"

mkdir -p "$LABEL_DIR" results

echo "=== Step 1: CVAT XML → sidecar JSONs ==="
echo "  XML:      $CVAT_XML"
echo "  Known mm: $KNOWN_MM"
echo "  Out:      $LABEL_DIR/"
python3 scripts/cvat_to_sidecar.py \
    --cvat-xml "$CVAT_XML" \
    --view lateral \
    --out-dir "$LABEL_DIR" \
    --known-mm "$KNOWN_MM"

echo ""
echo "=== Step 2: Measure → Excel ==="
echo "  Images: $IMAGE_DIR/"
echo "  Labels: $LABEL_DIR/"
echo "  Out:    $OUT_XLSX"
python3 -m fish_morpho.pipeline \
    --images "$IMAGE_DIR" \
    --labels "$LABEL_DIR" \
    --out "$OUT_XLSX" \
    --mode manual

echo ""
echo "=== Done ==="
echo "Open $OUT_XLSX in Excel to review measurements."
echo "(Measurements sheet = 30 traits; QC sheet = flagged outliers.)"
