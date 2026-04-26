# JCalipr

Automated morphometrics pipeline for brook trout (*Salvelinus fontinalis*)
specimens at the Cornell Museum of Vertebrates. Raw lab photos → CVAT
labeling → 30 morphometric traits → Excel.

A Python port and extension of MorFishJ, built around hybrid polygon +
keypoint annotations exported from [CVAT](https://app.cvat.ai). Replaces
hand-clicking measurements in ImageJ with a reproducible pipeline that
emits the same trait set (plus Cornell-specific extras) into a single
spreadsheet.

## What it measures

22 traits from the MorFishJ schema (SL, BD, HL, ED, CPd, PFl, etc.) plus
8 Cornell-specific extras (mouth width from frontal view, dorsal/pelvic/
anal fin heights and base lengths, fineness ratio, body area). All 30
land in one `Measurements` sheet with a parallel `QC` sheet flagging
outliers.

## Pipeline

```
raw lab photo (fish + ruler + mirror)
       │
       ├─► preprocess_cornell.py  (EXIF orient, mirror split, catalog naming)
       │
       ├─► CVAT labeling           (5 polygons + 19 keypoints + 2 ruler clicks)
       │
       ├─► cvat_to_sidecar.py      (CVAT XML → JSON sidecars per specimen)
       │
       └─► fish-morpho             (geometry → 30 traits → .xlsx)
```

## Quick start

```bash
# 1. Preprocess raw photos (EXIF orient + mirror split + catalog naming)
python3 scripts/preprocess_cornell.py \
    --raw-dir data/cornell_raw \
    --map-csv data/cornell_raw/specimen_map.csv \
    --out-dir data/cornell \
    --strain HRN

# 2. Label in CVAT (one-time setup):
#    - Create lateral project at app.cvat.ai
#    - Import labels: cvat/cvat_labels_lateral.json
#    - Upload data/cornell/lateral/ as a task
#    - See docs/CVAT_LABELING_GUIDE.txt for the per-image workflow

# 3. Export CVAT XML, then run the full pipeline:
bash scripts/run_pipeline.sh path/to/lateral_export.xml 100
#                                                       ^^^
#                                  known ruler span in mm
# → results/cornell_measurements.xlsx
```

## Catalog naming

Output crops follow the museum convention:
```
Salvelinus_fontinalis_{strain}_{specimen#}_L.JPEG   (lateral)
Salvelinus_fontinalis_{strain}_{specimen#}_F.JPEG   (frontal)
```

## Calibration

CVAT-native — drop two `ruler_point_a` / `ruler_point_b` keypoints on a
known ruler span, pass `--known-mm N` to `cvat_to_sidecar.py`, done.
Three precedence levels (highest first):

1. Per-fish JSON override (`--calibration-json`)
2. CVAT ruler keypoints + `--known-mm`
3. `--calibration-mode` fallback (`auto` runs ruler detector at process
   time; `none` forces explicit calibration)

The two ruler keypoints live in `CALIBRATION_KEYPOINTS`, disjoint from
the 19 anatomical keypoints, so they never pollute the measurement
schema.

## Layout

```
src/fish_morpho/
  landmark_config.py      Single source of truth: 5 polygons,
                          19 anatomical keypoints, 2 calibration
                          keypoints, 30 trait definitions.
  measurement_engine.py   Geometry: SL, polygon-area splits at the
                          peduncle, distance traits, fineness ratio.
  ruler_calibration.py    Auto + manual px-per-mm calibration.
  export.py               .xlsx writer (Measurements + QC sheets).
  pipeline.py             Orchestrator + CLI (fish-morpho).

scripts/
  preprocess_cornell.py   Raw lab photo → lateral + frontal crops.
  cvat_to_sidecar.py      CVAT XML 1.1 → per-image sidecar JSONs.
  export_cvat_config.py   Generate cvat_labels_*.json from schema.
  run_pipeline.sh         One-shot: CVAT XML → Excel.

cvat/                     Generated label configs (paste into CVAT).
docs/                     CVAT labeling cheat sheet.
tests/                    95 tests covering geometry, calibration,
                          schema, CVAT bridge, pipeline I/O.
```

## Install

```bash
pip install -e .
# or with dev deps:
pip install -e ".[dev]"
```

Requires Python 3.11+, NumPy ≥1.26, OpenCV ≥4.9, openpyxl ≥3.1.

## Tests

```bash
python3 -m pytest
# 95 passed
```

## Status

Manual mode (sidecar JSON in, Excel out) is production-ready. Auto mode
(DLC + SAM stack predicting polygons and keypoints) is stubbed — one
function to swap in once the model is trained on enough labeled
specimens.

## Acknowledgements

Built for the Cornell Museum of Vertebrates ichthyology collection.
Schema derives from MorFishJ (Lujan & Page) with Cornell extensions.
