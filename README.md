# caliPr

Automated morphometrics pipeline for brook trout (*Salvelinus fontinalis*)
specimens at the Cornell Museum of Vertebrates. Raw lab photos → landmark
labeling → 30 morphometric traits → Excel.

![Annotated brook trout: 5 polygons and 19 keypoints](docs/img/annotated_example.jpg)

A Python port and extension of MorFishJ. Replaces hand-clicking measurements in
ImageJ with a reproducible pipeline that emits the same trait set (plus
Cornell-specific extras) into a single spreadsheet — and keeps the annotations
as data, so the same labels can train a model to place them automatically.

**Validated against physical caliper measurements.** Across 35 hand-labeled
specimens spanning three hatchery strains, standard length agrees to a
**median 1.12%** (mean 1.39%) with a **+0.69% bias** — random scatter rather
than a measurement offset.

The specimen-to-caliper mapping was itself verified rather than assumed: an
offset sweep confirms ASN and HRN align at offset 0 (1.40% / 1.56% mean error,
versus 12–20% at any shift). Seven TXD rows disagree with their photos by
5–32% with mixed sign and no offset explains it; those measurements check out
independently (SL in *ruler spans*, which needs no millimetre assumption and no
choice of posterior endpoint), so the spreadsheet rows are the suspect party.
They are listed in `data/validation/caliper_exclusions.json` and excluded from
validation only — still valid for measurement and for training.

## What it measures

22 traits from the MorFishJ schema (SL, MBd, Hl, Ed, CPd, PFl, …) plus 8
Cornell extras (mouth width from the frontal mirror view, dorsal/pelvic/anal fin
heights and areas, lower jaw length). All 30 land in one `Measurements` sheet
with a parallel `QC` sheet carrying calibration provenance, missing landmarks,
and any recorded data compromise.

## Pipeline

```
raw lab photo (fish + ruler + mirror)
       │
       ├─► preprocess_jonah.py     EXIF orient, mirror split, catalog naming
       │
       ├─► label_server.py         browser labeler: 5 polygons + 19 keypoints
       │                           (writes sidecar JSON directly)
       │
       └─► fish-morpho             geometry → 30 traits → .xlsx
```

## Quick start

```bash
# 1. Preprocess raw photos into lateral + frontal crops
python scripts/preprocess_jonah.py --raw-dir data/cornell_raw/jonah \
    --out-dir data/cornell --lateral-margin 450

# 2. Label in the browser (http://localhost:8765)
python scripts/label_server.py

# 3. Measure everything that has been labeled
python -c "
import json,glob,sys; sys.path.insert(0,'src')
from pathlib import Path
from fish_morpho.pipeline import SpecimenInput, process_specimen
from fish_morpho.export import export_to_xlsx
recs=[]
for f in sorted(glob.glob('data/cornell/sidecars/*.json')):
    sc=json.load(open(f)); fid=sc['fish_id']
    recs.append(process_specimen(SpecimenInput(fid,
        Path(f'data/cornell/lateral/{fid}_L.JPEG'), Path(f), sc)))
export_to_xlsx(recs,'results/cornell_measurements.xlsx')"
```

## Labeling

`scripts/label_server.py` serves a dependency-free browser labeler that reads
the landmark schema live from `landmark_config.py`, so the UI can never drift
from what the measurement engine expects. It writes sidecar JSON directly — no
CVAT round-trip.

- **Reference panel** — an annotated example specimen beside the canvas, which
  zooms to whichever landmark is selected. Rebuild it from any labeled fish
  with `scripts/make_reference.py --specimen <stem>`.
- **Editing** — click the outline to insert a vertex, drag any point to move it,
  `Z` undoes the *selected* landmark.
- **Contrast/brightness** sliders for faint fins against shadow (display only —
  never touches coordinates).
- **Drafts autosave** to the browser and survive reload; **Save sidecar** writes
  the file the pipeline reads.

## Calibration

Two routes, both producing px/mm:

1. **Ruler auto-scale (preferred)** — `detect_tick_scale()` measures the
   millimetre ticks directly, no clicks and no typed span. The C-Thru rulers
   carry *both* metric and imperial scales, and 1/16 in = 1.5875 mm, so the
   imperial ticks appear as a period ~1.59× coarser; taking the strongest
   autocorrelation peak picks those and inflates every trait ~59%. The
   millimetre ticks are the *finest* true periodicity, so the detector takes the
   smallest period several bands agree on. Validated against 20 hand-clicked
   calibrations across two camera distances: **mean error 1.0%, max 2.5%, 20/20
   within 3%**.
2. **Two clicked points + known span** — the fallback. A mistyped span is the
   one calibration error nothing downstream can catch (the wrong scale is
   self-consistent), so the labeler shows a live px/mm readout and warns when a
   specimen drifts from the batch median.

## Data compromises

A clipped fin or snout must not silently produce a wrong number. Flagging a
compromise in the labeler records a `data_note` plus the traits it invalidates;
the pipeline then force-NaNs exactly those traits, logs the compromise at
runtime, and carries the reason onto the QC sheet. Everything still measurable
is kept — a fish with a clipped tail still yields its 26 body traits.

## Auto mode (in progress)

`predict_annotation()` in `pipeline.py` is the single integration point. The
plan is DeepLabCut for the 19 keypoints and SAM (prompted by those keypoints)
for the 5 polygons — so **only the keypoints require training data**; SAM is
zero-shot.

```bash
python scripts/build_dlc_dataset.py --out dlc      # sidecars → DLC format
python scripts/train_dlc.py --epochs 300           # train + per-keypoint eval
python scripts/dlc_report.py --project dlc_project/jcalipr-*  # error in mm
```

The split is stratified by strain, and absent landmarks are written as NaN
rather than a placeholder, so a clipped snout never teaches the model to predict
the frame edge.

## Layout

```
src/fish_morpho/
  landmark_config.py      Single source of truth: 5 polygons, 19 anatomical
                          keypoints, 2 calibration keypoints, 30 traits.
  measurement_engine.py   Geometry: SL, polygon-area split at the peduncle,
                          distance traits, areas, angles.
  ruler_calibration.py    Manual span, mm-tick auto-scale, ruler detector.
  export.py               .xlsx writer (Measurements + QC sheets).
  pipeline.py             Orchestrator + CLI (fish-morpho).

scripts/
  preprocess_jonah.py     Catalog-named batch → lateral + frontal crops.
  preprocess_cornell.py   Img####.JPG + specimen_map.csv → crops.
  label_server.py         Browser labeler (schema-driven, writes sidecars).
  make_reference.py       Rebuild the labeler's reference example.
  build_dlc_dataset.py    Sidecars → DeepLabCut project + stratified split.
  train_dlc.py            Train + evaluate the keypoint model.
  dlc_report.py           Per-landmark error in specimen millimetres.
  cvat_to_sidecar.py      CVAT XML 1.1 → sidecars (alternative to the labeler).

data/cornell/sidecars/    Hand-labeled annotations (the valuable artifact).
docs/                     Labeling guide + figures.
tests/                    96 tests: geometry, calibration, schema, I/O.
```

## Install

```bash
pip install -e ".[dev]"
```

Python 3.11+ (NumPy ≥1.26, OpenCV ≥4.9, openpyxl ≥3.1, Pillow ≥10). Editable
installs need pip ≥21.3. The DeepLabCut stack is heavy and pinned separately —
install it into its own environment rather than alongside the pipeline.

## Tests

```bash
python -m pytest        # 96 passed
```

## Status

Manual mode (sidecar JSON in, Excel out) is production-ready and validated
against calipers. 55 specimens are labeled (46 lateral, 35 frontal, 27 with
both views) and mouth width is populated for the first time.

Auto mode is in training. A pilot on 28 images reached a 0.84 mm *training*
error — better than the manual pipeline's own agreement with calipers — but
3.88 mm on held-out fish, a gap that says the architecture works and data is
the constraint. The labeled set is being built toward ~60–80.

## Acknowledgements

Built for the Cornell Museum of Vertebrates ichthyology collection.
Schema derives from MorFishJ (Ghilardi, M., 2022 — Leibniz Centre for
Tropical Marine Research; https://github.com/mattiaghilardi/MorFishJ,
doi:10.5281/zenodo.7275017) with Cornell extensions.
