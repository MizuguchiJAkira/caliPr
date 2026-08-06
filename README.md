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

22 traits from the MorFishJ schema (SL, MBd, Hl, Ed, CPd, PFl, …) plus 11
Cornell extras. All 33 land in one `Measurements` sheet with a parallel `QC`
sheet carrying calibration provenance, missing landmarks, and any recorded data
compromise.

The lab's own spreadsheet is the requirements list, and the pipeline now covers
**every one of its 22 photo-measurable columns**. Three of its columns are
deliberately out of scope: `weight(g)` is a mass, and `body_width` /
`caudal_peduncle_width` are lateral dimensions measured *across* the fish, which
a lateral photograph cannot show. (That is also why those two columns cannot be
used to validate body depth or peduncle depth — they measure a perpendicular
axis.) `tests/test_landmark_config.py` pins the coverage so a column cannot be
dropped silently.

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

## A caveat on the fin traits

Two separate things degrade the fin traits, and they need untangling.

**Tracing density.** A sparse outline is *imprecise*, and — this took a direct
test to establish — not biased in a predictable direction. The body outline is
traced with ~52 vertices and the fins were traced with 7–9, which is not enough.
Subsampling the body outline (dense enough to be its own ground truth) down to k
of its own real vertices loses area monotonically:

| vertices kept | 5 | 7 | 9 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|
| area error | −26% | −13% | −9% | −9% | −5% | −2% |

That measures *subsampling an accurate dense trace*, though, which is not what a
human does with 7 clicks — they place a few points and interpolate by eye, which
can run generous as easily as tight. Re-tracing one fish (HRN_5) at 46–86
vertices per fin and comparing to its own 7–11-vertex original:

| fin | old → new vertices | area change |
|---|---|---|
| anal | 7 → 55 | **+27.1%** |
| pelvic | 10 → 86 | +1.3% |
| pectoral | 11 → 46 | −3.1% |
| dorsal | 10 → 62 | **−7.8%** |

So the sign varies. `FIN_POLYGON_TARGET_VERTICES` is 16 because errors of that
size in *either* direction cannot be corrected after the fact, not because sparse
tracing reads low. The labeler shows a live `n/16` counter per fin and the
pipeline stamps a `data_note` on any specimen below it.

(An earlier version of this section cited a positive correlation between vertex
count and size-corrected fin area as evidence of a density bias. That is
confounded: click count also tracks fin *size* — r = +0.49 dorsal, +0.43 pelvic —
so a bigger fin gets both more clicks and more area without any bias at all.)

**Preservation.** The larger term, and the one no amount of tracing precision
fixes. Restricting to fins already traced at ≥8 vertices, size-corrected
variability is still 60.1% for dorsal fin area and 42.7% for pelvic, against
**5.4% for body area** on the same photographs and the same tracing. Alcohol
preservation dries the fins to where they cannot splay without fraying, so how
far a fin extends depends largely on how that specimen dried and was pinned.

Seven traits are affected (DFh, AFh, PlFl, DFs, PlFs, AFs, and to a lesser
extent PFs). They are still computed, but they record preservation state as much
as morphology. Some of the spread is genuine between-fish variation in fin size,
which this data cannot fully separate.

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
python scripts/build_dlc_dataset.py --out dlc --scale 0.15   # sidecars → DLC
python scripts/train_dlc.py --epochs 300 --batch-size 2      # train + evaluate
python scripts/dlc_report.py --project dlc_project/jcalipr-* # error in mm
```

The split is stratified by strain, and absent landmarks are written as NaN
rather than a placeholder, so a clipped snout never teaches the model to predict
the frame edge.

**Keep `--batch-size` small.** The default batch of 8 at 0.25 scale exhausts
memory on a 16 GB machine once the crops carry the lateral margin: training
wedges at epoch 3 with the process in uninterruptible disk wait and swap
effectively full. Batch 2 at 0.25 scale completes 300 epochs in ~37 min on
Apple MPS with room to spare. Dropping to `--scale 0.15` also works (~25 min)
but costs accuracy — see below.

### Current model

37 train / 9 held-out. **Median held-out error 0.81 mm** — comfortably better
than the manual pipeline's own caliper agreement of ~1.4 mm. Thirteen of
nineteen landmarks are under 1 mm and five are under 0.5 mm, led by `eye_dorsal`
at 0.22 mm.

Report medians, not means. DLC's summary CSV gives per-landmark *means*, and on
a 9-specimen held-out set a single failure dominates: `pectoral_ray_tip` reads
12.26 mm by mean and **0.94 mm by median**, because eight of nine specimens are
sub-millimetre and one is off by 104 mm. `pelvic_tip` is similarly 2.09 mean
against 0.77 median. `scripts/dlc_report.py` reads the raw prediction HDF5 and
reports both.

Resolution turned out to matter more than data volume for the posterior
landmarks. `peduncle_narrowest_dorsal`/`_ventral` sit only ~105 full-res px from
`caudal_base`; at 0.15 scale that is ~16 px, close enough that the model could
not separate three distinct landmarks. Retraining at 0.25 scale halved their
error (2.31 → 1.26 mm and 2.60 → 1.79 mm). This was worth chasing because the
pair defines reference line A, so it propagates into Bs, CFs, CFd, MBd, Eh, Mo,
PFi and PFb, and directly measures CPd.

Progression: 3.88 mm (28 images, 0.15) → 1.40 mm (37 images, 0.15) → 1.10 mm
(37 images, 0.25). The first step removed overfitting; the second removed a
resolution limit.

**Still weak:** `dorsal_tip` (2.10 mm) and `peduncle_narrowest_ventral`
(2.01 mm) are the only landmarks above 2 mm by median. `dorsal_tip` is one of
the two whose anatomical definition is still marked pending, and it also shows
the widest placement scatter in the hand labels — the definition is likely the
limiting factor, not the model.

**The model signals its own failures, but confidence is not comparable across
landmarks.** The single 104 mm `pectoral_ray_tip` miss came with likelihood 0.21
against 0.62–0.89 for that landmark's good predictions — clearly flagged. But
`caudal_base` sits at 0.24–0.45 on *every* specimen while being accurate to
0.97 mm, so a flat cutoff of 0.5 discards all nine good predictions. The report
therefore gates each landmark against its own median likelihood (`--relative`),
which rejects 2.4% of predictions instead of 19%. Rejected points become missing
landmarks, which the pipeline already handles by NaN-ing the dependent traits
with a reason — a declared gap rather than a confident wrong number.

That single failure is a size-generalisation gap, not a bad photo: HRN_46 is the
smallest fish in the set (SL 92 mm against a 138.6 mm median), and the model saw
mostly 130–160 mm specimens.

### Polygons: SAM

`scripts/eval_sam_polygons.py` benchmarks zero-shot SAM against the hand-traced
outlines at full frame; `scripts/eval_sam_zoom.py` adds cropping and
negative-point prompts. Median area error, **best prompt per polygon**, over
45–46 specimens:

| polygon | SAM | how | verdict |
|---|---|---|---|
| `body_plus_caudal` | **1.4%** | full frame, point prompts | matches hand tracing |
| anal | 11.1% | full frame | hand-trace |
| pelvic | 16.7% | crop + negative points | hand-trace |
| dorsal | 22.6% | full frame | hand-trace |
| pectoral | 29.2% | crop + ventral negatives | hand-trace |

The body outline is **52 of the 82 vertices** traced per fish, so SAM can take
62% of the polygon work at hand-tracing accuracy. No fin comes close, and the
gap is not a tuning problem — see below.

**Cropping helps only the two small fins, and hurts the other two.** SAM resizes
its input to 1024 px, so on a full 4400 px frame the pectoral is a handful of
pixels; cropping to it takes the pectoral from 238% to 31% and the pelvic from
27% to 17%. The dorsal and anal are large enough to survive the resize, and
cropping tight around them *removes* the body context that tells SAM where the
fin ends — dorsal degrades 23% → 45%, anal 11% → 18%. So the two strategies are
not interchangeable, and the table above mixes them deliberately.

**Negative points must go ventral for the pectoral, and only the pectoral.**
Prompting a point on a fin makes SAM return object-level masks — often the whole
fish — so negative points on the flank are what separate a part from its whole.
Where they go matters: brook trout pectorals angle dorsally and never run along
the belly margin, so a symmetric pair clips the fin's upper edge. Ventral-only
negatives fix that (48% → 29%). For the other three fins the same change is a
wash or worse (anal 18% → 27%), which is what you would expect — the asymmetry
is a fact about pectoral geometry, not a general prompting trick.

**Measured against dense re-tracings, SAM cannot do the fins.** The table above
used the old sparse outlines as reference, so it was partly measuring our own
tracing. Five specimens have now been re-traced at 38–86 vertices per fin; against
those:

| polygon | ASN_24 | ASN_27 | ASN_30 | HRN_42 | HRN_5 | median \|err\| |
|---|---|---|---|---|---|---|
| `body_plus_caudal` | +0.2% | +0.7% | +0.7% | +3.2% | −2.4% | **0.7%** |
| anal | −3.7% | −9.9% | −18.9% | −8.4% | +2.3% | 8.4% |
| pectoral | +23.9% | +40.5% | −8.4% | +18.2% | +15.3% | 18.2% |
| pelvic | +39.8% | −20.8% | +52.0% | +12.1% | +1.7% | 20.8% |
| dorsal | +3.0% | −67.1% | −59.7% | +12.8% | +35.4% | 35.4% |

The first fish re-traced (HRN_5) gave 1.7% pelvic and 2.3% anal, which looked
like SAM could take both fins and halve the remaining hand-tracing. It does not
replicate: the other four give +40%, −21%, +52%, +12% on the pelvic. One specimen
was not enough to see that, and the swing is not a bias that could be corrected —
it changes sign between fish.

So the recommendation is unchanged from the original benchmark, for a better
reason: **hand-trace all four fins, and let SAM take `body_plus_caudal` only.**
That one holds up under a sharper reference — 0.7% median, 3.2% worst — and it is
still 52 of the 82 vertices per fish.

The residual is contrast. Fin-to-surround separation is 5.7 intensity levels for
the pectoral, 10.6 pelvic, 17.7 dorsal, 29.7 anal, and SAM's full-frame error
tracks that almost monotonically. There is no edge to find. CLAHE did not help —
it amplifies styrofoam texture along with the boundary.

Subtracting the body mask to isolate the dorsal and anal fins does **not** work,
despite those fins sitting 96% outside the hand-traced body outline: SAM's mask
is the whole *animal* and already contains 55–70% of them.

## Layout

```
src/fish_morpho/
  landmark_config.py      Single source of truth: 5 polygons, 19 anatomical
                          keypoints, 2 calibration keypoints, 33 traits.
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
  eval_sam_polygons.py    Zero-shot SAM vs the hand-traced polygons.
  eval_sam_zoom.py        SAM with cropping + negative prompts (fins).
  cvat_to_sidecar.py      CVAT XML 1.1 → sidecars (alternative to the labeler).

data/cornell/sidecars/    Hand-labeled annotations (the valuable artifact).
docs/                     Labeling guide + figures.
tests/                    98 tests: geometry, calibration, schema, I/O.
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
python -m pytest        # 98 passed
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
