# caliPr

Automated morphometrics for museum fish specimens. Raw photographs → landmark
labeling → 33 morphometric traits → an Excel workbook that validates and explains
itself. Built at the Cornell University Museum of Vertebrates.

Two studies run on it today: **brook trout** (*Salvelinus fontinalis*) hatchery
strains, and **alewife** (*Alosa pseudoharengus*) landlocked versus migratory
populations for BIOEE 4761. Each declares its own landmark set, so one engine
serves both without either being able to drift from it.

![Annotated brook trout: 5 polygons and 23 keypoints](docs/img/annotated_example.jpg)

A Python port and extension of MorFishJ. Replaces hand-clicking measurements in
ImageJ with a reproducible pipeline that emits the same trait set (plus
Cornell-specific extras) into a single spreadsheet — and keeps the annotations as
data, so the same labels export to geomorph for shape analysis and train a model
to place them automatically.

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
Cornell extras, from 5 polygons and 23 lateral keypoints. A study that does not
collect a landmark gets **no column for the traits that needed it**, rather than
a column of blanks — an empty column reads as "measured and missing" when the
truth is "never in scope".

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
raw photograph
      │
      ├─► preprocess_jonah.py        EXIF orient, mirror split, catalog naming
      │                              (CUMV trout rig only; other rigs go straight in)
      │
      ├─► label_server.py            browser labeler, one sidecar JSON per specimen
      │   build_standalone_labeler   ...or a single .html for people without Python
      │
      ├─► export_measurements.py     geometry → validation → 6-sheet .xlsx
      ├─► export_tps.py              → .tps + landmark names, for geomorph in R
      └─► render_overlays.py         → each photo with its annotation drawn on
```

Sidecars are plain JSON, one per specimen, keyed by landmark name. They are the
durable artefact: every export is derived from them, and they are small enough to
keep in git alongside the code that reads them.

## Quick start

```bash
pip install -e .

# Label. Opens at http://localhost:8765 with a dropdown for every dataset
# under data/ that has a lateral/ folder.
python scripts/label_server.py

# Measure. Validates, then writes results/<dataset>/measurements.xlsx
python scripts/export_measurements.py --dataset cornell

# Landmarks for geomorph, and a folder of annotated photographs
python scripts/export_tps.py --sidecars data/cornell/sidecars \
    --images data/cornell/lateral --out results/cornell/tps
python scripts/render_overlays.py --dataset cornell
```

All three exports are also buttons in the labeler's sidebar, so a labelling
session never needs a terminal.

Only the CUMV trout rig needs preprocessing — one photograph there holds the
lateral view, a mirrored head shot and a ruler, and has to be split:

```bash
python scripts/preprocess_jonah.py --raw-dir data/cornell_raw/jonah \
    --out-dir data/cornell --lateral-margin 450
```

## Datasets

A dataset is any folder under `data/` containing a `lateral/`. It may also hold
`frontal/`, a `sidecars/` directory, and a `schema.json` declaring what the study
collects. The labeler discovers them and offers them in a dropdown.

```
data/alewife/
  lateral/        photographs (gitignored — large, and the museum's)
  sidecars/       one JSON per specimen (tracked: this is the data)
  schema.json     which landmarks and outlines this study uses
```

`schema.json` can only *remove* from the master schema, never add — anything a
dataset invented would be unknown to the measurement engine, so a typo weakens
the task list instead of silently creating a landmark. Editing it by hand is
optional: the × beside any landmark in the labeler writes it.

| | trout (`cornell`) | alewife |
|---|---|---|
| photographs | 131 | 181, 27 collection lots, 1932–1987 |
| labelled | 46 lateral, 35 frontal | 5 |
| collects | 5 polygons, 19 keypoints | 3 polygons, 23 keypoints |
| scale | C-Thru ruler on the tray | ruler on the tank glass |
| question | strain differences | landlocked vs migratory proportions |

## Labeling

`scripts/label_server.py` serves a dependency-free browser labeler that reads the
landmark schema live from `landmark_config.py`, so the UI can never drift from
what the measurement engine expects. It writes sidecar JSON directly — no CVAT
round-trip, no import step.

Three columns: controls on the left, the landmark task list beside them, the
specimen browser on the right, both dividers draggable. Everything the schema
defines is visible at once — a collapsed list saves pixels and costs a click per
landmark, which is the wrong trade when the click is the whole job.

- **Reference panel** — an annotated real specimen beside the canvas, which zooms
  to whichever landmark is selected, with the magnification chosen from the
  reference's own resolution rather than fixed. Rebuild it from any labelled fish
  with `scripts/make_reference.py --specimen <stem>`. Minimisable, and it comes
  back.
- **Editing** — click an outline to insert a vertex, drag any point to move it,
  `Z` undoes the *selected* landmark. A live ±px readout reports the placement
  precision the current zoom level actually affords.
- **Exclusion** — the × beside a landmark marks it out of scope for the whole
  study. It greys out in the canvas, drops out of the progress count, and its
  dependent traits lose their columns in the export. This is a project-level
  decision, written to the dataset's `schema.json`, not a per-specimen one.
- **Contrast/brightness** sliders for faint fins against shadow (display only —
  they never touch coordinates).
- **Drafts** autosave to the browser and survive reload; **Save sidecar** writes
  the file the pipeline reads. A draft older than its sidecar on disk is
  discarded, so an edit made outside the browser cannot be silently overwritten
  by a stale draft.
- **Fin retrace mode** walks one fin at a time — base, tip, outline — framed and
  zoomed, hiding every other landmark so the specimen is actually visible (hold
  `H` to reveal them). The specimen list becomes a worklist of fish whose fin
  outlines are not yet dense enough to trust an area from.
- **Export panel** — measurements, TPS and overlays as three buttons, each
  reporting what it wrote.

The labeler and its reference assets are in the repository, but **the
photographs are not** — they are large and they are the museum's. A fresh clone
opens to an empty specimen list until images exist under `data/<dataset>/lateral/`;
the server says so on startup. The sidecars *are* tracked, so the annotations and
everything computed from them travel with the repo.

### Labeling without installing anything

`scripts/build_standalone_labeler.py` emits a single ~24 KB HTML file with the
schema, the reference imagery and the whole editor inlined. It runs from a
double-click, needs no Python, no server and no network, and exports a zip of
sidecar JSON that `scripts/import_standalone_labels.py` folds back into a
dataset. Photographs are loaded by the person labelling and never leave their
machine.

```bash
python scripts/build_standalone_labeler.py --out caliPr-labeler.html --theme light
python scripts/import_standalone_labels.py --zip ~/Downloads/labels.zip --dataset alewife
```

It stamps the schema version and the git commit it was built from into the file,
and every sidecar it writes records who labelled it. Duplicate filenames across
contributors are detected on import rather than quietly overwriting each other —
the failure this is guarding against is two classmates both labelling `IMG_0042`.

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

   It originally searched a band where the trout rig always puts its ruler, and
   so found the alewife ruler in 10 photographs of 181. Scanning the full frame
   height instead takes that to **180/181**, and leaves the trout numbers
   unchanged (1.04% median error either way). Detection is not accuracy, though:
   only 9 of the 27 alewife lots are internally consistent to within 25%, so the
   per-lot median check below is what makes those scales usable, not the hit
   rate.
2. **Two clicked points + known span** — the fallback. A mistyped span is the
   one calibration error nothing downstream can catch (the wrong scale is
   self-consistent), so the labeler shows a live px/mm readout and warns when a
   specimen drifts from its **collection lot's** median. Comparing against a
   whole-batch median instead flags healthy specimens, because camera distance
   genuinely varies between lots.

3. **No scale at all** — a legitimate mode, not a failure. A study asking about
   *proportions* does not need millimetres. Those specimens measure in pixels,
   the workbook says so in a per-row `units` column, and the Ratios and Shape
   sheets stay valid because both are dimensionless. What the pipeline refuses to
   do is put pixels and millimetres in one column without saying which is which.

## Export

`export_measurements.py` writes one workbook of six sheets. The extra sheets are
not decoration — each answers a question that otherwise gets answered wrong in
a spreadsheet three months later.

| sheet | what it is for |
|---|---|
| **About** | What this file is, when and from which commit it was generated, how many specimens are in millimetres versus pixels, how many checks failed, and which traits were out of scope. Written so the file explains itself to someone who did not run it. |
| **Measurements** | One row per specimen, one column per trait, plus a per-row `units` column. |
| **Ratios** | Each length over SL, each area over SL². Dimensionless — but it assumes shape does not change with size. |
| **Shape** | Mosimann log-shape variables: each length over the geometric mean of all lengths, logged. This is the defensible size correction for comparing groups that differ in body size, and the sheet a population comparison should actually use. |
| **QC** | Calibration method and confidence per view, missing landmarks, recorded data compromises. |
| **Validation** | Every automated check, most severe first. |

### Validation

`fish_morpho/validation.py` runs eight checks before the workbook is written.
Each exists because its failure mode is **silent** — none of them throw, and none
of them look wrong once the number is in a cell.

| check | the silent failure |
|---|---|
| `orientation` | A mirrored specimen swaps `Bs` and `CFs` — two plausible numbers in the wrong columns. |
| `landmark_in_frame` | A coordinate outside the image means the sidecar belongs to a different photograph, usually after a re-crop. The geometry stays self-consistent. |
| `duplicate_id` | Two sidecars claiming one `fish_id` collapse to one row. |
| `mixed_units` | Millimetres and pixels under one header. |
| `calibration_outlier` | A mistyped span scales every trait on that specimen. Judged against the collection lot, not the batch. |
| `shape_outlier` | Size-corrected proportions far from peers — usually a misplaced landmark, and far cheaper to catch here than in a scatter plot at the end. |
| `sparse_outline` | A fin outline too coarse for its area to mean anything. |
| `incomplete` | Landmarks never placed. Not an error; labelling in progress looks exactly like this. |

Thresholds are conventional, not tuned to this data: robust z of 3.5 on a
median/MAD scale, and 25% calibration drift — wide, because it is hunting a
5× typo, not a 5% wobble. On the current 60 sidecars across both datasets the
pass returns **zero errors**, 5 shape-outlier warnings, and the incomplete notes
you would expect from labelling in progress.

### geomorph and R

`export_tps.py` writes the `.tps` that `readland.tps` reads, so `digitize2d`
never has to run. Two details in that format corrupt data silently and are
handled explicitly: TPS y is Cartesian from the **bottom** left (writing image y
unflipped mirrors every specimen, and Procrustes will happily align mirrored
shapes), and TPS has no NA — the convention is a negative coordinate read back
with `negNA = TRUE`, where writing a `0` would pin a landmark to the image corner
and drag the whole fit.

TPS identifies landmarks by row order, never by name, so the export ships a
`landmark_names.csv` and an R snippet that attaches them to the array's
dimnames. The order comes from `landmark_config`, which is the same order the
measurement engine and the pose model use.

```r
library(geomorph)
A <- readland.tps("landmarks.tps", specID = "ID", negNA = TRUE)
dimnames(A)[[1]] <- read.csv("landmark_names.csv")$name
gpa <- gpagen(A)              # Procrustes: strips size, position, rotation
plot(gm.prcomp(gpa$coords))   # shape space
```

`render_overlays.py` writes each photograph with its annotation drawn on, which
is what a supervisor or a lab notebook wants and what a `.tps` can never be.

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
plan is DeepLabCut for the keypoints and SAM (prompted by those keypoints) for
the polygons — so **only the keypoints require training data**; SAM is zero-shot.
The trained model covers the 19 keypoints the trout study collects; the four fin
base endpoints were added afterwards and land in the next training round.

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

`docs/what-we-tried.md` records every approach tried on the fins, including the
ones that failed and why.

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
  landmark_config.py      Single source of truth: 5 polygons, 23 anatomical
                          keypoints, 2 calibration keypoints, 33 traits.
                          traits_requiring() maps excluded landmarks to the
                          trait columns that must disappear with them.
  measurement_engine.py   Geometry: SL, polygon-area split at the peduncle,
                          distance traits, areas, angles.
  ruler_calibration.py    Manual span, mm-tick auto-scale, ruler detector.
  anatomy_constraints.py  Anatomical bounds that clip a predicted fin outline
                          where it has left the fin.
  validation.py           The eight pre-export checks.
  export.py               .xlsx writer: About, Measurements, Ratios, Shape,
                          QC, Validation.
  pipeline.py             Orchestrator + CLI (fish-morpho).

scripts/
  label_server.py             Browser labeler (schema-driven, writes sidecars).
  labeling_ui/                Its HTML/CSS/JS and reference assets.
  build_standalone_labeler.py One self-contained .html, no Python needed.
  import_standalone_labels.py Fold a contributor's zip back into a dataset.
  make_reference.py           Rebuild the labeler's reference example.

  export_measurements.py  Dataset → validated six-sheet workbook.
  export_tps.py           Dataset → .tps + landmark names + an R snippet.
  render_overlays.py      Dataset → annotated photographs.

  preprocess_jonah.py     Catalog-named batch → lateral + frontal crops.
  preprocess_cornell.py   Img####.JPG + specimen_map.csv → crops.
  audit_auto_calibration.py  Auto-scale vs hand-clicked, per lot.
  morfishj_validation.py     Trait definitions against the MorFishJ paper.

  build_dlc_dataset.py    Sidecars → DeepLabCut project + stratified split.
  train_dlc.py            Train + evaluate the keypoint model.
  dlc_report.py           Per-landmark error in specimen millimetres.
  eval_sam_polygons.py    Zero-shot SAM vs the hand-traced polygons.
  eval_sam_zoom.py        SAM with cropping + negative prompts (fins).
  cvat_to_sidecar.py      CVAT XML 1.1 → sidecars (alternative to the labeler).
  export_cvat_config.py   The schema as a CVAT label config.
  wipe_fin_keypoints.py   Clear fin annotations for a deliberate re-trace.
  harvest_idigbio.py      Pull comparison imagery from iDigBio.
  filter_fish_vista.py    Subset the FishVista corpus.

data/<dataset>/sidecars/  Hand-labelled annotations (the valuable artifact).
docs/                     Labeling guide, figures, and what-we-tried.md — a
                          ledger of every technique attempted, failures included.
tests/                    123 tests: geometry, calibration, schema, validation,
                          export, I/O.
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
python -m pytest        # 123 passed
```

## Future work

Ordered by what blocks what. `docs/what-we-tried.md` records the fixes and dead
ends behind these — including several approaches that look obviously correct and
are not, which is worth reading before re-attempting any of them.

**Blocking the next model**

1. **Re-trace the fins on the remaining 41 specimens** — all four fins, bases,
   tips and outlines, at the 16-vertex target. Five are done. SAM cannot take any
   of them (see the table above), so this is hand work. The labeler's retrace mode
   and its worklist exist for exactly this.
2. **Rebuild the DLC dataset and retrain** once that lands. The fin tips are the
   model's weakest landmarks (`dorsal_tip` 2.55 mm, `pelvic_tip` 0.77 mm median
   with a 7.95 mm tail), and they are weak partly because the labels were. Expect
   the train/test gap (currently 2.4×) to close as much from more data as from
   better tips — 46 labeled of 131 preprocessed.
3. **Rename the DLC `SCORER` / `PROJECT` constants** from `jcalipr`. Deferred
   because it forces a dataset rebuild and retrain; do it as part of step 2, not
   separately.

**Needs someone else's answer**

4. **Confirm the `CPl` definition.** Sheet #17 gives none, so the implementation
   (posterior end of the anal fin base → `caudal_base`) is an interpretation. It
   should be checked against how it was measured by hand before the numbers are
   used.
5. **Ask the lab about TXD spreadsheet rows 42, 44, 46–50.** Those rows do not
   describe those photographs — SL expressed in ruler spans, which needs no mm
   assumption and no endpoint choice, gives TXD_46 = 156 mm against a recorded
   117.87. Tracked in `data/validation/caliper_exclusions.json` and excluded from
   validation only; the measurements themselves are sound.
6. **Decide on the 11 MorFishJ traits the spreadsheet does not request** (TL, Bs,
   AO, POC, Eh, Mo, Jl, EMd, EMa, PFi, PFb). They are computed and exported today.
   Keeping them is free; the question is whether they are wanted.
7. **Get depth-dimension caliper measurements.** Validation currently rests on SL
   alone. The spreadsheet's `body_width` and `caudal_peduncle_width` are measured
   *across* the fish, so they cannot check body depth or peduncle depth.
8. **Map the 27 alewife CUMV lots to landlocked or migratory.** This is the
   analysis blocker for that study and not a code task — every landmark in the
   world is uninterpretable until the grouping variable exists.

**Open work**

9. **Measurement repeatability is unmeasured.** Labelling 8–10 fish twice, blind,
   and computing per-landmark error would put a number on how much of any
   between-group difference is the labeller. Cheap, and it is the first thing a
   reviewer should ask for.
10. **Auto mode is a stub.** `predict_annotation()` needs wiring: DLC for
   keypoints, SAM for `body_plus_caudal` only, anatomical constraints applied,
   low-confidence points demoted to missing landmarks rather than trusted.
11. **The pectoral and dorsal remain unsolved for automation** — 18% and 35% median
   area error, and the constraints only bring the dorsal to ~22%. The residual is
   contrast (5.7 intensity levels at the pectoral), so it likely needs a different
   imaging or annotation approach rather than a better prompt.
12. **Re-fit the anatomical allowances** once more specimens are densely re-traced.
    They are currently fitted against a distribution dominated by sparse outlines.
13. **ASN_31 has an empty sidecar**; 13 specimens have frontal crops under 700 px
    (boundary too far left) and cannot be used for mouth width.
14. **ASN_30's dorsal** has a single spike vertex worth an eye.

## Status

**Manual mode is production-ready** — sidecar JSON in, Excel out, 33 traits
covering all 22 photo-measurable columns of the lab's spreadsheet, validated
against calipers at 1.39% mean absolute difference on the 35-specimen verified
set. 46 trout are labelled on the lateral view, 35 on the frontal. The export
runs clean on both datasets: **zero validation errors**, with the warnings that
remain being five shape outliers worth a second look and the incomplete notes
that labelling in progress necessarily produces.

**The fin traces are being redone.** Every fin base, tip and outline was cleared
and is being re-traced at a 16-vertex target, because the originals at 7–9
vertices carried area errors of up to ±27% in an unpredictable direction. Five
specimens are done. Until a specimen is redone, its twelve fin-derived traits
export as blank with the reason attached rather than as a plausible wrong number.
Nothing outside the fins is affected.

**The alewife study is labelling.** 181 photographs across 27 CUMV lots, 5
labelled, three polygons excluded (the pelvic and anal fins have too little
contrast and too much fraying to trace honestly, so the study drops them rather
than recording guesses). The trout keypoint model does not transfer — median
likelihood 0.18, 98% of predictions below 0.5 — which is the correct behaviour:
confidence detects a domain shift cleanly even though it is a poor guide to error
within a domain. Those fish are being labelled by hand.

**Auto mode is not ready.** The keypoint model reaches 0.92 mm median error on
9 held-out specimens — the eye landmarks are good to 0.23–0.49 mm, the skeletal
ones are usable, and `dorsal_tip` (2.55 mm) and `peduncle_narrowest_ventral`
(2.00 mm) are not. The split is verified leak-free, but train error is 0.38 mm
against test 0.92 mm, so 37 training fish is still the binding constraint rather
than the architecture. Confidence gating does not rescue it: at any useful
threshold it catches 2 of the 24 errors over 2 mm.

For polygons, SAM segments `body_plus_caudal` to 0.7% median area error against a
dense hand tracing — 52 of the 82 vertices per fish, and worth automating. It
cannot do the fins: 8–35% median depending on which, with the sign changing
between specimens. Anatomical constraints help without solving it (pectoral
+15.3% → +12.3%, dorsal +35.9% → +22.0%, and they clip nothing on any of the 46
hand tracings). `predict_annotation()` remains a stub.

## Acknowledgements

Built for the Cornell Museum of Vertebrates ichthyology collection.
Schema derives from MorFishJ (Ghilardi, M., 2022 — Leibniz Centre for
Tropical Marine Research; https://github.com/mattiaghilardi/MorFishJ,
doi:10.5281/zenodo.7275017) with Cornell extensions.
