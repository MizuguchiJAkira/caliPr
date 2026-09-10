# caliPr

Reproducible morphometrics for museum fish collections. caliPr records landmarks
on specimen photographs and derives 33 morphometric traits, coordinates in TPS
format for geometric morphometrics, and annotated images, from a single stored
annotation per specimen.

It implements the trait schema of MorFishJ (Ghilardi 2022) with Cornell
extensions, and separates measurement from digitisation: landmarks are stored as
coordinates, and every trait, ratio and figure is derived from them. Changing a
trait definition is a re-run rather than a re-digitisation, and the same
coordinates train a model to place them automatically.

Standard length agrees with physical caliper measurements to a median of 1.12%
across 35 specimens (see [Accuracy](#accuracy)).

![Annotated brook trout: 5 polygons and 23 keypoints](docs/img/annotated_example.jpg)

## Installation

```bash
git clone https://github.com/MizuguchiJAkira/caliPr.git
cd caliPr
python3.11 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Python 3.11 or newer. Four dependencies (NumPy, OpenCV, openpyxl, Pillow),
approximately seven seconds. The system Python on macOS is 3.9 and will be
refused.

Automated landmarking requires PyTorch, DeepLabCut and Segment Anything, which
are installed separately and are not needed for manual labelling or export:

```bash
python3.11 -m venv .venv-train
./.venv-train/bin/pip install "deeplabcut[tf]" transformers torch
```

Development and all timings reported here are on Apple Silicon (M3, 8 GB) with
PyTorch on the MPS backend. There is no CUDA requirement.

## Usage

```bash
./.venv/bin/python scripts/label_server.py
```

This serves a browser interface at `http://localhost:8765`. Add a folder of
photographs, label specimens, and export. See
[docs/TUTORIAL.md](docs/TUTORIAL.md) for a walkthrough with screenshots.

Command-line equivalents, for scripted use:

```bash
python scripts/export_measurements.py --dataset cornell
python scripts/export_tps.py --sidecars data/cornell/sidecars \
    --images data/cornell/lateral --out results/cornell/tps
python scripts/render_overlays.py --dataset cornell
```

A dataset is a directory under `data/` containing `lateral/`, optionally
`frontal/`, `sidecars/`, and a `schema.json` declaring which landmarks the study
collects. `schema.json` may only remove from the master schema, never add.

## Measurements

Twenty-two traits from the MorFishJ schema and eleven Cornell extensions, derived
from 5 polygons and 23 lateral keypoints. The schema is defined in
`src/fish_morpho/landmark_config.py` and is the single source of truth for the
labeler, the measurement engine and the training pipeline.

Traits requiring a landmark the study does not collect are omitted from the
export rather than left blank.

Three columns of the reference spreadsheet are out of scope: `weight(g)` is a
mass, and `body_width` and `caudal_peduncle_width` are measured across the
specimen, which a lateral photograph cannot show.

## Outputs

**Workbook** (`.xlsx`), six sheets:

| sheet | contents |
|---|---|
| About | provenance: dataset, generation time, source commit, unit counts, check counts |
| Measurements | one row per specimen, one column per trait, with a per-row `units` column |
| Ratios | lengths over standard length, areas over SL²; dimensionless |
| Shape | Mosimann log-shape variables; the size correction for between-group comparison |
| QC | calibration method and confidence per view, missing landmarks, data compromises |
| Validation | automated checks, most severe first |

**TPS** for geomorph, with a landmark-name file and a loader snippet. Two format
conventions are handled explicitly: TPS y is Cartesian from the bottom left, and
missing landmarks are written as negative coordinates for `readland.tps(...,
negNA = TRUE)`.

```r
library(geomorph)
A <- readland.tps("landmarks.tps", specID = "ID", negNA = TRUE)
dimnames(A)[[1]] <- read.csv("landmark_names.csv")$name
gpa <- gpagen(A)
plot(gm.prcomp(gpa$coords))
```

**Annotations** are stored as one JSON sidecar per specimen, keyed by landmark
name. These are the durable artefact; all exports are derived from them.

**Overlays**: each photograph with its annotation drawn on.

## Accuracy

Standard length was compared against physical caliper measurements for 35
hand-labelled specimens spanning three hatchery strains: median difference 1.12%,
mean 1.39%, bias +0.69%.

The specimen-to-caliper correspondence was verified rather than assumed. An
offset sweep confirms alignment at offset 0 for two of three strains (1.40% and
1.56% mean error, against 12–20% at any shift). Seven TXD rows differ from their
photographs by 5–32% with mixed sign and no offset accounts for it; standard
length expressed in ruler spans, which requires no millimetre assumption,
supports the photographs. Those rows are listed in
`data/validation/caliper_exclusions.json` and excluded from validation only.

### Calibration

Three modes, all producing px/mm or declaring that none is available.

1. **Tick detection.** `detect_tick_scale()` measures millimetre ticks directly.
   Validated against 20 hand-clicked calibrations across two camera distances:
   mean error 1.0%, maximum 2.5%. C-Thru rulers carry both metric and imperial
   scales and 1/16 in = 1.5875 mm, so the strongest autocorrelation peak selects
   the imperial period and inflates every trait by approximately 59%; the
   detector takes the smallest period on which several frequency bands agree.
2. **Two points and a known span.** A mistyped span is self-consistent and
   undetectable downstream, so the labeler reports px/mm live and flags
   deviation from the collection lot's median.
3. **No scale.** Measurements are reported in pixels and marked as such. The
   Ratios and Shape sheets remain valid, being dimensionless.

Tick detection locates a ruler in 180 of 181 alewife photographs, but only 9 of
27 collection lots are internally consistent to within 25%. Detection rate and
accuracy are separate quantities.

### Validation

Eight checks run before the workbook is written: `orientation`,
`landmark_in_frame`, `duplicate_id`, `mixed_units`, `calibration_outlier`,
`shape_outlier`, `sparse_outline`, `incomplete`. Each targets a failure mode that
produces plausible values rather than an error.

Thresholds are conventional rather than fitted: robust z of 3.5 on a median/MAD
scale, and 25% calibration drift. Across the 60 sidecars in both datasets the
current pass returns no errors, five shape-outlier warnings, and incomplete
notes consistent with labelling in progress.

## Datasets

|  | brook trout (`cornell`) | alewife (`alewife`) |
|---|---|---|
| species | *Salvelinus fontinalis* | *Alosa pseudoharengus* |
| question | body-shape differences among three hatchery strains | proportional differences between landlocked and migratory populations |
| photographs | 131 | 181, across 27 CUMV lots, 1932–1987 |
| labelled | 46 lateral, 35 frontal | 5 |
| collects | 5 polygons, 19 keypoints | 3 polygons, 23 keypoints |
| scale reference | ruler on the tray | ruler on the tank glass |

Photographs are not included in the repository. Sidecars are.

Population assignment for the alewife lots has not been made; no comparison is
meaningful until it is. That study excludes the pelvic and anal fin outlines,
where fin-to-body contrast and fraying prevent reliable tracing.

## Limitations

**Fin traits reflect preservation state.** Restricting to fins traced at 8 or
more vertices, size-corrected variability is 60.1% for dorsal fin area and 42.7%
for pelvic, against 5.4% for body area on the same photographs. Alcohol
preservation dries fins so that extension depends on how the specimen dried and
was pinned. Seven traits are affected (DFh, AFh, PlFl, DFs, PlFs, AFs, and to a
lesser extent PFs) and should not carry a between-group comparison alone.

**Tracing density is imprecise but not directionally biased.** Subsampling a
dense body outline loses area monotonically (−26% at 5 vertices, −2% at 24), but
re-tracing one specimen at 46–86 vertices per fin changed areas by +27.1%, +1.3%,
−3.1% and −7.8% across the four fins. `FIN_POLYGON_TARGET_VERTICES` is 16 because
errors of that size in either direction cannot be corrected afterwards.

**Measurement repeatability has not been quantified.** No specimen has been
labelled twice blind, so the contribution of the operator to any between-group
difference is unknown.

**Declared compromises.** A clipped fin or snout is flagged in the labeler,
which records the affected traits; the pipeline sets those to NaN with the reason
carried onto the QC sheet.

## Automated landmarking

Under development and not required to use the pipeline. See
[docs/automation.md](docs/automation.md) for the full evaluation.

Current state: median held-out keypoint error 0.81 mm across 9 specimens, against
a manual pipeline that agrees with calipers to approximately 1.4 mm. Thirteen of
nineteen landmarks are under 1 mm; `dorsal_tip` (2.10 mm) and
`peduncle_narrowest_ventral` (2.01 mm) are not. The model does not transfer
across orders: on alewife it returns a median likelihood of 0.28 with every point
below the 0.6 threshold.

Segment Anything produces the body outline at IoU 0.937 against dense hand
tracings. It follows the dorsal and adipose fins rather than crossing their
bases; correcting this requires the four fin-base endpoint landmarks, which are
in the schema but not in the trained model.

Predictions are marked `source: predicted` and are refused by
`build_dlc_dataset.py`. Saved sidecars record which points a human corrected,
which were confirmed, and which were left unreviewed; these are not equivalent
evidence and are not pooled.

## Repository layout

```
src/fish_morpho/     schema, measurement engine, calibration, validation, export
scripts/             labeler, exporters, preprocessing, model training and evaluation
data/<dataset>/      lateral/, frontal/, sidecars/, schema.json
docs/                tutorial, automation reference, methods ledger
tests/               174 tests
```

`docs/what-we-tried.md` records approaches that were tested and rejected, with
the measurements that rejected them.

## Status

The manual pipeline is in use. Sidecar JSON in, validated workbook and TPS out,
33 traits covering all 22 photo-measurable columns of the reference spreadsheet.

Fin outlines are being re-traced at the 16-vertex target; 7 of 46 are complete.
Until a specimen is redone its twelve fin-derived traits export as blank with the
reason attached.

Automated landmarking is not ready for use.

Priorities, in order: quantify measurement repeatability; assign the alewife lots
to populations; complete the fin re-tracing; label the four fin-base endpoints
and retrain.

## Citation

caliPr is in preparation for publication. A Zenodo DOI will be given here on
release; until then cite this repository and the commit recorded on the About
sheet of the workbook.

Please also cite the trait schema:

> Ghilardi, M. (2022). *MorFishJ: an ImageJ plugin for morphometric analysis of
> fish.* Leibniz Centre for Tropical Marine Research.
> doi:[10.5281/zenodo.7275017](https://doi.org/10.5281/zenodo.7275017)

## License

MIT; see [LICENSE](LICENSE). The licence covers the code and the annotation
schema. The specimens and photographs belong to the Cornell University Museum of
Vertebrates. No MorFishJ source is included; the trait definitions are
reimplemented from its published documentation.

## Acknowledgements

Built for the ichthyology collection of the Cornell University Museum of
Vertebrates. Automated landmarking builds on DeepLabCut and Segment Anything.
