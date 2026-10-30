# Alewife (*Alosa pseudoharengus*) — CUMV series

Landlocked (Great Lakes) vs migratory populations, compared on **proportion**, so
no absolute scale is needed. Kept separate from the CUMV brook trout study so
nothing here can contaminate that dataset or its validation.

## Decisions made

- **Body outline follows the ventral keel of scutes**, not the body wall.
- All specimens are alewife. `Shad` in ~30 filenames is a labelling quirk, not
  *Alosa sapidissima*; do not treat it as a species split.
- No calibration. Photographs are of a specimen suspended mid-tank with the ruler
  taped to the near glass, so the scale sits in a different focal plane — parallax
  and refraction make any absolute measurement off it wrong by an unknown factor.
  The labeler writes `calibration: {"mode": "none"}` and values come out in
  PIXELS. Ratios (trait / SL) are exact regardless, which is all this study needs.
- **Population assignment is not in the filenames** (only one says `NonMig`). It
  has to come from the CUMV lot numbers and their localities. Without that
  mapping the landmarking cannot answer the question.

## Scope: pelvic and anal fins are not labelled

`schema.json` narrows the master schema for this dataset only. The pelvic has
almost no contrast against the flank and the anal frays too badly to give a
reliable margin, so neither outline nor either fin's keypoints are collected.
Three polygons remain (body_plus_caudal, pectoral, dorsal) and 15 of 19 lateral
keypoints.

Cost: 5 of 33 traits — PlFl, PlFs, AFh, AFs, and **CPl**.

CPl (caudal peduncle length) is the non-obvious one. It runs from the posterior
end of the anal fin base to `caudal_base`, so dropping the anal takes a peduncle
measurement with it. Re-adding just `anal_base_center` to the schema — one click
per fish, on the body rather than on the frayed margin — would recover it.

## Does the brook trout pose model transfer? No.

Tested 2026-08-06 on 8 specimens from lot CUMV 33050, cropped to the fish and
rescaled to the size the model trained at. Result: **overall median likelihood
0.18, with 98% of the 152 predictions below 0.5.** Visually the landmarks pile up
around the head instead of distributing to distinct points, with strays off the
animal entirely. See `results/alewife/dlc_on_alewife.jpg`.

There is weak signal — the clusters land near the head and near the caudal, so the
model has learned something about fish generally — but nothing usable.

**The useful finding is that confidence collapsed.** Within brook trout, the
confidence gate is nearly worthless: it catches 2 of 24 errors over 2 mm. Here it
flagged the failure wholesale. So likelihood detects *domain shift* well and
*within-domain error* badly, which are different jobs, and only the first can be
trusted as an automatic guard.

To get automatic landmarking on alewife the model needs alewife training data.
For scale: brook trout went 3.88 mm held-out error at 28 labeled images to
0.92 mm at 37.

## Labels from the ImageJ landmark files

113 of the specimens were landmarked earlier in ImageJ, all 23 BGNN points, and
saved one CSV per photograph (`analysis/cumv_landmarks_pca/landmarks/`) in units
set per photograph — cm, mm or pixels. In every file the pixel position is that
unit times one factor with no offset, so the hand labels survive exactly once the
factor is known. It was recovered per photograph by fitting the cross-species
model's predictions (`dlc_project_union`, 2026-10-04) to the labels by least
squares, on the landmarks the model was sure of, refitted after dropping strays.

Imported only where the fit pins the factor down: at least 7 landmarks, and the
factor's standard error moving the farthest landmark by under 0.5% of the fish's
length (a hand click's precision; typically 5-10 px). Nine photographs failed and
were not imported. Six whose factor differed from the rest of their lot (a
different zoom, or a file already in pixels) were checked on an overlay and kept.
`imagej_import.csv` lists each import's file, factor, how many landmarks fitted it
and its standard error. Two files label `1960_CUMV_36980_07` (one in cm, one in
mm); the closer fit was kept.

The coordinates are the hand labels, not the model's: the model only supplied the
one number per photograph that turns ImageJ's units back into pixels. Calibration
is `none`, as for the rest of this study.

## Running it

    .venv/bin/python scripts/label_server.py \
        --images data/alewife --out data/alewife/sidecars

Manual labeling only. The reference panel still shows a brook trout; rebuild it
from a labeled alewife with `scripts/make_reference.py --specimen <stem>`.
