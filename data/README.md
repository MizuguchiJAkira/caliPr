# `data/` — datasets and training-image provenance

This directory holds two different kinds of thing.

**Working datasets** — one folder per study (`cornell/`, `alewife/`), each with
a `lateral/` of photographs, a `sidecars/` of annotations, and an optional
`schema.json` narrowing the master schema to what that study collects. The
labeler discovers them by folder name. Photographs are git-ignored; **the
sidecars are tracked**, because they are the data.

**External harvest provenance** — brook trout (*Salvelinus fontinalis*) photos
pulled from public collections to enlarge the training pool, plus the scripts
and metadata needed to reproduce the harvest. None of the image bytes should be
committed (they're covered by `.gitignore` under `data/*/images/`); only the
CSVs and this README are source-controlled so provenance stays reviewable. The
rest of this file documents that harvest.

The goal of the harvest is to assemble a labeling pool for training the
**DLC + SAM hybrid model stack** used by `fish_morpho.pipeline` in auto mode:
DeepLabCut predicts the 25 anatomical keypoints (23 lateral + 2 frontal)
and Segment Anything consumes those keypoints as prompts to trace the 5
polygons (`body_plus_caudal`, `pectoral`, `dorsal`, `pelvic`, `anal`).
Both halves of the stack draw from the same labeling pool, which needs to
be large enough to cover preserved-specimen appearance variation (different
stains, preservation states, lighting) without requiring us to photograph
hundreds of new specimens in-house.

---

## What's here

```
data/
  cornell/           # brook trout study  — lateral/, frontal/, sidecars/, schema.json
  alewife/           # alewife study      — lateral/, sidecars/, schema.json
  validation/        # caliper reference measurements + exclusions
  idigbio/
    images/          # 41 JPEGs (git-ignored)
    metadata.csv     # per-image provenance (institution, catalog, license, …)
  fish_vista/
    csvs/            # 11 Fish-Vista split CSVs, cached for re-runs
    images/          # 2 JPEGs (git-ignored)
    metadata.csv     # per-image provenance for the 2 matched rows
  fishphenokey_request.md   # draft email asking for dataset access
  idigbio_harvest.log       # harvest run log (git-ignored)
  README.md                 # this file
```

## Source 1 — iDigBio media harvest

Script: `scripts/harvest_idigbio.py`
Run: `.venv/bin/python scripts/harvest_idigbio.py`

**Result: 41 specimen images** across 8 institutions (after filtering out
non-image media like 3D scans and field notes — iDigBio returned ~318 total
media records but only ~41 were actual downloadable photographs).

Institution breakdown:

| Institution | Count |
|---|---|
| USNM (Smithsonian) | 18 |
| NEON | 8 |
| YPM (Yale Peabody) | 5 |
| CMN (Canadian Museum of Nature) | 4 |
| NHMUK (Natural History Museum London) | 3 |
| INHS (Illinois Natural History Survey) | 1 |
| UCMP (UC Berkeley Museum of Paleontology) | 1 |
| MCZ (Harvard) | 1 |

Geographic spread: Maine (10), California (6), Massachusetts (4), Montana
(3), New York (3), Ontario (2), plus Oregon, Illinois, Québec, Nunavut,
Washington, and Pennsylvania (1 each).

### License notes

- **18 USNM** rows have custom Smithsonian "Usage Conditions Apply" terms —
  fine for internal research/training but **check before redistribution**
  of any derived models that encode pixel data.
- **5 YPM** rows are **CC0** (public domain) — unrestricted.
- **3 NHMUK** and **1 UCMP** are **CC BY 4.0** — fine to use, cite the
  institution.
- **8 NEON** rows are split between **CC0** and **CC BY-SA 4.0** (same
  photograph registered under both licenses depending on the portal
  serving it) — treat as CC BY-SA to be safe.
- **4 CMN** rows are **CC BY-NC 4.0** — non-commercial; OK for research
  training, potentially constrains future commercial use of the DLC model.
- **2 MCZ / INHS** rows are **CC BY-NC-SA 4.0**.

Bottom line: all 41 images are clearly usable for **non-commercial academic
research**. If the Cornell MV pipeline or DLC model is ever released
commercially, the CC-NC / USNM subset may need to be re-trained around.

### Transient harvest failures

Two records failed on the first run (both transient, worth retrying):
- `https://data.nhm.ac.uk/media/37ffc535-377a-4e63-8445-0c20ac5ea77e` — HTTP 500
- `https://iiif.mcz.harvard.edu/iiif/3/44358/full/max/0/default.jpg` — HTTP 404

The harvester is idempotent (`dest.exists() and size > 1024` short-circuits
successful downloads), so re-running `harvest_idigbio.py` will only attempt
the two failed records plus anything new that shows up in iDigBio.

## Source 2 — Fish-Vista

Script: `scripts/filter_fish_vista.py`
Run: `.venv/bin/python scripts/filter_fish_vista.py`

**Result: 2 images.** This is dramatically fewer than the 300–1000 estimate
from the earlier research pass. We scanned all 11 Fish-Vista splits
(classification/identification/segmentation × train/val/test) for rows
with `standardized_species == "salvelinus fontinalis"` and only
`identification_train.csv` returned matches:

| Filename | Source | Owner | License |
|---|---|---|---|
| `INHS_FISH_62305.jpg` | GLIN | INHS | CC BY-NC |
| `99407_Savlelinus_fontinalis.jpg` (sic) | iDigBio | MCZ | CC BY-NC-SA 3.0 |

Both images are already represented indirectly in the iDigBio harvest (the
INHS row matches `INHS_62305_*.jpg` and the MCZ row matches `MCZ_99407_*.jpg`),
so Fish-Vista contributes **zero net new training images** for brook trout.
The filter script is still kept in the repo because (a) `standardized_species`
coverage may improve in future Fish-Vista releases, and (b) the metadata
join is useful for cross-referencing annotation formats.

## Source 3 — FishPhenoKey (pending, pretraining only)

See `fishphenokey_request.md` for a draft access-request email. Even if
FishPhenoKey doesn't include *Salvelinus fontinalis*, we want it for
**pretraining** before fine-tuning on our own brook trout labels — the
landmark topology for teleost fins and body outline transfers well across
species, which should materially improve DLC accuracy given our small
downstream label set.

## Deliberately skipped sources

- **Riverine / aquaculture datasets** (live underwater video, hatchery
  production photos) — pose and framing don't match preserved-specimen
  morphometrics, so training on them risks degrading landmark accuracy
  rather than improving it. Our model only needs to work on lateral
  photos of fixed specimens on a board with a ruler.

---

## Pool size vs. training target

| Source | Usable brook trout images |
|---|---|
| iDigBio | 41 |
| Fish-Vista (net new) | 0 |
| Cornell MV in-house | TBD |
| **Pool for DLC labeling** | **41 + in-house** |

This is below the original 150-image DLC training target. To close the
gap we now need **one of the following**:

1. Photograph ~100 more brook trout specimens in-house using the same
   lateral + frontal mirror protocol already documented in
   `examples/sample_sidecar.json`. This is the clean path; everything
   will be consistent with the frontal-mirror mouth-width measurement.
2. Use FishPhenoKey pretraining (no species overlap needed) and accept
   a smaller fine-tuning set — DLC can reach decent accuracy with
   ~50–80 labeled frames if the backbone is well-pretrained.
3. Broaden to congeners (*Salvelinus* spp. — lake trout, Dolly Varden,
   Arctic char). Same genus, very similar landmark topology, and the
   iDigBio specimen pool jumps substantially. Re-run
   `harvest_idigbio.py` with the species list widened for this.

Option (1) combined with (2) is the recommended path — it gets us to a
usable labeled set faster and leaves the pipeline producing
morphometrics that are biologically clean (no mixed-species bias from
option 3).

## Labeling workflow

**The current path is the built-in browser labeler** (`scripts/label_server.py`,
or a self-contained HTML build for people without Python). It reads this
project's schema live, writes sidecar JSON directly, and needs no import step.
See the main README.

CVAT remains supported as an alternative for anyone who already runs it — the
schema exports to a CVAT label config, and `scripts/cvat_to_sidecar.py` converts
a CVAT XML dump back. It was the original path, chosen because the MorFishJ-port
schema mixes polygons and keypoints and CVAT natively supports both in one task,
where the DLC labeling GUI only handles keypoints and Label Studio's polygon UX
slows annotators down on fin outlines. The rest of this section describes that
route; **the labeling rules below apply to either.**

### One-time CVAT project setup

1. Generate the labels JSON straight from the schema so annotators see
   exactly the shapes the measurement engine expects:

   ```bash
   .venv/bin/python scripts/export_cvat_config.py --out-dir cvat/
   ```

   This writes two files:

   - `cvat/cvat_labels_lateral.json` — 5 polygon labels
     (`body_plus_caudal`, `pectoral`, `dorsal`, `pelvic`, `anal`) plus
     23 point labels for the lateral orbit cardinals, mouth/jaw/
     operculum keypoints, pectoral insertion and ray tip, peduncle
     narrowest pair, caudal base, and four fin base/tip anchors.
   - `cvat/cvat_labels_frontal.json` — 2 point labels (`mouth_left`,
     `mouth_right`) for the mirror head shot.

2. Create **two separate CVAT projects**, one per view: paste each
   file into the project constructor's "Raw" tab, or POST to
   `/api/projects` if you prefer the REST API. Each label carries its
   `description` and `labeling_hint` from `landmark_config.py` as
   read-only attributes, so annotators can hover any label in the CVAT
   sidebar to see the canonical definition.

3. Upload the 41 iDigBio photos (plus any in-house photos) to the
   lateral project; upload any paired mirror mouth-width photos to the
   frontal project. Images that don't have a frontal shot simply get
   left out of the frontal project — the pipeline treats mouth width
   as optional.

### Labeling rules for a single fish (lateral)

- **Orientation.** All photos must have the fish **facing left**
  (head at smaller x). Rotate any right-facing photo 180° before
  labeling — this matches MorFishJ's GUI convention and is what the
  measurement engine assumes.
- **`body_plus_caudal` polygon.** Trace the silhouette from the snout
  tip, along the dorsal outline, **step across the dorsal fin base**
  (do not follow the rayed dorsal margin), continue to the caudal
  peduncle, trace around the caudal fin margin, come back along the
  ventral outline, and **step across the pelvic and anal fin bases**
  the same way. The polygon is the fish minus the four unpaired/paired
  fins.
- **Fin polygons.** Trace each of `pectoral`, `dorsal`, `pelvic`,
  `anal` as its own closed outline.
- **Eye.** Drop **four** keypoints around the orbit at the cardinal
  compass points: `eye_anterior`, `eye_posterior`, `eye_dorsal`,
  `eye_ventral`. The engine averages them into a bounding-box
  centroid; if one or two are hard to click (specimen damage, stain
  obscuring the orbit) skip them and they surface as a missing-input
  NaN on the QC sheet rather than a silently-wrong number.
- **Peduncle narrowest.** Place `peduncle_narrowest_dorsal` and
  `peduncle_narrowest_ventral` on opposite sides of the peduncle at
  its visually narrowest point. These define line A, which the engine
  uses to split `body_plus_caudal` into body and caudal halves.
- **Exclude specimens that are:** not lateral, fin-damaged in a way
  that loses the polygon outlines we need, juvenile/larval (morphology
  differs enough to hurt the model), or missing a ruler for
  pixel-to-mm calibration.

### Labeling rules (frontal mirror)

Drop `mouth_left` and `mouth_right` on the outer corners of the
mouth in the mirror shot. That's the whole frontal task — mouth width
is the only trait on that view.

### Export from CVAT → sidecar JSON

CVAT's native export formats (CVAT XML, COCO JSON) don't match our
`examples/sample_sidecar.json` schema directly, so `scripts/cvat_to_sidecar.py`
walks a CVAT XML 1.1 dump and emits one sidecar per image:

```bash
.venv/bin/python scripts/cvat_to_sidecar.py --cvat-xml annotations.xml \
    --view lateral --out-dir data/cornell/sidecars
```

### Validating against MorFishJ

Once you have even a handful of labeled specimens that have **also**
been measured by hand in the MorFishJ GUI, run the port against
MorFishJ's own CSV output as an oracle:

```bash
.venv/bin/python scripts/morfishj_validation.py \
    --reference data/validation/morfishj_reference.csv \
    --labels    data/validation/labels/ \
    --tolerance-mm 0.2 --tolerance-mm2 2.0 --tolerance-deg 1.0
```

Nonzero exit = at least one trait deviated beyond tolerance. This is
how we catch regressions when the engine changes.

## Next steps

1. Stand up the CVAT project from `scripts/export_cvat_config.py` and
   walk the 41 iDigBio images in it, producing polygons + keypoints
   for every usable specimen.
2. Send the FishPhenoKey access request
   (`fishphenokey_request.md`) — pretraining for the DLC keypoint
   half of the hybrid stack.
3. Plan the in-house photo session to top up the training pool to
   ~150 labeled images, using the same lateral + frontal mirror
   protocol already documented in `examples/sample_sidecar.json`.
4. Run ~5 specimens through both MorFishJ (GUI) and the Python port
   and diff them with `scripts/morfishj_validation.py` — this gates
   "the port is good enough to replace MorFishJ for real work".
5. Wire the labeled data into the DLC + SAM training loop
   (`predict_annotation()` in `src/fish_morpho/pipeline.py` is the
   single integration point to fill in once the models exist).
