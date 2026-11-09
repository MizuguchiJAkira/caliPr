# What we tried

A ledger of fixes and dead ends, with the numbers that decided each one. Negative
results are here on purpose: several of these look obviously correct and are not,
and the reasoning is easier to re-derive from a record than from scratch.

Ordered by subsystem, not chronology.

---

## Photo preprocessing

### EXIF orientation is not trustworthy — detect from pixels

Of 133 lab photos, 102 carry EXIF orientation 6, **31 carry orientation 1**, and
one is stored portrait. The original `normalize_orientation` only rotated when the
image was portrait, so those 31 stayed 180° from canonical: fish facing right,
mirror on the right. The mirror split then cut the *left* side — the tail — and
silently severed the caudal fin on ~30 specimens.

**Fix:** `is_upside_down()` decides from pixels, not EXIF. The ruler's dense
vertical tick edges (horizontal Sobel) sit in the top band when canonical and the
bottom band when 180° off. Validated 133/133.

**Trap for anyone writing a similar check:** the handwritten label cards also
produce strong bottom-band edges. Restrict the x-range and key on the ruler's tick
*texture*, not raw edge energy.

### Mirror-split clipping — solved structurally after 7 failed detectors

`detect_mirror_boundary` keys on the strongest vertical edge in the left 35%, but
the ruler's tick marks can out-shout the mirror frame and push the split right,
through the snout. TXD_49 lost its whole head; TXD_41–45 lost snouts.

**Seven heuristics were tried to detect this from the crops and all failed** —
edge darkness, connected components, brightness step, snout search,
longest-dark-run, and variants. Every one broke, because a dark tray strip or a
mirror-ruler sliver sits at x=0 on many *correct* crops and is indistinguishable
from a clipped body by any threshold.

**Fix:** stop detecting, change the geometry. `process_one(lateral_margin=N)`
makes the lateral and frontal crops deliberately **overlap** — lateral starts at
`boundary - margin`, frontal keeps `[0:boundary]`. The cost is asymmetric: extra
mirror background in frame is harmless, a cut snout is unrecoverable. Re-cropped
at `--lateral-margin 450`.

Per-specimen forced splits live in `data/cornell_raw/boundary_overrides.json`.

**Note:** labeled specimens were deliberately *not* re-cropped, since that would
invalidate their sidecar coordinates. Each sidecar only has to match its own
image, so mixed conventions across the set are fine.

---

## Calibration

### The metric-vs-imperial harmonic trap

`detect_tick_scale()` recovers px/mm by autocorrelating column-darkness profiles
across horizontal bands. The C-Thru ruler carries **both** a mm and an imperial
scale, and 1/16 inch = 1.5875 mm — so the imperial ticks appear as a period ~1.59×
coarser than the millimetre ticks.

Taking the *strongest* autocorrelation peak picks the imperial ticks on ~40% of
photos and inflates every trait by ~59%.

**Fix:** millimetre ticks are the *finest* true periodicity — take the **smallest**
period supported by ≥3 bands. Validated against 20 hand-clicked calibrations
spanning two camera distances (20.5 and 25.2 px/mm): mean absolute error 1.0%, max
2.5%, 20/20 within 3%.

### Typed-span errors are invisible without a cross-check

A mistyped `known_mm` (10 for a 50 mm span) scales every trait for that specimen
and nothing downstream can tell — the geometry stays self-consistent, just wrong.
Five occurred in practice. The labeler now shows the batch median px/mm beside the
current one, which catches them while the annotator is still on the fish.

Root cause of one: switching view tabs inherited the other view's span.

---

## Fin segmentation with SAM

Benchmarks: `scripts/eval_sam_polygons.py` (full frame),
`scripts/eval_sam_zoom.py` (cropping and negative prompts).

### Worked: cropping — but only for the two small fins

SAM resizes its input to 1024 px, so on a full 4400 px frame the pectoral is a
handful of pixels by the time the model sees it. Cropping to the fin first:

| fin | full frame | cropped |
|---|---|---|
| pectoral | 238% | 31% |
| pelvic | 27% | 17% |
| dorsal | 23% | **45%** |
| anal | 11% | **18%** |

**It is not a general improvement.** The dorsal and anal are large enough to
survive the resize, so the crop buys no resolution and costs the body context that
tells SAM where the fin ends. Cropping must be applied per fin, not per pipeline.

### Worked: negative points, placed ventrally — for the pectoral only

A point prompt makes SAM return *object-level* masks, so prompting on a fin
happily returns the whole fish (the zoomed pelvic came back 15× too large).
Positive points cannot express "this fin, not the animal it is attached to";
negative points on the flank can.

Where they go matters. Brook trout pectorals angle **dorsally** and never run along
the belly margin, so a symmetric pair of negatives clips the fin's upper edge.
Ventral-only negatives: pectoral 48% → **29%**. The same change makes the anal
worse (18% → 27%), which is what you would expect — the asymmetry is a fact about
pectoral geometry, not a prompting trick.

### Worked: merging mask fragments across a pin

A fin is one structure, but a specimen pin laid across it splits SAM's output into
disconnected pieces (clearly visible on ASN_10 and TXD_7), and keeping only the
largest piece throws away real fin area.

`merge_fragments()` closes over gaps under a fraction of the crop's long edge,
then **intersects the result back with the original mask** so the morphological
closing contributes no area of its own — only the bridge is filled, and only where
it links genuine fin pixels. The component carrying the most original mask wins.

### Failed: subtracting the body mask

The idea: SAM traces `body_plus_caudal` well, so segment the body, subtract it,
and whatever remains in the dorsal/anal region is fin. The dorsal and anal sit 96%
outside the hand-traced body outline, so this should work.

**It does not.** SAM's "body" mask is the whole *animal* and already contains
55–70% of those fins. There is nothing left to subtract. Script removed rather
than kept, since working code that does not work invites someone to run it.

### Failed: CLAHE and contrast enhancement

Amplifies the styrofoam backing's texture along with the fin boundary. No net gain.

### Why the fins resist: it is contrast, and it is measurable

Fin-to-surround separation, in intensity levels: **pectoral 5.7, pelvic 10.6,
dorsal 17.7, anal 29.7**. SAM's full-frame error tracks that almost monotonically.
There is no edge to find — this is a property of alcohol-preserved fins on a pale
backing, not a tuning problem.

### The verdict, measured against dense re-tracings

Everything above was benchmarked against *sparse* hand outlines, which turned out
to be partly measuring our own tracing. Against five specimens re-traced at 38–86
vertices per fin:

| polygon | median \|err\| | worst |
|---|---|---|
| `body_plus_caudal` | **0.7%** | 3.2% |
| anal | 8.4% | 18.9% |
| pectoral | 18.2% | 40.5% |
| pelvic | 20.8% | 52.0% |
| dorsal | 35.4% | 67.1% |

**Hand-trace all four fins; automate `body_plus_caudal` only.**

A cautionary note on sample size: the first fish re-traced gave 1.7% pelvic and
2.3% anal, which read as "SAM was right all along, our reference was wrong." Four
more specimens killed it — the pelvic swings +52% to −21%. One specimen was not
enough, and the error changes *sign* between fish, so it is not a bias that could
be calibrated away.

---

## Anatomical constraints on predicted outlines

See `src/fish_morpho/anatomy_constraints.py`. A segmentation model has no anatomy;
these are facts about the animal, encoded as hard geometric limits.

### The allowance is the whole design

A bound placed exactly at the anatomical landmark **cuts real fin**. Two
independent demonstrations:

- The pectoral genuinely overlaps the gill cover: **all 44** specimens reach
  anterior of `operculum_posterior`, median 2.7% of SL. A hard cut there would
  have clipped real fin on every fish.
- Clipping the dorsal at exactly its insertion removed **up to 53%** of a real
  hand-traced fin, because a fin polygon is thin near its base and a dip of a few
  tenths of a percent of SL shaves a wide, shallow slice off it.

Every allowance is therefore fitted from how far the *hand* tracings cross the
line — they are the fin, by definition — then set clear of that.

### Placing the dorsal's ventral bound: flanking chord beats a fitted line

Two body-outline points flanking the fin base, with the chord between them as the
boundary, versus fitting a line to outline vertices within a radius window:

| method | wrongly cut real fin |
|---|---|
| radius PCA fit | 17 / 44 |
| flanking chord | 9 / 44 |
| flanking chord + 1.5% allowance | **0 / 46** |

The radius window also picks up the curvature of the back. Flanking points are
found by **arc length with interpolation along edges**, not by snapping to
vertices, so the chord does not depend on how densely the outline happens to be
traced.

### The distal bound is what removes the pin

The specimen pin sits *above* the fin, so a ventral bound never touches it.
Bounding at `dorsal_tip` does: SAM crosses that line by 3.47% of SL, hand tracings
by at most 1.30%.

Its allowance is wider (2% vs 1.5%) because it is anchored to a *keypoint* and so
inherits that keypoint's error — and `dorsal_tip` is the weakest landmark the pose
model predicts. Auto mode should pass a larger value than the hand-label default.

### Effect

On HRN_5, against its dense re-tracing: pectoral +15.3% → **+12.3%**, dorsal
+35.9% → **+22.0%**. Across 46 hand tracings the constraints cut nothing on the
dorsal (0/46) and one pectoral at 2.4%.

Real but partial. A half-plane can only remove area that *crosses* its line; it
cannot repair a mask that is wrong inside the allowed region.

---

## Keypoint model (DeepLabCut)

### Report medians, not means, on a small held-out set

DLC's summary CSV reports mean pixel error per landmark. On a 9-specimen test set
that is actively misleading: `pectoral_ray_tip` looked like a 12.26 mm catastrophe
by mean and is 0.94 mm by median — the difference was one fish.

### `rmse_pcutoff` is not a train/test comparison

An earlier reading of DLC's own output concluded overfitting was gone, from
`train rmse_pcutoff 6.21` vs `test rmse_pcutoff 5.28` px — test apparently better
than train. **That comparison is invalid.** `pcutoff` filters each set by the
model's own confidence, discarding exactly the hard cases, and it filters the two
sets by different amounts.

Computed honestly — median over all labeled points, in millimetres, using each
fish's own calibration — the gap is **train 0.38 mm vs test 0.92 mm, 2.4×**, and
2.3× at the 90th percentile too, so it is distribution-wide rather than
outlier-driven.

Leakage was separately ruled out: DLC's actual train indices in the shuffle pickle
match `dlc/split.json` exactly with zero overlap; max test-to-train image
similarity is 0.880 on landmark-cropped frames; and error versus
similarity-to-nearest-training-fish gives r = −0.24 on n=9, i.e. no
memorise-and-match signal. The held-out numbers are real; the model is simply
fitting 37 fish more closely than it generalises.

### Confidence gating is a catastrophe detector, not a quality gate

It catches **2 of the 24** held-out errors over 2 mm. It caught HRN_46's 103 mm
`pectoral_ray_tip` (likelihood 0.21) but missed ASN_27's 7.95 mm `pelvic_tip`
(0.56 against a 0.365 threshold). Do not present it as the safety net for bad
points.

Note also that likelihood is **not calibrated across landmarks** — `caudal_base`
sits at 0.24–0.45 while being accurate to 0.97 mm — so an absolute floor discards
good predictions wholesale. `dlc_report.py --relative` gates against each
landmark's own median instead.

### Training settings that actually run on this Mac

The defaults thrash it. Two runs stalled dead at epoch 3 with the process in state
`U` (uninterruptible disk wait), RSS collapsed to 8 MB, swap at 7249/8192 MB —
memory exhaustion, not a code fault.

Diagnose with `ps -eo pid,%cpu,rss,state` (look for state `U`) and
`sysctl vm.swapusage`, **not** by assuming it is slow.

---

## Hand tracing

### Sparse outlines are imprecise, not biased in a known direction

Subsampling the densely traced body outline to k of its own real vertices loses
area monotonically — −26% at k=5, −13% at k=7, −5% at k=16, −2% at k=24 — which
suggested sparse tracing reads *small*.

**That does not transfer to real hand tracing.** Subsampling keeps the survivors
exactly on the margin; a human with 7 clicks places a few points and interpolates
by eye, running generous as readily as tight. Re-tracing HRN_5 at 46–86 vertices
against its own 7–11 vertex original: anal **+27.1%**, pelvic +1.3%, pectoral
−3.1%, dorsal **−7.8%**.

The case for a 16-vertex target is that errors of that size in an *unknown*
direction cannot be corrected afterwards — not that sparse reads low.

### A correlation that looked like evidence and was not

Vertex count correlates positively with size-corrected fin area (pelvic r = +0.57),
which read as a density bias. It is confounded: click count also tracks fin
**size** (r = +0.49 dorsal, +0.43 pelvic), so a bigger fin draws both more clicks
and more area with no bias needed.

### `dorsal_tip` is the apex, not the farthest point

A QC check flagged `dorsal_tip` as 29–72% short of the farthest vertex of the
outline on all five re-traced specimens, while `anal_tip` and `pelvic_tip` matched
theirs exactly. The check was wrong, and it was reading the schema's own wording
back at itself.

The dorsal here is long and low — base 16–25 mm against a height of 3–11 mm — so a
corner of the base sits farther from the base centre than the apex does. The pin
was ruled out separately: only ASN_30's dorsal has its farthest vertex as a spike
(>1.6× its neighbours' radius); the other four are smooth margin.

### Self-intersections in dense tracings are harmless

Nine of 20 re-traced fin polygons self-intersect. Repairing them changes area by
at most **0.268%**, typically 0.00%.

Beware the obvious test: comparing shoelace area to a rasterised fill shows a
0.7–4.4% gap on *all* polygons including non-intersecting ones, because
`fillPoly` includes boundary pixels — an artifact of the test, matching a
predicted perimeter × 0.5 px almost exactly.

---

## Labeler

- **A localStorage draft shadowed the sidecar unconditionally**, so a file changed
  outside the browser — a bulk wipe, a hand fix, a `git checkout` — silently
  reappeared as the old values. Drafts now carry a timestamp and
  `/api/specimens` reports each sidecar's mtime; an older draft is discarded.
- **Hit-testing must be filtered by the same predicate as drawing.** Otherwise a
  drag near a fin margin can grab an invisible `body_plus_caudal` vertex and move
  it, corrupting the one polygon that is already correct, with no visual feedback.
- **`zoomToFin` must guard a zero-size canvas** the way `fit()` does; a hidden or
  not-yet-laid-out canvas measures 0×0 and clamps the scale to its floor.
- **Do not add a "densify" button** that inserts midpoints along existing chords.
  It would raise every counter to target and turn the QC flags green while the
  area stayed exactly as biased — a midpoint on a chord carries no information
  about where the margin actually is.

---

## Predicted body outline

Measured against dense hand tracings on five specimens, prompting SAM with six
predicted keypoints along the body axis.

- **Read IoU, not area error, for `body_plus_caudal`.** Area error is the right
  verdict for a fin, where the pipeline consumes one number. It is not for the
  body, and reading it that way hid a real defect for months. SAM wraps the
  dorsal and adipose fins instead of crossing their bases and dips into the
  shadow under the pelvic; those two errors cancel. ASN_30 scores **0.0% area
  error at IoU 0.953**. The polygon is also split at line A into *two* traits, so
  a shape error moves area between Bs and CFs while the total stays right.
  Baseline to beat: **IoU 0.937 median, 0.902 worst**.
- **Local smoothing does not remove a fin excursion.** A fin spans many vertices,
  so no single one is sharp against its neighbours. Across 55 sidecars,
  hand-traced body vertices sit within 2.08% of SL of their neighbours' chord at
  the 95th percentile, and SAM's fin excursions pass that test. Turn angle is
  worse still: hand tracings reach 91° at the 95th percentile, because a
  52-vertex polygon around a snout is genuinely angular.
- **Chords between margin landmarks help only where a landmark brackets the
  fin.** They bound the adipose and the pelvic notch, but on those spans SAM is
  already close to a hand tracing (3.25% vs 2.15% of SL on HRN_42), so the gain
  is small. The dorsal fin cannot be bounded at all this way: its excursion is
  centred on `dorsal_base_center` and there is no landmark forward of it.
- **Select chord spans along the outline, never by projection onto the chord.** A
  dorsal chord is projected onto by the entire ventral margin; pulling those
  vertices to it folds the outline flat, −70% area.
- **Negative point prompts at the predicted fin tips make it worse.** The
  technique that rescues the pectoral (48% → 29%) costs the body 0.937 → 0.934
  median IoU, on all five specimens. Rejected.
- **SAM vit-large is not worth it on this hardware.** IoU 0.937 → 0.946 for
  **1.67 s → 102.75 s per specimen** on MPS, a 61× cost for a 1% gain. It would
  turn a two-minute batch over 76 fish into two and a half hours.

What remains is the one that addresses the mechanism: **the four fin-base
endpoints**. `dorsal_base_anterior`, `dorsal_base_posterior`,
`anal_base_anterior`, `anal_base_posterior` are in the schema but in only 1–2 of
46 sidecars, and the trained model covers the 19 keypoints that predate them.
With those predicted the chord *is* the fin base and the cut is exact, which is
what a person does by hand. `MARGIN_CHORDS` in `predict_worker.py` already lists
them first and starts using them the moment they are predicted.

## Landmark plausibility bands

Confidence does not catch every bad landmark. `ASN_42` returned
`dorsal_base_center` half a fin base out of position at **0.914**, and `ASN_12`
put `anal_tip` on the caudal peduncle at **0.87**. Both would have entered a
workbook unchallenged.

**What works: axial position, measured per landmark.** The anterior and posterior
extremes of the `body_plus_caudal` outline give an axis the segmentation finds
even when every fin landmark fails. Each landmark occupies a narrow, repeatable
fraction of it across the 46 hand-labelled fish — `pelvic_base_center` 0.425–0.469,
`anal_base_center` 0.662–0.713. A prediction outside that range is a different
structure, not a near miss. Fitted by `scripts/fit_plausibility.py` into
`plausibility.json`; a dataset without one is not checked.

Validated leave-one-out against the hand labels: **4 of 553 hand-placed landmarks
(0.72%)** fall outside their own band. Across the batch predictions it drops 12
landmarks, 2 of which the model was confident about.

**Margin: half the observed spread, floored at 0.02 of body length — not a flat
allowance.** A landmark whose labelled positions span 0.04 is measured far more
precisely than one spanning 0.13. At a flat 0.05 the `ASN_42` dorsal base lands
exactly on the boundary and survives. Widening to the proportional rule raised
false positives from 1 to 4 in 553 and caught two confident errors instead of one.
The costs are not symmetric: a correct landmark wrongly dropped costs the seconds
it takes to place by hand, which is the no-automation baseline; a wrong landmark
wrongly kept can become data.

**Tried and rejected: which side of its base a fin tip falls on.** The most
obvious rule available, and the labels do not support it — `pelvic_tip` sits
*above* `pelvic_base_center` in 5 of the 6 fish carrying both. Either preserved
pelvics fold up against the flank more often than not, or those six labels
disagree. Six examples cannot tell the difference and a rule fitted to them would
reject correct predictions.

**Tried and rejected: distance from the body outline.** Fitted on hand tracings,
applied to SAM's outline — not the same curve. It wraps the fins a tracing cuts
across and is resampled to 52 even vertices. The rule flagged `premaxilla_tip` at
1.00 confidence and `operculum_posterior` at 0.87 where both were correct, because
the reference distribution was never the one being tested. Fitting it on predicted
outlines would mean fitting on the model's own output.

**Not a rule, a finding.** `operculum_posterior` sits posterior to
`pectoral_insertion_upper` in 17 of 17 predictions with both confident, and in the
hand labels (0.193 vs 0.182). That is the opercular flap trailing back over the
pectoral base, not an error. An anterior-to-posterior ordering check that includes
that pair reports a 100% violation rate and is measuring its own assumption.

**The real fix is labels, not rules.** Head and peduncle landmarks carry 46
examples each and predict at 0.002–0.008 SL. Every fin landmark carries **6 or 7**
(13–15% of the set), and those are precisely the ones that fail. The bands are a
backstop; labelling the fin landmarks on the remaining 39 fish is the fix.

## Predicted body outline withdrawn from Auto-label

SAM matches a dense hand tracing at IoU 0.937 overall, and the disagreement is
concentrated exactly where it matters: it wraps the adipose and anal fins rather
than crossing their bases. Reviewing that costs more than tracing the outline
fresh, so `data/cornell/schema.json` now carries
`exclude_predicted_polygons: ["body_plus_caudal"]` and Auto-label does not offer
one. The polygon stays in the labelling contract and is still traced by hand.

This is a different setting from `exclude_polygons`, deliberately. The outline is
still computed on every prediction, because its anterior and posterior extremes
are the axis the plausibility check measures every landmark against — and those
two extremes are the reliable part of it. The unreliable part is the middle.

Measured on TXD_18, warm:

| | time | result |
|---|---|---|
| outline computed, not offered | 2.29 s | 18/19 placed, `anal_tip` dropped |
| SAM skipped entirely | 0.84 s | 19/19 placed, `anal_tip` on the caudal tip |

So the check costs about 1.45 s per specimen, and without it the constraint layer
is inert rather than absent — it returns nothing to complain about because it has
no axis to measure against. Turning SAM off entirely means turning the check off.

## The check has to vet its own axis

Every landmark position is a fraction of the predicted outline, so an outline
that is not a fish rescales all nineteen at once. `ASN_48` — a small, curved
specimen filling 40% of the frame — segmented a lobe of foam above the animal,
and eight landmarks were rejected against an axis that was itself wrong.
`operculum_posterior` and `peduncle_narrowest_ventral` were among them, and those
carry 46 training examples each.

Two numbers separate a fish silhouette from a segmentation that has wandered:
what fraction of its bounding box the outline fills, and how many times longer
than deep it is. Across 46 hand tracings: **fill 0.497–0.724, aspect 3.33–5.49**.

| | fill | aspect | |
|---|---|---|---|
| ASN_48 | 0.54 | **2.01** | refused |
| ASN_51 | **0.44** | **3.03** | refused |
| ASN_45 | 0.70 | 5.00 | used |
| ASN_47 | 0.59 | 4.09 | used |
| TXD_18 | 0.65 | 4.06 | used |

An outline outside either range is refused and nothing is checked on that
specimen, reported as a warning rather than as a drop. Withholding the check is
the honest failure; rejecting correct landmarks on a bad axis is not.

## A fin tip against its own base, with no axis at all

Refusing a bad outline left a hole: on `HRN_15` the outline failed the shape test
(aspect 3.1), so nothing was checked, and `dorsal_tip` sat on the **adipose fin**
unchallenged — 935 px behind its own base at the same height.

A fin tip's offset from its own base does not need to know where the ends of the
fish are. Scaled by eye diameter — the two eye points carry 46 labelled examples
each and come back at 0.76–0.95 on specimens where every fin landmark fails —
it is available exactly when the axial check is not.

| fin | n | tip minus base, along the body (eye diameters) |
|---|---|---|
| dorsal | 7 | −0.44 to +1.06 |
| anal | 6 | +1.21 to +1.76 |
| pelvic | 6 | −0.21 to +2.12 |
| pectoral | 7 | +1.60 to +3.08 |

`HRN_15` predicted **+4.87**. More than four times outside the widest labelled
dorsal, and the adipose is the only structure there.

These bands rest on six or seven fish rather than forty-six, so leave-one-out
removes a sixth of the evidence and they need more slack than the axial ones.
Sweeping the margin:

| margin | false positives | HRN_15 |
|---|---|---|
| 0.5× spread | 8/559 | caught |
| 1.0× | 5/559 | caught |
| **1.5×** | **4/559** | **caught** |
| 3.0× | 4/559 | missed |

1.5 restores the false-positive rate to exactly what it was before this layer
existed (0.72%) while catching the adipose error with room to spare. On `ASN_48`,
whose outline is refused outright, the fin checks still drop three tips where the
broken axis had previously rejected eight landmarks including two with 46
training examples.

Jonah's observation that started this — *the dorsal doesn't extend past two
thirds of the fish* — is right and conservative: on a snout-to-caudal-base axis
the labelled `dorsal_tip` runs 0.529–0.616.

## Labels stranded by a re-crop

`HRN_4` was labelled on a lateral crop that started at the mirror boundary,
x=1656. On 2026-08-04 the lateral crops were re-cut with `--lateral-margin 450`,
starting it at x=1206. `process_one` overwrote the image unconditionally and
recorded nowhere where a crop began, so every landmark, the whole outline and
both ruler clicks stayed where they were and sat 450 px left of the fish -- in
the ruler -- for six weeks. It was one of the 37 fish the landmark model trained
on.

A uniform shift changes no distance, area or angle, so none of HRN_4's measured
traits were wrong: re-measuring it before and after the repair changed 0 of 20.
What broke was every use that pairs a coordinate with the pixel under it -- what
the labeler draws, and what the model learned.

Guarded at three points, because any one alone leaves a gap:

- **At the source.** Preprocessing refuses to re-crop a fish that has labels. With
  `--shift-labels` it moves them by the exact difference in crop start, and only
  once it has confirmed the file on disk is a slice of the same photograph (0.75
  grey levels of JPEG noise on HRN_4; a different photograph is refused outright).
- **Wherever the image changes for any other reason.** Every save records the size
  and hash of the image its coordinates sit on. A fish whose image no longer
  matches opens with its labels withheld, cannot be saved, and is skipped by the
  training-set builder; a browser draft made on another size is discarded. A crop
  anchored to the right edge that is N px wider started N px further left, so a
  recorded size makes the repair exact arithmetic.
- **Against stale saves.** A save names the version of the file its labels were
  loaded from and is refused if the file has changed since. Without this, the
  repair itself could be overwritten by a tab still holding the old coordinates --
  both image sizes match, so no fingerprint catches it.

`scripts/realign_labels.py --audit` checks every traced outline against its image:
aligned outlines score 20.6-37.6 in fish-versus-foam edge contrast across the 52
traced trout; HRN_4 scored 5.4, and 26.2 once shifted back. A four-point eye box
was tried for fish without an outline and rejected: it also scores highly on the
snout's edge against the foam, and flagged HRN_10 at -158 px when its points were
exactly right.

## Review records lost on a second save

Auto-label records, per landmark, whether a person accepted it, corrected it or
never looked, and the dataset builder leaves the unreviewed ones out of training.
That record lived only in the open page. Reopening a saved fish read its
coordinates back but not its record, so saving it again -- to fix a ruler span, to
add an outline -- wrote the file with no record at all. Every point still where
the model put it then trained as a hand label: the model learning its own output.

Found on 2026-09-16 while making Auto-label work on the frontal view. HRN_10,
HRN_16 and HRN_28 have no record, and 13, 17 and 16 of their lateral landmarks sit
within a pixel of the cached prediction. Which of those were looked at cannot be
recovered.

The record is now read back when a fish is opened, so a second save writes it
unchanged. Saves also keep the model's confidence for unreviewed points, which
older records lack; those show as `?` and are treated as unsure.

A single record shared by both views would have caused the same loss another
way, because a frontal Auto-label replaced it whole. Each view now has its own:
lateral under `metadata.assist` as before, frontal under
`metadata.assist_frontal`, and the builder reads the one for the view it builds.

The frontal model is pinned to epoch 200 by `model.json` in its project. Training
picked `snapshot-best-030` on its validation metric, but epoch 200 was better on
the seven held-out fish: median mouth-width error 0.73 mm, worst 1.7 mm.

## The ruler's scale is not the same along its length

Ruler auto-scale measures one tick spacing for the whole photograph. Laying that
spacing along the ruler as a dot per millimetre, and finding the real tick near
each dot, shows the dots on the ticks across the middle and drifting toward the
ends -- on TXD_46 a whole millimetre by each end, on ASN_3 two. The drift runs the
same way on both sides of the middle, so it is not lens distortion, which would
be symmetric: the millimetres widen from one end of the ruler to the other, as
they do when one end is nearer the camera. Checked without the tick tracking by
measuring the gaps between the long centimetre ticks: on TXD_46 they grow 247,
249, 250, 252, 254, 257, 257, 259 px from left to right; HRN_45, flatter in the
tracking, gives 252-254 px.

Across six Cornell photographs the two ends of the ruler differ by 1-7% in px/mm.
A single scale is therefore a few percent off at the ends of the ruler, and the
fish, which lies below it, is measured through the same tilt. The labeler now
draws the dots so this can be seen per photograph; it does not yet correct for it.

The dots also expose a wrong auto-scale: on ASN_42, read at 15.07 px/mm where the
rig is about 20.6, they fall between the ticks almost everywhere.

## A fin outliner trained on the lab's tracings

Segment Anything untrained missed fin area by a median 8-35%. A model trained on the
lab's own dense tracings (`scripts/build_fin_dataset.py`,
`scripts/train_fin_segmenter.py`): one network for all four fins, given a crop
framed from the fin's base and tip and sized from standard length, plus a map
marking that base and tip. ResNet-50 encoder, the one DeepLabCut uses. Measured by
five-fold cross-validation split by fish, as area error against the hand tracing.

| fin | tracings | median \|error\| | worst | IoU |
|---|---|---|---|---|
| pectoral | 24 → **37** | 5.7% → **3.2%** | 29% → 31% | 0.91 |
| dorsal | 18 → **19** | 13.7% → **11.0%** | 51% → 41% | 0.79 |
| pelvic | 22 → **27** | 10.0% → **6.9%** | 78% → 111% | 0.84 |
| anal | 25 → **33** | 4.7% → **4.6%** | 19% → 20% | 0.89 |

Two runs, 89 fins then 116, as the lab traced more. Every median improved, and the
amount each improved tracks how many tracings it gained: the pectoral gained 13 and
halved, the dorsal gained 1 and barely moved. An earlier run on 81 of the first 89
gave 4.3 / 8.3 / 8.8 / 5.2%, so with about twenty tracings a fin these still move
several points between runs -- read them as ranges, and read the dorsal's as the
widest.

**The dorsal is short of tracings, not short of ideas.** It is the only fin where
tracing density visibly matters: fins traced with fewer than 20 vertices come back
at 15.6% median against 8.7% for the denser ones (n=8 and 11). Across the other
three fins density makes no difference at all. 29 traced fins are below the
16-vertex floor and are not used -- 8 of them dorsal.

**The big misses are two different failures, and only one is the model's.**
Over-prediction is concentrated on the smallest fins: the bottom quarter by area
comes back at 7.2% median against 3.2-4.0% for the top half, and the three worst
(+111%, +80%, +50%, all pelvic) are all small fins where the model outlines a
longer structure than the tracing does. On TXD_17 that is the tracing -- its pelvic
tip already sat beyond its own outline, noted here when the crops were first built.
Under-prediction is the model's: HRN_14's dorsal at -26% is a large fin whose pale
upper membrane the model does not pick up, and the tracing does.

One fish, ASN_30, over-predicts on all four fins (+31, +40, +80, +8) on tracings of
38-57 vertices. Excluding it, the median over every remaining fin is 4.8%. A whole
fish biased one way is a question about that fish's tracings rather than about a
fin.

**Framing from predicted landmarks costs nothing measurable.** Scored again on the
26 fish the keypoint model never trained on, with crops framed from its
predictions: medians within a point of hand framing on all 65 fins. On the 29 fins
where predicted and labelled points actually differ -- including every fin of
ASN_24, ASN_27, ASN_30 and HRN_42, labelled wholly by hand -- the median is 6.0%
predicted against 7.0% hand-framed, with no fin consistently worse.

**What does cost: 6 of 71 fins got no crop**, because the plausibility check
dropped a predicted tip (HRN_15 dorsal; TXD_17 pectoral, dorsal and pelvic; TXD_18
anal; TXD_2 dorsal). Those fins would get no outline at all.

The large misses are small, folded or pinned fins, where a few pixels are a large
fraction of the area, and at least one is a labelling inconsistency: TXD_17's
pelvic tip sits well beyond its own traced outline.


## Species from the photograph itself: parked

Before going further with an image classifier for the shiners, BioCLIP 2
(Imageomics, Ohio State; ViT-L/14 trained on TreeOfLife-200M) was compared, zero-shot,
with the landmark identifier on the same fish, choosing among the same candidates.
Top 1 / top 2:

| fish | n | BioCLIP 2 | landmarks |
|---|---|---|---|
| reference photographs, 7 shiners | 78 | 49% / 73% | 59% / 74% (left out) |
| tray fish, points corrected | 15 | 47% / 60% | 60% / 80% |
| tray fish, model points | 87 | 40% / 67% | 34% / 49% |
| Catatonk minnows, among all 21 | 41 | 29% / 46% | 68% / 85% |

Their errors differ: the right species is in one or the other's top 2 for 87-94%
of fish, and a plain rank sum puts it in the top 2 for 82% of reference fish and
70% of model-point tray fish. Two caveats on BioCLIP's side, in opposite
directions: TreeOfLife-200M includes GBIF's preserved-specimen photographs, so it
may have trained on these very images; and pybioclip resizes every image to
224 x 224 without keeping proportions, so a shiner cut-out, about three times as
long as deep, reaches the model squeezed threefold along its length.

A closed-set classifier on frozen BioCLIP embeddings (letterboxed fish plus head,
mid-body and tail tiles; logistic regression with whole lots held out; a
leave-one-collection-out test) was written but not run. It is parked: whole-image
identification is well served by models like BioCLIP, and what caliPr adds is the
characters a person identifies by -- shape, and the counts below. Anyone picking
it up should hold out whole collections, not just lots: the reference set is
confounded by collection (47 of 50 *L. cornutus* and 47 of 49 *L. chrysocephalus*
photographs are FMNH's, 40 of 52 *N. rubellus* INHS's, 37 of 47 *C. spiloptera*
JFBM's), and a classifier can learn the light box.

## Fish scales: a view that shows them, and a count along a row

Shape cannot separate *Luxilus cornutus* from *L. chrysocephalus* (52%, chance, in
the size-standardised pair test), and the characters that do are scale counts
(Gilbert 1964). The labeler took counts typed in by hand; this is the attempt to
make the scales easier to see and to count them from the photograph.

### Seeing them: local contrast wins, and the dots are the problem

Compared on an FMNH, a CMN and a JFBM photograph: local contrast (CLAHE on
luminance, or harder on the green channel), a high-pass, a difference of Gaussians
and a black top-hat. **CLAHE brings the scale margins up best on all three**, down
to the FMNH Luxilus at about 23 px a scale. The high-pass is close behind; the
difference of Gaussians, at the one width tried, blurred the margins; the top-hat
lights every melanophore and is unreadable.

The obstacle is not faint margins but melanophore dots, as dark as the margins and
nearly as wide. A Hessian line measure (larger eigenvalue less the smaller) drops
a dot's centre, but its rim still reads as a short curved line: a 3 px dot keeps
about half the response of a 2 px margin line, and Frangi's blob ratio does no
better. Where a lateral stripe is dense with dots (*N. atherinoides*) the margins
under it cannot be read at all; the rows above and below can.

Built: the **scales** view in the Image tab (local contrast / margins only /
high-pass, strength, channel), filtered on the server at the photograph's
coordinates and sized from the fish's standard length (`fish_morpho.fish_scales`,
`scripts/scale_view.py`).

### Counting: spacing first, then one tick per margin

Counting peaks along a row fails on dots and glints. What works: straighten a
narrow band around the traced row (a fifth of the longest spacing allowed, so it
stays in one row and does not see the next row's margins half a scale on), read
the margin measure along it, take the margins' spacing from the band's
autocorrelation within bounds set by standard length, then place the ticks by
dynamic programming -- the best chain of margin crossings whose gaps stay near that
spacing (beat tracking, Ellis 2007) -- so a margin lost under a dot is bridged
rather than skipped, and a dot between margins is not counted. On synthetic flanks
(crescent margins, three rows offset half a scale, 900 dots, a glare ramp) the
count is exact to one at 18-30 px a scale.

Built: **Trace** beside each count. Click along the row; the server proposes the
ticks; clicking a tick drops it, clicking the row between ticks adds one. The row,
its ticks and the proposed count are saved with the fish (`metadata.count_traces`).

### Calibrated against published lateral-line counts

The counter was run, untraced, on 470 reference photographs: a straight path from
just behind the opercle to the end of the vertebral column, parallel to the
head-to-tail axis. A straight path crosses each oblique scale series once, so it
should give the lateral-line count without finding the lateral line. Published
counts for all 21 species came mainly from Jenkins & Burkhead 1994 (via the
Virginia agency's species booklets), Scott & Crossman 1973 and the Wisconsin Fish
ID key. For spottail shiner and bluntnose minnow, whose counts rise northward, the
northern ranges were used: the photographed fish are from Minnesota, Illinois and
Canada.

Before calibration, species means tracked the literature closely (r = 0.95 over 14
species with 3+ fish) but ran about one scale high, and fish of one species spread
by 2-5.5 scales. The tick chain was rewarded for every crossing, so a dot between
two margins earned a tick of its own. Calibrated, by a grid over the spacing cap,
a cost per tick and the gap tolerance, scored by leaving each species out:

- **Per-tick cost 0.25** (`TICK_COST`): species means within a median 0.7 scales
  of the published usual range's middle, worst 3.7, with no offset applied.
  Medians: *L. cornutus* 40 (usual 38-40), *L. chrysocephalus* 38 (37-39),
  *N. volucellus* 36 (36-38), *C. spiloptera* 37 (37-39), *N. procne* 36 (34-37),
  *C. analostana* 35 (35-37), creek chub 52 (52-58), stoneroller 48 (47-55).
- **The cap on the count stays at 64.** At 80 or 95 the spacing search locks onto
  half a scale on some shiners and the spread within species doubles (to 6.5-8.5).
  So the dace (61-75) are out of reach: one longnose dace read 41.
- **Per fish it is about ±3 scales** (within-species SD about 3, against a natural
  ±1-1.5) whatever the tick settings: the noise is in reading the photograph, not in
  placing ticks. Enough to tell small-scaled genera from shiners, not to split two
  shiners that differ by two scales; a traced, checked count is still the count.

The settings were chosen on the same 14 species they were scored on (15 tried), so
the 0.7 is a little optimistic.

### One clear scale, and the rest filled in: no better

The idea: on a photograph where the margins cannot be followed, find one scale
that can be measured, and fill the line with scales sized from it by how scale
length changes along the body. Nobody has measured that change along the lateral
line in any leuciscid (bitterling, salmon and bluegill studies measure removed
scales, and disagree on where the largest are), so it was measured here, on 118
clear photographs (22+ px a scale), as the local spacing in windows of about seven
scales, each window on its own:

| along the line | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| *Notropis* (81) | 1.00 | 1.02 | 1.02 | 1.00 | 0.96 | 0.93 | 0.97 | 1.00 | 0.93 |
| *Luxilus* (18) | 1.01 | 1.03 | 1.03 | 0.96 | 0.94 | 0.93 | 0.94 | 1.00 | 1.03 |
| *Cyprinella* (13) | 1.00 | 1.03 | 1.00 | 1.00 | 0.96 | 0.92 | 0.93 | 0.96 | 0.96 |

Nearly flat: within about 8%, a little shorter behind mid-body, the same in all
three genera. The elevated anterior scales of *Luxilus* and the mimic shiner do
not show as shorter ones, so they are taller rather than narrower.

Filled from k adjacent scales on those same fish, profile built without the fish,
against the full count: 1 scale, median error 3-5 and within ±1 on 10-18% of fish;
3 scales, 2-3 and 19-25%; 5 scales, 2 and 26-30%. The profile barely beats assuming
every scale the same length. The arithmetic says why: the count is the line's length
over the average scale length, so a count right to ±1 in 38 needs the average
scale length to 2.6%, and one scale at 15-25 px is not measured to 2.6% (one pixel
is 4-7%), nor is one scale the average of the others. Measuring the spacing across
every stretch that can be read, as the counter already does, is the same idea with
as many scales as the photograph allows. (The "full count" here is the counter's own
on clear photographs: this tests the filling-in, not the counter.)

### Built: filled in from every stretch that can be read, with a likely range

What the one-scale test left standing: use every scale the photograph lets you
read, not one. The row is cut into windows about six scales long, one scale apart;
a window counts as read when its own autocorrelation, at its own spacing, reaches
0.35 (0.4% of windows on a plain noisy ground do, a quarter of those on a shiner).
Each read window's spacing, divided by the relative scale length where it lies on
the body (`LENGTH_ALONG_BODY`, the table above re-measured by x/SL from the snout),
estimates the mid-body scale length; the median of those that agree within 30% is
taken (a window locked on half or double the spacing falls out), and the row is
filled with scales of it, plus one: a row so many scale lengths long holds a scale
part-way in at each end. Without the extra one the filled-in count ran 1-1.4 low.
Fewer than 8 scales read across all windows, and nothing is filled in.

The likely range: on the clear photographs, filling in from 10-23 scales strayed
from the whole row's count by 4.5-5% (robust SD) whatever the number measured, so
the range is mostly that floor -- about ±3 on a count of 38 at 90%. It says how far
the reading may be from what reading the whole row would give, not whether the
whole row's reading is itself right: that still waits on counts made by hand.

When to fill in rather than tick every margin: on 317 shiners where both could be
done, below 30% of the row readable the ticks fell inside the published range for
67% of fish and the filled-in count for 77%; above 30% the two were level. So a
tick at every margin only when the read stretches cover 30% of the row
(`TICKS_COVERAGE`), and filled in otherwise.

On 460 reference photographs, untraced flank path: 358 read margin by margin (count
inside the published range for 84%, its likely range overlapping it for 96%), 46
filled in that were refused before (76%; 93%), 56 still refused. In the labeler a
filled-in count reads "≈37 (likely 33-41), filled in from 13 scales measured", its
ticks drawn only where it measured and not editable, since the count is not them.

### Against counts made by hand: a blind check, waiting on the counts

Every check above compares the counter with itself or with a species' published
range; neither says whether one fish's count is right. So 32 shiners are being
counted by hand: three of each of the nine look-alike species (a small, a middle
and a large photograph, one of them filled in where the species has any), two creek
chub, two stoneroller and a fallfish, drawn at random from fish whose ticks or
enhanced flank had never been looked at. Blind twice over: the counter's counts, on
the path a batch would use, were frozen before any counting (the frozen file's
sha256 is checked when they are compared), and the photographs are cropped to the
fish under neutral names in shuffled order, so neither a museum tag nor the
filename gives the species and its textbook count. Trace is left alone, since it
shows the counter's answer. The rule was set beforehand: within one scale for at
least 80% of fish, and the count can be proposed on photographs like these;
otherwise it stays a tool checked by hand.

### Ticks on the scales' pale rims, where a row shows them

On *Cyprinella analostana* 7, three lateral-line scales outlined by hand showed
the ticks sitting mid-scale: the outlines cross the line at thin pale rims, and
the ticks landed on the dark band of pigment just inside each scale, 7-12 px
short of the edge, with one tick inside the third scale. The count was right (35,
published (33) 35-37 (38)), since there is one band per scale.

Tried and dropped: moving each tick to the low point after its band. The lowest
point within half a scale fitted the outlines, and the first low point nearly so,
but both moved the ticks on thin margin lines by a fifth to a quarter of a scale
(*N. rubellus* 10-13 px, *N. procne* 9.5-16 px) with nothing to say that was
right, and neither fixed the third scale, where two ticks had been placed 18 px
apart. Ticks at the dark-to-light step after each band were 4-8 px early.

Built: the same measure on the inverted image finds pale lines. Where they repeat
strongly along the row (autocorrelation 0.25 or more, at the margins' spacing
within 15%) and give the same count, the ticks go on them: on the three scales,
1-5 px from the outlines and none inside. Otherwise the ticks stay on the dark
margins as before. On 99 hand-labelled shiners, 5 rows go to the rims and no count
changes. A row traced from the tail is now read from the head, so the answer does
not depend on which end was clicked first. The five rows on rims have not been
checked against outlines.

## Characters from the photograph, where the landmarks say to look

`fish_morpho.photo_characters`, compared beside shape in batch identification.

| character | read as | worked? |
|---|---|---|
| dorsal origin behind pelvic origin | along the fish, over SL, from landmarks | yes: emerald, rosyface, comely +0.07 SL; both *Luxilus* 0.00 |
| caudal spot | darkness at the end of the vertebral column against the lower flank | yes, once moved onto the body: spottail 0.42, the rest about 0 or below |
| snout | darkness between premaxilla and orbit | yes: blacknose and bridle +0.28-0.30, spottail, mimic and *Luxilus* below +0.05 |
| side stripe | darkest disc across the side, median of three places along it | yes: blacknose against mimic 87% of pairs |
| upright scale margins | share of margin strength in near-vertical lines, front of the side | partly: *Luxilus*, *Cyprinella*, creek chub, stoneroller 0.61-0.72 against *Notropis* 0.48-0.55; *L. cornutus* against *L. chrysocephalus* 70% of pairs, both nearly all FMNH so not the light box; does not separate mimic from spottail |
| dorsal fin spots, front and rear | darkness of the membrane | **no**: fins are translucent over the background (white on most light boxes, black on CMN's) and often folded; normalising to the background above the fin did not bring up the spotfin's blotch |

The caudal spot first sat 0.03 SL behind the vertebral column, which is fin over
the light box, and read every species pale; the side stripe first read one disc on
the axis, which misses a stripe that rides higher on some fish.

Identification, each hand-labelled fish left out (119 fish, 13 species with 4+):
top 1 60% → 63%, top 2 75% → 79%; *L. cornutus* against *L. chrysocephalus* 43% →
57% (chance for shape alone). Tray fish on model points (102): top 1 35% → 37%, top 2
49% → 51%. Modest. One likely reason: each character's spread per species is
estimated from the 4-12 hand-labelled fish of that species, while across all ~45
photographs per species (model points placing the characters) the pairs separate
clearly -- spottail against mimic in 88% of pairs on the caudal spot alone. Using
every identified photograph for the characters, not only the hand-labelled ones, is
the next thing to try.

Reference fish's characters are kept in `<study>/photo_characters.json` with a
fingerprint of the landmarks they were read at (`scripts/photo_characters.py`;
the labeler reruns it for changed fish before scoring a batch), and travel in the
packaged reference set, which has no photographs. Tray fish's are kept in the
tray's cache under the same kind of fingerprint.
