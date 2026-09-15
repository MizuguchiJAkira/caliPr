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
