# Tutorial

Getting from a folder of photographs to a spreadsheet of measurements. No
terminal beyond the two lines below, and nothing to configure.

```bash
git clone https://github.com/MizuguchiJAkira/caliPr.git
cd caliPr
python3.11 -m venv .venv          # 3.11 or newer
./.venv/bin/pip install -e .
./.venv/bin/python scripts/label_server.py
```

Open <http://localhost:8765>. Everything after this happens in the browser.

Four packages come down — NumPy, OpenCV, openpyxl, Pillow — and it takes about
ten seconds. **Check your Python first:** macOS still ships 3.9, which is too
old, and `pip install` will refuse with a `requires-python` error rather than
anything more helpful. `python3 --version` tells you; Homebrew or python.org
gets you a newer one.

Automated landmarking is a separate, much heavier install (PyTorch, DeepLabCut,
Segment Anything, about 1.7 GB) and is **not needed to label by hand or to
export**. The trained models are downloaded separately, not kept in the
repository. Skip all of it until you want it:

```bash
python3.11 -m venv .venv-train
./.venv-train/bin/pip install "deeplabcut==3.0.1" transformers
./.venv/bin/python scripts/fetch_model.py
```

Without it the labeler runs normally, and Auto-label tells you what to run.

---

## 1. Add your photographs

Open the **dataset** menu and choose **Add folder — a new study**, then pick a
folder of photographs. It becomes a *study*, named after the folder.

![Add folder](img/tutorial/1-add-folder.png)

Two boxes to tick on the way in:

- **Use the same settings as** the study you have open. Leave it ticked for more
  fish from the same rig: the study then collects the same landmarks, reads the
  strain from each file name, and gets Auto-label's anatomy check. Untick it for
  a different species or setup.
- **Each photo also shows the head-on view in a mirror.** For photographs from
  the Cornell rig, where a mirror on the left shows the fish's head. Each one is
  split into a lateral and a frontal image. The original is kept, and a photo
  where no mirror edge can be found is added whole and named in a message. Check
  a few on the **frontal** view to see the head was not cut.

  A study can also keep **one photograph per fish** and not be cut up at all
  (`"single_photo": true` in its `schema.json`). The **frontal** tab then opens
  the same photograph, zoomed to the mirror, both views' landmarks are placed in
  the one coordinate system, and Auto-label crops for the model in memory. This
  is the better arrangement: a crop on disk is cut once, and a boundary found in
  the wrong place cuts a head off or strands the labels already placed on it.
  Where the mirror's edge cannot be found at all, the lateral view still runs on
  the whole frame and the head-on view is left for you to label by hand rather
  than predicted somewhere in the fish's flank.

You can also drag a folder straight onto the page. Subfolders are walked, and
their names are kept: `Lake_2026/Site_A/IMG_01.jpg` arrives as
`Site_A_IMG_01.jpg`, so a folder-per-site survives into the filenames. Anything
that is not an image is ignored.

To add more later, hover a study in the same menu and click **+ photos**.
**remove** takes a study out of the list by moving its folder to `data/.trash/`;
nothing is deleted, so moving the folder back restores it. A study with saved
labels asks you to type its name first.

A study is a plain folder under `data/`, so you can also just make one yourself
and it will appear without restarting the server.

---

## 2. The three columns

Controls on the left, the landmarks to place in the middle, the specimen list
below the controls. Click a specimen to open it.

![Layout](img/tutorial/2-layout.png)

`/` focuses the search box. It matches catalogue numbers as you type, and also
takes words like `todo`, `labelled`, `heldout`. Terms combine, so `todo txd`
gives the TXD specimens still to do.

---

## 3. Auto-label, then check what it flags

**Auto-label** runs the trained model on the open specimen's current view. On
the lateral view that is the landmark model; on the frontal view it is a separate
model that places the two mouth corners.

![Auto-label](img/tutorial/3-auto-label.png)

Points arrive coloured by the model's own confidence:

| | |
|---|---|
| **green** | confident |
| **orange, ringed** | not confident — these are the ones to look at |

Whether it also draws the body outline depends on the study; the brook trout
study traces it by hand. The **pectoral and anal fins** come back outlined — those
are the two the fin outliner matches hand tracings on closely enough to be worth
starting from (3.2% and 4.6% median area error, inside the spread between two
tracings of the same fin). The dorsal and pelvic are not offered; trace those
yourself. The sidebar lists what still needs review.

A predicted outline is drawn differently from one you traced and is **never
accepted by `A`** — `A` walks landmarks only. Drag its vertices onto the fin, click
the line to insert one, or press `Z` to take one away. Any of those marks the
outline as yours: until then it stays recorded as the model's, kept out of what
the next model trains on, and named in the workbook's QC note beside the fin area
that came from it.

Some landmarks come back missing, with a note saying where the model tried to put
one. Those failed an anatomy check — not a confidence threshold, but a fact about
where the structure can be on a fish: this landmark's place between the eye and
the caudal base, or two structures coming back in an order that would mean they
had been swapped. Both are measured from the study's own labels. A point the
anatomy rules out is not offered at all, because you can put a missing point back
but you cannot un-see a confident one in the wrong place.

Then work down the points with one key:

- **press `A`** to accept the point in hand and move to the next one waiting.
- **drag it** to correct it, then press `A` to move on. A faint line stays
  behind showing where the model had it, so you can see the size of the fix.

`A` walks every predicted point, in the order the list shows them — head,
pectoral, dorsal, pelvic, anal, then the peduncle and caudal at the back. Fin
landmarks are included, and the message names the fin when the walk reaches one:
fins are where the model is least reliable, and an accepted point is used for
training, so those are the ones worth a second look.

Accepting is a deliberate keypress, never assumed from leaving a point alone.
Those are different claims and the sidecar records them separately. `Z` undoes
anything — an accept, a drag, a vertex — and **Clear all** empties the view in
one step, also undoable.

Each view keeps its own record of what was accepted, corrected and left
unreviewed, and the record is saved with the fish. Reopening a saved fish brings
it back, so points you never reviewed stay marked and `A` still walks them.

On the frontal view there is no ruler auto-scale: click the two ruler points and
set **known span** to the millimetres between them. Without it the fish still
exports, with mouth width left blank and the reason in the QC sheet.

On the frontal view, check both corners whatever their colour. On the seven fish
held out from training, the worst corner pair was 1.7 mm off in mouth width at
0.95 confidence, and the best was placed at 0.27.

The first Auto-label of a session takes a few seconds while the model loads,
closer to a minute on a fresh install; after that it is about a second. **Auto-label all unlabelled** runs the whole
backlog in one pass. Each fish then opens with its points already placed and
the first one selected, so `A` works from the first press; a **P** badge marks
the ones still to review.

### A different landmark scheme

Under the **Landmarks** header, **scheme** chooses what a study collects:

| | |
|---|---|
| **caliPr (23 landmarks, 33 traits)** | the default: what the traits, the trained model and this tutorial use |
| **BGNN 2D body landmarks (23)** | the Fish-AIR / BGNN shape landmarks, in their numbered order |

A study on another scheme asks for that scheme's points, in its numbering, and
exports them to TPS in that order — which is what geomorph reads. No trait is
computed there: every trait is defined in code against caliPr's landmarks, and a
mapping between two schemes would be a claim about anatomy rather than a
conversion. The measurements export says so instead of writing a sheet of blanks,
and Auto-label is off, because the model only knows the landmarks it was trained
on.

Switching is reversible and changes nothing already saved. Labels stay in their
sidecars exactly as clicked; the other scheme's points simply stop being asked
for until you switch back.

### Landmarks of your own

Every study collects the landmarks defined in the code. A study can also add its
own: **＋ add a landmark to this study**, at the end of the list. It is then
offered on every specimen in that study, saved with the labels, and written to the
TPS export. No trait is computed from it — the 33 traits are defined in code
against fixed names — and the model does not learn it.

The **✎** on any landmark renames it for this study. That changes what the labeler
and the TPS names file call it, never the name it is stored under: the traits, the
trained model and every sidecar already saved refer to that name. Renaming one of
your own to nothing removes it, and it asks first if the point is already placed.

---

## 4. Save, and export

![Save and export](img/tutorial/4-save.png)

**Save sidecar** writes one JSON file per specimen into `sidecars/`. That file
is the durable artefact — every export is derived from it, and it records which
points you corrected and which you accepted.

If every landmark is still exactly where the model put it, Save asks first. A
sidecar saved that way is model output entering the training set as ground
truth, which is worth one deliberate click to avoid.

Open **EXPORT**. Each export is saved under `results/<study>/` and opened on
your computer rather than downloaded: the workbook in your spreadsheet app, the
rest shown in Finder.

| | |
|---|---|
| **Measurements (.xlsx)** | one row per specimen, 33 traits, plus About / Ratios / Shape / QC / Validation sheets |
| **Annotations for R (.zip)** | the coordinates — `.tps` for geomorph, `landmarks_imagej.csv` in the shape ImageJ writes, landmark names, and a loader snippet |
| **Annotations (.zip)** | the labels themselves, to send back for training |
| **Labelled images (.zip)** | each photograph with its annotation drawn on |

Read the **Validation** sheet before analysing. It lists the checks that failed,
most severe first.

The coordinates are not in the workbook. They are an input to geomorph rather
than something to read in a spreadsheet, so they go out with the rest of the R
material: `landmarks_imagej.csv` is one row per landmark per specimen, the
landmark named and the photograph repeated in `Label`, exactly as ImageJ's
Multi-Measure writes it, so a series digitised here can be pooled with one
digitised in ImageJ. `load_landmarks.R` reads it into geomorph.

Mind the y axis. Both files use **image coordinates, y downward**, as ImageJ
does; the `.tps` file uses Cartesian y, because that is what `readland.tps`
expects. The two are mirror images, so pick one and stay with it. Coordinates are
millimetres where the specimen has a scale and pixels where it does not, stated
per row in a `units` column.

---

## What to expect from the automation

It is a starting point, not finished work.

- The head, jaws, operculum, pectoral and peduncle landmarks are usually right.
- **Fin tips are often wrong**, especially on specimens whose fins dried folded
  flat — and the model flags those itself, which is why the orange points are
  worth more attention than the green ones.
- The body outline follows the dorsal and anal fins instead of crossing their
  bases. Drag those vertices onto the body wall.
- **Predicted fin outlines run generous on small fins.** The smallest quarter by
  area comes back at 7.2% median error against 3.2–4.0% for the largest half, and
  the worst cases are small folded fins where the outline runs past the rays.
- On a species the model was not trained on, expect *everything* to come back
  orange. That is the correct answer, not a failure.

Accuracy figures and their caveats are in the [README](../README.md).

---

## No scale?

If there is no ruler in the frame, the labeler says so and measurements come out
in **pixels**, marked as such in a `units` column. That is a valid way to work:
the Ratios and Shape sheets are dimensionless, so a study comparing proportions
does not need millimetres.

With a ruler, click **use ruler auto-scale**, or place the two ruler points and
type the span you clicked across.

To see how far to trust the auto-scale, tick **show mm dots on the ruler**: a dot
for every millimetre at the detected scale, green within ¼ mm of its tick, amber
within ½ mm, red beyond. **check** zooms to the end of the ruler that drifts most.
Dots that go red toward the ends mean the ruler is nearer the camera at one end,
so its millimetres are wider there; dots that are red almost everywhere, or no
dots at all, mean the auto-scale is wrong for that photograph.

---

## Sharing the work

For a collaborator without Python:

```bash
python scripts/build_standalone_labeler.py --out calipr-labeler.html
```

One self-contained HTML file. They open it, label, and send back a bundle
containing their photographs and labels; `scripts/import_standalone_labels.py`
folds it in and verifies the photographs are the ones that were labelled.
