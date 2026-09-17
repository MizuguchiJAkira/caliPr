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
study traces it by hand. The sidebar lists what still needs review.

Then work down the points with one key:

- **press `A`** to accept the point in hand and move to the next one waiting.
- **drag it** to correct it, then press `A` to move on. A faint line stays
  behind showing where the model had it, so you can see the size of the fix.

`A` passes over fin landmarks. Fins are where the model is least reliable, and
an accepted point is used for training, so they are left for you to click and
review deliberately.

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
| **Landmarks for R (.tps)** | geomorph-ready, with landmark names and a loader snippet |
| **Annotations (.zip)** | the labels themselves, to send back for training |
| **Labelled images (.zip)** | each photograph with its annotation drawn on |

Read the **Validation** sheet before analysing. It lists the checks that failed,
most severe first.

---

## What to expect from the automation

It is a starting point, not finished work.

- The head, jaws, operculum, pectoral and peduncle landmarks are usually right.
- **Fin tips are often wrong**, especially on specimens whose fins dried folded
  flat — and the model flags those itself, which is why the orange points are
  worth more attention than the green ones.
- The body outline follows the dorsal and anal fins instead of crossing their
  bases. Drag those vertices onto the body wall.
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
