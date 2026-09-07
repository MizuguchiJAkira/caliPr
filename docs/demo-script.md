# caliPr — silent demo script

A ~2:30 screen recording with on-screen captions and no narration. Captions carry
the whole argument, so each one states a fact the viewer can see happening.

Two rules that shaped the wording below. **Say what is on screen, not what you
were thinking** — a caption that comments rather than describes reads as a
voiceover with the voice missing. And **show the tool being wrong on purpose**:
the flagged landmarks are the most persuasive thing here, because anyone
evaluating this has already seen demos where everything works.

---

## Before you record

| | |
|---|---|
| **Warm the model.** | Click Auto-label once on any fish before recording. The first call loads the snapshot (~5 s); every one after is ~1 s. A five-second pause on camera reads as a hang. |
| **Protect the real labels.** | `python scripts/label_server.py --out /tmp/demo-sidecars` — Save then works on camera but writes nowhere near `data/cornell/sidecars/`. |
| **Window.** | ~1600×1000. Wider and the fish gets small; narrower and the landmark list crowds. |
| **Dataset.** | Open `cornell` first. The alewife shot comes at the end. |
| **Hero specimen.** | `ASN_37` — 13 confident, 6 flagged, and its dorsal fin is dried folded flat, which is what makes shot 5 work. |
| **Contrast specimen.** | `ASN_35` — 9 confident, 10 flagged. Optional; use if you want the harder case. |
| **Hide the clock and notifications.** | |

Verify before rolling: the specimen list header should read **46/131**.

---

## Shot list

### 1 — Open on the specimen list · 6 s

*On screen:* the labeler, `cornell` loaded, list of specimens visible.

> **Museum fish photographs. 131 of them.**
>
> **46 have been landmarked by hand. That took hours.**

---

### 2 — Type in the search box · 5 s

*On screen:* type `todo` — the list filters to 85.

> **85 still to do.**

---

### 3 — Click ASN_37 · 5 s

*On screen:* the photograph loads. Landmark list on the right shows 0/30 (23 keypoints, 5 outlines, 2 ruler points).

> **A specimen the model has never seen.**

---

### 4 — Click Auto-label · 8 s

*On screen:* ~1 s pause, then 19 points appear, green and orange.

> **One click. Nineteen landmarks, in about a second.**
>
> **Green: the model is confident. Orange: it is not.**

Let the colours sit for a beat before the next caption. This is the shot the
whole demo is built around.

---

### 5 — Zoom to the dorsal fin · 12 s

*On screen:* scroll-zoom onto the back of the fish, where `dorsal_tip` and
`dorsal_base_center` are floating above the body, ringed in orange.

> **These two are wrong — they are off the fish entirely.**
>
> **This specimen's dorsal fin dried folded flat. There is nothing there to find.**
>
> **The model flagged both of them itself.**

The single most important caption in the demo is the third one. Give it 4 s.

---

### 6 — Correct the flagged points · 20 s

*On screen:* drag `dorsal_tip` onto the real fin margin, then
`dorsal_base_center`, then one more flagged point. A faint line stays behind
showing where the model had put each one.

> **Correcting one takes a second.**
>
> **The line shows how far the model was off.**
>
> **Corrections are recorded. Accepting a point takes a keypress.**

---

### 7 — Press A on a few confident points · 8 s

*On screen:* select two or three green points, press `A` on each. The counter in
the sidebar moves.

> **Confirming a point is deliberate, never assumed.**
>
> **"I checked it" and "I never looked" are different claims.**

---

### 8 — Save · 5 s

*On screen:* click Save sidecar; the toast confirms.

> **Saved — with a record of which points were corrected, and which were confirmed.**

---

### 9 — Terminal: correction report · 15 s

*On screen:*
`python scripts/correction_report.py --dataset data/cornell --px-per-mm 20.71`

> **Every correction feeds back.**
>
> **This is where the model is weakest, measured on fish it had never seen.**
>
> **Corrections train it. Confirmations only count — a human agreeing with a
> suggestion is weaker evidence than one placing a landmark cold.**

Third caption is optional if you want a shorter cut. It is the one a
methodologically careful viewer will most want to hear.

---

### 10 — Switch to the alewife dataset, Auto-label · 15 s

*On screen:* dataset dropdown → `alewife`, pick any specimen, Auto-label. All 19
points come back orange.

> **A different order of fish. The model was trained only on trout.**
>
> **Every landmark flagged. Median confidence 0.28.**
>
> **It does not know this animal, and it says so instead of guessing.**

This is the answer to "how do you know it isn't just confidently wrong". Worth
the 15 seconds.

---

### 11 — Close · 6 s

*On screen:* back to the trout, full annotated fish, or the overlay image.

> **Thirteen of nineteen landmarks placed. Six flagged for review.**
>
> **Standard length, measured automatically off the ruler in frame: 140.4 mm.**
>
> **The uncertainty is the output that matters.**

---

## Captions, plain list

For pasting into an editor. Numbers match the shots above.

```
1a  Museum fish photographs. 131 of them.
1b  46 have been landmarked by hand. That took hours.
2   85 still to do.
3   A specimen the model has never seen.
4a  One click. Nineteen landmarks, in about a second.
4b  Green: the model is confident. Orange: it is not.
5a  These two are wrong — they are off the fish entirely.
5b  This specimen's dorsal fin dried folded flat. There is nothing there to find.
5c  The model flagged both of them itself.
6a  Correcting one takes a second.
6b  The line shows how far the model was off.
6c  Corrections are recorded. Accepting a point takes a keypress.
7a  Confirming a point is deliberate, never assumed.
7b  "I checked it" and "I never looked" are different claims.
8   Saved — with a record of which points were corrected, and which were confirmed.
9a  Every correction feeds back.
9b  This is where the model is weakest, measured on fish it had never seen.
9c  Corrections train it. Confirmations only count.
10a A different order of fish. The model was trained only on trout.
10b Every landmark flagged. Median confidence 0.28.
10c It does not know this animal, and it says so instead of guessing.
11a Thirteen of nineteen landmarks placed. Six flagged for review.
11b Standard length, measured automatically off the ruler in frame: 140.4 mm.
11c The uncertainty is the output that matters.
```

---

## A 60-second cut

If you need something shorter, keep shots 1, 4, 5, 6, 10 and these seven
captions. The argument survives: here is the problem, here is one click, here is
it being wrong and saying so, here is the fix, here is it refusing to guess on an
animal it does not know.

```
Museum fish photographs. 131 of them. 46 landmarked by hand.
One click. Nineteen landmarks, in about a second.
Green: the model is confident. Orange: it is not.
These two are wrong — the dorsal fin dried folded flat.
The model flagged them itself.
Correcting one takes a second, and the correction feeds back.
On a fish it was never trained on, it flags all nineteen rather than guessing.
```

---

## Every number a caption claims

Check these still hold on the day; a wrong number on screen is worse than no
caption. Regenerate with `scripts/predict_landmarks.py` and
`scripts/dlc_report.py`.

| claim | value | where it comes from |
|---|---|---|
| photographs | 131 | `data/cornell/lateral/` |
| hand-labelled | 46 lateral | badge key in the labeler |
| still to do | 85 | 131 − 46 |
| landmarks placed | 19 | the trout study's keypoint set |
| ASN_37 split | 13 confident / 6 flagged | `predict_landmarks.py` |
| time per prediction | ~1 s warm, ~5 s cold | measured on Apple MPS |
| alewife median confidence | 0.28, and 114/114 points flagged over 6 specimens | `predict_landmarks.py` on `data/alewife` |
| automatic SL, ASN_37 | 140.4 mm | 2907 px ÷ 20.71 px/mm |
| held-out error | 0.81 mm median | `dlc_report.py`, 9 specimens |

Two claims worth **not** making on screen, because they would not survive a
question: that the model is accurate enough to replace hand labelling (it is
not — 0.81 mm median hides two landmarks over 2 mm), and that the 140.4 mm SL is
validated (`caudal_base` is one of the flagged points on that fish, so that
number rests on a landmark the model is unsure of).
