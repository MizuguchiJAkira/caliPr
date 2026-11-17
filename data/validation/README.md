# `data/validation/` — MorFishJ oracle validation

This directory drives `scripts/morfishj_validation.py`, which diffs the
`fish_morpho` Python engine against **MorFishJ** (the ImageJ plugin by
Mattia Ghilardi, Leibniz-ZMT — https://github.com/mattiaghilardi/MorFishJ)
as the reference. It checks that caliPr's reimplementation of the MorFishJ
trait definitions gives MorFishJ's values on the same specimens.

## What lives here

```
morfishj_reference.EXAMPLE.csv   SYNTHETIC: one made-up fish, to show the format.
labels/                  Sidecar JSONs for the SAME specimens (fish_id = filename stem).
auto_calibration_audit.* Ruler-detector audit on the iDigBio pool (separate concern).
pool_triage.csv          iDigBio pool triage (separate concern).
```

> ⚠️ **No comparison with MorFishJ has been run yet.** The only reference in
> this folder, `morfishj_reference.EXAMPLE.csv`, is synthetic: its one fish
> (`BKT-2025-EXAMPLE`, the same as `examples/sample_sidecar.json`) was generated
> from caliPr's *own* output to test the plumbing, so it agrees by construction
> and says nothing about fidelity. Create the real `morfishj_reference.csv` from
> MorFishJ GUI output as described below.
>
> Standard length has been checked separately against physical caliper
> measurements (`scripts/caliper_check.py`); that is an independent check of
> calibration and landmarking, not of the MorFishJ trait definitions.

## How to produce a real validation

1. **Pick ~5 labeled specimens** that you can measure both ways.
2. **MorFishJ (oracle):** open each specimen photo in ImageJ/Fiji, run the
   MorFishJ complete morphometric analysis, and collect its CSV output.
   Assemble one row per fish into `morfishj_reference.csv` with a
   `fish_id` column plus one column per MorFishJ trait code
   (`TL, SL, MBd, Hl, Hd, Ed, Eh, Snl, POC, AO, EMd, EMa, Mo, Jl, Bs,
   CPd, CFd, CFs, PFs, PFl, PFi, PFb`). Blank cells are skipped. The 8
   Cornell extras (`LJl, DFh, DFs, PlFl, PlFs, AFh, AFs, MW`) have no
   MorFishJ oracle and are ignored if present.
3. **caliPr:** produce a sidecar JSON for each of those same
   specimens in `labels/`, named `<fish_id>.json` (the stem must equal
   the `fish_id` in the CSV). Use `scripts/cvat_to_sidecar.py` on a CVAT
   export, or hand-author from `examples/sample_sidecar.json`.
4. **Run the diff:**
   ```bash
   python scripts/morfishj_validation.py \
       --reference data/validation/morfishj_reference.csv \
       --labels    data/validation/labels/ \
       --tolerance-mm 0.2 --tolerance-mm2 2.0 --tolerance-deg 1.0
   ```
   Exit 0 = every compared trait within tolerance. Exit 1 = at least one
   deviated (printed with Python-vs-MorFishJ values and the delta). Exit
   2 = input error.

## Tolerances

Defaults (0.2 mm, 2.0 mm², 1.0°) are the hand-measurement noise floor.
Revisit them once a real reference set exists — if MorFishJ and caliPr
agree far tighter than 0.2 mm, tighten the gate.

## Definitions that differ from MorFishJ

Apart from the numeric comparison, the engine's trait *formulas* were checked
against MorFishJ's published definitions (manual v0.2.2, `main_traits`). Most
follow them. Three do not:

| trait | MorFishJ | caliPr |
|---|---|---|
| `SL` | from D, the most anterior point of the head, to the caudal base | from `premaxilla_tip` to `caudal_base` |
| `Snl` | from D to the anterior margin of the orbit | from `premaxilla_tip` to `eye_anterior` |
| `AO` | from the orbit to the head margin along the horizontal line through the eye (J) | from D to `eye_anterior` |

caliPr's `AO` is therefore MorFishJ's `Snl`. SL and Snl start at the premaxilla
so that neither depends on how far the lower jaw projects or the mouth gapes. On
a closed-mouth fish whose premaxilla is the most anterior point, caliPr's SL and
Snl equal MorFishJ's; where the lower jaw projects, they are shorter.
