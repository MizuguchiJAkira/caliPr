# `data/validation/` — MorFishJ oracle validation

This directory drives `scripts/morfishj_validation.py`, which diffs the
`fish_morpho` Python engine against **MorFishJ** (the ImageJ plugin by
Mattia Ghilardi, Leibniz-ZMT — https://github.com/mattiaghilardi/MorFishJ)
as the ground-truth oracle. A green run is the gate for "the port is good
enough to replace MorFishJ for real work."

## What lives here

```
morfishj_reference.csv   Ground-truth trait values from the MorFishJ GUI.
labels/                  Sidecar JSONs for the SAME specimens (fish_id = filename stem).
auto_calibration_audit.* Ruler-detector audit on the iDigBio pool (separate concern).
pool_triage.csv          iDigBio pool triage (separate concern).
```

> ⚠️ **`morfishj_reference.csv` is deliberately not committed.** The harness
> was proven end-to-end against a placeholder generated from the engine's
> *own* output, which validates the plumbing but not the science — shipping
> that file would risk it being mistaken for ground truth. Create the real
> one from MorFishJ GUI output as described below.
>
> Note that the pipeline has since been validated against **physical caliper
> measurements** (median 1.05% agreement on standard length across 17
> specimens), which is an independent oracle; the MorFishJ comparison remains
> useful for checking trait-by-trait definition fidelity.

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
3. **Python port:** produce a sidecar JSON for each of those same
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
Revisit them once a real reference set exists — if MorFishJ and the port
agree far tighter than 0.2 mm, tighten the gate.

## Definition-fidelity note

Independent of the numeric diff, the engine's trait *formulas* were
checked against MorFishJ's published definitions (manual v0.2.2,
`main_traits.html`) and track them faithfully, including the subtlety
that `Snl` anchors on the premaxilla landmark while `AO` anchors on
reference line D (body anterior extremum) — the two coincide only on a
closed-mouth specimen where the premaxilla is the anterior-most point.
