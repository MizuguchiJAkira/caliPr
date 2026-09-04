# Alewife (Alosa pseudoharengus) — BIOEE 4761

Scratch dataset for testing the pipeline on a body plan it was not built for.
Not part of the CUMV brook trout study; kept separate so nothing here can
contaminate that dataset or its validation.

Drop lateral photographs into `lateral/`. No preprocessing needed if each photo
is already one fish plus a scale reference — `preprocess_jonah.py` is specific to
the CUMV rig (mirror split, ruler band, label card) and should be skipped here.

Run the labeler against it:

    .venv/bin/python scripts/label_server.py \
        --images data/alewife --out data/alewife/sidecars

Known caveats for a non-salmonid:

- The DLC keypoint model is trained on brook trout and will not transfer. Use
  manual labeling only.
- `anatomy_constraints` allowances are fitted to brook trout and do not apply.
- The dorsal polygon's hint says to exclude the adipose fin, which alewife do not
  have. Harmless, but ignore it.
- Alewife carry a ventral keel of scutes; decide once whether the body outline
  follows the keel or the body wall, and be consistent.
