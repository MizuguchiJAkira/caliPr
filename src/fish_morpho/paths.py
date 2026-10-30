"""Where caliPr keeps what it writes.

Run from a clone, everything stays in the clone, as the README describes: studies
in ``data/``, exports in ``results/``, the trained models and the Auto-label
environment beside them. The Mac app's code sits inside the application, which is
not to be written to (macOS may even run it from a read-only copy), so the app
points these elsewhere:

- ``CALIPR_HOME``: studies and exports, ``~/Documents/caliPr`` in the app, where
  the person using it will look for them.
- ``CALIPR_MODELS``: the trained models and the environment Auto-label runs in,
  ``~/Library/Application Support/caliPr`` in the app. Nobody opens these by hand.

Standard library only: the model downloader imports this and runs without NumPy.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The code: ``scripts/``, ``src/`` and the labeler's page. Read, never written.
CODE = Path(__file__).resolve().parents[2]

#: Holds ``data/`` (one folder per study) and ``results/`` (exports).
WORK = Path(os.environ.get("CALIPR_HOME") or CODE).expanduser()

#: Holds the model projects (``dlc_project*/``, ``fin_seg_runs/``) and ``.venv-train``.
MODELS = Path(os.environ.get("CALIPR_MODELS") or CODE).expanduser()

#: The interpreter with PyTorch and DeepLabCut, if Auto-label has been installed.
TRAINING_PYTHON = MODELS / ".venv-train" / "bin" / "python"
