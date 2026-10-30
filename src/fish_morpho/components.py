"""The parts of caliPr downloaded separately from the labeler.

The labeler -- placing landmarks, measuring, exporting -- is the download. Two parts
are installed on top of it, each when it is wanted:

- **Auto-label**: the environment the landmark model runs in and the trained models
  (``install_autolabel.py``).
- **Batch identification**: a reference set of identified, landmarked specimens that
  the fish of a photograph are compared with (``install_component.py batch``). It
  needs Auto-label to find and landmark those fish.

``scripts/components.json`` says which parts are released for download. A part that
is not released is not offered, but one already on a machine keeps working: what
decides whether a feature runs is what is installed, not what is published.

Standard library only: the installers import this and run without NumPy.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import paths

MANIFEST = paths.CODE / "scripts" / "components.json"

#: The optional parts, in the order they are offered.
NAMES = ("autolabel", "batch")


def manifest(path: Path = MANIFEST) -> dict:
    """The manifest, or an empty one if it cannot be read: nothing is then offered."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def released(name: str, path: Path = MANIFEST) -> bool:
    """Whether ``name`` can be downloaded now."""
    return bool((manifest(path).get(name) or {}).get("released"))
