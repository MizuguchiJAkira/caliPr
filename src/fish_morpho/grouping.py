"""Which comparison group a specimen belongs to.

Morphometrics is comparative: a workbook of 500 rows is only useful once each row
carries the population, strain or site it belongs to. This resolves that label
from three sources, most specific first, so a study can supply it whichever way
suits the data it already has.

1. **The sidecar.** ``metadata.group``, set in the labeler. Wins over everything
   else, because it is the one a person entered deliberately about that specimen.
2. **A table.** ``groups.csv`` in the dataset directory, two columns
   ``fish_id,group``. This is the right shape when the grouping comes from
   somewhere outside the photographs — collection lot numbers mapped to
   localities, say, which is how the alewife populations will have to be
   assigned.
3. **The filename**, but only where the study asks for it. ``schema.json`` may
   carry ``group_from_filename``, a regular expression whose first capture group
   is the label. The trout series encodes strain as ``..._ASN_12``, so
   ``"_([A-Z]{2,4})_\\d+$"`` recovers it without anyone typing anything.

Guessing is deliberately not a fallback. A specimen with no group returns the
empty string and appears in an "ungrouped" bucket, which is visible; inferring a
label from a filename that happens to look right produces a grouping variable
nobody chose, and a comparison between groups that were never defined.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

#: Column name used throughout the exports.
GROUP_COLUMN = "group"

#: What an ungrouped specimen is called in summaries. Empty in the cell itself,
#: so a spreadsheet filter shows it as blank rather than as a real group.
UNGROUPED = "(ungrouped)"


def load_group_table(dataset_dir: Path) -> dict[str, str]:
    """``groups.csv`` as {fish_id: group}, or empty if absent or unreadable.

    A malformed table is not worth failing an export over: the group is
    additional information, and losing it degrades the workbook rather than
    invalidating it.
    """
    path = Path(dataset_dir) / "groups.csv"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
    except Exception:
        return {}
    if not rows:
        return {}
    # A header is optional; skip it when the first row looks like one.
    start = 1 if rows[0] and rows[0][0].strip().lower() in ("fish_id", "id",
                                                           "specimen") else 0
    for row in rows[start:]:
        if len(row) >= 2 and row[0].strip():
            out[row[0].strip()] = row[1].strip()
    return out


def filename_pattern(dataset_dir: Path) -> str | None:
    """``group_from_filename`` from the dataset's schema.json, if declared."""
    path = Path(dataset_dir) / "schema.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get("group_from_filename") or None
    except Exception:
        return None


def resolve(fish_id: str, metadata: dict | None = None,
            table: dict[str, str] | None = None,
            pattern: str | None = None) -> str:
    """The specimen's group, or "" when no source supplies one."""
    if metadata:
        explicit = str(metadata.get(GROUP_COLUMN, "") or "").strip()
        if explicit:
            return explicit
    if table:
        from_table = str(table.get(fish_id, "") or "").strip()
        if from_table:
            return from_table
    if pattern:
        try:
            m = re.search(pattern, fish_id)
        except re.error:
            return ""
        if m:
            return (m.group(1) if m.groups() else m.group(0)).strip()
    return ""


def summarise(groups) -> dict[str, int]:
    """{group: count}, ungrouped last, so a reader sees the design at a glance."""
    counts: dict[str, int] = {}
    for g in groups:
        key = g or UNGROUPED
        counts[key] = counts.get(key, 0) + 1
    named = {k: v for k, v in sorted(counts.items()) if k != UNGROUPED}
    if UNGROUPED in counts:
        named[UNGROUPED] = counts[UNGROUPED]
    return named
