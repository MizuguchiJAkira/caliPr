"""Who placed each landmark.

The labeler writes an operator ID beside every landmark, outline, ruler point and
midline, per view: ``lateral.operators = {"premaxilla_tip": "JC", ...}``. It is
whoever last placed, moved or accepted the point; a model's point nobody has
reviewed is ``"model"``. Sidecars saved before this existed carry none, and read
as not recorded rather than as anybody in particular.

Measurement error has an operator component, and a comparison between groups
labelled by different people carries it. Knowing who placed what is what lets a
repeatability study separate the two, and lets a shared dataset credit its
contributors.
"""

from __future__ import annotations

MODEL = "model"


def landmark_operators(sidecar: dict) -> dict[str, str]:
    """{landmark or outline name: operator} over both views."""
    out: dict[str, str] = {}
    for view in ("lateral", "frontal"):
        for name, op in ((sidecar.get(view) or {}).get("operators") or {}).items():
            if op:
                out[name] = str(op)
    return out


def people(sidecar: dict) -> list[str]:
    """The people who placed a sidecar's landmarks, most landmarks first."""
    count: dict[str, int] = {}
    for op in landmark_operators(sidecar).values():
        if op != MODEL:
            count[op] = count.get(op, 0) + 1
    return sorted(count, key=lambda o: (-count[o], o))


def summary(sidecar: dict) -> str:
    """One line for a table: "JC", "JC, AB", "JC; 3 unreviewed model points"."""
    ops = landmark_operators(sidecar)
    who = ", ".join(people(sidecar))
    unreviewed = sum(1 for o in ops.values() if o == MODEL)
    if unreviewed:
        who = "; ".join(filter(None, [who, f"{unreviewed} unreviewed model point"
                                            + ("s" if unreviewed > 1 else "")]))
    return who


def determined_by(sidecar: dict, names) -> str:
    """Who placed the given landmarks: the operators behind one trait."""
    ops = landmark_operators(sidecar)
    found = sorted({ops[n] for n in names if n in ops})
    return ", ".join(o for o in found if o != MODEL) + (
        ("; " if len(found) > 1 else "") + MODEL if MODEL in found else "")
