"""Alternative landmark schemes a study can be switched to.

caliPr's own scheme (``landmark_config``) exists to compute 33 traits, so its
landmarks are chosen for what each trait needs. Geometric morphometrics asks a
different question -- shape, after Procrustes -- and uses its own published point
sets, which no trait here is defined against.

A study picks a scheme in its ``schema.json``::

    {"scheme": "bgnn_2d"}

and the labeler then asks for that scheme's landmarks instead. The points are
saved in the sidecar like any other, and exported to TPS **in the scheme's own
order**, which is what makes row N mean the same thing in every specimen. No trait
is computed for such a study: every trait names caliPr landmarks, and inventing a
mapping between two schemes would be a claim about anatomy, not a conversion.

Adding a scheme is adding an entry here: a list of landmarks in the order they are
numbered in the protocol, each with what to click.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Landmark order is the protocol's numbering. Do not reorder: TPS identifies a
#: landmark by its row, so renumbering silently redefines every exported file.
BGNN_2D = {
    "title": "BGNN 2D body landmarks (23)",
    "source": "Bagheri et al., Fish-AIR / BGNN minnow protocol",
    "note": "Shape landmarks for geometric morphometrics. No caliPr trait is "
            "computed from them; export to TPS and analyse in geomorph.",
    "view": "lateral",
    "landmarks": [
        ("dentary_anterior", "Anteriormost point of dentary", "head",
         "The forward-most point of the lower jaw's dentary bone."),
        ("mouth_posterior", "Posteriormost point of mouth", "head",
         "The back corner of the mouth, where the jaws meet."),
        ("orbit_anterior", "Anterior limit of orbital", "eye",
         "The front edge of the bony eye socket, not of the pupil."),
        ("orbit_posterior", "Posterior limit of orbital", "eye",
         "The back edge of the bony eye socket."),
        ("orbit_dorsal", "Dorsal limit of orbital", "eye",
         "The top edge of the bony eye socket."),
        ("orbit_ventral", "Ventral limit of orbital", "eye",
         "The bottom edge of the bony eye socket."),
        ("orbit_center", "Center of orbit", "eye",
         "The middle of the eye socket, between the four limits."),
        ("supraoccipital_tip", "Posterodorsal tip of supraoccipital", "head",
         "The rear point of the supraoccipital crest, on the midline behind the skull roof."),
        ("dorsal_fin_base_anterior", "Anteriormost base of the first dorsal-fin ray",
         "dorsal fin", "Where the front edge of the first dorsal ray meets the body."),
        ("dorsal_fin_base_posterior", "Posteriormost base of the last dorsal-fin ray",
         "dorsal fin", "Where the back edge of the last dorsal ray meets the body."),
        ("caudal_crease_dorsal", "Dorsal-most point of caudal crease", "caudal",
         "Top of the crease left when the tail is flexed, at the caudal-fin base."),
        ("vertebral_centrum_posterior",
         "Compound vertebral centrum at posterior of vertebral column", "caudal",
         "The last vertebral centrum, at the end of the vertebral column."),
        ("caudal_crease_ventral", "Ventral-most point of caudal crease", "caudal",
         "Bottom of the crease at the caudal-fin base."),
        ("anal_fin_base_posterior", "Posteriormost point of the base of the anal fin",
         "anal fin", "Where the back of the anal-fin base meets the body."),
        ("anal_fin_base_anterior", "Anteriormost base of the first anal-fin ray",
         "anal fin", "Where the front edge of the first anal ray meets the body."),
        ("pelvic_fin_base_posterior", "Posteriormost point of the base of the pelvic fin",
         "pelvic fin", "Where the back of the pelvic-fin base meets the body."),
        ("pelvic_fin_base_anterior", "Anteriormost point of the base of the pelvic fin",
         "pelvic fin", "Where the front of the pelvic-fin base meets the body."),
        ("pectoral_fin_origin", "Pectoral fin origin (top point)", "pectoral fin",
         "The upper corner where the pectoral fin leaves the body."),
        ("pectoral_fin_inner", "Innermost point of pectoral fin (bottom point)",
         "pectoral fin", "The lower, inner corner of the pectoral-fin base."),
        ("cleithrum", "Cleithrum", "shoulder",
         "The cleithrum's posterior margin, the bony ridge behind the gill opening."),
        ("opercle_medial", "Medialmost point of opercle", "opercle",
         "The inner-most point of the gill cover's margin."),
        ("opercle_dorsal", "Dorsalmost point of opercle", "opercle",
         "The top point of the gill cover."),
        ("opercle_ventral", "Ventralmost point of opercle", "opercle",
         "The bottom point of the gill cover."),
    ],
}

#: Schemes defined in code: published protocols, and read-only for that reason.
#: Adding a point to one would make this lab's exports stop matching everyone
#: else's files under the same protocol's name, which is the one thing a named
#: scheme exists to prevent. Adding to one copies it instead -- see :func:`save`.
BUILTIN: dict[str, dict] = {"bgnn_2d": BGNN_2D}

#: Schemes someone made here, one JSON file each. They live beside the studies
#: rather than inside one, because a scheme is a protocol: the whole point of
#: row N meaning the same landmark is that it means it in every study that
#: follows it, not just the one it was first drawn up for.
USER_DIR_NAME = "schemes"
_user_root: Path | None = None


def use_data_root(root) -> None:
    """Where user-defined schemes live: ``<root>/schemes/``."""
    global _user_root
    _user_root = Path(root) / USER_DIR_NAME if root else None


def _user_dir() -> Path | None:
    return _user_root


def _valid(doc: dict) -> bool:
    # An empty landmark list is valid: a scheme starts empty when it is not
    # copied from one, and is filled a point at a time. Requiring at least one
    # made a new scheme vanish from the list the moment it was created.
    lms = doc.get("landmarks")
    return bool(doc.get("title") and isinstance(lms, list)
                and all(isinstance(k, (list, tuple)) and len(k) == 4 for k in lms))


def user_schemes() -> dict[str, dict]:
    """Every scheme defined here, by name. A malformed file is skipped, not raised:
    one bad file must not stop the labeler listing the rest."""
    out: dict[str, dict] = {}
    d = _user_dir()
    if d is None or not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        if _valid(doc):
            doc = dict(doc, landmarks=[tuple(k) for k in doc["landmarks"]], editable=True)
            out[path.stem] = doc
    return out


def all_schemes() -> dict[str, dict]:
    """Built-in and user-defined together. A user file never shadows a built-in."""
    out = dict(user_schemes())
    out.update(BUILTIN)
    return out


#: What a study is using when it names no scheme: caliPr's own, the one every
#: trait is defined against.
DEFAULT = "calipr"


def get(name: str | None) -> dict | None:
    """The named scheme, or None for caliPr's own."""
    if not name or name == DEFAULT:
        return None
    return all_schemes().get(name)


def listing() -> list[dict]:
    """Every scheme a study can be switched to, caliPr's own first."""
    out = [{"name": DEFAULT, "title": "caliPr (23 landmarks, 33 traits)",
            "landmarks": None, "editable": False,
            "note": "The scheme the traits, the trained model and the tutorial use."}]
    out += [{"name": k, "title": v["title"], "landmarks": len(v["landmarks"]),
             "note": v.get("note", ""), "source": v.get("source", ""),
             "editable": bool(v.get("editable"))}
            for k, v in sorted(all_schemes().items(),
                               key=lambda kv: (bool(kv[1].get("editable")), kv[0]))]
    return out


def save(name: str, doc: dict) -> Path:
    """Write a user-defined scheme. Only ever called for an editable one."""
    d = _user_dir()
    if d is None:
        raise RuntimeError("no data root set for user schemes")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    keep = {k: v for k, v in doc.items() if k != "editable"}
    keep["landmarks"] = [list(k) for k in doc["landmarks"]]
    path.write_text(json.dumps(keep, indent=2) + "\n")
    return path


def keypoints(name: str, view: str) -> list[dict]:
    """A scheme's landmarks for one view, in the protocol's order, numbered."""
    scheme = get(name)
    if not scheme or scheme.get("view", "lateral") != view:
        return []
    return [
        {"name": n, "label": f"{i}. {label}", "description": label,
         "hint": hint, "group": group, "excluded": False, "scheme": name, "number": i}
        for i, (n, label, group, hint) in enumerate(scheme["landmarks"], start=1)
    ]


def order(name: str, view: str = "lateral") -> tuple[str, ...]:
    """Landmark names in the order TPS rows must follow."""
    return tuple(k["name"] for k in keypoints(name, view))


def study_landmarks(dataset_dir, view: str = "lateral") -> tuple[tuple[str, ...], dict[str, str]]:
    """What a study collects, in the order its exports must use, and its names.

    One answer for every consumer -- the workbook, the TPS folder, the labeler --
    so a study's landmarks cannot mean one thing in one export and another
    elsewhere. Returns (order, labels): the landmark names in order, and what this
    study calls each one where that differs from the name.

    A study following another protocol exports that protocol's landmarks in its
    numbering. Otherwise it is caliPr's own, minus anything the study does not
    collect, plus any landmark it added for itself.
    """
    import json
    from pathlib import Path

    from .landmark_config import KEYPOINTS, View

    doc: dict = {}
    if dataset_dir is not None:
        path = Path(dataset_dir) / "schema.json"
        if path.is_file():
            try:
                doc = json.loads(path.read_text())
            except Exception:
                doc = {}

    scheme = doc.get("scheme")
    if get(scheme):
        kps = keypoints(scheme, view)
        return tuple(k["name"] for k in kps), {k["name"]: k["label"] for k in kps}

    drop = set(doc.get("exclude_keypoints") or [])
    want = View.LATERAL if view == "lateral" else View.FRONTAL
    order = [k.name for k in KEYPOINTS if k.view == want and k.name not in drop]
    labels = {}
    for k in doc.get("extra_keypoints") or []:
        if not isinstance(k, dict) or not k.get("name") or k["name"] in drop:
            continue
        if (k.get("view") or "lateral") == view:
            order.append(k["name"])
            labels[k["name"]] = k.get("label") or k["name"]
    labels.update(doc.get("labels") or {})
    return tuple(order), labels
