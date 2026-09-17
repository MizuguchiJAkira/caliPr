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

SCHEMES: dict[str, dict] = {"bgnn_2d": BGNN_2D}

#: What a study is using when it names no scheme: caliPr's own, the one every
#: trait is defined against.
DEFAULT = "calipr"


def get(name: str | None) -> dict | None:
    """The named scheme, or None for caliPr's own."""
    if not name or name == DEFAULT:
        return None
    return SCHEMES.get(name)


def listing() -> list[dict]:
    """Every scheme a study can be switched to, caliPr's own first."""
    out = [{"name": DEFAULT, "title": "caliPr (23 landmarks, 33 traits)",
            "landmarks": None,
            "note": "The scheme the traits, the trained model and the tutorial use."}]
    out += [{"name": k, "title": v["title"], "landmarks": len(v["landmarks"]),
             "note": v["note"], "source": v.get("source", "")}
            for k, v in SCHEMES.items()]
    return out


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
