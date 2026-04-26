"""Annotation schema for brook trout morphometrics.

This module is the single source of truth for the ML model's output
contract: which polygons, which keypoints, and which named reference
lines an annotated fish image must carry in order to compute all 22
MorFishJ traits plus our extras. The same declarations feed:

  * the CVAT / Label Studio project config (so labelers see exactly the
    shapes the measurement engine expects, no more and no less),
  * the measurement engine's input validation,
  * the labeling guide that ships with the data README.

Three shape types live here:

1. **Polygons** — closed point sequences tracing fin and body outlines.
   Five of them: ``body_plus_caudal``, ``pectoral``, and the three
   extras ``dorsal``, ``pelvic``, ``anal``. The body+caudal polygon is
   split at line A (narrowest peduncle) by the measurement engine into
   "body" and "caudal" halves, matching MorFishJ's convention.

2. **Keypoints** — single ``(x, y)`` clicks for anatomical landmarks
   (eye cardinals, premaxilla tip, mouth corner, pectoral insertion,
   etc.) plus the endpoints for line A (narrowest peduncle) and the
   anchor points for lines H (caudal fin base) and I (posterior
   operculum). Two frontal keypoints (``mouth_left``, ``mouth_right``)
   live in the mirror view for mouth width.

3. **Reference lines** — named horizontal or vertical construction
   lines MorFishJ uses to define its measurements. Three are
   user-provided via keypoints: A (from the two peduncle-narrowest
   keypoints), H (vertical through ``caudal_base``), I (vertical
   through ``operculum_posterior``). Nine are derived by the engine at
   measurement time: B/C (dorsal/ventral body extrema), D/E (anterior
   body / posterior caudal extrema), F/G (dorsal/ventral caudal
   extrema), J/K (horizontal/vertical through eye centroid), L
   (vertical through ``pectoral_insertion_upper``).

Orientation convention
----------------------
All lateral photos must have the fish **facing left** (head toward the
smaller x values). If a photo has the head pointing right, rotate it
180° before labeling. This is the same convention MorFishJ's "Fish
facing left or right" step handles in its GUI; we enforce it as a
pre-labeling step instead.

Coordinate convention
---------------------
Pixel space, +x to the right, +y downward (standard OpenCV / image
processing). "Highest body edge" in the MorFishJ manual means visually
highest = smallest y; "lowest body edge" = largest y.

Trait source
------------
Every trait declares whether it comes from MorFishJ (22 of them) or
from our extras (8 of them, pending full anatomical definitions from
the Cornell reference sheets). Extras carry a reference-sheet number so
the export column order can group them alongside the existing manual
workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Literal


class View(str, Enum):
    """Which image context an annotation lives in."""

    LATERAL = "lateral"  # main left-side-up photo
    FRONTAL = "frontal"  # mirror-reflected head shot for mouth width


class TraitSource(str, Enum):
    MORFISHJ = "morfishj"  # defined in MorFishJ's main_traits page
    EXTRAS = "extras"  # Cornell lab measurements MorFishJ doesn't cover


class Unit(str, Enum):
    MM = "mm"
    MM2 = "mm^2"
    DEG = "deg"


# ---------------------------------------------------------------------------
# Shape definitions (what the annotator / model must produce)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Polygon:
    """A closed point sequence tracing an outline on the image."""

    name: str
    description: str
    view: View
    labeling_hint: str


@dataclass(frozen=True)
class Keypoint:
    """A single ``(x, y)`` click for an anatomical landmark."""

    name: str
    description: str
    view: View
    labeling_hint: str


@dataclass(frozen=True)
class UserReferenceLine:
    """A named reference line derived from user-placed keypoints.

    ``source_keypoints`` lists the keypoint(s) that anchor the line.
    For vertical/horizontal lines with a single keypoint, the line
    passes through that keypoint in the specified orientation. For
    lines with two keypoints (like A from the two peduncle-narrowest
    points), the line connects them.
    """

    name: str
    description: str
    orientation: Literal["horizontal", "vertical", "segment"]
    source_keypoints: tuple[str, ...]


@dataclass(frozen=True)
class DerivedReferenceLine:
    """A named reference line computed at measurement time from a polygon.

    ``extremum`` says which polygon vertex the line passes through.
    ``min_y`` = smallest y = visually highest in image space;
    ``max_y`` = largest y = visually lowest; ``min_x`` = leftmost
    (anterior in a left-facing fish); ``max_x`` = rightmost (posterior).
    """

    name: str
    description: str
    orientation: Literal["horizontal", "vertical"]
    source_polygon: str
    extremum: Literal["min_x", "max_x", "min_y", "max_y"]
    # If the source polygon is body_plus_caudal and this line is about the
    # caudal fin (F or G), we restrict the extremum search to the caudal
    # half of the polygon (x > line_A_x). "body" restricts to the body half.
    polygon_half: Literal["whole", "body", "caudal"] = "whole"


@dataclass(frozen=True)
class Trait:
    """A single numeric trait extracted from the annotation."""

    code: str  # "TL", "SL", "DFh", etc.
    label: str  # "Total Length"
    description: str
    unit: Unit
    view: View
    source: TraitSource
    required_polygons: tuple[str, ...] = ()
    required_keypoints: tuple[str, ...] = ()
    # Reference-sheet number for extras (None for MorFishJ traits,
    # which have no lab reference-sheet number).
    number: int | None = None


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------

POLYGONS: tuple[Polygon, ...] = (
    Polygon(
        "body_plus_caudal",
        "Closed outline of the body and caudal fin together, excluding the "
        "dorsal, pectoral, pelvic, and anal fins. Split by line A "
        "(narrowest peduncle) into a body half and a caudal half.",
        View.LATERAL,
        "Trace the silhouette starting at the snout tip, along the dorsal "
        "outline. When you reach the dorsal fin base, step across its "
        "insertion line (do NOT follow the fin margin) and continue along "
        "the back toward the caudal peduncle. Go around the caudal fin "
        "margin, then back along the ventral outline, stepping across the "
        "pelvic and anal fin bases the same way. The resulting polygon is "
        "the fish minus the four unpaired/paired fins.",
    ),
    Polygon(
        "pectoral",
        "Outline of the pectoral fin.",
        View.LATERAL,
        "Trace the pectoral fin margin starting at the upper insertion "
        "(pectoral_insertion_upper), around the distal margin, and back to "
        "the lower insertion.",
    ),
    Polygon(
        "dorsal",
        "Outline of the first (rayed) dorsal fin. Excludes the adipose fin.",
        View.LATERAL,
        "Trace the first dorsal fin margin. Do NOT include the adipose fin "
        "behind it.",
    ),
    Polygon(
        "pelvic",
        "Outline of the pelvic fin.",
        View.LATERAL,
        "Trace the pelvic fin margin.",
    ),
    Polygon(
        "anal",
        "Outline of the anal fin.",
        View.LATERAL,
        "Trace the anal fin margin.",
    ),
)


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------

KEYPOINTS: tuple[Keypoint, ...] = (
    # ---- Eye (represented as 4 cardinal orbit points) ----
    Keypoint(
        "eye_anterior",
        "Anterior edge of the orbit on its horizontal centerline.",
        View.LATERAL,
        "Click the forward-most point of the bony eye socket, at eye height.",
    ),
    Keypoint(
        "eye_posterior",
        "Posterior edge of the orbit on its horizontal centerline.",
        View.LATERAL,
        "Click the rearward-most point of the bony eye socket, at eye height.",
    ),
    Keypoint(
        "eye_dorsal",
        "Dorsal edge of the orbit on its vertical centerline.",
        View.LATERAL,
        "Click the top of the bony eye socket, directly above the eye center.",
    ),
    Keypoint(
        "eye_ventral",
        "Ventral edge of the orbit on its vertical centerline.",
        View.LATERAL,
        "Click the bottom of the bony eye socket, directly below the eye "
        "center.",
    ),

    # ---- Mouth ----
    Keypoint(
        "premaxilla_tip",
        "Anterior-most tip of the upper jaw (premaxilla).",
        View.LATERAL,
        "Click the forward-most point of the closed upper jaw. MorFishJ's "
        "EMd / EMa / Mo / Jl / Snl all anchor on this point.",
    ),
    Keypoint(
        "maxilla_mandible_intersection",
        "Intersection of the maxilla (upper) and mandible (lower jaw) at "
        "the posterior corner of the mouth. NOT where the flesh of the lips "
        "meets — the bony joint.",
        View.LATERAL,
        "Click the back corner of the mouth at the bony joint between the "
        "upper and lower jaw.",
    ),
    Keypoint(
        "lower_jaw_tip",
        "Anterior-most tip of the mandible (lower jaw).",
        View.LATERAL,
        "Click the forward-most point of the lower jaw. Coincides with "
        "premaxilla_tip if the mouth is fully closed.",
    ),

    # ---- Operculum (anchors reference line I) ----
    Keypoint(
        "operculum_posterior",
        "Posterior-most point of the bony operculum (gill cover). Anchors "
        "reference line I (vertical through this x).",
        View.LATERAL,
        "Click the trailing edge of the bony operculum, NOT the flexible "
        "branchiostegal membrane below it.",
    ),

    # ---- Pectoral fin anchors ----
    Keypoint(
        "pectoral_insertion_upper",
        "Upper (dorsal) insertion of the pectoral fin base on the body. "
        "Anchors reference line L (vertical through this x).",
        View.LATERAL,
        "Click the top corner where the pectoral fin meets the body behind "
        "the operculum.",
    ),
    Keypoint(
        "pectoral_ray_tip",
        "Distal tip of the longest pectoral fin ray. Used with "
        "pectoral_insertion_upper to compute PFl (pectoral fin length).",
        View.LATERAL,
        "Click the farthest-out point of the pectoral fin when held in "
        "natural position.",
    ),

    # ---- Caudal peduncle anchors (line A) and caudal base (line H) ----
    Keypoint(
        "peduncle_narrowest_dorsal",
        "Dorsal body outline at the narrowest point of the caudal peduncle. "
        "Paired with peduncle_narrowest_ventral to define line A and to "
        "measure CPd.",
        View.LATERAL,
        "Find the narrowest visible part of the peduncle and click the top "
        "silhouette there.",
    ),
    Keypoint(
        "peduncle_narrowest_ventral",
        "Ventral body outline at the narrowest point of the caudal peduncle, "
        "directly opposite peduncle_narrowest_dorsal.",
        View.LATERAL,
        "Click the bottom silhouette at the same x position.",
    ),
    Keypoint(
        "caudal_base",
        "Posterior end of the vertebral column at the hypural plate; the "
        "start of the caudal fin rays on midline. Anchors reference line H "
        "(vertical through this x) and is the posterior endpoint of SL.",
        View.LATERAL,
        "Click where the last scale meets the caudal fin rays, along the "
        "midline of the fish.",
    ),

    # ---- Extras: fin-height anchors (pending reference sheets) ----
    Keypoint(
        "dorsal_base_center",
        "Midpoint of the first dorsal fin base at the body outline. Used "
        "with dorsal_tip to compute dorsal fin height (extras).",
        View.LATERAL,
        "Click the center of the dorsal fin's base where it inserts on the "
        "back. Exact definition pending Cornell reference sheet.",
    ),
    Keypoint(
        "dorsal_tip",
        "Distal tip of the first dorsal fin (farthest from base). Used with "
        "dorsal_base_center to compute dorsal fin height.",
        View.LATERAL,
        "Click the outermost point of the dorsal fin margin.",
    ),
    Keypoint(
        "pelvic_base_center",
        "Midpoint of the pelvic fin base at the body outline.",
        View.LATERAL,
        "Click the center of the pelvic fin's base where it inserts on the "
        "ventral body. Exact definition pending Cornell reference sheet.",
    ),
    Keypoint(
        "pelvic_tip",
        "Distal tip of the pelvic fin.",
        View.LATERAL,
        "Click the outermost point of the pelvic fin margin.",
    ),
    Keypoint(
        "anal_base_center",
        "Midpoint of the anal fin base at the body outline.",
        View.LATERAL,
        "Click the center of the anal fin's base where it inserts on the "
        "ventral body. Exact definition pending Cornell reference sheet.",
    ),
    Keypoint(
        "anal_tip",
        "Distal tip of the anal fin.",
        View.LATERAL,
        "Click the outermost point of the anal fin margin.",
    ),

    # ---- Frontal (mirror) view ----
    Keypoint(
        "mouth_left",
        "Left corner of the open mouth in frontal view.",
        View.FRONTAL,
        "On the mirror image, click the left outer edge of the premaxilla.",
    ),
    Keypoint(
        "mouth_right",
        "Right corner of the open mouth in frontal view.",
        View.FRONTAL,
        "On the mirror image, click the right outer edge of the premaxilla.",
    ),
)


# ---------------------------------------------------------------------------
# Calibration keypoints (NOT part of the measurement schema)
# ---------------------------------------------------------------------------
#
# These sit in CVAT alongside the anatomical keypoints so a labeler can
# establish the pixel-to-millimeter scale by clicking the ends of a known
# ruler span as part of the same task. The measurement engine never reads
# them — they feed the calibration block only — so they deliberately live
# outside ``KEYPOINTS`` to keep trait validation, missing-landmark
# tracking, and the 19-lateral-keypoint invariant clean.
#
# The auto-ruler detector in :mod:`fish_morpho.ruler_calibration` fails on
# ~90% of our iDigBio pool (museum photos have too many ruler types,
# catalog cards, and bone trays for tick-period autocorrelation to work
# reliably), so two CVAT clicks per fish is the minimum-friction fallback
# that still produces trustworthy morphometrics.

CALIBRATION_KEYPOINTS: tuple[Keypoint, ...] = (
    Keypoint(
        "ruler_point_a",
        "First endpoint of a known-length ruler span in the lateral photo. "
        "Paired with ruler_point_b to synthesize a manual calibration "
        "(px_per_mm) at sidecar-conversion time.",
        View.LATERAL,
        "Click one end of a metric ruler or scale bar in the frame. The "
        "distance between this point and ruler_point_b must match the "
        "--known-mm value you pass to cvat_to_sidecar.py (or the known_mm "
        "you supply in a calibration JSON override).",
    ),
    Keypoint(
        "ruler_point_b",
        "Second endpoint of the known-length ruler span, separated from "
        "ruler_point_a by the value of --known-mm.",
        View.LATERAL,
        "Click the other end of the same metric span. For an 8cm checkered "
        "scale bar, that's the opposite corner (known_mm=80); for a "
        "standard mm ruler pick two tick labels you can hit precisely (e.g. "
        "the 0 and 100 mm marks, known_mm=100).",
    ),
)


# ---------------------------------------------------------------------------
# Reference lines
# ---------------------------------------------------------------------------
#
# Naming follows MorFishJ's main_traits page. User-provided lines are
# anchored by keypoints the annotator clicks; derived lines are computed
# at measurement time by the engine.

USER_REFERENCE_LINES: tuple[UserReferenceLine, ...] = (
    UserReferenceLine(
        "A",
        "Narrowest caudal peduncle line. Splits the body_plus_caudal "
        "polygon into body (for Bs) and caudal (for CFs) halves. CPd is "
        "the length of this segment.",
        "segment",
        ("peduncle_narrowest_dorsal", "peduncle_narrowest_ventral"),
    ),
    UserReferenceLine(
        "H",
        "Vertical line through the caudal fin base (end of vertebral "
        "column). Posterior endpoint of standard length (SL).",
        "vertical",
        ("caudal_base",),
    ),
    UserReferenceLine(
        "I",
        "Vertical line through the posterior operculum margin. Posterior "
        "endpoint of head length (Hl).",
        "vertical",
        ("operculum_posterior",),
    ),
)


DERIVED_REFERENCE_LINES: tuple[DerivedReferenceLine, ...] = (
    DerivedReferenceLine(
        "B",
        "Horizontal line at the highest body edge (smallest y). Defines "
        "the top of MBd.",
        "horizontal",
        "body_plus_caudal",
        "min_y",
        polygon_half="body",
    ),
    DerivedReferenceLine(
        "C",
        "Horizontal line at the lowest body edge (largest y). Defines the "
        "bottom of MBd and the reference floor for Eh, Mo, PFi.",
        "horizontal",
        "body_plus_caudal",
        "max_y",
        polygon_half="body",
    ),
    DerivedReferenceLine(
        "D",
        "Vertical line at the most anterior body tip (smallest x in "
        "left-facing convention). Anterior endpoint of TL, SL, Hl, Snl, AO.",
        "vertical",
        "body_plus_caudal",
        "min_x",
        polygon_half="whole",
    ),
    DerivedReferenceLine(
        "E",
        "Vertical line at the most posterior caudal tip (largest x). "
        "Posterior endpoint of TL.",
        "vertical",
        "body_plus_caudal",
        "max_x",
        polygon_half="whole",
    ),
    DerivedReferenceLine(
        "F",
        "Horizontal line at the highest caudal fin edge (smallest y of the "
        "caudal half of the body polygon). Top of CFd.",
        "horizontal",
        "body_plus_caudal",
        "min_y",
        polygon_half="caudal",
    ),
    DerivedReferenceLine(
        "G",
        "Horizontal line at the lowest caudal fin edge (largest y of the "
        "caudal half). Bottom of CFd.",
        "horizontal",
        "body_plus_caudal",
        "max_y",
        polygon_half="caudal",
    ),
    # J, K, L are keypoint-anchored but derived at measurement time from
    # keypoints rather than from the body polygon, so they sit in a
    # separate dispatch — see measurement_engine._eye_centroid and
    # _vertical_through_keypoint.
)


# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------
#
# The 22 MorFishJ traits plus 8 Cornell extras. Each trait declares which
# polygons and which keypoints it needs so the engine can report missing
# inputs cleanly instead of silently NaN-ing. The actual computation is
# in :mod:`fish_morpho.measurement_engine` keyed by ``code``.

TRAITS: tuple[Trait, ...] = (
    # ============================================================
    # MorFishJ traits (22)
    # ============================================================
    Trait(
        code="TL",
        label="Total Length",
        description="Distance from anterior body tip (D) to posterior "
        "caudal fin tip (E), along the horizontal.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
    ),
    Trait(
        code="SL",
        label="Standard Length",
        description="Distance from anterior body tip (D) to caudal fin "
        "base (H), along the horizontal.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=("caudal_base",),
    ),
    Trait(
        code="MBd",
        label="Maximum Body Depth",
        description="Vertical distance between the highest (B) and lowest "
        "(C) body edges.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=(
            "peduncle_narrowest_dorsal",
            "peduncle_narrowest_ventral",
        ),
    ),
    Trait(
        code="Hl",
        label="Head Length",
        description="Horizontal distance from anterior head (D) to "
        "posterior operculum margin (I).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=("operculum_posterior",),
    ),
    Trait(
        code="Hd",
        label="Head Depth",
        description="Head depth (B-to-C span) along reference K (vertical "
        "through eye centroid).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=(
            "eye_anterior",
            "eye_posterior",
            "eye_dorsal",
            "eye_ventral",
        ),
    ),
    Trait(
        code="Ed",
        label="Eye Diameter",
        description="Horizontal orbit diameter (distance from eye_anterior "
        "to eye_posterior).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_keypoints=("eye_anterior", "eye_posterior"),
    ),
    Trait(
        code="Eh",
        label="Eye Position",
        description="Vertical distance from eye centroid to body bottom (C).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=(
            "eye_anterior",
            "eye_posterior",
            "eye_dorsal",
            "eye_ventral",
            "peduncle_narrowest_dorsal",
            "peduncle_narrowest_ventral",
        ),
    ),
    Trait(
        code="Snl",
        label="Snout Length",
        description="Horizontal distance from the anterior tip of the "
        "snout (premaxilla_tip) to the anterior orbit margin "
        "(eye_anterior). On a closed-mouth specimen this coincides with "
        "AO, because premaxilla_tip is the body polygon's anterior "
        "extremum.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_keypoints=("premaxilla_tip", "eye_anterior"),
    ),
    Trait(
        code="POC",
        label="Posterior of Orbit Centroid",
        description="Horizontal distance from eye centroid to posterior "
        "operculum margin (I).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_keypoints=(
            "eye_anterior",
            "eye_posterior",
            "eye_dorsal",
            "eye_ventral",
            "operculum_posterior",
        ),
    ),
    Trait(
        code="AO",
        label="Anterior of Orbit",
        description="Horizontal distance from anterior orbit margin "
        "(eye_anterior) to anterior body tip (D).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=("eye_anterior",),
    ),
    Trait(
        code="EMd",
        label="Eye-Mouth Distance",
        description="Euclidean distance from eye centroid to premaxilla tip.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_keypoints=(
            "eye_anterior",
            "eye_posterior",
            "eye_dorsal",
            "eye_ventral",
            "premaxilla_tip",
        ),
    ),
    Trait(
        code="EMa",
        label="Eye-Mouth Angle",
        description="Angle (degrees) between EMd and the horizontal line "
        "through the premaxilla tip. Positive when the eye is above the "
        "mouth in image space.",
        unit=Unit.DEG,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_keypoints=(
            "eye_anterior",
            "eye_posterior",
            "eye_dorsal",
            "eye_ventral",
            "premaxilla_tip",
        ),
    ),
    Trait(
        code="Mo",
        label="Oral Gape Position",
        description="Vertical distance from premaxilla tip to body bottom "
        "(C).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=(
            "premaxilla_tip",
            "peduncle_narrowest_dorsal",
            "peduncle_narrowest_ventral",
        ),
    ),
    Trait(
        code="Jl",
        label="Maxillary Jaw Length",
        description="Distance from premaxilla tip to maxilla-mandible "
        "intersection.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_keypoints=("premaxilla_tip", "maxilla_mandible_intersection"),
    ),
    Trait(
        code="Bs",
        label="Body Surface Area",
        description="Area of the body portion of body_plus_caudal "
        "(everything anterior to line A).",
        unit=Unit.MM2,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=(
            "peduncle_narrowest_dorsal",
            "peduncle_narrowest_ventral",
        ),
    ),
    Trait(
        code="CPd",
        label="Caudal Peduncle Depth",
        description="Distance between peduncle_narrowest_dorsal and "
        "peduncle_narrowest_ventral (the endpoints of line A).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_keypoints=(
            "peduncle_narrowest_dorsal",
            "peduncle_narrowest_ventral",
        ),
    ),
    Trait(
        code="CFd",
        label="Caudal Fin Depth",
        description="Vertical distance between the highest (F) and lowest "
        "(G) edges of the caudal half of body_plus_caudal.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=(
            "peduncle_narrowest_dorsal",
            "peduncle_narrowest_ventral",
        ),
    ),
    Trait(
        code="CFs",
        label="Caudal Fin Surface Area",
        description="Area of the caudal portion of body_plus_caudal "
        "(everything posterior to line A).",
        unit=Unit.MM2,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=(
            "peduncle_narrowest_dorsal",
            "peduncle_narrowest_ventral",
        ),
    ),
    Trait(
        code="PFs",
        label="Pectoral Fin Surface Area",
        description="Area of the pectoral fin polygon.",
        unit=Unit.MM2,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("pectoral",),
    ),
    Trait(
        code="PFl",
        label="Pectoral Fin Length",
        description="Distance from pectoral_insertion_upper to "
        "pectoral_ray_tip.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_keypoints=("pectoral_insertion_upper", "pectoral_ray_tip"),
    ),
    Trait(
        code="PFi",
        label="Pectoral Fin Position",
        description="Vertical distance from pectoral_insertion_upper to "
        "body bottom (C).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=(
            "pectoral_insertion_upper",
            "peduncle_narrowest_dorsal",
            "peduncle_narrowest_ventral",
        ),
    ),
    Trait(
        code="PFb",
        label="Body Depth at Pectoral Fin",
        description="Body depth along reference L (vertical through "
        "pectoral_insertion_upper).",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.MORFISHJ,
        required_polygons=("body_plus_caudal",),
        required_keypoints=(
            "pectoral_insertion_upper",
            "peduncle_narrowest_dorsal",
            "peduncle_narrowest_ventral",
        ),
    ),

    # ============================================================
    # Cornell extras (8) — reference-sheet numbers are placeholders
    # until the actual sheets are provided.
    # ============================================================
    Trait(
        code="LJl",
        label="Lower Jaw Length",
        description="Distance from lower_jaw_tip to "
        "maxilla_mandible_intersection.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.EXTRAS,
        required_keypoints=("lower_jaw_tip", "maxilla_mandible_intersection"),
        number=32,
    ),
    Trait(
        code="DFh",
        label="Dorsal Fin Height",
        description="Distance from dorsal_base_center to dorsal_tip. Exact "
        "base-point definition pending Cornell reference sheet.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.EXTRAS,
        required_keypoints=("dorsal_base_center", "dorsal_tip"),
        number=8,
    ),
    Trait(
        code="DFs",
        label="Dorsal Fin Surface Area",
        description="Area of the dorsal fin polygon.",
        unit=Unit.MM2,
        view=View.LATERAL,
        source=TraitSource.EXTRAS,
        required_polygons=("dorsal",),
        number=18,
    ),
    Trait(
        code="PlFl",
        label="Pelvic Fin Length",
        description="Distance from pelvic_base_center to pelvic_tip. Exact "
        "base-point definition pending Cornell reference sheet.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.EXTRAS,
        required_keypoints=("pelvic_base_center", "pelvic_tip"),
        number=12,
    ),
    Trait(
        code="PlFs",
        label="Pelvic Fin Surface Area",
        description="Area of the pelvic fin polygon.",
        unit=Unit.MM2,
        view=View.LATERAL,
        source=TraitSource.EXTRAS,
        required_polygons=("pelvic",),
        number=20,
    ),
    Trait(
        code="AFh",
        label="Anal Fin Height",
        description="Distance from anal_base_center to anal_tip. Exact "
        "base-point definition pending Cornell reference sheet.",
        unit=Unit.MM,
        view=View.LATERAL,
        source=TraitSource.EXTRAS,
        required_keypoints=("anal_base_center", "anal_tip"),
        number=13,
    ),
    Trait(
        code="AFs",
        label="Anal Fin Surface Area",
        description="Area of the anal fin polygon.",
        unit=Unit.MM2,
        view=View.LATERAL,
        source=TraitSource.EXTRAS,
        required_polygons=("anal",),
        number=21,
    ),
    Trait(
        code="MW",
        label="Mouth Width",
        description="Distance from mouth_left to mouth_right in the mirror "
        "(frontal) view.",
        unit=Unit.MM,
        view=View.FRONTAL,
        source=TraitSource.EXTRAS,
        required_keypoints=("mouth_left", "mouth_right"),
        number=25,
    ),
)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def polygon_names(view: View | None = None) -> tuple[str, ...]:
    if view is None:
        return tuple(p.name for p in POLYGONS)
    return tuple(p.name for p in POLYGONS if p.view == view)


def keypoint_names(view: View | None = None) -> tuple[str, ...]:
    if view is None:
        return tuple(k.name for k in KEYPOINTS)
    return tuple(k.name for k in KEYPOINTS if k.view == view)


def calibration_keypoint_names(view: View | None = None) -> tuple[str, ...]:
    """Names of the CVAT-native calibration keypoints (outside the measurement schema)."""
    if view is None:
        return tuple(k.name for k in CALIBRATION_KEYPOINTS)
    return tuple(k.name for k in CALIBRATION_KEYPOINTS if k.view == view)


def polygon_by_name(name: str) -> Polygon:
    for p in POLYGONS:
        if p.name == name:
            return p
    raise KeyError(f"Unknown polygon: {name!r}")


def keypoint_by_name(name: str) -> Keypoint:
    for k in KEYPOINTS:
        if k.name == name:
            return k
    raise KeyError(f"Unknown keypoint: {name!r}")


def calibration_keypoint_by_name(name: str) -> Keypoint:
    for k in CALIBRATION_KEYPOINTS:
        if k.name == name:
            return k
    raise KeyError(f"Unknown calibration keypoint: {name!r}")


def trait_by_code(code: str) -> Trait:
    for t in TRAITS:
        if t.code == code:
            return t
    raise KeyError(f"Unknown trait code: {code!r}")


def traits_by_source(source: TraitSource) -> tuple[Trait, ...]:
    return tuple(t for t in TRAITS if t.source == source)


def trait_column_order() -> list[str]:
    """Stable column ordering for export: MorFishJ traits first (by code
    alphabetical within the 22), then extras (by reference-sheet number)."""
    morfishj = sorted(traits_by_source(TraitSource.MORFISHJ), key=lambda t: t.code)
    extras = sorted(
        traits_by_source(TraitSource.EXTRAS),
        key=lambda t: (t.number if t.number is not None else 10**6, t.code),
    )
    return [t.code for t in (*morfishj, *extras)]


def trait_labels() -> dict[str, str]:
    """Map trait code → ``"CODE — Label (unit)"`` for xlsx headers."""
    return {t.code: f"{t.code} — {t.label} ({t.unit.value})" for t in TRAITS}


# ---------------------------------------------------------------------------
# Schema validation (runs at import time)
# ---------------------------------------------------------------------------


def _flatten(items: Iterable[str]) -> set[str]:
    return set(items)


def validate_schema() -> None:
    """Raise ``ValueError`` if the schema is internally inconsistent.

    Checks:
      * Polygon names, keypoint names, and trait codes are each unique.
      * Every trait's required_polygons / required_keypoints reference
        shapes that actually exist.
      * Every reference line's ``source_keypoints`` / ``source_polygon``
        references exist.
      * Every trait's view matches the view of each required shape
        (e.g. a LATERAL trait can't depend on FRONTAL keypoints).
      * Extras traits carry a reference-sheet number.
    """
    poly_names = {p.name for p in POLYGONS}
    if len(poly_names) != len(POLYGONS):
        raise ValueError("Duplicate polygon names in POLYGONS")

    kp_names = {k.name for k in KEYPOINTS}
    if len(kp_names) != len(KEYPOINTS):
        raise ValueError("Duplicate keypoint names in KEYPOINTS")

    cal_kp_names = {k.name for k in CALIBRATION_KEYPOINTS}
    if len(cal_kp_names) != len(CALIBRATION_KEYPOINTS):
        raise ValueError("Duplicate keypoint names in CALIBRATION_KEYPOINTS")
    if cal_kp_names & kp_names:
        raise ValueError(
            "CALIBRATION_KEYPOINTS must be disjoint from KEYPOINTS: "
            f"overlap={sorted(cal_kp_names & kp_names)}"
        )

    trait_codes = {t.code for t in TRAITS}
    if len(trait_codes) != len(TRAITS):
        raise ValueError("Duplicate trait codes in TRAITS")

    for t in TRAITS:
        for p in t.required_polygons:
            if p not in poly_names:
                raise ValueError(
                    f"Trait {t.code!r} requires unknown polygon {p!r}"
                )
            if polygon_by_name(p).view != t.view:
                raise ValueError(
                    f"Trait {t.code!r} (view={t.view.value}) requires "
                    f"polygon {p!r} from view {polygon_by_name(p).view.value}"
                )
        for k in t.required_keypoints:
            if k not in kp_names:
                raise ValueError(
                    f"Trait {t.code!r} requires unknown keypoint {k!r}"
                )
            if keypoint_by_name(k).view != t.view:
                raise ValueError(
                    f"Trait {t.code!r} (view={t.view.value}) requires "
                    f"keypoint {k!r} from view {keypoint_by_name(k).view.value}"
                )
        if t.source == TraitSource.EXTRAS and t.number is None:
            raise ValueError(
                f"Extras trait {t.code!r} must carry a reference-sheet number"
            )

    for line in USER_REFERENCE_LINES:
        for kp in line.source_keypoints:
            if kp not in kp_names:
                raise ValueError(
                    f"User reference line {line.name!r} references unknown "
                    f"keypoint {kp!r}"
                )

    for line in DERIVED_REFERENCE_LINES:
        if line.source_polygon not in poly_names:
            raise ValueError(
                f"Derived reference line {line.name!r} references unknown "
                f"polygon {line.source_polygon!r}"
            )


# Run validation at import time — fail loud if the schema has drifted.
validate_schema()
