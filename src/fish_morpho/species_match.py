"""How typical one fish's shape is of each species in a reference set.

An aid to identification, not a verdict. For every species in the reference --
fish a person has landmarked and a museum has identified -- the score is the share
of that species' fish expected to look *less* like a typical member than this one
does: 1.00 for a fish at the centre of the species, falling towards 0.00 as it
moves to where that species' fish are seldom found. Each species is scored on its
own, so the scores do not add up to one. Two look-alikes can both score high,
which says the shape cannot tell them apart; every species can score low, which
says the fish is like none of them -- a species missing from the reference, a
damaged specimen, or landmarks that went wrong.

**Size.** Shiner shape changes as the fish grows, so a fish is compared with each
species at its own standard length when both it and the reference have a scale:
the expected shape of a species at that length, from a growth slope shared by the
species (one slope fitted to every species, since a handful of fish each cannot
support one each). A length outside the species' reference range is reported, not
quietly extrapolated from.

**Landmarks.** Only the points the model places reliably and that nearly every
reference fish has (``RELIABLE``); a reference fish missing one or two of them is
kept, the gaps estimated from the average fish (never its own species'). The
caudal-crease points are never shape: where the crease sits along the body follows
how the tail was flexed, so they give peduncle depth and nothing else.

**Misplaced landmarks.** A point far from where the rest of the fish puts it --
a fin origin set on a folded fin, a jaw tip on an open mouth -- makes a fish look
like no species at all. ``identify`` finds such points against the average fish
and scores the fish without them, saying which it left out.

**Length.** A fish longer than a species gets is unlikely to be it, whatever its
shape. Each species' longest recorded length (``species_sizes.json``: FishBase and
the field guides it cites, converted to SL) cuts its score as a fish nears and
passes it (``size_factor``); below most of that length, length counts for nothing.

**Drainage.** Where the lot was caught narrows the field: ``with_drainage`` ranks
the species museums have recorded from its drainage before those they have not,
and says when the shape fits one of the others better. The scores stay the
shape's -- record counts measure collecting, not how common a fish is, so they
weigh nothing.

**Measurements.** Beside shape, the ones the keys use (``TRAITS``): snout length
over orbit diameter, peduncle depth over SL, greatest body depth over peduncle
depth, and how far the dorsal fin starts behind the pelvic fins -- each at the
fish's own size, each only where every species has reference fish that have it.

**Characters from the photograph** (``PHOTO_TRAITS``, ``photo_characters``): how dark
the caudal spot, the snout and the side stripe are against the lower flank, and how
upright the scale margins are along the front of the side. Read from the photograph
at places the landmarks set, so a reference fish has them only where they were
read from its photograph (``photo_characters.json`` in the study, by
``scripts/photo_characters.py``) and a fish being identified only when its own
photograph is given. They enter exactly as the measurements do.

The statistic is a Mahalanobis distance in the first few principal axes of
within-species shape variation plus the measurements, its tail probability read
from chi-square, and widened for the uncertainty in each species' mean
(``n / (n + 1)``). With a few reference fish per species the scores are rough;
they firm up as the reference grows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fish_morpho import photo_characters

#: Landmarks the cross-species model places well on shiners (run 4, median error
#: under 1 % of SL on held-out shiners), under the BGNN 2D names.
RELIABLE = (
    "dentary_anterior", "premaxilla_tip", "mouth_posterior",
    "orbit_anterior", "orbit_posterior", "orbit_dorsal", "orbit_ventral",
    "dorsal_fin_base_anterior", "dorsal_fin_base_posterior", "pectoral_fin_origin",
    "anal_fin_base_anterior", "anal_fin_base_posterior", "vertebral_centrum_posterior",
)
#: A landmark enters the comparison only if this share of reference fish have it.
MIN_SHARE = 0.8
#: A reference fish missing at most this many of the landmarks compared is kept,
#: the missing ones estimated from the average fish.
MAX_GAPS = 2
#: Fewer landmarks than this and there is no shape to compare.
MIN_LANDMARKS = 6
#: A species with fewer reference fish than this is not scored.
MIN_FISH = 3
#: Within-species shape axes the distance is measured in. Few, and the directions
#: that tell species apart -- often ones they vary little along -- are lost; many,
#: and the scores stop meaning what they say. On the 69 hand-labelled shiners
#: (leave-one-out), 8 gave the true species in the top two for 81 % with its own
#: score under 0.25 for 25 % of fish, as a typicality should; 4 gave 74 %, 12 87 %
#: but 32 %.
AXES = 8
#: How far, in shape units (a fish of centroid size 1), the landmarks of the fish
#: being identified are expected to be off -- the model's error, since its points
#: are placed by the model and the reference's by hand. Left out, a few pixels of
#: model error read as a fish unlike any species.
QUERY_NOISE = 0.0
#: A landmark this many times the usual distance from where the rest of the fish
#: puts it is taken to be misplaced, and the fish is scored without it. Measured
#: with the landmark left out of the fit, against the same distance on the reference
#: fish. On the 69 hand-labelled shiners the worst landmark of a fish was under 2.6
#: for 95 % and over 4 for one -- an anal fin whose two ends had been swapped; on
#: batch photographs, dorsal-fin origins put on the nape by the model came out 8-9.
MISPLACED = 4.0
#: At most this many are left out. A fish with more wrong is one to look at, not to
#: score around.
MAX_LEFT_OUT = 2
#: Below this for every species, the fish is like none of them. With hand-placed
#: landmarks that was 4 of 69 reference fish (left out in turn); on batch
#: photographs it was most fish, every one with points the model had misplaced.
NO_MATCH = 0.05
#: Species within this of the best score are tied with it, if they score NO_MATCH
#: or more: the scores do not tell them apart. Leaving out any one reference fish of a species moved its score by up
#: to 0.10 (the median over 32 scores of fish on four batch photographs), so a
#: smaller difference is the reference set, not the fish.
TIE = 0.10
#: Set aside from the shape of a fish whose mouth is open. Prying a preserved fish's
#: mouth open lifts the whole head -- on five gaping Luxilus cornutus the snout sat
#: about 10 degrees higher against the body than on any closed-mouth reference fish
#: -- so the jaw and eye points are where no reference fish has them. What is left
#: is the body: the dorsal and anal fin bases, the pectoral origin, the caudal base.
MOUTH_OPEN = ("dentary_anterior", "premaxilla_tip", "mouth_posterior",
              "orbit_anterior", "orbit_posterior", "orbit_dorsal", "orbit_ventral")
#: Length counts against a species only near the longest it is recorded to reach:
#: nothing below this share of the record; from there the score is cut, to
#: ``SIZE_AT_RECORD`` of what shape gave it at the record itself, and by a further
#: factor of e for every ``SIZE_BEYOND`` longer than that. A record is the one
#: biggest fish anyone wrote down, so most adults stop well short of it.
SIZE_FREE = 0.85
SIZE_AT_RECORD = 0.1
SIZE_BEYOND = 0.05
#: Snout to caudal base: the length a fish is scaled by.
SNOUT = ("premaxilla_tip", "dentary_anterior")
TAIL = "vertebral_centrum_posterior"


@dataclass
class RefFish:
    fid: str
    species: str
    kp: dict
    sl_mm: float | None = None
    chars: dict = field(default_factory=dict)   # photo_characters, where read


def standard_length_px(kp: dict) -> float | None:
    snout = next((kp[n] for n in SNOUT if kp.get(n)), None)
    tail = kp.get(TAIL)
    if snout is None or tail is None:
        return None
    return math.dist(snout[:2], tail[:2])


def px_per_mm(calibration: dict | None) -> float | None:
    """A ruler's two points and the millimetres between them, as px per mm."""
    c = calibration or {}
    try:
        if c.get("mode", "manual") == "manual" and float(c["known_mm"]) > 0:
            return math.dist(c["point_a"], c["point_b"]) / float(c["known_mm"])
    except (KeyError, TypeError, ValueError):
        pass
    return None


def load_reference(study_dir: Path, species_of, terminal_mouth=()) -> list[RefFish]:
    """The study's hand-labelled fish, each with the species ``species_of`` gives
    for its id ("" to leave a fish out) and its standard length where it has a
    ruler.

    ``terminal_mouth``: groups (by the start of their name, e.g. "Luxilus") whose
    mouth is terminal, so that the tip of the upper jaw and the tip of the lower
    are one point: a fish of these without a premaxilla tip takes its dentary tip.
    """
    terminal = tuple(terminal_mouth or ())
    out = []
    for f in sorted((Path(study_dir) / "sidecars").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        fid = d.get("fish_id") or f.stem
        sp = species_of(fid)
        lat = d.get("lateral") or {}
        kp = {k: tuple(v[:2]) for k, v in (lat.get("keypoints") or {}).items() if v}
        if not sp or len(kp) < MIN_LANDMARKS:
            continue
        chars = read_characters(study_dir, fid, kp)     # read with the landmarks as saved
        if terminal and sp.startswith(terminal) and "premaxilla_tip" not in kp and "dentary_anterior" in kp:
            kp["premaxilla_tip"] = kp["dentary_anterior"]
        sl, ppm = standard_length_px(kp), px_per_mm(lat.get("calibration"))
        out.append(RefFish(fid, sp, kp, sl / ppm if sl and ppm else None, chars))
    return out


#: Where a study keeps the characters read from its fish's photographs.
CHARACTERS_FILE = "photo_characters.json"
_CHARACTERS: dict = {}


def read_characters(study_dir: Path, fid: str, kp: dict) -> dict:
    """The characters read from ``fid``'s photograph -- if they were read with the
    landmarks it has now. Read with other landmarks, they were read in other places
    and are left out until read again."""
    path = Path(study_dir) / CHARACTERS_FILE
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return {}
    if _CHARACTERS.get(path, (None,))[0] != stamp:
        try:
            _CHARACTERS[path] = (stamp, json.loads(path.read_text()).get("fish") or {})
        except (OSError, ValueError):
            _CHARACTERS[path] = (stamp, {})
    e = _CHARACTERS[path][1].get(fid) or {}
    return dict(e.get("characters") or {}) if e.get("landmarks") == photo_characters.landmarks_key(kp) else {}


# ------------------------------------------------------------------- geometry

def _centre_scale(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    return x / np.sqrt((x ** 2).sum())


def _rotate_onto(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``x`` rotated (never reflected) to fit ``ref`` best; both centred, unit size."""
    u, _, vt = np.linalg.svd(x.T @ ref)
    d = np.sign(np.linalg.det(u @ vt))
    return x @ (u @ np.diag([1.0, d]) @ vt)


def gpa(shapes: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Procrustes: (n, k, 2) -> aligned shapes and their consensus."""
    X = np.array([_centre_scale(s) for s in shapes])
    ref = X[0]
    for _ in range(iters):
        X = np.array([_rotate_onto(s, ref) for s in X])
        new = _centre_scale(X.mean(axis=0))
        if np.allclose(new, ref, atol=1e-12):
            break
        ref = new
    return X, ref


def _tps(src: np.ndarray, dst: np.ndarray):
    """The thin-plate spline taking ``src`` onto ``dst``, as a function of points."""
    n = len(src)
    r2 = ((src[:, None] - src[None]) ** 2).sum(-1)
    K = np.where(r2 > 0, r2 * np.log(r2 + 1e-300), 0.0)
    P = np.hstack([np.ones((n, 1)), src])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
    W = np.linalg.solve(L + 1e-9 * np.eye(n + 3), np.vstack([dst, np.zeros((3, 2))]))

    def warp(q):
        d2 = ((q[:, None] - src[None]) ** 2).sum(-1)
        U = np.where(d2 > 0, d2 * np.log(d2 + 1e-300), 0.0)
        return U @ W[:n] + np.hstack([np.ones((len(q), 1)), q]) @ W[n:]
    return warp


def _fit_partial(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``x`` (missing landmarks NaN) moved onto ``ref`` by the landmarks it has."""
    have = ~np.isnan(x[:, 0])
    xs, rs = x[have], ref[have]
    xc, rc = xs.mean(axis=0), rs.mean(axis=0)
    x0, r0 = xs - xc, rs - rc
    s = math.sqrt(float((r0 ** 2).sum()) / max(float((x0 ** 2).sum()), 1e-12))
    u, _, vt = np.linalg.svd(x0.T @ r0)
    d = np.sign(np.linalg.det(u @ vt))
    return (x - xc) * s @ (u @ np.diag([1.0, d]) @ vt) + rc


def _fill_gaps(shapes: list[np.ndarray], consensus: np.ndarray) -> np.ndarray:
    """Each fish's missing landmarks (NaN) estimated by warping the consensus onto
    the landmarks it has -- the average fish, not any species', so the estimate
    cannot lean a fish towards its own species."""
    out = []
    for s in shapes:
        have = ~np.isnan(s[:, 0])
        if not have.all():
            s = s.copy()
            s[~have] = _tps(consensus[have], s[have])(consensus[~have])
        out.append(s)
    return np.array(out)


def chi2_sf(x: float, k: int) -> float:
    """P(chi-square with ``k`` degrees of freedom > x), for whole ``k``."""
    h = max(x, 0.0) / 2
    if k % 2 == 0:
        term, total = 1.0, 1.0
        for i in range(1, k // 2):
            term *= h / i
            total += term
        return float(math.exp(-h) * total)
    total = math.erfc(math.sqrt(h))
    term = math.sqrt(h) / math.gamma(1.5)
    for i in range(1, (k + 1) // 2):
        total += math.exp(-h) * term
        term *= h / (i + 0.5)
    return float(min(1.0, total))


def _snout_over_orbit(kp: dict) -> float | None:
    """Snout length (premaxilla tip to the front of the orbit) over orbit diameter."""
    if not all(kp.get(n) for n in ("premaxilla_tip", "orbit_anterior", "orbit_posterior")):
        return None
    eye = math.dist(kp["orbit_anterior"][:2], kp["orbit_posterior"][:2])
    return math.dist(kp["premaxilla_tip"][:2], kp["orbit_anterior"][:2]) / eye if eye > 0 else None


def _peduncle_depth(kp: dict) -> float | None:
    """Caudal-peduncle depth -- between the two caudal-crease points, placed at the
    least depth of the peduncle and good for nothing else -- over snout-to-caudal-base
    length."""
    sl = standard_length_px(kp)
    if not sl or not all(kp.get(n) for n in ("caudal_crease_dorsal", "caudal_crease_ventral")):
        return None
    return math.dist(kp["caudal_crease_dorsal"][:2], kp["caudal_crease_ventral"][:2]) / sl


def _depth_over_peduncle(kp: dict) -> float | None:
    """Greatest body depth over least caudal-peduncle depth: deep-bodied Luxilus
    against the slimmer Notropis they share shape with otherwise."""
    if not all(kp.get(n) for n in ("body_depth_dorsal", "body_depth_ventral",
                                   "caudal_crease_dorsal", "caudal_crease_ventral")):
        return None
    ped = math.dist(kp["caudal_crease_dorsal"][:2], kp["caudal_crease_ventral"][:2])
    return math.dist(kp["body_depth_dorsal"][:2], kp["body_depth_ventral"][:2]) / ped if ped > 0 else None


#: Measurements the keys use, compared beside shape. One is used only where the fish
#: has it and every species has ``MIN_FISH`` reference fish with it.
TRAITS = {"snout length / orbit": _snout_over_orbit, "peduncle depth / SL": _peduncle_depth,
          "body depth / peduncle depth": _depth_over_peduncle,
          "dorsal origin behind pelvic / SL": photo_characters.dorsal_behind_pelvic}
#: Characters read from the photograph (``photo_characters``), used the same way.
PHOTO_TRAITS = photo_characters.NAMES
#: The landmarks each measurement's shape comes from: one is not used when any of
#: them is set aside. (Peduncle depth is over SL, which a raised head barely moves.)
#: A photograph character is read where landmarks put it, so it goes with them.
TRAIT_USES = {"snout length / orbit": {"premaxilla_tip", "orbit_anterior", "orbit_posterior"},
              "peduncle depth / SL": {"caudal_crease_dorsal", "caudal_crease_ventral"},
              "body depth / peduncle depth": {"body_depth_dorsal", "body_depth_ventral",
                                              "caudal_crease_dorsal", "caudal_crease_ventral"},
              "dorsal origin behind pelvic / SL": {"dorsal_fin_base_anterior", "pelvic_fin_base_anterior"},
              "caudal spot": {"vertebral_centrum_posterior"},
              "snout": {"premaxilla_tip", "orbit_anterior"},
              "side stripe": {"vertebral_centrum_posterior", "pelvic_fin_base_anterior"},
              "upright scale margins": {"dorsal_fin_base_anterior"}}


# ---------------------------------------------------------------------- match

def _core(q: dict, ref: list[RefFish], set_aside=()) -> list[str]:
    """The landmarks compared: reliable ones the fish has and nearly every
    reference fish has too, less any ``set_aside``."""
    share = {n: sum(n in r.kp for r in ref) / max(len(ref), 1) for n in RELIABLE}
    return [n for n in RELIABLE if n in q and share[n] >= MIN_SHARE and n not in set_aside]


def _aligned_reference(core: list[str], ref: list[RefFish]):
    """The reference fish compared on ``core`` and their shapes: (fish, raw shapes
    with gaps as NaN, aligned shapes with the gaps filled, consensus), or an error."""
    # A reference fish missing a landmark or two is kept, the gaps estimated.
    fish = [r for r in ref if sum(n not in r.kp for n in core) <= MAX_GAPS]
    counts: dict[str, int] = {}
    for r in fish:
        counts[r.species] = counts.get(r.species, 0) + 1
    fish = [r for r in fish if counts[r.species] >= MIN_FISH]
    if not fish:
        return "no species has enough reference fish yet"
    X = [np.array([r.kp.get(n, (np.nan, np.nan)) for n in core], float) for r in fish]
    for x in X:
        x[:, 1] *= -1                                  # image y runs down
    whole = [x for x in X if not np.isnan(x).any()]
    if len(whole) < MIN_FISH:
        return "too few reference fish have every landmark compared"
    _, consensus0 = gpa(np.array(whole))
    aligned, consensus = gpa(_fill_gaps([_fit_partial(x, consensus0) for x in X], consensus0))
    return fish, X, aligned, consensus


def _fit_without(x: np.ndarray, j: int, consensus: np.ndarray) -> np.ndarray:
    """``x`` moved onto the consensus by every landmark it has but ``j``: where the
    rest of the fish puts each point, ``j`` included."""
    o = [i for i in range(len(x)) if i != j and not np.isnan(x[i, 0])]
    xs, rs = x[o], consensus[o]
    xc, rc = xs.mean(axis=0), rs.mean(axis=0)
    x0, r0 = xs - xc, rs - rc
    s = math.sqrt(float((r0 ** 2).sum()) / max(float((x0 ** 2).sum()), 1e-12))
    u, _, vt = np.linalg.svd(x0.T @ r0)
    d = np.sign(np.linalg.det(u @ vt))
    return (x - xc) * s @ (u @ np.diag([1.0, d]) @ vt) + rc


def misplaced(query: dict, ref: list[RefFish], set_aside=()) -> list[dict]:
    """Landmarks of ``query`` far from where the rest of the fish puts them, worst
    first: ``[{"landmark", "times"}]``, ``times`` being how many times the reference
    fish's usual distance it is. Measured against the average fish of every species
    together, so it cannot lean a fish towards any one.

    The suspect is the landmark whose leaving out lets the others fit the average
    fish best -- a wild point pulls the fit, so the point that looks furthest out
    while it is in may be a good one at the end of the fish -- and it is left out
    only if it is itself far. Then the next, up to ``MAX_LEFT_OUT``."""
    q = {k: tuple(v[:2]) for k, v in query.items() if v}
    out: list[dict] = []
    while len(out) < MAX_LEFT_OUT:
        core = _core(q, ref, set_aside)
        if len(core) <= MIN_LANDMARKS:
            break
        got = _aligned_reference(core, ref)
        if isinstance(got, str):
            break
        _, X, aligned, consensus = got
        have = ~np.isnan(np.array(X)[:, :, 0])           # gaps were filled: not evidence
        spread = np.array([math.sqrt(float((((aligned[have[:, i], i] - consensus[i]) ** 2).sum(-1)).mean()))
                           for i in range(len(core))])
        qx = np.array([q[n] for n in core], float)
        qx[:, 1] *= -1
        fits = [_fit_without(qx, j, consensus) for j in range(len(core))]
        rest = [sum((np.linalg.norm(f[i] - consensus[i]) / spread[i]) ** 2
                    for i in range(len(core)) if i != j) for j, f in enumerate(fits)]
        j = int(np.argmin(rest))
        usual = [np.linalg.norm(_fit_without(a, j, consensus)[j] - consensus[j])
                 for a, h in zip(aligned, have) if h[j]]
        u = math.sqrt(sum(v * v for v in usual) / len(usual)) if usual else 0.0
        times = float(np.linalg.norm(fits[j][j] - consensus[j]) / u) if u > 0 else 0.0
        if times < MISPLACED:
            break
        out.append({"landmark": core[j], "times": round(times, 1)})
        q.pop(core[j])
    return out


def size_factor(sl_mm: float | None, max_sl_mm: float | None) -> float:
    """How much a fish's length leaves of a species' score: 1 well short of the
    longest the species is recorded to reach, falling towards 0 beyond it."""
    if not sl_mm or not max_sl_mm:
        return 1.0
    r = sl_mm / max_sl_mm
    if r <= SIZE_FREE:
        return 1.0
    if r <= 1.0:
        return SIZE_AT_RECORD ** (math.log(r / SIZE_FREE) / math.log(1 / SIZE_FREE))
    return SIZE_AT_RECORD * math.exp(-math.log(r) / SIZE_BEYOND)


SIZES = Path(__file__).with_name("species_sizes.json")


def sizes_for(species, path: Path = SIZES) -> dict:
    """Each species (as a study names it -- "Notropis_hudsonius" or "Notropis
    hudsonius", or any synonym GBIF lists) -> its entry in ``species_sizes.json``.
    A name filed there under two species is left out: it says neither."""
    try:
        table = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    ambiguous = set(table.get("ambiguous_names") or ())
    by_name = {n.lower(): e for e in table.get("species") or () for n in e.get("names") or ()}
    out = {}
    for sp in species:
        key = sp.replace("_", " ").lower()
        if key in by_name and key not in ambiguous:
            out[sp] = by_name[key]
    return out


def identify(query: dict, ref: list[RefFish], px_per_mm: float | None = None,
             set_aside=(), sizes: dict | None = None, query_chars: dict | None = None) -> dict:
    """``match``, after leaving out the landmarks ``misplaced`` finds, with the fish's
    standard length from its landmarks and the photograph's scale. ``set_aside``:
    landmarks kept out of the shape, as ``MOUTH_OPEN`` for a fish with its mouth
    open; they still give its length. ``sizes``: ``sizes_for``'s table, to weigh
    the fish's length against each species' longest recorded (when it has one).
    ``query_chars``: the characters read from its photograph (``match``).

    Adds ``left_out`` (``misplaced``'s list), ``with_all`` (the best score had they
    been kept), ``sl_mm``, ``set_aside``, ``no_match`` (no species scores
    ``NO_MATCH``) and ``tied`` (the species within ``TIE`` of the best, best first)."""
    q = {k: v for k, v in query.items() if v}
    aside = tuple(n for n in RELIABLE if n in set(set_aside))
    left = misplaced(q, ref, aside)
    kept = {k: v for k, v in q.items() if k not in {x["landmark"] for x in left}}
    sl_px = standard_length_px(kept)
    sl_mm = round(sl_px / px_per_mm, 1) if sl_px and px_per_mm else None
    # characters read where a misplaced landmark put them are left out with it
    gone = aside + tuple(x["landmark"] for x in left)
    m = match(kept, ref, sl_mm, set_aside=gone, query_chars=query_chars)
    m.update(left_out=left, sl_mm=sl_mm, with_all=None, set_aside=list(aside))
    if left:
        full = match(q, ref, sl_mm, set_aside=aside, query_chars=query_chars)
        m["with_all"] = full["matches"][0]["score"] if full.get("ok") else None
    if m.get("ok") and sizes and sl_mm:
        for x in m["matches"]:
            e = sizes.get(x["species"])
            if not e:
                continue
            f = size_factor(sl_mm, e.get("max_sl_mm"))
            x["size"] = {"max_sl_mm": e["max_sl_mm"], "max_tl_mm": e.get("max_tl_mm"),
                         "source": e.get("max_source"), "factor": round(f, 3)}
            if f < 1:
                x["shape_score"] = x["score"]
                x["score"] = round(x["score"] * f, 2)
        m["matches"].sort(key=lambda x: -x["score"])
    m["no_match"] = bool(m.get("ok")) and m["matches"][0]["score"] < NO_MATCH
    # tied with the best: within TIE of it, and a contender at all (NO_MATCH or more)
    m["tied"] = ([x["species"] for x in m["matches"]
                  if x["score"] >= max(m["matches"][0]["score"] - TIE, NO_MATCH)]
                 if m.get("ok") and not m["no_match"] else [])
    return m


def with_drainage(m: dict, records: dict) -> dict:
    """``identify``'s answer for a fish whose lot was caught in a drainage, where
    ``records`` gives each species' museum specimens from it (missing or 0: none).

    The species recorded there come first, best shape first, then the rest; the
    best of the first is what the fish is taken for, and ties and "like none" are
    judged among them. ``better_elsewhere``: a species with no record there whose
    shape fits better than any that has one -- a range extension, an introduction,
    or an old misidentification, never to be hidden. ``none_here``: no reference
    species is recorded there, so the order is the shape's alone."""
    if not m.get("ok"):
        return m
    for x in m["matches"]:
        x["drainage_records"] = int(records.get(x["species"]) or 0)
    here = [x for x in m["matches"] if x["drainage_records"]]
    away = [x for x in m["matches"] if not x["drainage_records"]]
    m["matches"] = here + away
    m["drainage_ranked"] = True
    m["none_here"] = not here
    top = here[0]["score"] if here else None
    m["better_elsewhere"] = ({"species": away[0]["species"], "score": away[0]["score"]}
                             if here and away and away[0]["score"] > top and away[0]["score"] >= NO_MATCH else None)
    pool = here or m["matches"]
    m["no_match"] = pool[0]["score"] < NO_MATCH
    m["tied"] = ([x["species"] for x in pool if x["score"] >= max(pool[0]["score"] - TIE, NO_MATCH)]
                 if not m["no_match"] else [])
    return m


def match(query: dict, ref: list[RefFish], query_sl_mm: float | None = None,
          query_noise: float | None = None, set_aside=(), query_chars: dict | None = None) -> dict:
    """Every reference species, scored for how typical ``query`` (landmark name ->
    (x, y)) is of it, best first. ``query_noise``: see ``QUERY_NOISE``.
    ``set_aside``: landmarks the shape and the measurements leave out.
    ``query_chars``: ``photo_characters.characters`` of the fish's photograph."""
    noise = QUERY_NOISE if query_noise is None else query_noise
    q = {k: tuple(v[:2]) for k, v in query.items() if v}
    if query_sl_mm:
        # compared at its own length: with the reference fish that have one. Without
        # this, two fish lacking a ruler turned size off for the whole comparison.
        ref = [r for r in ref if r.sl_mm]
    core = _core(q, ref, set_aside)
    if len(core) < MIN_LANDMARKS:
        return {"ok": False, "error": f"only {len(core)} of the landmarks shape is compared "
                                      f"on were found on this fish"}
    got = _aligned_reference(core, ref)
    if isinstance(got, str):
        return {"ok": False, "error": got}
    fish, _, aligned, consensus = got
    species = sorted({r.species for r in fish})
    Z = (aligned - consensus).reshape(len(fish), -1)
    qx = np.array([q[n] for n in core], float)
    qx[:, 1] *= -1
    qz = (_rotate_onto(_centre_scale(qx), consensus) - consensus).reshape(-1)
    lab = np.array([species.index(r.species) for r in fish])

    # size: only when the fish and every reference fish used have a length
    sized = query_sl_mm is not None and query_sl_mm > 0 and all(r.sl_mm for r in fish)
    logsl = np.log([r.sl_mm for r in fish]) if sized else None
    means = np.array([Z[lab == s].mean(axis=0) for s in range(len(species))])
    W = Z - means[lab]
    slope = None
    if sized:
        lmean = np.array([logsl[lab == s].mean() for s in range(len(species))])
        lw = logsl - lmean[lab]
        if float(lw @ lw) > 1e-9:
            slope = (W.T @ lw) / float(lw @ lw)
            W = W - np.outer(lw, slope)

    dof = len(fish) - len(species) - (1 if slope is not None else 0)
    k = min(AXES, dof - 1 if dof > 1 else 0)
    if k < 2:
        return {"ok": False, "error": "too few reference fish to measure variation within species"}
    _, s, vt = np.linalg.svd(W, full_matrices=False)
    axes = vt[:k]
    var = (s[:k] ** 2) / dof
    var = var + 1e-3 * var[0]                          # no axis with no spread at all
    var = var + noise ** 2                             # the query's own error, on every axis

    # the measurements: each its own term, at the fish's size where there is one
    trait_terms = []                                   # (name, value of the query, {species: (expect, var)})
    qchars = query_chars or {}
    getters = [(n, fn, None) for n, fn in TRAITS.items()] + [(n, None, n) for n in PHOTO_TRAITS]
    for name, fn, key in getters:
        if TRAIT_USES.get(name, set()) & set(set_aside):
            continue
        qv = fn(q) if fn else qchars.get(key)
        vals = [(fn(r.kp) if fn else r.chars.get(key), li, r.sl_mm) for r, li in zip(fish, lab)]
        vals = [(v, li, sl) for v, li, sl in vals if v is not None]
        per = [sum(1 for _, li, _ in vals if li == s) for s in range(len(species))]
        if qv is None or min(per) < MIN_FISH:
            continue
        y = np.array([v for v, _, _ in vals]); tl = np.array([li for _, li, _ in vals])
        ym = np.array([y[tl == s].mean() for s in range(len(species))])
        res = y - ym[tl]
        b, lm_t = 0.0, None
        if slope is not None:
            ls = np.log([sl for _, _, sl in vals])
            lm_t = np.array([ls[tl == s].mean() for s in range(len(species))])
            lw_t = ls - lm_t[tl]
            if float(lw_t @ lw_t) > 1e-9:
                b = float(res @ lw_t) / float(lw_t @ lw_t)
                res = res - b * lw_t
        tdof = len(y) - len(species) - (1 if b else 0)
        if tdof < 2:
            continue
        tvar = float(res @ res) / tdof
        expect_t = {s: (float(ym[s] + (b * (math.log(query_sl_mm) - lm_t[s]) if lm_t is not None else 0)),
                        tvar, per[s]) for s in range(len(species))}
        trait_terms.append((name, qv, expect_t))

    out = []
    for i, sp in enumerate(species):
        n = int((lab == i).sum())
        expect = means[i].copy()
        size_note = None
        rng = None
        if sized:
            sls = [r.sl_mm for r, li in zip(fish, lab) if li == i]
            rng = [round(min(sls), 1), round(max(sls), 1)]
            if slope is not None:
                expect = expect + slope * (math.log(query_sl_mm) - lmean[i])
            if query_sl_mm < 0.9 * rng[0] or query_sl_mm > 1.1 * rng[1]:
                size_note = f"{query_sl_mm:.0f} mm is outside the {rng[0]:.0f}-{rng[1]:.0f} mm reference"
        d = (qz - expect) @ axes.T
        d2 = float((d ** 2 / var).sum()) * n / (n + 1)
        for _, qv, expect_t in trait_terms:
            mu, tvar, nt = expect_t[i]
            d2 += (qv - mu) ** 2 / tvar * nt / (nt + 1)
        out.append({"species": sp, "score": round(chi2_sf(d2, k + len(trait_terms)), 2), "n": n,
                    "sl_range_mm": rng, "size_note": size_note})
    out.sort(key=lambda m: -m["score"])
    return {"ok": True, "matches": out, "landmarks": core, "axes": k,
            "traits": [t[0] for t in trait_terms],
            "size_matched": slope is not None, "reference_fish": len(fish)}
