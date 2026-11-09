"""Fish scales: a photograph enhanced to show their margins, and a count along a row.

"Scale" here is the fish's, not the ruler's (that is ``ruler_calibration``).

In a preserved shiner the scales show as their margins: a crescent of pigment along
each scale pocket, faint under the light-box glare and crossed by melanophore dots.
``enhance`` brings the margins up for the eye -- local contrast (CLAHE), a
margin map that answers to lines and not to dots, or a high-pass -- at the
photograph's own size, so a point placed on the enhanced view is placed on the
photograph.

``count_along`` counts the scales along a path someone traces (a lateral line, or
Gilbert's anterior dorso-lateral row). It straightens a narrow band around the
path, reads the margin map along it, finds the spacing of the margins from the
band's autocorrelation, and places one tick per margin by dynamic programming --
the best chain of dark crossings whose gaps stay near that spacing -- so a margin
lost under a dot or a glint is bridged rather than skipped. The ticks are returned
so a person can see every one and correct the count; the number is a proposal.

Where the margins cannot be followed the whole way, the count is filled in: every
stretch of the row regular enough to read gives the scale length there, those are
pooled (as many scales as the photograph allows -- one scale is not a ruler fine
enough), and the rest of the row is filled with scales of that length, shaded by
how scale length changes along the body (``LENGTH_ALONG_BODY``). Either way the
answer carries a likely range, wider the fewer scales were measured.

numpy and OpenCV only, so it runs in the labeler's own environment.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

#: Display modes ``enhance`` offers.
MODES = ("local", "margins", "highpass")
CHANNELS = ("luminance", "red", "green", "blue")
#: Spacing of the margins along a row, as a share of standard length: 30-64
#: lateral-line scales over the ~0.78 SL from the shoulder girdle to the hypural.
SPACING_SL = (0.78 / 64, 0.78 / 30)
#: Below this the band's autocorrelation at the chosen spacing says there is no
#: regular pattern to count, and the count is withheld.
MIN_PERIODICITY = 0.12
#: What each tick costs the chain, in standard deviations of the margin signal.
#: Without it every crossing adds to the chain's score, and a dot between two
#: margins earns a tick of its own. Calibrated against published lateral-line
#: counts for 14 species (docs/what-we-tried.md): at 0.25 the species' average
#: counts land within a median 0.7 scales of the literature with no correction.
TICK_COST = 0.25
#: Scale length along the body, relative to mid-body, by position (x / SL from the
#: snout): the median local spacing, in windows of about seven scales, on 115 clear
#: shiner photographs (Notropis 81, Luxilus 18, Cyprinella 13). Nearly flat, and
#: alike in the three genera: a little shorter behind mid-body, longer at the base
#: of the caudal fin.
LENGTH_ALONG_BODY = ((0.29, 1.00), (0.36, 1.02), (0.44, 1.02), (0.51, 1.00), (0.59, 0.97),
                     (0.66, 0.94), (0.74, 0.95), (0.81, 0.97), (0.89, 0.97), (0.96, 1.07))
#: A stretch is read when its own autocorrelation, at its own spacing, reaches this:
#: 0.4% of windows on a plain, noisy ground do, against a quarter of those on a shiner.
READABLE = 0.35
#: Fewer scales than this, across every stretch read, and the count is not filled in.
MIN_MEASURED = 8
#: How far a stretch's scale length may stray from the others' before it is taken
#: for a misreading (a half or double spacing) and left out.
STRAY = 0.3
#: Error of a count, relative to the count, from n scales measured:
#: sqrt(ERR_PER_SCALE**2 / n + ERR_FLOOR**2). On 115 clear photographs, filling in
#: from 10-23 scales strayed from the whole row's count by 4.5-5% (robust SD)
#: whatever the number measured, so the floor carries it; the per-scale term is a
#: modest allowance for the few rows read on fewer than ten.
ERR_PER_SCALE, ERR_FLOOR = 0.10, 0.045
#: The likely range given with a count: this many standard errors either side (90%).
RANGE_Z = 1.645
#: A tick at every margin only where the stretches read cover this share of the
#: row; less, and the count is filled in from them. On 317 shiners read both ways,
#: below 30% the ticks fell inside the published range for 67% of fish and the
#: filled-in count for 77%; above it the two were level.
TICKS_COVERAGE = 0.3
#: Where a row's scales show pale rims (the free edge, behind a band of pigment, as on
#: Cyprinella analostana), the ticks go on the rims rather than the dark bands, which
#: lie inside each scale: when the rims' own autocorrelation reaches this, at a spacing
#: within 15% of the margins', and they give the same count. On three scales outlined
#: by hand the rims were 1-5 px from the outlines, the bands 7-12 px inside them.
RIM_PERIODICITY = 0.25


def _channel(image: np.ndarray, channel: str) -> np.ndarray:
    if image.ndim == 2:
        return image
    if channel == "luminance":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
    return image[..., {"blue": 0, "green": 1, "red": 2}[channel]]


def _clahe(gray: np.ndarray, clip: float, scale_size_px: float | None) -> np.ndarray:
    h, w = gray.shape
    if scale_size_px:                      # tiles about four scales across
        tile = max(8.0, 4 * scale_size_px)
        grid = (max(2, round(w / tile)), max(2, round(h / tile)))
    else:
        grid = (8, max(2, round(8 * h / w)))
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=grid).apply(gray)


def margin_map(gray: np.ndarray, scale_size_px: float | None = None) -> np.ndarray:
    """How much each pixel looks like a dark line, 0 where it looks like a dot.

    From the Hessian of the smoothed image: across a dark line the larger
    eigenvalue is large and the smaller near 0; on a dark dot both are large, and
    their difference is what is kept. Best over two widths.
    """
    g = gray.astype(np.float32)
    widths = [1.2, 2.2] if not scale_size_px else [max(1.0, 0.05 * scale_size_px), max(1.6, 0.09 * scale_size_px)]
    best = np.zeros_like(g)
    for s in widths:
        b = cv2.GaussianBlur(g, (0, 0), s)
        dxx = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3)
        half, root = (dxx + dyy) / 2, np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
        big, small = half + root, half - root
        best = np.maximum(best, s * s * np.clip(big - np.abs(small), 0, None))
    return best


def _stretch(a: np.ndarray, lo: float = 1, hi: float = 99.5) -> np.ndarray:
    p, q = np.percentile(a, [lo, hi])
    return np.clip((a - p) * (255.0 / max(q - p, 1e-6)), 0, 255).astype(np.uint8)


def enhance(image: np.ndarray, *, mode: str = "local", strength: float = 4.0,
            scale_size_px: float | None = None, channel: str = "luminance") -> np.ndarray:
    """A grey image the photograph's size with the scale margins brought up.

    ``local``: CLAHE, ``strength`` its clip limit (2 gentle, 8 harsh).
    ``margins``: dark lines drawn dark on white, dots left out (``margin_map``).
    ``highpass``: the photograph less its blur at about half a scale, then CLAHE.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}")
    gray = _channel(image, channel)
    if mode == "local":
        return _clahe(gray, strength, scale_size_px)
    if mode == "highpass":
        g = gray.astype(np.float32)
        sigma = 0.6 * scale_size_px if scale_size_px else max(4.0, max(g.shape) / 250)
        return _clahe(_stretch(g - cv2.GaussianBlur(g, (0, 0), sigma)), max(1.0, strength / 2), scale_size_px)
    m = margin_map(_clahe(gray, 3.0, scale_size_px), scale_size_px)
    return 255 - _stretch(m, 50, 100 - min(4.0, strength / 2))


def _resample(path: list, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step`` px along a polyline, and the unit normal at each."""
    p = np.asarray(path, np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    p, seg = p[keep], seg[seg > 1e-9]
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], step)
    pts = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)
    d = np.gradient(pts, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return pts, np.stack([-d[:, 1], d[:, 0]], axis=1)


def _ticks(signal: np.ndarray, period: float, tightness: float = 3.0) -> list[int]:
    """The best chain of high points in ``signal`` with gaps near ``period``.

    Beat tracking (Ellis 2007): best[i] = signal[i] + max over earlier j of
    best[j] - tightness * log((i - j) / period)^2, for gaps 0.55-1.7 periods;
    the chain may start within the first period and must end within the last.
    """
    n = signal.size
    lo, hi = max(1, int(0.55 * period)), max(2, int(math.ceil(1.7 * period)))
    best = signal.astype(np.float64).copy()
    back = np.full(n, -1)
    for i in range(n):
        j0, j1 = max(0, i - hi), i - lo
        if j1 < j0:
            if i > period:
                best[i] = -np.inf
            continue
        js = np.arange(j0, j1 + 1)
        cand = best[js] - tightness * np.log((i - js) / period) ** 2
        k = int(np.argmax(cand))
        if cand[k] > -np.inf and (i > period or cand[k] > 0):
            best[i] += cand[k]
            back[i] = js[k]
        elif i > period:
            best[i] = -np.inf
    tail = np.arange(max(0, n - int(math.ceil(period))), n)
    i = int(tail[np.argmax(best[tail])])
    out = []
    while i >= 0:
        out.append(i)
        i = int(back[i])
    return out[::-1]


def band_signal(image: np.ndarray, path: list, *, sl_px: float | None = None,
                spacing_px: tuple[float, float] | None = None, width_px: float | None = None,
                pale: bool = False):
    """The margin strength along ``path``, detrended: (points, normals, signal, lo, hi),
    or an error string. ``count_along`` reads its spacing and ticks from this.
    ``pale``: the strength of pale lines instead, the scales' rims."""
    pts, normals = _resample(path)
    n = len(pts)
    if n < 24:
        return "The path is too short to count along."
    if spacing_px:
        lo, hi = spacing_px
    elif sl_px:
        lo, hi = SPACING_SL[0] * sl_px, SPACING_SL[1] * sl_px
    else:
        lo, hi = n / 120, n / 4
    lo, hi = max(3.0, lo), max(lo + 2, min(hi, n / 2))
    half = (width_px or max(3.0, hi / 5)) / 2

    # The margin map on a crop around the path only
    pad = int(half + 4 * hi)
    x0, y0 = (np.floor(pts.min(0)) - pad).astype(int).clip(0)
    x1, y1 = np.ceil(pts.max(0) + pad).astype(int)
    x1, y1 = min(x1, image.shape[1]), min(y1, image.shape[0])
    crop = image[y0:y1, x0:x1]
    lum = _clahe(_channel(crop, "luminance"), 3.0, (lo + hi) / 2)
    m = margin_map(255 - lum if pale else lum, (lo + hi) / 2)
    offs = np.linspace(-half, half, max(3, int(2 * half) | 1))
    mx = (pts[:, 0, None] - x0 + normals[:, 0, None] * offs).astype(np.float32)
    my = (pts[:, 1, None] - y0 + normals[:, 1, None] * offs).astype(np.float32)
    band = cv2.remap(m, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    profile = band.mean(axis=1)
    trend = cv2.GaussianBlur(profile.reshape(-1, 1).astype(np.float32), (0, 0), 1.5 * hi).ravel()
    return pts, normals, profile - trend, lo, hi


def spacing(sig: np.ndarray, lo: float, hi: float) -> tuple[float, float] | str:
    """The margins' spacing along the band, and how regular they are (the
    autocorrelation there, 0-1): the strongest repeat between ``lo`` and ``hi`` px."""
    n = sig.size
    x = sig - sig.mean()
    size = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:n]
    ac = ac / ac[0] if ac[0] > 0 else ac
    a, b = int(math.floor(lo)), min(n - 2, int(math.ceil(hi)))
    if b <= a + 1:
        return "The path is too short for the spacing searched."
    window = ac[a:b + 1]
    peaks = [i for i in range(1, window.size - 1) if window[i] >= window[i - 1] and window[i] >= window[i + 1]]
    if not peaks:
        return "No regular spacing along this path."
    k = max(peaks, key=lambda i: window[i])
    # a harmonic at twice the spacing can edge out the fundamental; take the
    # shorter one when it is nearly as strong
    for i in sorted(peaks):
        if i < k and abs((a + k) / (a + i) - 2) < 0.25 and window[i] > 0.7 * window[k]:
            k = i
            break
    return float(a + k), float(window[k])


def relative_length(x_sl) -> np.ndarray:
    """``LENGTH_ALONG_BODY`` at body positions ``x_sl`` (x / SL from the snout)."""
    xs, vs = zip(*LENGTH_ALONG_BODY)
    return np.interp(np.asarray(x_sl, float), xs, vs)


def _rim_ticks(image: np.ndarray, path: list, period: float, n_ticks: int, **band) -> list[int] | None:
    """The ticks on the scales' pale rims, if the row shows them (``RIM_PERIODICITY``)
    and they count the same scales as the dark margins; else None."""
    got = band_signal(image, path, pale=True, **band)
    if isinstance(got, str):
        return None
    sig, lo, hi = got[2], got[3], got[4]
    sp = spacing(sig, lo, hi)
    if isinstance(sp, str) or sp[1] < RIM_PERIODICITY or abs(sp[0] / period - 1) > 0.15:
        return None
    ticks = _ticks(sig / (sig.std() or 1.0) - TICK_COST, period)
    return ticks if len(ticks) == n_ticks else None


def body_positions(pts: np.ndarray, axis) -> np.ndarray | None:
    """x / SL from the snout of each point, given ``axis`` = (snout, end of the
    vertebral column); None without one."""
    if not axis:
        return None
    a, b = np.asarray(axis[0], float)[:2], np.asarray(axis[1], float)[:2]
    sl = float(np.linalg.norm(b - a))
    if sl <= 0:
        return None
    return (np.asarray(pts, float) - a) @ ((b - a) / sl) / sl


def read_stretches(sig: np.ndarray, lo: float, hi: float, guess: float) -> list[tuple[int, int, float]]:
    """The windows of the band regular enough to read, each with its own spacing:
    [(start, end, spacing px)], windows about six scales long, one scale apart."""
    w = int(math.ceil(max(3.2 * hi, 6 * guess)))
    out = []
    for a in range(0, sig.size - w + 1, max(1, int(guess))):
        r = spacing(sig[a:a + w], lo, hi)
        if not isinstance(r, str) and r[1] >= READABLE:
            out.append((a, a + w, r[0]))
    return out


def fill_in(stretches: list, rel: np.ndarray) -> dict | None:
    """The count of a row ``rel.size`` px long from the stretches read on it.

    Each stretch's spacing, divided by the relative scale length where it lies,
    estimates the mid-body scale length; the median of those that agree (within
    ``STRAY``) is taken, and the row is filled with scales of it times ``rel``.
    ``measured`` is how many scales the stretches cover."""
    if not stretches:
        return None
    mid = np.array([s / rel[min(rel.size - 1, (a + b) // 2)] for a, b, s in stretches])
    med = float(np.median(mid))
    keep = [k for k, m in enumerate(mid) if abs(m / med - 1) <= STRAY]
    unit = float(np.median(mid[keep]))
    covered = np.zeros(rel.size, bool)
    for k in keep:
        covered[stretches[k][0]:stretches[k][1]] = True
    measured = float(covered.sum()) / unit
    if measured < MIN_MEASURED:
        return None
    # the row's length in scale lengths, and one more: a scale part-way in at each end
    count = float(np.sum(1.0 / (unit * rel))) + 1
    err = count * math.sqrt(ERR_PER_SCALE ** 2 / max(measured, 1.0) + ERR_FLOOR ** 2)
    return {"count": count, "unit": unit, "measured": measured, "err": err, "covered": covered,
            "kept": [stretches[k] for k in keep]}


def count_along(image: np.ndarray, path: list, *, sl_px: float | None = None,
                spacing_px: tuple[float, float] | None = None, width_px: float | None = None,
                axis=None) -> dict:
    """The scales crossed along ``path`` ([[x, y], ...] in image pixels).

    The search for the margins' spacing is bounded by ``spacing_px`` (lo, hi) if
    given, else by the fish's standard length (``SPACING_SL``), else by the path
    (4 to 120 margins along it). The band read across the path is ``width_px``
    wide, by default a fifth of the longest spacing allowed -- narrow, so it stays
    inside one row and does not see the next row's margins half a scale on.
    ``axis`` (snout, end of the vertebral column) places the path on the body, so
    a count filled in follows how scale length changes along it.

    ``method`` says how the count was made: "every margin" (a tick at each, the
    count is the ticks) or "filled in" (the row's regular stretches measured, the
    rest filled; the ticks show only the stretches measured, and ``measured`` how
    many scales they cover) -- filled in when the row as a whole is not regular
    enough, or its readable stretches cover less than ``TICKS_COVERAGE`` of it.
    ``range`` is the likely range either way. A row traced from the tail is read
    from the head, so the answer does not depend on which end was clicked first;
    its ticks come back in the order traced.
    """
    if axis is not None and len(path) >= 2:
        d = np.asarray(axis[1], float)[:2] - np.asarray(axis[0], float)[:2]
        if (np.asarray(path[-1], float)[:2] - np.asarray(path[0], float)[:2]) @ d < 0:
            res = count_along(image, path[::-1], sl_px=sl_px, spacing_px=spacing_px, width_px=width_px, axis=axis)
            if res.get("ticks"):
                res["ticks"] = res["ticks"][::-1]
            if res.get("stretches"):
                res["stretches"] = [[b, a] for a, b in res["stretches"][::-1]]
            return res
    got = band_signal(image, path, sl_px=sl_px, spacing_px=spacing_px, width_px=width_px)
    if isinstance(got, str):
        return {"ok": False, "error": got}
    pts, _, sig, lo, hi = got
    x = body_positions(pts, axis)
    rel = relative_length(x) if x is not None else np.ones(len(pts))
    whole = spacing(sig, lo, hi)
    whole = None if isinstance(whole, str) else whole
    guess = whole[0] if whole else math.sqrt(lo * hi)
    est = fill_in(read_stretches(sig, lo, hi, guess), rel)
    z = sig / (sig.std() or 1.0) - TICK_COST
    xy = lambda i: [round(float(pts[i, 0]), 1), round(float(pts[i, 1]), 1)]
    base = {"length_px": round(float(len(pts)), 1), "searched_px": [round(lo, 1), round(hi, 1)]}

    readable = est is None or est["measured"] / est["count"] >= TICKS_COVERAGE
    if whole and whole[1] >= MIN_PERIODICITY and readable:
        ticks = _ticks(z, whole[0])
        rims = _rim_ticks(image, path, whole[0], len(ticks), sl_px=sl_px, spacing_px=spacing_px, width_px=width_px)
        ticks = rims or ticks
        n = len(ticks)
        measured = est["measured"] if est else float(n)
        err = n * math.sqrt(ERR_PER_SCALE ** 2 / max(measured, 1.0) + ERR_FLOOR ** 2)
        return {**base, "ok": True, "method": "every margin", "count": n, "ticks": [xy(i) for i in ticks],
                "ticks_on": "pale rims" if rims else "dark margins",
                "spacing_px": round(whole[0], 1), "periodicity": round(whole[1], 3),
                "measured": round(measured, 1),
                "range": [int(math.floor(n - RANGE_Z * err)), int(math.ceil(n + RANGE_Z * err))]}
    if est:
        n = est["count"]
        mid_spacing = est["unit"] * float(np.median(rel))
        ticks = [i for i in _ticks(z, mid_spacing) if est["covered"][i]]
        return {**base, "ok": True, "method": "filled in", "count": int(round(n)),
                "ticks": [xy(i) for i in ticks], "spacing_px": round(mid_spacing, 1),
                "periodicity": round(whole[1], 3) if whole else 0.0,
                "measured": round(est["measured"], 1),
                "stretches": [[xy(a), xy(min(b, len(pts) - 1))] for a, b, _ in est["kept"]],
                "range": [int(math.floor(n - RANGE_Z * est["err"])), int(math.ceil(n + RANGE_Z * est["err"]))]}
    return {**base, "ok": False, "periodicity": round(whole[1], 3) if whole else 0.0,
            "error": "No stretch of this row is regular enough to measure the scales on."}
