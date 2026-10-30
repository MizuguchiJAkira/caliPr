"""Each fish in a photograph of several, laid out on one background.

Museum lots and field samples are photographed together -- a tray of fish on a
light box, or on a dark cloth -- with a ruler and a label. The background is one
colour, so a fish is whatever differs from it, and the fish are told from the
ruler, the label and its text by shape: a fish is long and tapers at both ends,
where a ruler, a label or a tag is a rectangle, and the letters printed on them
are small.

The outline found here only says where each fish is. The landmarks are placed by
the landmark model on each fish's own rectangle of the photograph, as for a
photograph of one.

Needs OpenCV, so it runs in the training environment, beside the model.
"""

from __future__ import annotations

import numpy as np

#: The photograph is searched at this size (longest side), then scaled back.
WORK_SIDE = 1600
#: Long side over short side of the fish's bounding rectangle, at least / at most.
#: A shiner is four to six times as long as deep; the deepest, a breeding
#: Luxilus, about three and a half. Logo panels and tags are squatter.
MIN_ELONGATION, MAX_ELONGATION = 3.0, 14.0
#: Colour distance from the background above which a pixel is certainly not
#: background. A black ruler is far further than that; capping it keeps the ruler
#: from setting the threshold, so pale fish on a light box are still found.
DIST_CAP = 40.0
#: Area over the area of its bounding rectangle: a fish fills 50-80 % of it, a
#: label or ruler nearly all of it.
MAX_FILL = 0.86
#: A fish has at least this share of a typical fish's area (the median of the
#: candidates): smaller things are letters, debris and scale bars.
MIN_SHARE_OF_TYPICAL = 0.25
#: Share of a candidate's outline actually covered: a fish is solid, a word -- its
#: letters run together -- is mostly holes.
MIN_SOLIDITY = 0.72
#: And at least this share of the photograph.
MIN_SHARE_OF_PHOTO = 0.0004
#: Each fish's rectangle, widened by this share of its length on every side.
MARGIN = 0.12
#: Snout to caudal base, in pixels, below which the landmark model's points are not
#: to be relied on. Held-out shiners shrunk to tray size kept their accuracy and
#: their scores at 600 and 450 px; at 300 the true species' median score fell from
#: 0.59 to 0.07.
SMALLEST_SL_PX = 400


def background(lab8: np.ndarray) -> np.ndarray:
    """The background behind each pixel: the photograph with everything as thin as a
    fish blurred away (a median over an eighth of the photograph). A light box is
    brighter in the middle than at the corners, and a label or colour card along
    one edge is not the background, so one colour for the whole photograph will
    not do."""
    import cv2

    h, w = lab8.shape[:2]
    s = 400 / max(h, w)
    tiny = cv2.resize(lab8, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    # Fish packed close together would pass for background in a plain median, so
    # whatever is clearly not background -- far from the colour of the edge -- is
    # painted over with that colour first, and only then is the rest smoothed.
    band = 4
    edge = np.concatenate([tiny[:band].reshape(-1, 3), tiny[-band:].reshape(-1, 3),
                           tiny[:, :band].reshape(-1, 3), tiny[:, -band:].reshape(-1, 3)])
    rough = np.median(edge, axis=0)
    far = np.linalg.norm(tiny.astype(np.float32) - rough, axis=2)
    d8 = (np.clip(far, 0, DIST_CAP) * (255.0 / DIST_CAP)).astype(np.uint8)
    _, obj = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    obj = cv2.dilate(obj, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    filled = tiny.copy()
    filled[obj > 0] = rough.astype(np.uint8)
    bg = cv2.medianBlur(filled, (max(tiny.shape[:2]) // 8) | 1)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def find_fish(bgr: np.ndarray) -> list[dict]:
    """Every fish found, left to right then top to bottom: its rectangle (with a
    margin, in the photograph's pixels), its outline and its length in pixels."""
    import cv2

    h, w = bgr.shape[:2]
    s = min(1.0, WORK_SIDE / max(h, w))
    small = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    lab8 = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    dist = np.linalg.norm(lab8.astype(np.float32) - background(lab8), axis=2)
    d8 = cv2.GaussianBlur((np.clip(dist, 0, DIST_CAP) * (255.0 / DIST_CAP)).astype(np.uint8),
                          (5, 5), 0)
    _, fg = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, round(min(small.shape[:2]) / 300)) | 1
    disk = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))  # noqa: E731
    # Opening cuts the strings tags hang from.
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, disk(k))
    photo_area = small.shape[0] * small.shape[1]
    # Rulers, labels, colour cards and the panels they sit on are rectangles. Taken
    # out before anything is joined up, they cannot be joined to a fish.
    for c in cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if area > MIN_SHARE_OF_PHOTO * photo_area and rw * rh > 0 and area / (rw * rh) > MAX_FILL:
            cv2.drawContours(fg, [c], -1, 0, -1)
    # Closing fills the pale fins and the gaps a light belly leaves against a light
    # background, without reaching across to the next fish.
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, disk(2 * k + 1))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_SHARE_OF_PHOTO * photo_area:
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        long_, short = max(rw, rh), min(rw, rh)
        if short < 1:
            continue
        if not MIN_ELONGATION <= long_ / short <= MAX_ELONGATION:
            continue
        if area / (rw * rh) > MAX_FILL:
            continue
        inside = np.zeros_like(fg)
        cv2.drawContours(inside, [c], -1, 255, -1)
        if cv2.countNonZero(cv2.bitwise_and(fg, inside)) < MIN_SOLIDITY * cv2.countNonZero(inside):
            continue
        cands.append((area, long_, c))
    if not cands:
        return []
    typical = float(np.median([a for a, _, _ in cands]))
    fish = []
    for area, long_, c in cands:
        if area < MIN_SHARE_OF_TYPICAL * typical:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        m = MARGIN * long_
        box = [max(0, (x - m) / s), max(0, (y - m) / s),
               min(w, (x + bw + m) / s), min(h, (y + bh + m) / s)]
        eps = 0.004 * cv2.arcLength(c, True)
        outline = (cv2.approxPolyDP(c, eps, True).reshape(-1, 2) / s).round(1).tolist()
        fish.append({"box": [round(v) for v in box], "outline": outline,
                     "length_px": round(long_ / s, 1)})
    # reading order: rows of fish, top to bottom, each left to right
    fish.sort(key=lambda f: ((f["box"][1] + f["box"][3]) / 2, f["box"][0]))
    rows: list[list[dict]] = []
    for f in fish:
        cy = (f["box"][1] + f["box"][3]) / 2
        if rows and abs(cy - np.mean([(g["box"][1] + g["box"][3]) / 2 for g in rows[-1]])) \
                < 0.5 * (f["box"][3] - f["box"][1]):
            rows[-1].append(f)
        else:
            rows.append([f])
    return [f for row in rows for f in sorted(row, key=lambda g: g["box"][0])]
