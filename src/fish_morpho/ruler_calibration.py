"""Ruler detection and pixels-per-millimeter calibration.

Each specimen photo contains a metric ruler laid next to the fish. We need to
convert pixel distances into millimeters, so this module exposes two things:

  1. ``detect_ruler_scale`` — an OpenCV-based detector that tries to find the
     ruler automatically, identify tick marks along one of its long edges,
     and return a pixels-per-mm factor.
  2. ``scale_from_known_span`` — a deterministic fallback used when a human
     (or a JSON annotation file) provides two points and the real-world
     distance between them. This is always available and is what the
     pipeline uses when ``--mode manual`` is passed or when auto-detection
     fails.

The frontal-view mirror ruler is smaller and oriented perpendicular to the
main ruler, but the algorithm is identical — we just run it on the mirror
region of interest.

Design notes
------------
* We do not require a specific ruler brand or colour. The detector works on
  the assumption that the ruler is the dominant straight, high-contrast,
  elongated object in its ROI and that its tick marks are a regular 1D
  pattern perpendicular to its long axis.
* Tick spacing is recovered via an FFT-based peak in the 1D intensity
  profile along the ruler's long axis. This is more robust than finding
  every individual tick (which tends to fail for faint sub-cm marks and
  over-count when numerals are printed on the ruler).
* The detector returns a ``CalibrationResult`` that includes a ``confidence``
  score and the intermediate artifacts (ROI, profile, detected period in
  pixels) so the caller can decide to fall back to manual calibration.
* We do not depend on a specific OpenCV build at import time; the heavy
  ``cv2`` import is deferred into functions that need it. This keeps unit
  tests that only exercise ``scale_from_known_span`` lightweight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationResult:
    """Result of a ruler calibration attempt.

    Attributes
    ----------
    px_per_mm:
        Conversion factor. Multiply a pixel distance by ``1 / px_per_mm`` to
        get millimeters, or divide.
    method:
        Human-readable tag: ``"auto"`` or ``"manual"``.
    confidence:
        Value in ``[0, 1]``. For ``manual`` calibrations this is 1.0.
        For ``auto`` calibrations it is a heuristic derived from the FFT
        peak prominence; values below ~0.3 should be treated as unreliable.
    roi:
        Bounding box ``(x, y, w, h)`` of the detected ruler, or ``None`` for
        manual calibrations. Useful for overlaying a sanity-check annotation
        on the original image.
    notes:
        Free-form diagnostic message for logging.
    """

    px_per_mm: float
    method: str
    confidence: float = 1.0
    roi: tuple[int, int, int, int] | None = None
    notes: str = ""

    def px_to_mm(self, px: float) -> float:
        return px / self.px_per_mm

    def area_px_to_mm2(self, area_px: float) -> float:
        return area_px / (self.px_per_mm ** 2)


# ---------------------------------------------------------------------------
# Manual calibration
# ---------------------------------------------------------------------------


def scale_from_known_span(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    known_mm: float,
) -> CalibrationResult:
    """Build a calibration from two clicked points with a known real distance.

    Raises ``ValueError`` if the two points coincide or ``known_mm`` is not
    positive — either of those would produce a meaningless scale factor.
    """
    if known_mm <= 0:
        raise ValueError(f"known_mm must be positive, got {known_mm}")

    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    dist_px = math.hypot(dx, dy)
    if dist_px <= 0:
        raise ValueError("Calibration points must be distinct")

    return CalibrationResult(
        px_per_mm=dist_px / known_mm,
        method="manual",
        confidence=1.0,
        notes=f"manual span {dist_px:.2f} px = {known_mm} mm",
    )


# ---------------------------------------------------------------------------
# Automatic calibration
# ---------------------------------------------------------------------------


def detect_ruler_scale(
    image: "NDArray[np.uint8]",
    roi: tuple[int, int, int, int] | None = None,
    min_ticks: int = 10,
) -> CalibrationResult:
    """Attempt to auto-detect the ruler and compute px/mm.

    Parameters
    ----------
    image:
        BGR or grayscale image (numpy array as returned by ``cv2.imread``).
    roi:
        Optional ``(x, y, w, h)`` hint telling the detector where to look.
        If omitted the whole image is searched — in practice the lab's photos
        put the ruler below the fish, so passing an ROI is encouraged.
    min_ticks:
        Minimum number of tick periods that must fit in the detected profile
        for the result to be considered reliable. Fewer than this is a strong
        signal the detector hit noise rather than a real ruler.

    Returns
    -------
    ``CalibrationResult`` with ``method="auto"``. The caller should check
    ``confidence`` before trusting the value.

    Raises
    ------
    RuntimeError
        If no plausible ruler-like region can be found at all.
    """
    import cv2  # local import — keeps tests that don't need cv2 lightweight

    # --- 1. Prepare grayscale ROI ------------------------------------------------
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h_img, w_img = gray.shape

    if roi is not None:
        x, y, w, h = roi
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        gray_roi = gray[y : y + h, x : x + w]
    else:
        x, y = 0, 0
        gray_roi = gray

    # --- 2. Find the dominant rectangular object (the ruler) --------------------
    #
    # Strategy: threshold to isolate the (typically bright) ruler body, find the
    # largest near-rectangular contour, fit a rotated rectangle to recover the
    # long-axis orientation and extent. We try Otsu first because it handles
    # most lab lighting conditions; if that fails we fall back to adaptive
    # thresholding.
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    _, th = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Try both polarities — some rulers are bright, some are dark on bright paper.
    candidates = [th, cv2.bitwise_not(th)]

    best_rect = None
    best_area = 0.0
    for mask in candidates:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.01 * gray_roi.size:
                # Ignore tiny blobs.
                continue
            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = rect
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect < 4.0:
                # Rulers are long and skinny; skip square-ish blobs.
                continue
            if area > best_area:
                best_area = area
                best_rect = rect

    if best_rect is None:
        raise RuntimeError(
            "No ruler-like region found (no elongated high-contrast object). "
            "Try providing an ROI hint or use scale_from_known_span."
        )

    # --- 3. Rotate the ROI so the ruler's long axis is horizontal --------------
    (cx, cy), (rw, rh), angle = best_rect
    if rw < rh:
        # minAreaRect returns the width along the first edge; normalize so
        # that w is the long side.
        rw, rh = rh, rw
        angle += 90.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        gray_roi,
        M,
        (gray_roi.shape[1], gray_roi.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Crop to the ruler's bounding box in the rotated frame.
    half_w = rw / 2.0
    half_h = rh / 2.0
    x0 = int(round(cx - half_w))
    x1 = int(round(cx + half_w))
    y0 = int(round(cy - half_h))
    y1 = int(round(cy + half_h))
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(rotated.shape[1], x1)
    y1 = min(rotated.shape[0], y1)
    ruler_strip = rotated[y0:y1, x0:x1]

    if ruler_strip.size == 0 or ruler_strip.shape[1] < 32:
        raise RuntimeError("Rotated ruler strip is degenerate; cannot profile ticks.")

    # --- 4. Build a 1D intensity profile along the long axis -------------------
    #
    # Tick marks show up as dark vertical bands. Averaging across the short
    # axis turns them into periodic minima. We subtract a rolling mean to
    # remove low-frequency illumination drift.
    profile = ruler_strip.mean(axis=0).astype(np.float32)
    if profile.size < 32:
        raise RuntimeError("Profile too short to analyze.")

    kernel_size = max(9, profile.size // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    rolling = _boxcar(profile, kernel_size)
    detrended = profile - rolling
    detrended -= detrended.mean()

    # --- 5. Recover tick period via autocorrelation ----------------------------
    #
    # We use autocorrelation rather than a plain FFT peak because sharp tick
    # marks look like a pulse train and their power spectrum has roughly equal
    # amplitude at every harmonic of the fundamental — which means the raw
    # FFT peak can land on an odd harmonic of the true period. Autocorrelation
    # puts a clean peak at the fundamental (and its integer multiples), so
    # taking the FIRST strong peak after a minimum lag reliably gives us the
    # true tick period.
    #
    # The 1 mm ticks on the lab's rulers map directly to px/mm; if you are
    # working with rulers where the finest labeled tick is at a different
    # spacing, divide the returned period by that spacing in mm.
    period_px = _period_via_autocorrelation(
        detrended,
        min_ticks=min_ticks,
    )
    if period_px is None:
        raise RuntimeError("No periodic tick signal detected.")

    # --- 6. Confidence score ---------------------------------------------------
    #
    # Re-derive the peak value and normalize against the autocorrelation mean
    # over the search window. Strong periodic signals give prominence > 10;
    # noise gives ~1. We clip to [0, 1] for reporting.
    prominence = _autocorr_peak_prominence(detrended, period_px)
    confidence = float(min(1.0, prominence / 8.0))

    # Translate ruler bounding box back into ORIGINAL image coordinates for
    # the ``roi`` return field. We use the original unrotated minAreaRect
    # center, because that's what users will want to visualize.
    box = cv2.boxPoints(best_rect).astype(np.int32)
    box[:, 0] += x
    box[:, 1] += y
    bx, by, bw, bh = cv2.boundingRect(box)

    return CalibrationResult(
        px_per_mm=float(period_px),
        method="auto",
        confidence=confidence,
        roi=(int(bx), int(by), int(bw), int(bh)),
        notes=(
            f"autocorrelation period={period_px:.3f} px, "
            f"prominence={prominence:.1f}"
        ),
    )


def _boxcar(signal: "NDArray[np.float32]", window: int) -> "NDArray[np.float32]":
    """Centered rolling mean via convolution, edges replicated.

    Implemented without scipy to keep the dependency surface small.
    """
    if window <= 1:
        return signal.copy()
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")[: signal.size]


def _autocorrelation(signal: "NDArray[np.float32]") -> "NDArray[np.float32]":
    """Biased autocorrelation at non-negative lags, normalized so ``ac[0] == 1``.

    Uses FFT so it stays O(n log n) on realistic profile lengths.
    """
    n = signal.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    x = signal - signal.mean()
    fft_size = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    spectrum = np.fft.rfft(x, n=fft_size)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[:n].real
    if ac[0] <= 0:
        return ac.astype(np.float32)
    return (ac / ac[0]).astype(np.float32)


def _period_via_autocorrelation(
    signal: "NDArray[np.float32]",
    *,
    min_ticks: int,
    min_period_px: float = 2.5,
    max_period_fraction: float = 0.1,
    peak_threshold: float = 0.2,
) -> float | None:
    """Recover the fundamental period of a (roughly) periodic signal.

    Parameters
    ----------
    signal:
        Detrended 1D intensity profile along the ruler's long axis.
    min_ticks:
        Minimum number of tick periods that must fit across the profile for
        the result to be considered plausible. Rejects "period" values that
        would imply fewer ticks than this.
    min_period_px:
        Lower bound on the search window. Sub-pixel periods don't correspond
        to physically meaningful ticks on any real ruler.
    max_period_fraction:
        Upper bound expressed as a fraction of the profile length. Caps how
        many pixels a single period can span so the first-peak search doesn't
        wander into global-trend territory.
    peak_threshold:
        Minimum normalized autocorrelation value required to accept a peak.

    Returns
    -------
    Period in pixels, or ``None`` if no valid peak was found.
    """
    n = signal.size
    if n < 8:
        return None
    ac = _autocorrelation(signal)

    min_lag = max(int(round(min_period_px)), 2)
    max_lag = max(min_lag + 2, int(n * max_period_fraction))
    max_lag = min(max_lag, n - 2)
    # If min_ticks would force a tighter upper bound, honor it.
    max_lag = min(max_lag, n // max(min_ticks, 1))
    if max_lag <= min_lag + 1:
        return None

    window = ac[min_lag : max_lag + 1]

    # Find the first interior local maximum that clears peak_threshold. The
    # "first" part is load-bearing: harmonics show up as later peaks and we
    # want the fundamental.
    first_peak: int | None = None
    for i in range(1, window.size - 1):
        if (
            window[i] > window[i - 1]
            and window[i] >= window[i + 1]
            and window[i] > peak_threshold
        ):
            first_peak = i
            break

    if first_peak is None:
        # No peak crossed the threshold — ruler-quality signal probably
        # isn't there. Bail.
        return None

    peak_lag = min_lag + first_peak

    # Sub-lag interpolation via parabolic fit around the peak for extra
    # precision: the true maximum of a triangle function's autocorrelation
    # rarely sits exactly on an integer lag.
    y_minus = float(ac[peak_lag - 1])
    y_zero = float(ac[peak_lag])
    y_plus = float(ac[peak_lag + 1])
    denom = y_minus - 2 * y_zero + y_plus
    if denom != 0:
        offset = 0.5 * (y_minus - y_plus) / denom
        offset = max(-1.0, min(1.0, offset))
    else:
        offset = 0.0
    return float(peak_lag + offset)


def _autocorr_peak_prominence(
    signal: "NDArray[np.float32]", period_px: float
) -> float:
    """Ratio of the autocorrelation at the chosen period to the window mean.

    Used only for the ``confidence`` field of ``CalibrationResult``.
    """
    ac = _autocorrelation(signal)
    peak_lag = int(round(period_px))
    if peak_lag <= 0 or peak_lag >= ac.size:
        return 0.0
    search_lo = max(2, peak_lag // 4)
    search_hi = min(ac.size, peak_lag * 4 + 1)
    window = ac[search_lo:search_hi]
    if window.size == 0:
        return 0.0
    baseline = float(np.mean(np.abs(window))) or 1e-6
    return float(ac[peak_lag]) / baseline


def calibrate(
    image: "NDArray[np.uint8]" | None = None,
    *,
    roi: tuple[int, int, int, int] | None = None,
    manual_span: tuple[tuple[float, float], tuple[float, float], float] | None = None,
    min_confidence: float = 0.3,
) -> CalibrationResult:
    """High-level entry point used by the pipeline.

    Tries automatic detection first (if ``image`` is provided), falls back to
    the manual span if auto-detect fails or the confidence is below
    ``min_confidence``. At least one of ``image`` or ``manual_span`` must be
    supplied.
    """
    if image is None and manual_span is None:
        raise ValueError("calibrate() needs either image or manual_span")

    auto_error: str | None = None
    if image is not None:
        try:
            result = detect_ruler_scale(image, roi=roi)
        except RuntimeError as exc:
            auto_error = str(exc)
        else:
            if result.confidence >= min_confidence:
                return result
            auto_error = (
                f"auto confidence {result.confidence:.2f} below threshold "
                f"{min_confidence}"
            )

    if manual_span is None:
        raise RuntimeError(
            f"Auto-calibration failed and no manual span provided: {auto_error}"
        )
    a, b, known_mm = manual_span
    result = scale_from_known_span(a, b, known_mm)
    if auto_error:
        result = CalibrationResult(
            px_per_mm=result.px_per_mm,
            method=result.method,
            confidence=result.confidence,
            roi=result.roi,
            notes=f"{result.notes}; auto fallback reason: {auto_error}",
        )
    return result
