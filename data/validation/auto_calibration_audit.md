# Auto-calibration audit on iDigBio brook trout pool

Every image was run through `detect_ruler_scale` with no ROI hint. `ok` means the detector returned a result at or above the 0.3 confidence threshold the pipeline's `calibrate()` helper uses to decide whether to fall back to manual. `low_confidence` means a number came back but the pipeline would reject it.

## Overall

- Total images: **41**
- `ok`: **2** (5%)
- `low_confidence`: **2** (5%)
- `detect_failed`: **37** (90%)
- `image_read_failed`: **0** (0%)

## By institution

| Institution | n | ok | low_conf | detect_fail | read_fail |
|---|---:|---:|---:|---:|---:|
| CMN | 4 | 0 | 1 | 3 | 0 |
| INHS | 1 | 0 | 0 | 1 | 0 |
| MCZ | 1 | 0 | 0 | 1 | 0 |
| NEON | 8 | 2 | 0 | 6 | 0 |
| NHMUK | 3 | 0 | 1 | 2 | 0 |
| UCMP | 1 | 0 | 0 | 1 | 0 |
| USNM | 18 | 0 | 0 | 18 | 0 |
| YPM | 5 | 0 | 0 | 5 | 0 |

## `ok` px/mm distribution

- min: 37.065
- median: 37.065
- max: 37.065

## Per-image failures (not `ok`)

| File | Institution | Outcome | Conf | Notes |
|---|---|---|---:|---|
| CMN_CMNFI_1981-0881.1_98d57bf2.jpg | CMN | `low_confidence` | 0.21 | autocorrelation period=29.693 px, prominence=1.7 |
| CMN_CMNFI_1989-0225.2_0eb6f828.jpg | CMN | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| CMN_CMNFI_1989-0225.2_c82ae90f.jpg | CMN | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| CMN_CMNFI_2005-0048.1_79236b1e.jpg | CMN | `detect_failed` |  | No periodic tick signal detected. |
| INHS_62305_f39e9dba.jpg | INHS | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| MCZ_99407_702ff616.jpg | MCZ | `detect_failed` |  | No periodic tick signal detected. |
| NEON_NEON054P2_663b9d32.jpg | NEON | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| NEON_NEON054P2_c8df7fd8.jpg | NEON | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| NEON_NEON054P3_7675d274.jpg | NEON | `detect_failed` |  | No periodic tick signal detected. |
| NEON_NEON054P3_c9e31208.jpg | NEON | `detect_failed` |  | No periodic tick signal detected. |
| NEON_NEON06X4A_41a0eca5.jpg | NEON | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| NEON_NEON06X4A_bb335ece.jpg | NEON | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| NHMUK_1936.8.15.1_909ea656.jpg | NHMUK | `detect_failed` |  | No periodic tick signal detected. |
| NHMUK_2021.8.31.135_37f8ba64.jpg | NHMUK | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| NHMUK_2021.8.31.135_6fcc4c9a.jpg | NHMUK | `low_confidence` | 0.23 | autocorrelation period=22.499 px, prominence=1.9 |
| UCMP_140673_d01896ae.jpeg | UCMP | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_110145_23e6932f.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_110339_0126e263.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_110343_4d561d6f.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_110343_80d98122.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_110355_230c4456.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_15470_449003e2.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_15758_4ae7e02e.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_20950_91f56467.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_259555_592c01af.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_259556_c9d79275.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_283652_238fadcb.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_283652_2ba2aa25.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_283658_ff9f072e.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_283659_031eec8b.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_283659_2b381b16.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_283660_3d1d6db3.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_283660_5ef26d12.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| USNM_USNM_39933_77f6ace4.jpg | USNM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| YPM_YPM_ICH_011792_7c558858.jpg | YPM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| YPM_YPM_ICH_011793_b04a34ff.jpg | YPM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| YPM_YPM_ICH_012073_f1a034f4.jpg | YPM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| YPM_YPM_ICH_025874_04a6c11a.jpg | YPM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
| YPM_YPM_ICH_029984_35f7f9df.jpg | YPM | `detect_failed` |  | No ruler-like region found (no elongated high-contrast object). Try providing... |
