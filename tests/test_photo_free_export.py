"""The exporters work from the sidecars alone, as a clone of the repository has them.

The photographs are not in the repository, but every sidecar records its
landmarks, outlines, calibration and the size of the photograph it was labelled
on. A reviewer must be able to reproduce the measurements and the landmark
export from those files, so neither exporter may need the photographs unless
the calibration itself has to be read from the ruler's pixels.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

FISH = ["Salvelinus_fontinalis_ASN_2", "Salvelinus_fontinalis_HRN_10", "Salvelinus_fontinalis_LT_14"]


@pytest.fixture()
def study(tmp_path):
    base = tmp_path / "data" / "trout"
    (base / "sidecars").mkdir(parents=True)
    for name in ("schema.json", "groups.csv"):
        src = ROOT / "data" / "cornell" / name
        if src.exists():
            shutil.copy(src, base / name)
    for fid in FISH:
        shutil.copy(ROOT / "data" / "cornell" / "sidecars" / f"{fid}.json", base / "sidecars")
    return tmp_path, base


def test_measurements_export_without_photographs(study):
    import openpyxl
    import export_measurements

    root, base = study
    out = root / "measurements.xlsx"
    assert export_measurements.main(["--dataset", "trout", "--data-root", str(root / "data"),
                                     "--out", str(out)]) == 0
    rows = list(openpyxl.load_workbook(out, read_only=True)["Measurements"].iter_rows(values_only=True))
    header, body = rows[0], rows[1:]
    assert sorted(r[0] for r in body) == sorted(FISH)
    sl = header.index(next(h for h in header if str(h).startswith("SL")))
    assert all(isinstance(r[sl], float) and 50 < r[sl] < 300 for r in body)
    assert all(r[header.index("units")] == "mm" for r in body)
    # The command that reproduces the workbook is written beside it.
    rerun = (out.parent / (out.name + ".rerun.txt")).read_text()
    assert "python scripts/export_measurements.py --dataset trout" in rerun
    assert "caliPr " in rerun and "commit " in rerun


def test_tps_export_without_photographs(study):
    import export_tps

    root, base = study
    out = root / "tps"
    assert export_tps.main(["--sidecars", str(base / "sidecars"), "--images", str(base / "lateral"),
                            "--out", str(out)]) == 0
    tps = (out / "landmarks.tps").read_text()
    assert tps.count("IMAGE=") == len(FISH)
    assert "python scripts/export_tps.py --sidecars" in (out / "RERUN.txt").read_text()
    # y is flipped against the recorded photograph height, as it is with the photo
    sc = json.loads((base / "sidecars" / f"{FISH[0]}.json").read_text())
    h = sc["metadata"]["images"]["lateral"]["height"]
    x, y = sc["lateral"]["keypoints"]["caudal_base"][:2]
    assert f"{float(x):.4f} {h - float(y):.4f}" in tps
