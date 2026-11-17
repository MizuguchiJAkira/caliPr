"""Batch identification from the command line, against the batch page's own answer.

The study of the batch page's tests: sixteen hand-labelled reference fish of two
made-up species and a photograph of three, with the model stood in for. The
command line gives the scores the page gives -- with a fish's points put right, a
scale set by hand and the lot's drainage looked up -- landmarks a photograph only
where Auto-label is installed, leaves each reference fish out in turn, and writes
the command that reproduces it beside what it writes.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_batch_routes import (  # noqa: E402,F401
    GROUPS, SCRIPTS, _FakeDrainageTool, _get, _load, _post, shape, srv,
)


@pytest.fixture
def cli():
    return _load("batch_identify", SCRIPTS / "batch_identify.py")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_command_line_gives_the_page_s_scores(srv, cli, tmp_path, monkeypatch):
    url, st, calls, truth = srv
    import label_server as ls
    monkeypatch.setattr(ls, "_drainage_tool", lambda: _FakeDrainageTool)
    _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    # fish 2 put right by hand, a ruler set by hand, the drainage looked up
    _post(url, "/api/batch/correct?dataset=shiners", {"photo": "tray.jpg", "n": 2})
    rng = np.random.default_rng(3)
    (st.parent / "shiners_batch" / "sidecars" / "tray_fish_2.json").write_text(json.dumps({
        "fish_id": "tray_fish_2", "lateral": {"keypoints": shape(rng, at=(10, 70), **GROUPS["Notropis_beta"])}}))
    _post(url, "/api/batch/scale?dataset=shiners",
          {"photo": "tray.jpg", "point_a": [0, 0], "point_b": [80, 0], "known_mm": 10})
    _post(url, "/api/batch/drainage?dataset=shiners", {"photo": "tray.jpg", "lat": 42.5, "lon": -76.9})
    code, page = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    assert code == 200 and page["fish"][1]["checked"] and page["scale_from"] == "hand"
    assert page["drainage"]["huc8"] == "04140201"

    out = tmp_path / "out"
    assert cli.main(["--study", str(st), "--out", str(out)]) == 0
    doc = json.loads((out / "batch_scores.json").read_text())
    assert doc["not_landmarked"] == [] and len(doc["photos"]) == 1
    mine = doc["photos"][0]
    assert mine.pop("landmarked_with")["model"] == "test-model"
    page.pop("cached")
    assert mine == page                                    # every score, ranking and note
    assert calls["detect"] == 1                            # scored as kept, never landmarked again

    rows = _rows(out / "batch_scores.csv")
    assert [int(r["fish"]) for r in rows] == [1, 2, 3] and {r["photo"] for r in rows} == {"tray.jpg"}
    for r, f in zip(rows, page["fish"]):
        m = f["match"]
        assert r["best_species"] == m["matches"][0]["species"]
        assert float(r["best_score"]) == m["matches"][0]["score"]
        assert r["second_species"] == m["matches"][1]["species"]
        assert float(r["sl_mm"]) == f["sl_mm"]
        assert r["checked"] == str(f["checked"]) and r["drainage_ranked"] == "True"
        assert r["no_match"] == str(m["no_match"])
    assert [r["checked"] for r in rows] == ["False", "True", "False"]


def test_photographs_are_landmarked_only_where_auto_label_is_installed(srv, cli, tmp_path, monkeypatch, capsys):
    url, st, calls, _ = srv
    import label_server as ls
    out = tmp_path / "out"
    cli.main(["--study", str(st), "--out", str(out)])
    assert json.loads((out / "batch_scores.json").read_text())["not_landmarked"] == ["tray.jpg"]
    assert "--detect" in capsys.readouterr().out

    monkeypatch.setattr(ls, "autolabel_installed", lambda: False)
    cli.main(["--study", str(st), "--out", str(out), "--detect"])
    assert "Auto-label is not installed" in capsys.readouterr().out
    assert calls["detect"] == 0 and not (st / "batch_cache").exists()

    monkeypatch.setattr(ls, "autolabel_installed", lambda: True)
    cli.main(["--study", str(st), "--out", str(out), "--detect"])
    doc = json.loads((out / "batch_scores.json").read_text())
    assert calls["detect"] == 1 and len(doc["photos"][0]["fish"]) == 3
    assert all(lms and "dentary_anterior" in lms for _, lms in calls["predict"])   # the study's names
    # landmarked as the page would: the page takes them as its own, unchanged
    code, page = _get(url, "/api/batch/fish/tray.jpg?dataset=shiners&cache=only")
    assert page["cached"] is True and calls["detect"] == 1
    assert [f["keypoints"] for f in page["fish"]] == [f["keypoints"] for f in doc["photos"][0]["fish"]]


def test_leave_one_out_on_a_small_reference(srv, cli, tmp_path):
    url, st, *_ = srv
    out = tmp_path / "loo"
    assert cli.main(["loo", "--reference", str(st), "--out", str(out), "--axes", "4", "8"]) == 0
    doc = json.loads((out / "loo.json").read_text())
    assert doc["reference_fish"] == 16 and [r["axes"] for r in doc["runs"]] == [4, 8]
    for run in doc["runs"]:
        s = run["summary"]
        assert s["overall"]["fish"] == 16 and s["not_tested"] == []
        assert set(s["species"]) == {"Notropis_alpha", "Notropis_beta"}
        assert s["overall"]["top2"] == 16                  # two species: always in the first two
        assert s["overall"]["top1"] >= 14                  # and they differ plainly in shape
        assert s["overall"]["own_scored"] == 16
    rows = _rows(out / "loo.csv")
    assert len(rows) == 32 and {r["fish_id"] for r in rows} == {f"Notropis_{s}_{i}" for s in ("alpha", "beta")
                                                                 for i in range(8)}
    # each fish is scored without itself: its own species then has seven
    one = next(r for r in doc["runs"][1]["fish"] if r["fish_id"] == "Notropis_alpha_0")
    assert one["rank"] in (1, 2) and one["own_score"] is not None


def test_a_species_with_too_few_others_is_not_tested(srv, cli, tmp_path):
    url, st, *_ = srv
    for i in range(3, 8):                                  # three alpha left: two without the one out
        (st / "sidecars" / f"Notropis_alpha_{i}.json").unlink()
    out = tmp_path / "loo"
    cli.main(["loo", "--reference", str(st), "--out", str(out)])
    s = json.loads((out / "loo.json").read_text())["runs"][0]["summary"]
    assert s["not_tested"] == ["Notropis_alpha"] and s["overall"]["fish"] == 8


def test_a_rerun_record_is_written(srv, cli, tmp_path):
    url, st, *_ = srv
    _get(url, "/api/batch/fish/tray.jpg?dataset=shiners")
    out = tmp_path / "out"
    cli.main(["--study", str(st), "--out", str(out)])
    rerun = (out / "RERUN.txt").read_text()
    assert f"python scripts/batch_identify.py --study {st} --out {out}" in rerun
    assert "caliPr" in rerun and "Python" in rerun
    cli.main(["loo", "--reference", str(st), "--out", str(tmp_path / "loo")])
    assert f"scripts/batch_identify.py loo --reference {st}" in (tmp_path / "loo" / "RERUN.txt").read_text()
