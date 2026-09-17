"""A downloaded model installs where the labeler looks, or not at all."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_model as F  # noqa: E402
import package_model as P  # noqa: E402

PROJECT = "jcalipr-jcalipr-2026-09-16"
TRAIN = "dlc-models-pytorch/iteration-0/jcaliprSep16-trainset81shuffle1/train"
CONFIG = ("Task: jcalipr\n# Project path (change when moving around)\nproject_path:\n"
          "  /Users/someone/JCalipr/dlc_project_frontal/jcalipr-jcalipr-2026-09-16\n"
          "pose_config_path:\nengine: pytorch\nbodyparts:\n- mouth_left\n- mouth_right\n"
          "start: 0.0\n")
PYCFG = (f"net_type: resnet_50\nmetadata:\n  project_path: /Users/someone/x\n"
         f"  pose_config_path: /Users/someone/x/{TRAIN}/pytorch_config.yaml\n  bodyparts:\n"
         f"  - mouth_left\n")


def _package(tmp: Path, extra: dict[str, bytes] | None = None) -> tuple[Path, dict]:
    z = tmp / "model.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(f"{PROJECT}/config.yaml", P.set_paths(CONFIG, {"project_path": P.PLACEHOLDER}))
        f.writestr(f"{PROJECT}/{TRAIN}/pytorch_config.yaml",
                   P.set_paths(PYCFG, {k: P.PLACEHOLDER for k in P._PATH_KEYS}))
        f.writestr(f"{PROJECT}/{TRAIN}/snapshot-200.pt", b"weights")
        for name, body in (extra or {}).items():
            f.writestr(name, body)
    data = z.read_bytes()
    return z, {"file": "model.zip", "url": str(z), "sha256": hashlib.sha256(data).hexdigest(),
               "bytes": len(data), "folder": "dlc_project_frontal", "project": PROJECT,
               "snapshot": "snapshot-200.pt"}


def test_paths_are_set_whether_on_the_line_or_folded_below_it():
    out = F.set_paths(CONFIG, {"project_path": "/a b/it's here"})
    assert "project_path: '/a b/it''s here'\n" in out          # quoted: spaces and quotes survive
    assert "/Users/someone" not in out
    assert "pose_config_path:\nengine: pytorch" in out          # an empty key stays empty
    assert out.endswith("start: 0.0\n")                         # nothing after is disturbed


def test_a_model_installs_where_the_labeler_looks(tmp_path):
    _, entry = _package(tmp_path)
    root = tmp_path / "repo"
    assert F.install("frontal", entry, root)
    dest = root / "dlc_project_frontal" / PROJECT
    assert (dest / TRAIN / "snapshot-200.pt").read_bytes() == b"weights"
    assert f"project_path: '{dest.resolve()}'" in (dest / "config.yaml").read_text()
    cfg = (dest / TRAIN / "pytorch_config.yaml").read_text()
    assert f"pose_config_path: '{dest.resolve() / TRAIN / 'pytorch_config.yaml'}'" in cfg
    assert P.PLACEHOLDER not in cfg
    assert [p.name for p in (root / "dlc_project_frontal").iterdir()] == [PROJECT]  # no temp left


def test_a_download_that_does_not_match_its_checksum_is_refused(tmp_path, capsys):
    _, entry = _package(tmp_path)
    entry["sha256"] = "0" * 64
    root = tmp_path / "repo"
    assert not F.install("frontal", entry, root)
    assert "REFUSED" in capsys.readouterr().out
    assert list((root / "dlc_project_frontal").iterdir()) == []


def test_an_archive_reaching_outside_the_model_folder_is_refused(tmp_path, capsys):
    _, entry = _package(tmp_path, {"../../escape.txt": b"x"})
    root = tmp_path / "repo"
    assert not F.install("frontal", entry, root)
    assert "outside the model folder" in capsys.readouterr().out
    assert not (tmp_path / "escape.txt").exists()
    assert list((root / "dlc_project_frontal").iterdir()) == []


def test_an_existing_model_is_never_replaced(tmp_path, capsys):
    """It may be one trained on this machine."""
    _, entry = _package(tmp_path)
    mine = tmp_path / "repo" / "dlc_project_frontal" / PROJECT
    mine.mkdir(parents=True)
    (mine / "config.yaml").write_text("mine")
    assert F.install("frontal", entry, tmp_path / "repo")
    assert "left as it is" in capsys.readouterr().out
    assert (mine / "config.yaml").read_text() == "mine"


def test_bodyparts_are_read_from_the_project_config():
    assert P.bodyparts(CONFIG) == ["mouth_left", "mouth_right"]


def test_the_published_manifest_matches_the_scripts(tmp_path):
    manifest = Path(F.MANIFEST)
    if not manifest.is_file():
        return
    models = json.loads(manifest.read_text())["models"]
    for view, m in models.items():
        assert m["folder"] == {"lateral": "dlc_project", "frontal": "dlc_project_frontal"}[view]
        assert m["url"].endswith("/" + m["file"]) and len(m["sha256"]) == 64
