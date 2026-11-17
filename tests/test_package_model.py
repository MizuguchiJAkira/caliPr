"""A packaged model must not carry paths from the machine it was trained on."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import package_model as pm  # noqa: E402


CONFIG = """net_type: resnet_50
runner:
  resume_training_from: /Users/someone/Desktop/JCalipr/dlc_project_union/run/snapshot-best-220.pt
  snapshots:
    max_snapshots: 5
train_settings:
  weights_init:
    dataset: /home/someone/data/superanimal
  folded_path:
    /Users/someone/Desktop/a/very/long/path/that/deeplabcut/folded/onto/the/next/line.pt
  epochs: 120
  note: keep me
"""


def test_absolute_paths_are_cleared_wherever_they_sit():
    out = pm.strip_local_paths(CONFIG)
    assert "/Users/" not in out and "/home/" not in out
    assert "  resume_training_from: null" in out
    assert "    dataset: null" in out
    assert "  folded_path: null" in out
    assert "epochs: 120" in out and "note: keep me" in out and "max_snapshots: 5" in out


def test_a_home_path_is_detected():
    assert pm.mentions_this_machine("resume: /Users/anyone/Desktop/x.pt\n")
    assert pm.mentions_this_machine(f"x: {Path.home()}/y\n")
    assert not pm.mentions_this_machine("project_path: 'SET-ON-INSTALL'\n")


def test_the_card_states_the_licence_and_points_to_the_credits():
    out = pm.card("lateral", Path("p"), Path("s.pt"), 0.25, ["a", "b"], "CC BY 4.0", credits=True)
    assert "Licence for these weights: CC BY 4.0." in out
    assert "CREDITS.md" in out
    assert "as stated on the release page" not in out
    plain = pm.union_card(Path("p"), Path("s.pt"), ["a"], {"fish_train": [1], "groups": {}},
                          "CC BY-NC 4.0")
    assert "Licence for these weights: CC BY-NC 4.0." in plain and "CREDITS.md" not in plain
