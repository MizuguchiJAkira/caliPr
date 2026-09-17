"""An auto-labelled point nobody reviewed must not train as a hand label."""

import importlib.util
import pathlib

import pytest

pytest.importorskip("pandas")     # the dataset builder is training-stack only

_spec = importlib.util.spec_from_file_location(
    "build_dlc_dataset",
    pathlib.Path(__file__).parent.parent / "scripts/build_dlc_dataset.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def test_unreviewed_reads_both_records():
    d = {"metadata": {"assist": {"unreviewed": ["a"]},
                      "unreviewed_predictions": ["b"]}}
    assert B.unreviewed(d) == {"a", "b"}


def test_accepted_and_corrected_are_not_unreviewed():
    d = {"metadata": {"assist": {"accepted": ["a"], "corrected": {"b": 3.0},
                                 "unreviewed": ["c"]}}}
    assert B.unreviewed(d) == {"c"}


def test_hand_label_with_no_assist_record_is_untouched():
    assert B.unreviewed({"metadata": {"source": "hand-labeled"}}) == set()


def test_each_view_reads_only_its_own_record():
    """A frontal record read as lateral would exclude nothing, and silently."""
    d = {"metadata": {"assist": {"unreviewed": ["pelvic_tip"]},
                      "unreviewed_predictions": ["anal_tip"],
                      "assist_frontal": {"unreviewed": ["mouth_left"],
                                         "accepted": ["mouth_right"]}}}
    assert B.unreviewed(d, "lateral") == {"pelvic_tip", "anal_tip"}
    assert B.unreviewed(d, "frontal") == {"mouth_left"}
    assert B.unreviewed({"metadata": {"assist": {"unreviewed": ["x"]}}}, "frontal") == set()
