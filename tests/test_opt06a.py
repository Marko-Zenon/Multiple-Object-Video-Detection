"""OPT-06A: the c4 candidate is frozen to the OPT-02 grid entry, and the competition gate constants are as approved."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from movd import ensemble

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def opt06a():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("opt06a_c4", SCRIPTS / "opt06a_c4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_is_the_frozen_opt02_c4(opt06a):
    cand = opt06a.candidate()
    assert cand["id"] == "c4_w075_nms06"
    assert (cand["w_A"], cand["w_B"], cand["iou"], cand["route"]) == (
        1.0,
        0.75,
        0.6,
        None,
    )
    assert opt06a.o2.candidates()["max_det"] == 500


def test_gate_constants(opt06a):
    assert opt06a.GATE == {
        "min_mAP": 0.2840,
        "max_AP_small_drop": 0.005,
        "max_density_ge100_drop": 0.010,
    }


def test_c4_merge_downweights_f_and_uses_nms_06():
    def rec(cat, score, x):
        return {
            "image_id": 1,
            "category_id": cat,
            "bbox": [x, 0.0, 10.0, 10.0],
            "score": score,
        }

    a = [rec(1, 0.5, 0.0)]
    f = [
        rec(1, 0.6, 1.0),
        rec(1, 0.4, 3.0),
    ]  # IoU with A: 9/11 = 0.82 (suppressed), 7/13 = 0.54 (kept at 0.6)
    out = ensemble.merge_frame([a, f], [1.0, 0.75], iou=0.6)
    assert out[0]["score"] == 0.5
    assert [r["bbox"][0] for r in out] == [0.0, 3.0]
    assert out[1]["score"] == pytest.approx(0.3)
