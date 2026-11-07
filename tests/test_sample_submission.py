"""Tests for scripts/make_sample_submission.py (CH-02)."""

from __future__ import annotations

import importlib.util

import pytest

from movd import data

spec = importlib.util.spec_from_file_location("make_sample_submission", data.REPO_ROOT / "scripts" / "make_sample_submission.py")
mss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mss)

IDS = ["clip_001_0000001", "clip_001_0000002", "clip_002_0000001"]


def test_written_file_is_valid(tmp_path):
    path = tmp_path / "sample_submission.csv"
    mss.write(path, IDS)
    assert path.read_text(encoding="utf-8") == "image_id,PredictionString\n" + "".join(f"{i},\n" for i in IDS)
    assert mss.validate(path, IDS) == []


@pytest.mark.parametrize("content, problem", [
    ("ImageId,PredictionString\na,\nb,\n", "header"),
    ("image_id,PredictionString\na,\n", "data rows"),
    ("image_id,PredictionString\nb,\na,\n", "order"),
    ("image_id,PredictionString\na,\na,\n", "duplicate"),
    ("image_id,PredictionString\na,\nc,\n", "unknown"),
    ("image_id,PredictionString\na,0 0.5 1 2 3 4\nb,\n", "not empty"),
])
def test_validator_catches_problems(tmp_path, content, problem):
    path = tmp_path / "s.csv"
    path.write_text(content, encoding="utf-8")
    assert any(problem in p for p in mss.validate(path, ["a", "b"]))


@pytest.mark.skipif(not (data.DATASET_ROOT / "sample_submission.csv").exists(), reason="sample not generated")
def test_local_sample_submission_is_valid():
    ids = mss.expected_ids(None)
    assert len(ids) == 3811
    assert mss.validate(data.DATASET_ROOT / "sample_submission.csv", ids) == []
