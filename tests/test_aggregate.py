"""Tests for CH-12: scripts/aggregate_results.py, the selection record and configs/runs/final.json (CPU only)."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json

import pytest

from movd import data, ultra

OUT = data.REPO_ROOT / "results" / "ch-12-comparison"
FINAL = data.REPO_ROOT / "configs" / "runs" / "final.json"


def aggregate():
    spec = importlib.util.spec_from_file_location(
        "aggregate_results", data.REPO_ROOT / "scripts" / "aggregate_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_run_has_the_same_metrics_and_a_verdict():
    rows, compat = aggregate().summary_rows()
    assert len(rows) == len(compat) == 14
    families = {r["family"] for r in rows}
    assert len(families) == 4
    for row in rows:
        assert all(isinstance(row[f"val_full_{k}"], float) for k in aggregate().KEYS)
    excluded = {r["run"]: r["failed_checks"] for r in rows if not r["comparable"]}
    assert excluded == {
        "yolo_imgsz1600": "long_side_1344",
        "rtdetr_lr1e4": "single_variable_clean",
    }


def test_selection_follows_the_rule():
    module = aggregate()
    rows, _ = module.summary_rows()
    chosen, checks = module.selection(rows)
    assert chosen["finalist_run"] == "rtdetr_baseline" and not chosen["tie"]
    assert chosen["lead_mAP"] > chosen["run_to_run_spread_val_full_mAP"]
    assert all(c["leader_ahead"] for c in checks)
    stored = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(chosen))


def test_summary_csv_is_current():
    rows, _ = aggregate().summary_rows()
    with open(data.REPO_ROOT / "results" / "summary.csv", encoding="utf-8") as f:
        stored = list(csv.DictReader(f))
    assert [r["run"] for r in stored] == [r["run"] for r in rows]
    for old, new in zip(stored, rows):
        assert float(old["val_full_mAP"]) == new["val_full_mAP"]


def test_final_config_is_a_valid_run_config_of_the_finalist():
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    config = ultra.validate_run_config(final["run_config"])
    baseline = json.loads(
        (data.REPO_ROOT / "configs" / "runs" / "rtdetr_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    for key in (
        "family",
        "weights",
        "epochs",
        "imgsz",
        "batch_size",
        "workers",
        "amp",
        "overrides",
    ):
        assert config[key] == baseline[key]
    assert final["train_split"]["num_images"] * config["epochs"] == 36432
    assert final["expected_runtime"]["total_hours_with_buffer"] < 10


@pytest.mark.skipif(
    not data.coco_path("train").exists(), reason="dataset not extracted"
)
def test_train_final_split_matches_the_frozen_hash():
    final = json.loads(FINAL.read_text(encoding="utf-8"))["train_split"]
    ids = data.every_kth_per_clip(data.load_coco("train")["images"], 4)
    assert len(ids) == final["num_images"]
    assert (
        hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        == final["image_ids_sha256"]
    )
    assert (
        len(set(ids) & set(data.split_image_ids("train_rubric")))
        == final["overlap_with_train_rubric"]
    )
