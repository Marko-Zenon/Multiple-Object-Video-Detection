"""CH-14: the final run trains on train_final (every 4th train frame), never silently on train_rubric.

Covers the frozen train_final split, the CH-14 run configs, `movd.ultra.train_manifest` (the explicit,
checked choice of the training list) and `scripts/train_ultra.py` failing before training on a mismatch.
CPU only; the frame-rule test needs the dataset and is skipped without it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from movd import data, ultra

RUNS = data.REPO_ROOT / "configs" / "runs"
FINAL = json.loads((RUNS / "final.json").read_text(encoding="utf-8"))
HAS_DATASET = (data.DATASET_ROOT / "train" / "annotations_coco.json").exists()
needs_dataset = pytest.mark.skipif(
    not HAS_DATASET, reason="dataset not extracted at DATASET_ROOT"
)


def _split(name: str) -> dict:
    return json.loads((data.SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the frozen split


def test_train_final_split_matches_final_json():
    split = _split("train_final")
    ids = ultra.train_split_ids("train_final")
    assert (
        split["num_images"]
        == len(ids)
        == len(set(ids))
        == FINAL["train_split"]["num_images"]
        == 6072
    )
    assert split["num_clips"] == 56 and len(split["frames_per_clip"]) == 56
    # The id-list hash CH-12 recorded in final.json, in movd.data.every_kth_per_clip order.
    assert (
        hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        == FINAL["train_split"]["image_ids_sha256"]
    )
    assert (
        len(set(ids) & set(ultra.train_split_ids("train_rubric")))
        == FINAL["train_split"]["overlap_with_train_rubric"]
    )


def test_train_final_has_no_val_or_test_frames():
    split = _split("train_final")
    assert split["source_split"] == "train"
    clips = {Path(f).parent.name for f in split["file_names"]}
    val_clips = {Path(f).parent.name for f in _split("val_full")["file_names"]}
    assert (
        clips == set(_split("train_rubric")["frames_per_clip"]) and len(val_clips) == 7
    )
    assert not clips & val_clips
    if (data.DATASET_ROOT / "test_images.csv").exists():
        test_clips = {im["clip"] for im in data.load_test_images()}
        assert len(test_clips) == 48 and not clips & test_clips


@needs_dataset
def test_train_final_is_every_fourth_frame_of_every_clip():
    coco = data.load_coco("train")
    by_clip: dict[str, list[dict]] = {}
    for image in coco["images"]:
        by_clip.setdefault(data.clip_name(image), []).append(image)
    expected = []
    for clip in sorted(by_clip):
        frames = sorted(by_clip[clip], key=lambda im: im["frame_id"])
        expected.extend(im["id"] for im in frames[::4])
    assert ultra.train_split_ids("train_final") == expected
    counts = Counter(data.clip_name(im) for im in coco["images"])
    assert _split("train_final")["frames_per_clip"] == {
        c: -(-n // 4) for c, n in sorted(counts.items())
    }


# ---------------------------------------------------------------- run configs


def test_final_rtdetr_config_is_final_json_plus_explicit_train_split():
    config = ultra.load_run_config(RUNS / "final_rtdetr.json")
    frozen = FINAL["run_config"]
    assert config["train_split"] == "train_final"
    assert {k: v for k, v in config.items() if k not in ("train_split", "note")} == {
        k: v for k, v in frozen.items() if k != "note"
    }
    assert config["note"].startswith(frozen["note"])
    kwargs = ultra.train_kwargs(config, config["data_yaml"], "out", "0")
    assert kwargs["epochs"] * FINAL["train_split"]["num_images"] == 36432
    assert (kwargs["imgsz"], kwargs["batch"], kwargs["amp"], kwargs["seed"]) == (
        1344,
        1,
        True,
        42,
    )
    assert "fraction" not in kwargs


def test_smoke_config_uses_the_same_data_path():
    smoke = ultra.load_run_config(RUNS / "final_rtdetr_smoke.json")
    final = ultra.load_run_config(RUNS / "final_rtdetr.json")
    for key in (
        "family",
        "weights",
        "imgsz",
        "batch_size",
        "amp",
        "data_yaml",
        "train_split",
        "overrides",
    ):
        assert smoke[key] == final[key], key
    assert 40 <= smoke["smoke_fraction"] * 6072 <= 60


def test_existing_configs_stay_on_train_rubric():
    for path in sorted(RUNS.glob("*.json")):
        if not path.stem.startswith(("yolo_", "rtdetr_")):
            continue
        config = ultra.load_run_config(path)
        assert config.get("train_split", ultra.DEFAULT_TRAIN_SPLIT) == "train_rubric", (
            path.name
        )


def test_run_config_rejects_unknown_train_split():
    with pytest.raises(ValueError, match="train_split"):
        ultra.validate_run_config(
            {
                "run_id": "r",
                "weights": "w.pt",
                "batch_size": 1,
                "train_split": "val_full",
            }
        )
    with pytest.raises(ValueError, match="must be str"):
        ultra.validate_run_config(
            {"run_id": "r", "weights": "w.pt", "batch_size": 1, "train_split": 4}
        )


# ---------------------------------------------------------------- train_manifest


def _fake_export(
    root: Path, name: str, yaml_train: str | None = None, changes: dict | None = None
) -> Path:
    """data.yaml + list file + manifest for `name`, built from the frozen split (no images needed)."""
    root.mkdir(parents=True, exist_ok=True)
    split = _split(name)
    lines = [
        "./" + (Path(name) / "images" / Path(f).parent.name / Path(f).name).as_posix()
        for f in split["file_names"]
    ]
    (root / f"{name}.txt").write_text(
        "".join(line + "\n" for line in lines), encoding="utf-8"
    )
    manifest = {
        "name": name,
        "source_split": "train",
        "num_images": len(lines),
        "masked": True,
        **(changes or {}),
    }
    (root / f"{name}_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root.as_posix()}\ntrain: {yaml_train or name + '.txt'}\nval: val_fast.txt\nnc: 8\n",
        encoding="utf-8",
    )
    return yaml


def test_train_manifest_selects_train_final(tmp_path):
    yaml = _fake_export(tmp_path, "train_final")
    config = ultra.load_run_config(RUNS / "final_rtdetr.json")
    manifest = ultra.train_manifest(config, yaml)
    assert manifest["name"] == "train_final" and manifest["num_images"] == 6072


def test_train_manifest_is_backward_compatible(tmp_path):
    yaml = _fake_export(tmp_path, "train_rubric")
    config = ultra.load_run_config(RUNS / "rtdetr_baseline.json")
    assert "train_split" not in config
    assert ultra.train_manifest(config, yaml)["num_images"] == 2800


def test_train_manifest_rejects_train_rubric_export_for_final_run(tmp_path):
    """The CH-12 bug: the final config pointed at an export whose train list is train_rubric."""
    config = ultra.load_run_config(RUNS / "final_rtdetr.json")
    with pytest.raises(ValueError, match="train_rubric.txt"):
        ultra.train_manifest(config, _fake_export(tmp_path / "a", "train_rubric"))
    # A train_final data.yaml with only a train_rubric manifest next to it.
    yaml = _fake_export(tmp_path / "b", "train_rubric", yaml_train="train_final.txt")
    with pytest.raises(FileNotFoundError, match="train_final"):
        ultra.train_manifest(config, yaml)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"name": "train_rubric"}, "not a masked train export"),
        ({"masked": False}, "not a masked train export"),
        ({"source_split": "val"}, "not a masked train export"),
        ({"num_images": 6000}, "frozen split"),
    ],
)
def test_train_manifest_rejects_mismatched_manifest(tmp_path, changes, message):
    yaml = _fake_export(tmp_path, "train_final", changes=changes)
    with pytest.raises(ValueError, match=message):
        ultra.train_manifest(ultra.load_run_config(RUNS / "final_rtdetr.json"), yaml)


def test_train_manifest_rejects_changed_list_file(tmp_path):
    yaml = _fake_export(tmp_path, "train_final")
    lines = (tmp_path / "train_final.txt").read_text(encoding="utf-8").splitlines()
    config = ultra.load_run_config(RUNS / "final_rtdetr.json")
    for bad in (
        lines[:-1] + [lines[0]],
        lines[:-1] + ["./train_final/images/clip_x/0.jpg"],
    ):
        (tmp_path / "train_final.txt").write_text(
            "\n".join(bad) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="differ"):
            ultra.train_manifest(config, yaml)


def test_train_manifest_missing_files(tmp_path):
    config = ultra.load_run_config(RUNS / "final_rtdetr.json")
    with pytest.raises(FileNotFoundError, match="data.yaml"):
        ultra.train_manifest(config, tmp_path / "data.yaml")
    yaml = _fake_export(tmp_path, "train_final")
    (tmp_path / "train_final_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="train_final_manifest.json"):
        ultra.train_manifest(config, yaml)


def test_train_ultra_fails_before_training_on_mismatch(tmp_path):
    yaml = _fake_export(tmp_path / "export", "train_rubric")
    config = {
        **ultra.load_run_config(RUNS / "final_rtdetr.json"),
        "run_id": "ch14_pytest_mismatch",
        "data_yaml": str(yaml),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(data.REPO_ROOT / "scripts" / "train_ultra.py"),
            "--config",
            str(path),
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=data.REPO_ROOT,
        timeout=300,
    )
    assert (
        result.returncode != 0 and "selects train_split 'train_final'" in result.stderr
    )
    assert not (data.REPO_ROOT / "results" / "ch14_pytest_mismatch").exists()
    assert not (data.REPO_ROOT / "outputs" / "ch14_pytest_mismatch").exists()
