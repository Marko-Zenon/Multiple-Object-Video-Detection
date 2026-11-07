"""Tests for movd/data.py (CH-02). Dataset-dependent tests are skipped when the dataset is missing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from movd import data

HAS_DATASET = (data.DATASET_ROOT / "val" / "annotations_coco.json").exists()
needs_dataset = pytest.mark.skipif(not HAS_DATASET, reason="dataset not extracted at DATASET_ROOT")


# ---------------------------------------------------------------- label map


def test_label_map_round_trip():
    assert data.NUM_CLASSES == 8
    assert [data.category_to_class(c) for c in data.CATEGORY_IDS] == list(range(8))
    assert [data.class_to_category(k) for k in range(8)] == list(data.CATEGORY_IDS)
    assert data.CLASS_NAMES[data.category_to_class(3)] == "car"


@pytest.mark.parametrize("bad", [0, 9, -1])
def test_label_map_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        data.category_to_class(bad)


def test_class_to_category_rejects_out_of_range():
    with pytest.raises(ValueError):
        data.class_to_category(8)


def test_visdrone_merge_covers_all_object_classes():
    assert sorted(data.VISDRONE_ORIG_TO_CATEGORY) == list(range(1, 11))
    assert set(data.VISDRONE_ORIG_TO_CATEGORY.values()) == set(data.CATEGORY_IDS)


# ---------------------------------------------------------------- ignore loaders


def test_read_ignore_file_converts_normalised_cxcywh(tmp_path):
    path = tmp_path / "0000001.txt"
    path.write_text("0.5 0.5 0.2 0.4\n\n0.1 0.1 0.2 0.2\n")
    boxes = data.read_ignore_file(path, width=100, height=50)
    assert np.allclose(boxes, [[40, 15, 20, 20], [0, 0, 20, 10]])


def test_read_ignore_file_missing_means_no_regions(tmp_path):
    assert data.read_ignore_file(tmp_path / "none.txt", 100, 100) == []


def test_read_ignore_file_rejects_malformed_line(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 0.5 0.5 0.2 0.4\n")
    with pytest.raises(ValueError):
        data.read_ignore_file(path, 100, 100)


def _toy_coco():
    region = [10, 10, 20, 20]
    anns = [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [50, 50, 10, 10], "iscrowd": 0}]
    anns += [{"id": 10 + c, "image_id": 1, "category_id": c, "bbox": region, "iscrowd": 1, "ignore": 1}
             for c in data.CATEGORY_IDS]
    images = [{"id": 1, "file_name": "images/a/0000001.jpg", "width": 100, "height": 100, "video": "a", "frame_id": 1},
              {"id": 2, "file_name": "images/a/0000002.jpg", "width": 100, "height": 100, "video": "a", "frame_id": 2}]
    return {"images": images, "annotations": anns, "videos": [{"id": 1, "name": "a"}]}


def test_split_annotations_collapses_per_category_crowd_copies():
    index = data.split_annotations(_toy_coco())
    assert [a["id"] for a in index[1]["targets"]] == [1]
    assert index[1]["ignore"] == [[10, 10, 20, 20]]
    assert index[2] == {"targets": [], "ignore": []}


# ---------------------------------------------------------------- masking (training policy)


def test_mask_fills_ignore_region_but_keeps_targets():
    image = np.full((50, 60, 3), 7, dtype=np.uint8)
    out = data.mask_ignore_regions(image, ignore_boxes=[[10, 10, 20, 20]], keep_boxes=[[25, 25, 10, 10]])
    fill = np.array(data.IGNORE_FILL_RGB, dtype=np.uint8)
    assert (out[12, 12] == fill).all()          # inside the ignore region
    assert (out[27, 27] == 7).all()             # inside a target that overlaps the region
    assert (out[45, 55] == 7).all()             # outside everything
    assert (image == 7).all()                   # the input is not modified
    assert data.ignore_mask(50, 60, [[10, 10, 20, 20]]).sum() == 400


def test_mask_clips_boxes_to_image_and_rounds_outwards():
    mask = data.ignore_mask(10, 10, [[-5, 8.5, 20, 5]])
    assert mask[8:, :].all() and not mask[:8, :].any()


def test_target_arrays_clips_and_drops_degenerate():
    anns = [{"bbox": [-5, 0, 10, 10], "category_id": 1},
            {"bbox": [95, 95, 10, 10], "category_id": 8},
            {"bbox": [100, 10, 5, 5], "category_id": 2}]
    boxes, labels = data.target_arrays(anns, width=100, height=100)
    assert np.allclose(boxes, [[0, 0, 5, 10], [95, 95, 100, 100]])
    assert labels.tolist() == [1, 8] and labels.dtype == np.int64


# ---------------------------------------------------------------- frame subsampling


def _clip_images(n_clips=3, n_frames=20):
    return [{"id": c * 100 + f, "video": f"clip{c}", "frame_id": f}
            for c in range(n_clips) for f in range(n_frames, 0, -1)]


def test_evenly_spaced_per_clip_is_deterministic_and_stratified():
    images = _clip_images()
    a = data.evenly_spaced_per_clip(images, 5)
    assert a == data.evenly_spaced_per_clip(list(reversed(images)), 5)
    assert len(a) == 15
    assert a[:5] == [1, 5, 10, 15, 20]          # first and last frame of each clip included
    assert len(data.evenly_spaced_per_clip(images, 50)) == 60   # never more than a clip has


def test_every_kth_per_clip():
    assert data.every_kth_per_clip(_clip_images(1, 9), 4) == [1, 5, 9]


# ---------------------------------------------------------------- committed protocol and splits


def test_protocol_matches_module_constants():
    protocol = data.load_protocol()
    assert protocol["seed"] == data.SEED == 42
    assert protocol["label_map"]["names"] == list(data.CLASS_NAMES)
    assert protocol["label_map"]["category_ids"] == list(data.CATEGORY_IDS)
    assert protocol["preprocessing"]["ignore_fill_rgb"] == list(data.IGNORE_FILL_RGB)
    budget = protocol["rubric_budget"]
    assert budget["images_seen"] == budget["epochs"] * protocol["splits"]["train_rubric"]["num_images"]
    assert budget["input_long_side"] % 32 == 0


@pytest.mark.parametrize("name", data.FRAME_SPLITS)
def test_split_files_are_consistent(name):
    split = data.load_split(name)
    ids = split["image_ids"]
    assert len(ids) == len(set(ids)) == split["num_images"] == len(split["file_names"])
    assert split["num_images"] == data.load_protocol()["splits"][name]["num_images"]
    assert sum(split["frames_per_clip"].values()) == split["num_images"]


def test_train_and_val_splits_share_no_clip():
    train = set(data.load_split("train_rubric")["frames_per_clip"])
    val = set(data.load_split("val_full")["frames_per_clip"])
    assert len(train) == 56 and len(val) == 7
    assert not train & val
    assert set(data.load_split("val_fast")["frames_per_clip"]) == val


def test_val_fast_is_subset_of_val_full():
    assert set(data.split_image_ids("val_fast")) <= set(data.split_image_ids("val_full"))


def test_unknown_split_name_rejected():
    with pytest.raises(ValueError):
        data.load_split("test")


# ---------------------------------------------------------------- real dataset


@needs_dataset
def test_splits_regenerate_identically():
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "scripts/make_splits.py", "--check"],
                            cwd=data.REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_dataset
def test_split_ids_exist_and_file_names_match():
    for name, source in [("train_rubric", "train"), ("val_full", "val")]:
        images = {im["id"]: im for im in data.load_coco(source)["images"]}
        split = data.load_split(name)
        for image_id, file_name in zip(split["image_ids"], split["file_names"]):
            assert images[image_id]["file_name"] == file_name


@needs_dataset
def test_no_clip_overlap_in_coco_files():
    assert data.clip_overlap(data.load_coco("train"), data.load_coco("val")) == set()


@needs_dataset
def test_coco_crowd_regions_equal_ignore_files_on_val():
    coco = data.load_coco("val")
    index = data.split_annotations(coco)
    for image in coco["images"][::25]:
        from_txt = data.read_ignore_file(data.ignore_path("val", image), image["width"], image["height"])
        from_coco = index[image["id"]]["ignore"]
        assert len(from_txt) == len(from_coco)
        for a, b in zip(sorted(from_coco), sorted(from_txt)):
            assert np.allclose(a, b, atol=1.0)


@needs_dataset
def test_crowd_annotations_replicated_for_every_category():
    coco = data.load_coco("val")
    cats_per_region = {}
    for ann in coco["annotations"]:
        if ann["iscrowd"]:
            assert ann["ignore"] == 1
            cats_per_region.setdefault((ann["image_id"], tuple(ann["bbox"])), set()).add(ann["category_id"])
    assert cats_per_region
    assert all(c == set(data.CATEGORY_IDS) for c in cats_per_region.values())


@needs_dataset
def test_yolo_labels_use_category_minus_one():
    coco = data.load_coco("val")
    image = coco["images"][0]
    targets = data.split_annotations(coco)[image["id"]]["targets"]
    file_name = data.image_path("val", image)
    label_file = file_name.parents[2] / "labels" / file_name.parent.name / f"{file_name.stem}.txt"
    lines = [line.split() for line in label_file.read_text().splitlines() if line.strip()]
    assert [int(line[0]) for line in lines] == [data.category_to_class(a["category_id"]) for a in targets]


@needs_dataset
def test_training_sample_masks_ignore_pixels():
    coco = data.load_coco("val")
    index = data.split_annotations(coco)
    image = next(im for im in coco["images"] if index[im["id"]]["ignore"])
    entry = index[image["id"]]
    rgb, boxes, labels = data.training_sample("val", image, entry)
    assert rgb.shape == (image["height"], image["width"], 3)
    assert len(boxes) == len(labels) == len(entry["targets"])
    mask = data.ignore_mask(rgb.shape[0], rgb.shape[1], entry["ignore"], [a["bbox"] for a in entry["targets"]])
    assert mask.any()
    assert (rgb[mask] == np.array(data.IGNORE_FILL_RGB, dtype=np.uint8)).all()


@needs_dataset
def test_test_images_mapping():
    rows = data.load_test_images()
    assert len(rows) == 3811
    assert [r["id"] for r in rows] == list(range(1, 3812))
    assert len({r["image_key"] for r in rows}) == 3811
    assert data.test_image_path(rows[0]).exists()
    assert json.loads(json.dumps(rows[0]))["image_key"] == "clip_001_0000001"
