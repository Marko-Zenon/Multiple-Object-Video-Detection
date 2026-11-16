"""Tests for the CH-10 YOLO11s-P2 model and its partial weight transfer (movd/yolo_p2.py), and the CH-10 configs.

CPU only. The model tests need the pinned `ultralytics`; the transfer test also needs the git-ignored
`outputs/weights/yolo11s.pt` and is skipped without it.
"""

from __future__ import annotations

import json

import pytest
import torch

from movd import data, ultra, yolo_p2

try:
    import ultralytics  # noqa: F401

    HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    HAS_ULTRALYTICS = False
needs_ultralytics = pytest.mark.skipif(
    not HAS_ULTRALYTICS, reason="ultralytics not installed"
)
PRETRAINED = data.REPO_ROOT / "outputs" / "weights" / "yolo11s.pt"
RUNS = data.REPO_ROOT / "configs" / "runs"
BASELINE = RUNS / "yolo_baseline_marko_3050.json"


def test_map_key_moves_only_the_detect_layer():
    assert yolo_p2.map_key("model.0.conv.weight") == "model.0.conv.weight"
    assert (
        yolo_p2.map_key("model.22.m.0.cv1.conv.weight")
        == "model.22.m.0.cv1.conv.weight"
    )
    assert (
        yolo_p2.map_key("model.23.cv2.0.0.conv.weight")
        == "model.26.cv2.0.0.conv.weight"
    )
    assert yolo_p2.map_key("model.23.dfl.conv.weight") == "model.26.dfl.conv.weight"
    with pytest.raises(ValueError):
        yolo_p2.map_key("model.24.conv.weight")


def test_new_keys_are_exactly_the_p2_branch():
    assert yolo_p2.is_new_key("model.25.cv1.conv.weight")
    assert yolo_p2.is_new_key("model.26.cv2.3.0.conv.weight")
    assert yolo_p2.is_new_key("model.26.cv3.3.2.bias")
    assert not yolo_p2.is_new_key("model.26.cv2.0.0.conv.weight")
    assert not yolo_p2.is_new_key("model.26.dfl.conv.weight")
    assert not yolo_p2.is_new_key("model.16.cv1.conv.weight")


@needs_ultralytics
def test_p2_model_has_stride4_level_and_pretrained_head_widths():
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(yolo_p2.P2_YAML), nc=8, verbose=False)
    head = model.model[-1]
    assert [int(s) for s in head.stride] == [8, 16, 32, 4]
    # P3 first: the hidden widths equal plain yolo11s (128-channel P3), so the pretrained head transfers.
    assert [m[0].conv.in_channels for m in head.cv2] == [128, 256, 512, 64]
    outputs = model.eval()(torch.zeros(1, 3, 128, 128))
    preds = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    anchors = sum((128 // s) ** 2 for s in (8, 16, 32, 4))
    assert preds.shape == (1, 4 + 8, anchors)


@needs_ultralytics
@pytest.mark.skipif(
    not PRETRAINED.exists(), reason="outputs/weights/yolo11s.pt missing"
)
def test_build_p2_init_transfers_every_baseline_tensor(tmp_path):
    from ultralytics.nn.tasks import load_checkpoint

    report = yolo_p2.build_p2_init(PRETRAINED, tmp_path / "init.pt")
    assert report["baseline_tensors"] == report["transferred_tensors"] == 499
    assert (
        report["p2_tensors"] == report["transferred_tensors"] + report["fresh_tensors"]
    )
    assert all(yolo_p2.is_new_key(k) for k in report["fresh_keys"])
    # As the baseline trainer: COCO class rows copied by name (names set before load).
    assert set(report["cls_rows_copied_from_coco_by_name"]) == {
        "person",
        "bicycle",
        "car",
        "truck",
        "bus",
    }
    model, _ = load_checkpoint(str(tmp_path / "init.pt"))
    assert model.names == dict(enumerate(data.CLASS_NAMES))
    # Same seed -> identical checkpoint content (the fresh branch is reproducible).
    again = yolo_p2.build_p2_init(PRETRAINED, tmp_path / "again.pt")
    other, _ = load_checkpoint(str(tmp_path / "again.pt"))
    first, second = model.state_dict(), other.state_dict()
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert again["transferred_tensors"] == 499


@pytest.mark.parametrize("name", ["yolo_imgsz1600", "yolo_p2"])
def test_ch10_variants_change_one_variable(name):
    base = ultra.load_run_config(BASELINE)
    variant = ultra.load_run_config(RUNS / f"{name}.json")
    shared = ("family", "epochs", "imgsz", "batch_size", "workers", "amp")
    assert {k: variant.get(k) for k in shared} == {k: base.get(k) for k in shared}
    base_kwargs = ultra.train_kwargs(base, "d.yaml", "p", "0")
    kwargs = ultra.train_kwargs(variant, "d.yaml", "p", "0")
    changed = {k for k in kwargs if kwargs[k] != base_kwargs.get(k)}
    if name == "yolo_imgsz1600":
        assert changed == {"imgsz"} and kwargs["imgsz"] == 1600
        assert variant["weights"] == base["weights"]
    else:
        assert changed == set()
        assert variant["weights"] == "outputs/weights/yolo11s-p2-init.pt"


def test_p2_transfer_report_is_tracked_and_consistent():
    path = data.REPO_ROOT / "results" / "ch-10-yolo-tuning" / "p2_transfer.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["transferred_tensors"] == report["baseline_tensors"] == 499
    assert report["detect_strides"] == [8, 16, 32, 4]
    assert report["pretrained_sha256"].startswith("85a76fe8")
    assert len(report["cls_rows_copied_from_coco_by_name"]) == 5
