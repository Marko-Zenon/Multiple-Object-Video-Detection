import numpy as np
import pytest
import torch
from PIL import Image

from movd.tv_train import VisDroneDataset, build_model, collate_fn


def test_training_sample_preserves_targets_in_ignore_region(tmp_path, monkeypatch):
    relative = "clip_001/0000001.jpg"
    path = tmp_path / "train" / relative
    path.parent.mkdir(parents=True)
    Image.fromarray(np.full((10, 12, 3), 200, dtype=np.uint8)).save(path)
    coco = {
        "images": [
            {"id": 1, "file_name": relative, "height": 10, "width": 12},
            {"id": 2, "file_name": relative, "height": 10, "width": 12},
        ],
        "annotations": [
            {"image_id": 1, "bbox": [2, 2, 3, 3], "category_id": 3, "iscrowd": 0},
            {"image_id": 1, "bbox": [0, 0, 9, 9], "category_id": 3, "iscrowd": 1},
            {"image_id": 2, "bbox": [0, 0, 9, 9], "category_id": 3, "iscrowd": 1},
        ],
    }
    monkeypatch.setattr("movd.tv_train.random.random", lambda: 0.0)
    dataset = VisDroneDataset("train", [1, 2], coco, tmp_path, training=True)
    images, targets = collate_fn([dataset[0], dataset[1]])
    assert images[0].shape == (3, 10, 12)
    torch.testing.assert_close(images[0][:, 0, -1], torch.tensor([124, 116, 104]) / 255)
    torch.testing.assert_close(images[0][:, 2, 8], torch.full((3,), 200 / 255))
    torch.testing.assert_close(
        targets[0]["boxes"], torch.tensor([[7.0, 2.0, 10.0, 5.0]])
    )
    assert targets[0]["labels"].tolist() == [3]
    assert targets[1]["boxes"].shape == (0, 4)
    assert targets[1]["labels"].shape == (0,)
    assert targets[1]["labels"].dtype == torch.long

    val_image, val_target = VisDroneDataset("train", [1], coco, tmp_path)[0]
    assert val_target == {"image_id": 1}
    torch.testing.assert_close(val_image, torch.full((3, 10, 12), 200 / 255))


@pytest.mark.parametrize("family", ["frcnn", "retina"])
def test_detector_uses_protocol_defaults_and_custom_anchors(family):
    config = {
        "family": family,
        "pretrained": False,
        "anchors": {
            "sizes": [[16], [32], [64], [128], [256]],
            "aspect_ratios": [[1.0]] * 5,
        },
    }
    if family == "frcnn":
        config.update(rpn={"rpn_nms_thresh": 0.6}, roi={"box_nms_thresh": 0.4})
    else:
        config.update(focal_alpha=0.5, focal_gamma=1.5)
    model = build_model(config)
    assert model.transform.min_size == (756,)
    assert model.transform.max_size == 1344
    if family == "frcnn":
        assert model.roi_heads.detections_per_img == 500
        assert model.rpn.anchor_generator.num_anchors_per_location() == [1] * 5
        assert model.rpn.head.cls_logits.out_channels == 1
        assert model.rpn.nms_thresh == 0.6
        assert model.roi_heads.nms_thresh == 0.4
    else:
        assert model.detections_per_img == 500
        assert model.anchor_generator.num_anchors_per_location() == [1] * 5
        assert model.head.classification_head.cls_logits.out_channels == 8
        assert model.head.classification_head.alpha == 0.5
        assert model.head.classification_head.gamma == 1.5
        assert model.head.regression_head.bbox_reg.out_channels == 4
