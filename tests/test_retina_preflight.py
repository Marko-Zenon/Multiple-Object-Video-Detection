import json
from pathlib import Path

import pytest
import torch
from torchvision.ops import sigmoid_focal_loss

from movd import data
from movd.tv_train import build_model

RUNS = ("baseline", "anchors", "focal_gamma15")


def config(variant):
    return json.loads((Path("configs/runs") / f"retina_{variant}.json").read_text())


@pytest.mark.parametrize("variant", RUNS)
def test_retina_experiment_config_and_model(variant):
    run = config(variant)
    baseline = config("baseline")
    stats = json.loads(Path("results/ch-08-retina/anchor_stats.json").read_text())
    changes = {
        "baseline": {},
        "anchors": {"anchors": stats["anchors"]},
        "focal_gamma15": {"focal_gamma": 1.5},
    }
    assert run == {**baseline, **changes[variant], "run_id": f"retina_{variant}"}
    budget = data.load_protocol()["rubric_budget"]
    assert run["pretrained"] and run["amp"]
    assert (
        run["epochs"] * len(data.split_image_ids(run["train_split"]))
        == budget["images_seen"]
    )
    model = build_model(run, initialize_weights=False)
    assert model.transform.min_size == (756,)
    assert model.transform.max_size == budget["input_long_side"]
    assert model.detections_per_img == 500
    assert model.anchor_generator.num_anchors_per_location() == [9] * 5
    head = model.head.classification_head
    assert head.cls_logits.out_channels == 9 * 8
    assert model.head.regression_head.bbox_reg.out_channels == 9 * 4
    assert head.alpha == 0.25
    assert head.gamma == (1.5 if variant == "focal_gamma15" else 2.0)
    if variant == "anchors":
        assert [list(level) for level in model.anchor_generator.sizes] == stats[
            "anchors"
        ]["sizes"]
        assert [list(level) for level in model.anchor_generator.aspect_ratios] == stats[
            "anchors"
        ]["aspect_ratios"]


def test_retina_focal_gamma_changes_loss_with_same_weights():
    baseline = build_model(config("baseline"), initialize_weights=False)
    tuned = build_model(config("focal_gamma15"), initialize_weights=False)
    tuned.load_state_dict(baseline.state_dict(), strict=True)
    logits = torch.tensor([[[0.0] * 8, [-2.0] * 8, [5.0] * 8]])
    targets = [{"labels": torch.tensor([0])}]
    matches = [torch.tensor([0, -1, -2])]
    expected_classes = torch.zeros(2, 8)
    expected_classes[0, 0] = 1
    losses = []
    for model, gamma in ((baseline, 2.0), (tuned, 1.5)):
        loss = model.head.classification_head.compute_loss(
            targets, {"cls_logits": logits}, matches
        )
        expected = sigmoid_focal_loss(
            logits[0, :2], expected_classes, alpha=0.25, gamma=gamma, reduction="sum"
        )
        torch.testing.assert_close(loss, expected)
        losses.append(loss)
    assert losses[1] > losses[0]
