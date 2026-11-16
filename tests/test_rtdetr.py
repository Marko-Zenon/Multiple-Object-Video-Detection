"""Tests for movd/rtdetr.py and the RT-DETR path through the CH-09 Ultralytics code (CH-11).

CPU only. The model tests need the pinned `ultralytics`; the pretrained-weight test also needs
`outputs/weights/rtdetr-l.pt` (git-ignored) and is skipped without it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from movd import data, rtdetr, ultra
from movd import evaluate as ev

try:
    import ultralytics

    HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    HAS_ULTRALYTICS = False
needs_ultralytics = pytest.mark.skipif(
    not HAS_ULTRALYTICS, reason="ultralytics not installed"
)
PRETRAINED = data.REPO_ROOT / "outputs" / "weights" / "rtdetr-l.pt"
RUNS = data.REPO_ROOT / "configs" / "runs"


def _packaged_yaml() -> Path:
    return (
        Path(ultralytics.__file__).parent
        / "cfg"
        / "models"
        / "rt-detr"
        / "rtdetr-l.yaml"
    )


# ---------------------------------------------------------------- model YAML and family dispatch


@needs_ultralytics
def test_q500_yaml_is_packaged_rtdetr_l_with_only_the_query_count_changed():
    ours = yaml.safe_load(rtdetr.Q500_YAML.read_text(encoding="utf-8"))
    packaged = yaml.safe_load(_packaged_yaml().read_text(encoding="utf-8"))
    assert ours["backbone"] == packaged["backbone"]
    assert ours["head"][:-1] == packaged["head"][:-1]
    assert ours["head"][-1][:3] == packaged["head"][-1][:3]
    assert packaged["head"][-1][3] == ["nc"] and ours["head"][-1][3] == ["nc", 256, 500]
    assert {k: v for k, v in ours.items() if k not in ("backbone", "head")} == {
        k: v for k, v in packaged.items() if k not in ("backbone", "head")
    }
    assert rtdetr.yaml_queries(rtdetr.Q500_YAML) == 500
    assert rtdetr.yaml_queries(_packaged_yaml()) == rtdetr.PRETRAINED_QUERIES == 300


@needs_ultralytics
def test_model_class_dispatch():
    from ultralytics import RTDETR, YOLO

    assert rtdetr.model_class("rtdetr") is RTDETR
    assert rtdetr.model_class("yolo11") is YOLO
    with pytest.raises(ValueError, match="unknown Ultralytics family"):
        rtdetr.model_class("detr")


@needs_ultralytics
def test_q500_model_builds_with_500_queries_and_same_parameter_shapes():
    """No parameter shape depends on the query count, so pretrained weights transfer fully."""
    q500 = rtdetr.load_model(rtdetr.Q500_YAML, "rtdetr")
    q300 = rtdetr.load_model(_packaged_yaml(), "rtdetr")
    assert rtdetr.num_queries(q500) == 500 and rtdetr.num_queries(q300) == 300
    assert rtdetr.max_detections_per_image(q300, ev.HEADLINE_MAX_DETS) == 300
    assert rtdetr.max_detections_per_image(q500, ev.HEADLINE_MAX_DETS) == 500
    shapes = lambda m: {k: v.shape for k, v in m.model.state_dict().items()}  # noqa: E731
    assert shapes(q500) == shapes(q300)


@needs_ultralytics
@pytest.mark.skipif(
    not PRETRAINED.exists(), reason="outputs/weights/rtdetr-l.pt not downloaded"
)
def test_pretrained_checkpoint_loads_as_rtdetr_and_rejects_yolo():
    model = rtdetr.load_model(PRETRAINED, "rtdetr")
    assert rtdetr.num_queries(model) == 300
    assert (
        len(model.names) == 80
    )  # COCO head; fine-tuning replaces it with the 8 VisDrone classes


@needs_ultralytics
def test_yolo_model_is_rejected_as_rtdetr():
    from ultralytics import YOLO

    yolo_yaml = (
        Path(ultralytics.__file__).parent / "cfg" / "models" / "11" / "yolo11.yaml"
    )
    with pytest.raises(TypeError, match="not RTDETRDecoder"):
        rtdetr.decoder(YOLO(str(yolo_yaml)))
    with pytest.raises(TypeError):  # the RT-DETR class cannot even build a YOLO head
        rtdetr.load_model(yolo_yaml, "rtdetr")


# ---------------------------------------------------------------- original-pixel geometry


@needs_ultralytics
def test_rtdetr_predictor_scales_normalised_boxes_to_original_pixels():
    """The RT-DETR predictor stretches frames to a square and rescales by the original width/height."""
    from ultralytics.models.rtdetr.predict import RTDETRPredictor

    predictor = RTDETRPredictor(
        overrides={"conf": 0.001, "max_det": 500, "imgsz": 1344}
    )
    predictor.model = type("M", (), {"names": dict(enumerate(data.CLASS_NAMES))})()
    width, height = 1904, 1071
    original = np.zeros((height, width, 3), dtype=np.uint8)
    predictor.batch = (["frame.jpg"], [original], [""])
    # One query: normalised cx, cy, w, h = 0.5, 0.25, 0.1, 0.2, score 0.9, class 3 (van). Second query below conf.
    preds = torch.tensor(
        [[[0.5, 0.25, 0.1, 0.2, 0.9, 3.0], [0.1, 0.1, 0.05, 0.05, 0.0005, 0.0]]]
    )
    (result,) = predictor.postprocess(preds, torch.zeros(1, 3, 1344, 1344), [original])
    dets = ultra.result_to_coco(result, image_id=7)
    assert len(dets) == 1
    x, y, w, h = dets[0]["bbox"]
    assert (x, y, w, h) == pytest.approx(
        (0.45 * width, 0.15 * height, 0.1 * width, 0.2 * height), rel=1e-5
    )
    assert dets[0]["category_id"] == 4 and dets[0]["score"] == pytest.approx(0.9)


@needs_ultralytics
def test_rtdetr_preprocessing_stretches_to_square_without_letterbox_padding():
    from ultralytics.models.rtdetr.predict import RTDETRPredictor

    assert RTDETRPredictor.scale_fill is True
    predictor = RTDETRPredictor(overrides={"imgsz": 1344})
    predictor.imgsz = (1344, 1344)
    predictor.model = type("M", (), {"stride": 32, "format": "pt"})()
    (out,) = predictor.pre_transform([np.full((1071, 1904, 3), 255, dtype=np.uint8)])
    assert out.shape[:2] == (1344, 1344)
    assert out.min() == 255  # no grey padding: the whole frame is resized (scale_fill)


# ---------------------------------------------------------------- run configs


def _kwargs(name: str) -> dict:
    config = ultra.load_run_config(RUNS / f"{name}.json")
    return ultra.train_kwargs(config, "d.yaml", "out", "0")


def test_rtdetr_run_configs_are_valid_and_keep_the_frozen_budget():
    names = {p.stem for p in RUNS.glob("rtdetr_*.json")}
    assert names >= {
        "rtdetr_baseline",
        "rtdetr_q500",
        "rtdetr_lr1e4",
        "rtdetr_lr1e4_wb0",
    }
    for name in names:
        config = ultra.load_run_config(RUNS / f"{name}.json")
        assert config["family"] == "rtdetr" and config["run_id"] == name
    for name in ("rtdetr_baseline", "rtdetr_q500", "rtdetr_lr1e4", "rtdetr_lr1e4_wb0"):
        kwargs = _kwargs(name)
        assert kwargs["epochs"] * 2800 == 16800 and kwargs["imgsz"] == 1344
        assert kwargs["seed"] == 42 and kwargs["amp"] is True and kwargs["batch"] == 1
        assert kwargs["save_period"] == 1 and kwargs["max_det"] == 500
        assert (
            "fraction" not in kwargs
            and kwargs["fliplr"] == 0.5
            and kwargs["mosaic"] == 0
        )


def test_rtdetr_variants_change_exactly_one_declared_variable():
    base = _kwargs("rtdetr_baseline")
    q500 = _kwargs("rtdetr_q500")
    lr = _kwargs("rtdetr_lr1e4")
    diff = lambda a, b: {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}  # noqa: E731
    assert diff(base, q500) == {
        "pretrained"
    }  # model YAML (500 queries) + the same pretrained file
    assert q500["pretrained"] == "outputs/weights/rtdetr-l.pt"
    assert diff(base, lr) == {"optimizer", "lr0", "momentum"}
    assert lr["optimizer"] == "AdamW" and lr["momentum"] == 0.9 and lr["lr0"] == 1e-4
    # rtdetr_lr1e4 is kept but confounded: an explicit optimizer skips the auto branch that zeroes
    # warmup_bias_lr. rtdetr_lr1e4_wb0 sets it back to 0.0, so only lr0 differs from what auto picks.
    clean = _kwargs("rtdetr_lr1e4_wb0")
    assert diff(lr, clean) == {"warmup_bias_lr"} and clean["warmup_bias_lr"] == 0.0
    q500_config = ultra.load_run_config(RUNS / "rtdetr_q500.json")
    assert rtdetr.yaml_queries(data.REPO_ROOT / q500_config["weights"]) == 500


@needs_ultralytics
def test_optimizer_auto_matches_the_explicit_learning_rate_variant_except_lr0(tmp_path):
    """What optimizer=auto picks (and changes) for the baseline, so the corrected variant differs only in lr0."""
    ITERATIONS = (
        math.ceil(2800 / 64) * 6
    )  # BaseTrainer: ceil(len(dataset) / max(batch, nbs)) * epochs
    from types import SimpleNamespace

    from ultralytics.cfg import get_cfg
    from ultralytics.engine.trainer import BaseTrainer

    def built(name: str) -> SimpleNamespace:
        kwargs = _kwargs(name)
        args = get_cfg(
            overrides={
                k: v
                for k, v in kwargs.items()
                if k not in ("data", "project", "device")
            }
        )
        trainer = SimpleNamespace(args=args, data={"nc": len(data.CLASS_NAMES)})
        optimizer = BaseTrainer.build_optimizer(
            trainer,
            torch.nn.Linear(2, 2),
            args.optimizer,
            args.lr0,
            args.momentum,
            args.weight_decay,
            ITERATIONS,
        )
        return SimpleNamespace(args=args, optimizer=optimizer)

    auto, clean = built("rtdetr_baseline"), built("rtdetr_lr1e4_wb0")
    assert type(auto.optimizer).__name__ == type(clean.optimizer).__name__ == "AdamW"
    assert (
        auto.optimizer.defaults["betas"][0]
        == clean.optimizer.defaults["betas"][0]
        == 0.9
    )
    assert auto.args.warmup_bias_lr == clean.args.warmup_bias_lr == 0.0
    assert auto.args.lr0 == pytest.approx(0.000833) and clean.args.lr0 == 1e-4


@needs_ultralytics
def test_rtdetr_train_kwargs_are_valid_ultralytics_arguments():
    from ultralytics.cfg import get_cfg

    for name in ("rtdetr_baseline", "rtdetr_q500", "rtdetr_lr1e4"):
        cfg = get_cfg(overrides=_kwargs(name))
        assert cfg.imgsz == 1344 and cfg.epochs == 6


def test_run_config_rejects_unknown_family(tmp_path):
    with pytest.raises(ValueError, match="family"):
        ultra.validate_run_config(
            {"run_id": "x", "weights": "w.pt", "batch_size": 1, "family": "detr"}
        )
