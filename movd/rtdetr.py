"""Query-based detector (Ultralytics RT-DETR) helpers for CH-11.

RT-DETR reuses the CH-09 Ultralytics path unchanged: the masked YOLO-format export (`movd.ultra.export_split`),
the training arguments (`movd.ultra.train_kwargs`), the prediction adapter (`movd.ultra.predict_coco`, with the
shared 500-per-image cap) and the project evaluator. This module holds only what differs for a query model:

* the model class: RT-DETR checkpoints must be opened with `ultralytics.RTDETR`, not `YOLO`. `YOLO` would give
  them the YOLO trainer, loss and NMS predictor;
* the number of decoder queries. RT-DETR has no NMS. Every query gives at most one box, so
  `num_queries` is a hard upper bound on detections per image (300 for the pretrained `rtdetr-l`), below the
  evaluator's maxDets 500. The tuned variant `configs/models/rtdetr-l-q500.yaml` raises it to 500;
* the inference geometry. The RT-DETR predictor and dataset stretch every frame to a square `imgsz x imgsz`
  (`LetterBox(auto=False, scale_fill=True)`) and scale normalised boxes back by the original width and height,
  so boxes come out in original pixels like YOLO's (checked in tests/test_rtdetr.py).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from movd import data
from movd.ultra import ULTRALYTICS_FAMILIES as FAMILIES

PRETRAINED_QUERIES = 300
Q500_YAML = data.REPO_ROOT / "configs" / "models" / "rtdetr-l-q500.yaml"


def model_class(family: str):
    """The Ultralytics model class for a run-config `family` (default family: yolo11)."""
    if family not in FAMILIES:
        raise ValueError(
            f"unknown Ultralytics family {family!r}; expected one of {FAMILIES}"
        )
    if family == "rtdetr":
        from ultralytics import RTDETR

        return RTDETR
    from ultralytics import YOLO

    return YOLO


def load_model(weights: str | Path, family: str = "yolo11"):
    """Open a checkpoint (`.pt`) or model YAML with the right Ultralytics class for its family."""
    model = model_class(family)(str(weights))
    if family == "rtdetr":
        # The official rtdetr-l.pt (saved by Ultralytics 8.1.25) unpickles as a generic DetectionModel with an
        # RTDETRDecoder head; RTDETRTrainer rebuilds an RTDETRDetectionModel (RT-DETR loss) and loads it.
        decoder(model)
    return model


def decoder(model):
    """The `RTDETRDecoder` head of an Ultralytics RT-DETR model (wrapper or `nn.Module`)."""
    module = getattr(model, "model", model)
    head = module.model[-1]
    if type(head).__name__ != "RTDETRDecoder":
        raise TypeError(f"last layer is {type(head).__name__}, not RTDETRDecoder")
    return head


def num_queries(model) -> int:
    """Decoder queries = the hard cap on RT-DETR detections per image."""
    return int(decoder(model).num_queries)


def yaml_queries(path: str | Path) -> int:
    """Query count declared by an RT-DETR model YAML (`RTDETRDecoder` args `[nc, hd, nq, ...]`)."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for layer in spec["head"]:
        if layer[2] == "RTDETRDecoder":
            args = layer[3]
            return int(args[2]) if len(args) > 2 else PRETRAINED_QUERIES
    raise ValueError(f"{path}: no RTDETRDecoder layer")


def max_detections_per_image(model, max_det: int) -> int:
    """Effective detection cap for an RT-DETR model at a given `max_det`: min(queries, max_det)."""
    return min(num_queries(model), max_det)
