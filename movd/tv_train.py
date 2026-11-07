"""Torchvision models, data adapters and epoch-boundary training."""

from __future__ import annotations

import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import sigmoid_focal_loss

from movd import data


class VisDroneDataset(Dataset):
    """Keep original pixels here. The detector handles resizing and normalization."""

    def __init__(self, split, image_ids, coco, root=None, training=False):
        self.split = split
        self.root = root
        self.training = training
        images = {image["id"]: image for image in coco["images"]}
        self.images = [images[image_id] for image_id in image_ids]
        selected = set(image_ids)
        self.entries = {}
        if training:
            self.entries = data.split_annotations(
                {
                    "images": self.images,
                    "annotations": [
                        ann
                        for ann in coco["annotations"]
                        if ann["image_id"] in selected
                    ],
                }
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image_id = image["id"]
        target = {"image_id": image_id}
        if self.training:
            rgb, boxes, labels = data.training_sample(
                self.split, image, self.entries[image_id], self.root
            )
            if random.random() < 0.5:
                rgb = rgb[:, ::-1].copy()
                boxes[:, [0, 2]] = rgb.shape[1] - boxes[:, [2, 0]]
            target.update(
                boxes=torch.from_numpy(boxes), labels=torch.from_numpy(labels)
            )
        else:
            with Image.open(data.image_path(self.split, image, self.root)) as frame:
                rgb = np.array(frame.convert("RGB"))
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255), target


def collate_fn(batch):
    return tuple(zip(*batch))


class FocalClassificationHead(RetinaNetClassificationHead):
    """Torchvision's foreground-normalized focal loss with configurable alpha/gamma."""

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        for target, logits, matches in zip(
            targets, head_outputs["cls_logits"], matched_idxs
        ):
            foreground = matches >= 0
            classes = torch.zeros_like(logits)
            classes[foreground, target["labels"][matches[foreground]]] = 1
            valid = matches != self.BETWEEN_THRESHOLDS
            loss = sigmoid_focal_loss(
                logits[valid],
                classes[valid],
                alpha=self.alpha,
                gamma=self.gamma,
                reduction="sum",
            )
            losses.append(loss / foreground.sum().clamp_min(1))
        return torch.stack(losses).mean()


def build_model(config, initialize_weights=True):
    protocol = data.load_protocol()
    cap = protocol["inference"]["max_detections_per_image"]
    classes = len(protocol["label_map"]["category_ids"])
    max_size = config.get("max_size", protocol["rubric_budget"]["input_long_side"])
    resize = {
        "min_size": config.get("min_size", round(max_size * 9 / 16)),
        "max_size": max_size,
    }
    pretrained = config.get("pretrained", True) and initialize_weights
    anchors = None
    if "anchors" in config:
        anchors = AnchorGenerator(**config["anchors"])

    if config["family"] == "frcnn":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=5 if weights else None,
            box_detections_per_img=cap,
            **resize,
            **config.get("rpn", {}),
            **config.get("roi", {}),
        )
        predictor = model.roi_heads.box_predictor
        model.roi_heads.box_predictor = FastRCNNPredictor(
            predictor.cls_score.in_features, classes + 1
        )
        if anchors is not None:
            old_count = model.rpn.anchor_generator.num_anchors_per_location()[0]
            count = shared_anchor_count(anchors, model.rpn.anchor_generator)
            if count != old_count:
                model.rpn.head = RPNHead(
                    model.backbone.out_channels, count, conv_depth=2
                )
            model.rpn.anchor_generator = anchors
        return model

    if config["family"] != "retina":
        raise ValueError("family must be frcnn or retina")
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = retinanet_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=5 if weights else None,
        detections_per_img=cap,
        **resize,
    )
    old_count = model.anchor_generator.num_anchors_per_location()[0]
    count = old_count
    if anchors is not None:
        count = shared_anchor_count(anchors, model.anchor_generator)
        model.anchor_generator = anchors
    old_head = model.head.classification_head
    head = FocalClassificationHead(model.backbone.out_channels, count, classes)
    head.conv = old_head.conv
    head.alpha = config.get("focal_alpha", 0.25)
    head.gamma = config.get("focal_gamma", 2.0)
    model.head.classification_head = head
    if count != old_count:
        regression = model.head.regression_head
        regression.bbox_reg = nn.Conv2d(
            model.backbone.out_channels, 4 * count, 3, padding=1
        )
        nn.init.normal_(regression.bbox_reg.weight, std=0.01)
        nn.init.zeros_(regression.bbox_reg.bias)
    return model


def shared_anchor_count(anchors, defaults):
    counts = anchors.num_anchors_per_location()
    if len(counts) != len(defaults.sizes) or len(set(counts)) != 1 or not counts[0]:
        raise ValueError(
            "Use the same positive number of anchors per location across all FPN levels"
        )
    return counts[0]


def seed_epoch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset, config, shuffle=False, seed=0):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def training_epoch(model, optimizer, scaler, loader, family, amp, max_steps):
    model.train()
    total_loss, steps, images_seen = 0.0, 0, 0
    for images, targets in loader:
        images = [image.cuda(non_blocking=True) for image in images]
        targets = [
            {
                "boxes": target["boxes"].cuda(non_blocking=True),
                "labels": target["labels"].cuda(non_blocking=True)
                - int(family == "retina"),
            }
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            loss = sum(model(images, targets).values())
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(
                f"Non-finite loss at iteration {steps + 1}: {value}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += value
        steps += 1
        images_seen += len(images)
        if steps == max_steps:
            break
    return {"loss": total_loss / steps, "steps": steps, "images_seen": images_seen}


@torch.inference_mode()
def predict_coco(model, loader, family, amp):
    model.eval()
    predictions = []
    for images, targets in loader:
        with torch.autocast("cuda", enabled=amp):
            outputs = model([image.cuda(non_blocking=True) for image in images])
        for output, target in zip(outputs, targets):
            boxes = output["boxes"].float().cpu()
            boxes[:, 2:] -= boxes[:, :2]
            labels = output["labels"].cpu() + int(family == "retina")
            for bbox, category_id, score in zip(
                boxes.tolist(), labels.tolist(), output["scores"].tolist()
            ):
                predictions.append(
                    {
                        "image_id": target["image_id"],
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": score,
                    }
                )
    return predictions


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def save_checkpoint(path, checkpoint):
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


def train(
    config, resume=False, training_only=False, evaluate_epoch=None, evaluate_final=None
):
    if not training_only and (evaluate_epoch is None or evaluate_final is None):
        raise RuntimeError(
            "Evaluation requires CH-03. Use --training-only for a smoke run."
        )
    if training_only and not config.get("smoke_steps_per_epoch"):
        raise ValueError("--training-only is restricted to smoke runs")
    if config["epochs"] < 1 or config.get("smoke_steps_per_epoch", 1) < 1:
        raise ValueError("Epoch and smoke step counts must be positive")
    run_id = config["run_id"]
    if Path(run_id).name != run_id or run_id in ("", ".", ".."):
        raise ValueError("run_id must be a single directory name")
    output = Path("outputs") / run_id
    results = Path("results") / run_id
    if resume:
        paths = sorted(output.glob("epoch_*.pt"))
        if not paths:
            raise FileNotFoundError(f"No complete epoch checkpoints in {output}")
        previous = torch.load(paths[-1], map_location="cpu", weights_only=True)
        if previous["config"] != config or previous["training_only"] != training_only:
            raise ValueError(
                "Resume requires the original run configuration and evaluation mode"
            )
    else:
        if results.exists():
            raise FileExistsError(
                f"Evidence already exists in {results}; use a new run_id"
            )
        output.mkdir(parents=True, exist_ok=False)
        previous = None
    results.mkdir(parents=True, exist_ok=True)

    protocol = data.load_protocol()
    seed = protocol["seed"]
    start = time.perf_counter()
    seed_epoch(seed)
    # Stable epoch ordering and augmentation across process restarts.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    amp = config.get("amp", True)
    family = config["family"]
    root = config.get("dataset_root")
    train_split = config.get("train_split", protocol["rubric_budget"]["train_split"])
    dataset = VisDroneDataset(
        "train",
        data.split_image_ids(train_split),
        data.load_coco("train", root),
        root,
        training=True,
    )
    model = build_model(config, initialize_weights=not resume).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    if previous is not None:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        history = previous["history"]
        del previous
        print(
            f"Resuming at epoch {len(history) + 1}, global_step={history[-1]['global_step']}",
            flush=True,
        )

    save_json(
        results / "config.json",
        {
            **config,
            "seed": seed,
            "training_only": training_only,
            "protocol_version": protocol["version"],
            "train_split": train_split,
            "train_split_size": len(dataset),
            "amp": amp,
            "weights": "COCO_V1" if config.get("pretrained", True) else None,
            "min_size": model.transform.min_size[0],
            "max_size": model.transform.max_size,
            "detections_per_img": protocol["inference"]["max_detections_per_image"],
        },
    )
    save_json(
        results / "hardware.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    )
    save_json(results / "curve.json", history)
    max_steps = config.get("smoke_steps_per_epoch")
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        seed_epoch(seed + epoch)
        loader = make_loader(dataset, config, shuffle=True, seed=seed + epoch)
        torch.cuda.reset_peak_memory_stats()
        row = training_epoch(model, optimizer, scaler, loader, family, amp, max_steps)
        row["train_seconds"] = time.perf_counter() - epoch_start
        row["train_peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        row["epoch"] = epoch
        row["global_step"] = (history[-1]["global_step"] if history else 0) + row[
            "steps"
        ]
        row["val_fast"] = None if training_only else evaluate_epoch(model, config)
        row["epoch_seconds"] = time.perf_counter() - epoch_start
        history.append(row)
        save_checkpoint(
            output / f"epoch_{epoch:04d}.pt",
            {
                "config": config,
                "training_only": training_only,
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
        )
        save_json(results / "curve.json", history)
        print(
            f"Checkpoint epoch {epoch}, global_step={row['global_step']}, loss={row['loss']:.5f}",
            flush=True,
        )

    metrics, speed = (None, None) if training_only else evaluate_final(model, config)
    save_json(
        results / "metrics.json",
        {
            "val_full": metrics,
            "status": "training_only" if training_only else "complete",
        },
    )
    save_json(
        results / "speed.json",
        {
            "inference": speed,
            "status": "pending_evaluation" if training_only else "complete",
        },
    )
    save_json(
        results / "runtime.json",
        {
            "this_process_seconds": time.perf_counter() - start,
            "train_seconds": sum(row["train_seconds"] for row in history),
            "completed_epoch_seconds": sum(row["epoch_seconds"] for row in history),
            "images_seen": sum(row["images_seen"] for row in history),
            "global_step": history[-1]["global_step"],
        },
    )
