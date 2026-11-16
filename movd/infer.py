"""Inference-side building blocks of the final pipeline (CH-13): raw RT-DETR passes, decoding, tiling, TTA, merging.

The idea: one GPU pass per *view* of a frame (the full frame at some imgsz, its horizontal flip, or a tile) gives
the RT-DETR decoder's raw output, 300 query boxes and 8 class scores per query. Everything after that (top-k
decoding, the confidence threshold, NMS, merging several views, weighted boxes fusion, the 500 cap) is cheap CPU
work. So CH-13 caches the raw outputs once per view and evaluates every post-processing variant from the cache;
`predict_image` runs the same functions live, for the speed benchmark and for test inference.

Reference behaviour (checked in tests/test_infer.py): `decode(raw, k=300, conf=0.001)` reproduces the Ultralytics
8.4.207 RT-DETR predictor, which takes the top 300 (query, class) pairs of the sigmoid scores
(`RTDETRDecoder.postprocess`), keeps scores > conf and scales the normalised boxes by the original width and height.

Boxes are xyxy in original-frame pixels throughout; classes are 0-7.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.ops import batched_nms

from movd import ultra

# ---------------------------------------------------------------- containers


@dataclass
class Raw:
    """Raw decoder output of one view of one frame, mapped to frame pixels: boxes (Q, 4) xyxy, scores (Q, nc)."""

    boxes: np.ndarray
    scores: np.ndarray


@dataclass
class Dets:
    """Detections of one frame: boxes (N, 4) xyxy frame pixels, scores (N,), classes (N,) int 0-7."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def __len__(self) -> int:
        return len(self.scores)

    def take(self, index) -> Dets:
        return Dets(self.boxes[index], self.scores[index], self.classes[index])


def empty_dets() -> Dets:
    return Dets(
        np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    )


def concat(parts: Sequence[Dets]) -> Dets:
    parts = [p for p in parts if len(p)]
    if not parts:
        return empty_dets()
    return Dets(
        np.concatenate([p.boxes for p in parts]),
        np.concatenate([p.scores for p in parts]),
        np.concatenate([p.classes for p in parts]),
    )


# ---------------------------------------------------------------- views (full frame, flip, tiles)


def tile_windows(
    width: int, height: int, grid: int = 2, frac: float = 0.6
) -> list[tuple[int, int, int, int]]:
    """`grid x grid` windows (x0, y0, x1, y1) of `frac` of the frame size, evenly spaced, covering the frame.

    Each window keeps the frame's aspect ratio, so a tile is stretched to the square input exactly like a
    training frame, only zoomed in (1 / frac).
    """
    if grid < 2 or not 0 < frac < 1 or frac * grid < 1:
        raise ValueError("tiles need grid >= 2 and 1/grid <= frac < 1")
    tw, th = round(frac * width), round(frac * height)
    xs = np.linspace(0, width - tw, grid).round().astype(int)
    ys = np.linspace(0, height - th, grid).round().astype(int)
    return [(int(x), int(y), int(x) + tw, int(y) + th) for y in ys for x in xs]


def view_names(views: str, grid: int = 2) -> list[str]:
    """Names of the per-view caches of a view family: full, flip, or tiles -> tile0..tile{grid^2 - 1}."""
    if views == "tiles":
        return [f"tile{i}" for i in range(grid * grid)]
    if views in ("full", "flip"):
        return [views]
    raise ValueError(f"unknown view family {views!r}")


def crop_view(
    image: np.ndarray, view: str, grid: int = 2, frac: float = 0.6
) -> tuple[np.ndarray, tuple]:
    """The BGR array a view feeds the model, plus its window (x0, y0, x1, y1) in the frame."""
    h, w = image.shape[:2]
    if view == "full":
        return image, (0, 0, w, h)
    if view == "flip":
        return np.ascontiguousarray(image[:, ::-1]), (0, 0, w, h)
    if view.startswith("tile"):
        x0, y0, x1, y1 = tile_windows(w, h, grid, frac)[int(view[4:])]
        return np.ascontiguousarray(image[y0:y1, x0:x1]), (x0, y0, x1, y1)
    raise ValueError(f"unknown view {view!r}")


# ---------------------------------------------------------------- raw RT-DETR pass


def live_network(model) -> torch.nn.Module:
    """The `RTDETRDetectionModel` that `model.predict` actually runs (a prepared copy, not `model.model`)."""
    predictor = getattr(model, "predictor", None)
    if predictor is None or predictor.model is None:
        raise RuntimeError("call prepare(model, ...) before capturing raw outputs")
    return predictor.model.model


def prepare(model, imgsz: int, device: str) -> None:
    """Set up the Ultralytics predictor once (it also runs its warm-up), so later passes can be hooked."""
    if getattr(model, "predictor", None) is None:
        model.predict(
            np.zeros((imgsz, imgsz, 3), np.uint8), **ultra.predict_kwargs(imgsz, device)
        )


class RawCapture(contextlib.AbstractContextManager):
    """Forward hooks that record the decoder's raw output (and the network input shape) during `model.predict`.

    In inference the `RTDETRDecoder` returns `(y, (dec_bboxes, dec_scores, ...))`: dec_bboxes (1, bs, Q, 4)
    normalised cxcywh, dec_scores (1, bs, Q, nc) logits. The sigmoid is taken on the device, as Ultralytics does.
    """

    def __init__(self, model):
        # After the first `predict`, Ultralytics runs its own prepared copy of the network
        # (`model.predictor.model.model`, inside AutoBackend): hook that one.
        self.net = live_network(model)
        self.head = self.net.model[-1]
        if type(self.head).__name__ != "RTDETRDecoder":
            raise TypeError(
                f"last layer is {type(self.head).__name__}, not RTDETRDecoder"
            )
        self.outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.input_shapes: list[tuple[int, ...]] = []

    def __enter__(self):
        def on_head(_module, _inputs, output):
            boxes, logits = output[1][0], output[1][1]
            self.outputs.append(
                (boxes[-1].float().cpu(), logits[-1].sigmoid().float().cpu())
            )

        def on_input(_module, inputs):
            self.input_shapes.append(tuple(inputs[0].shape))

        self.handles = [
            self.head.register_forward_hook(on_head),
            self.net.register_forward_pre_hook(on_input),
        ]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        return False


def raw_pass(
    model,
    image: np.ndarray,
    imgsz: int,
    device: str,
    view: str = "full",
    grid: int = 2,
    frac: float = 0.6,
    return_result: bool = False,
):
    """One RT-DETR pass over one view of a BGR frame -> `Raw` in frame pixels (and the Ultralytics Results)."""
    sub, (x0, y0, x1, y1) = crop_view(image, view, grid, frac)
    kwargs = ultra.predict_kwargs(imgsz, device)
    prepare(model, imgsz, device)
    with RawCapture(model) as cap:
        results = model.predict(sub, **kwargs)
    if len(cap.outputs) != 1 or cap.input_shapes[0][-2:] != (imgsz, imgsz):
        raise RuntimeError(
            f"unexpected capture: {len(cap.outputs)} outputs, input {cap.input_shapes}"
        )
    boxes_n, scores = cap.outputs[0]
    boxes_n, scores = boxes_n[0], scores[0]  # batch 1
    w, h = x1 - x0, y1 - y0
    # Same operations, in the same order and precision, as RTDETRPredictor.postprocess.
    xy = boxes_n[:, :2]
    half = boxes_n[:, 2:] / 2
    boxes = torch.cat([xy - half, xy + half], dim=1)
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    boxes = boxes.numpy()
    if view == "flip":
        boxes = np.stack(
            [w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], axis=1
        )
    if x0 or y0:
        boxes = boxes + np.array([x0, y0, x0, y0], dtype=np.float32)
    raw = Raw(boxes.astype(np.float32), scores.numpy().astype(np.float32))
    return (raw, results[0]) if return_result else raw


# ---------------------------------------------------------------- decoding and merging (CPU)


def decode(raw: Raw, k: int = 300, conf: float = 0.001) -> Dets:
    """Top-k (query, class) pairs by score (Ultralytics RT-DETR decoding), then score > conf."""
    scores = torch.from_numpy(raw.scores)
    nc = scores.shape[1]
    values, index = scores.flatten().topk(min(k, scores.numel()))
    query, cls = index // nc, index % nc
    keep = values > conf
    return Dets(
        raw.boxes[query[keep].numpy()],
        values[keep].numpy(),
        cls[keep].numpy().astype(np.int64),
    )


def nms(dets: Dets, iou: float) -> Dets:
    """Class-wise NMS (torchvision `batched_nms`); result sorted by score, highest first."""
    if not len(dets):
        return dets
    keep = batched_nms(
        torch.from_numpy(np.ascontiguousarray(dets.boxes, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(dets.scores, dtype=np.float32)),
        torch.from_numpy(dets.classes),
        iou,
    )
    return dets.take(keep.numpy())


def _iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    iw = np.clip(
        np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None
    )
    inter = iw * ih
    union = (
        (box[2] - box[0]) * (box[3] - box[1])
        + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        - inter
    )
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def wbf(parts: Sequence[Dets], iou: float, num_views: int | None = None) -> Dets:
    """Weighted boxes fusion (Solovyev et al., 2021) of several views or models, class by class.

    Boxes are visited by descending score; each joins the same-class cluster whose fused box overlaps it most
    (IoU > `iou`), or starts a new cluster. A fused box is the score-weighted mean of its members; its score is
    the members' mean score x min(members, T) / T, with T = `num_views` (a view that missed the object counts 0).
    """
    num_views = num_views or len(parts)
    dets = concat(parts)
    if not len(dets):
        return dets
    out_boxes, out_scores, out_classes = [], [], []
    for c in np.unique(dets.classes):
        sel = np.flatnonzero(dets.classes == c)
        sel = sel[np.argsort(-dets.scores[sel], kind="stable")]
        fused = np.zeros((len(sel), 4))
        wsum = np.zeros((len(sel), 4))
        ssum = np.zeros(len(sel))
        count = np.zeros(len(sel), dtype=int)
        n = 0
        for i in sel:
            box, score = dets.boxes[i].astype(float), float(dets.scores[i])
            j = -1
            if n:
                ious = _iou_one(box, fused[:n])
                best = int(np.argmax(ious))
                if ious[best] > iou:
                    j = best
            if j < 0:
                j, n = n, n + 1
            wsum[j] += score * box
            ssum[j] += score
            count[j] += 1
            fused[j] = wsum[j] / ssum[j]
        mean = ssum[:n] / count[:n]
        out_boxes.append(fused[:n])
        out_scores.append(mean * np.minimum(count[:n], num_views) / num_views)
        out_classes.append(np.full(n, c, dtype=np.int64))
    merged = Dets(
        np.concatenate(out_boxes).astype(np.float32),
        np.concatenate(out_scores).astype(np.float32),
        np.concatenate(out_classes),
    )
    return merged.take(np.argsort(-merged.scores, kind="stable"))


def cap(dets: Dets, max_det: int = 500) -> Dets:
    """Keep the `max_det` highest-scoring detections of the frame (stable on ties)."""
    order = np.argsort(-dets.scores, kind="stable")[:max_det]
    return dets.take(order)


# ---------------------------------------------------------------- a full pipeline configuration


def pass_specs(config: dict) -> list[dict]:
    """Every raw pass a pipeline config needs, in merge order: {model, imgsz, view, k}.

    A pass entry may set its own decoder top-k (for example 500 for a 500-query model); otherwise the config's
    `decode.k` (default 300, the pretrained query count) applies.
    """
    k_default = config.get("decode", {}).get("k", 300)
    specs = []
    for p in config["passes"]:
        for view in view_names(p.get("views", "full"), config.get("tile_grid", 2)):
            specs.append(
                {
                    "model": p["model"],
                    "imgsz": int(p["imgsz"]),
                    "view": view,
                    "k": int(p.get("k", k_default)),
                }
            )
    return specs


def combine(raws: Sequence[Raw], config: dict) -> Dets:
    """Decode every pass, merge them (NMS or WBF), optional post-hoc NMS, then the per-frame cap."""
    conf = config.get("decode", {}).get("conf", 0.001)
    specs = pass_specs(config)
    if len(specs) != len(raws):
        raise ValueError(f"config needs {len(specs)} passes, got {len(raws)}")
    parts = [decode(r, s["k"], conf) for r, s in zip(raws, specs)]
    merge = config.get("merge")
    if len(parts) == 1:
        dets = parts[0]
    elif merge is None:
        raise ValueError("several passes need a merge rule")
    elif merge["method"] == "nms":
        dets = nms(concat(parts), merge["iou"])
    elif merge["method"] == "wbf":
        dets = wbf(parts, merge["iou"])
    else:
        raise ValueError(f"unknown merge method {merge['method']!r}")
    if config.get("post_nms") is not None:
        dets = nms(dets, config["post_nms"])
    return cap(dets, config.get("max_det", 500))


def predict_image(models: dict, image: np.ndarray, config: dict, device: str) -> Dets:
    """The live pipeline for one BGR frame: every raw pass of `config`, then `combine`."""
    grid, frac = config.get("tile_grid", 2), config.get("tile_frac", 0.6)
    raws = [
        raw_pass(models[s["model"]], image, s["imgsz"], device, s["view"], grid, frac)
        for s in pass_specs(config)
    ]
    return combine(raws, config)


def to_coco(image_id: int, dets: Dets) -> list[dict]:
    """Detections -> COCO-results dicts (category_id 1-8), through the shared CH-09 converter and its checks."""
    return ultra.boxes_to_coco(image_id, dets.boxes, dets.classes, dets.scores)
