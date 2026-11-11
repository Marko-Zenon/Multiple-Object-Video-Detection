"""OPT-10: a better estimate of the fine-tuned weights behind the protected E6 pipeline.

E6 (OPT-08) = soup w080 (0.8 x Model A epoch0.pt + 0.2 x epoch1.pt) -> CH-13 merge with inner tile-border boxes
dropped -> five-frame persistence boost. OPT-10 changes only the checkpoint; the inference views, the merge and the
post-processing stay E6's (`movd.opt09.E6`).

Two tools, both in weight space:

* a weighted average of several checkpoints (`weighted_state`): the OPT-08 soup arithmetic for any number of
  checkpoints. For the two E6 checkpoints with weights (0.8, 0.2) it gives E6's soup exactly;
* BatchNorm re-estimation (`reestimate_bn`): the model was trained at batch size 1 with BatchNorm momentum 0.03, so
  the stored running statistics average only the last ~33 training frames. Re-estimation replaces them by the plain
  average over a fixed list of 1,024 training frames, with the weights unchanged.

`bn_batch_stats` is a diagnostic only (D1): BatchNorm uses the statistics of the current image at inference, as it
did in training, and the stored statistics are left byte-identical.

The grid, the stages, the gate and the diagnostic thresholds below were frozen in
results/opt-10-soup-bn/selection_rule.md before any OPT-10 result and are never changed after one.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable, Iterable, Sequence

from movd.opt09 import E6  # the protected post-processing settings, reused unchanged

# Every OPT-10 candidate keeps E6's soup-independent settings: border drop 0.005, five-frame boost IoU 0.5.
POSTPROCESS = {k: E6[k] for k in ("factor", "margin", "iou", "strength")}

IMGSZ = 1344
BN_FRAMES = 1024  # frames used for BatchNorm re-estimation
D2_FRAMES = 714  # train_final frames of diagnostic D2 (as many as val_fast)

# Seed replicates of Model A's recipe (configs/runs/final_rtdetr.json). The frozen protocol seed is 42 (Model A);
# seeds 43-45 are the approved plan.md section 11 exception, only for these replicate runs.
REPLICATE_SEEDS = {"R1": 43, "R2": 44, "R3": 45}

# Frozen candidates. `runs`: which fine-tuning runs are averaged ("A" = Model A's run); `epoch_weights`: the weight
# of each run's epoch-1 and epoch-2 checkpoint (E6 uses 0.8 / 0.2 on run A alone); `bn`: BatchNorm re-estimation.
GRID: dict[str, dict] = {
    "c0": {"bn": True, "epoch_weights": [0.8, 0.2], "phase": 0, "runs": ["A"]},
    "m2": {"bn": False, "epoch_weights": [0.8, 0.2], "phase": 1, "runs": ["A", "R1"]},
    "m4": {
        "bn": False,
        "epoch_weights": [0.8, 0.2],
        "phase": 2,
        "runs": ["A", "R1", "R2", "R3"],
    },
    "m4e1": {
        "bn": False,
        "epoch_weights": [1.0, 0.0],
        "phase": 2,
        "runs": ["A", "R1", "R2", "R3"],
    },
    "m4bn": {
        "bn": True,
        "epoch_weights": [0.8, 0.2],
        "phase": 2,
        "runs": ["A", "R1", "R2", "R3"],
    },
}
PHASE2_SELECTION = ("m4", "m4e1", "m4bn")  # S1 picks one of these
M4BN_NEEDS_C0_DISCOVERY_DELTA = (
    0.0  # m4bn is built only if c0's discovery delta is >= this
)

STAGE = {"S1_min_delta_discovery": 0.0010, "S2_min_delta_confirmation": 0.0}
GATE = {
    "G1_min_delta_mAP": 0.0010,
    "G2_min_delta_mAP_width_le_1920": 0.0010,
    "G3_min_clips_non_negative": 5,
    "G4_min_delta_mAP_without_4k_clip": 0.0010,
    "G5_max_AP_small_loss": 0.002,
    "G6_max_density_ge100_loss": 0.002,
    "G7_min_leave_one_clip_out_delta_exclusive": 0.0,
}
SELECTION_TIER = {"min_delta_mAP": 0.0015, "public_must_exceed": 0.2947349001}
TARGET_LABEL_VAL_FULL = 0.2999  # label T (upload priority only, not a gate)
BARRIER_M2_DISCOVERY_DELTA = -0.0020  # below this, no R2 / R3

# Diagnostics (reported, never selected from). D1: BatchNorm with per-image statistics at inference, on val_fast.
# D2: normal inference on train_final frames. Thresholds of the interpretation table, fixed before running.
DIAG_EPOCHS = (1, 3, 6)  # final_rtdetr checkpoints epoch0.pt, epoch2.pt, epoch5.pt
DIAG = {
    "anchor_model_A_val_fast_mAP": 0.27896670695801273,  # results/final_rtdetr/curve.json, epoch 1
    "anchor_tolerance": 0.0005,
    "D1_H1_min_lift_epoch6": 0.010,
    "c0_H2_min_delta_discovery": 0.0010,
    "D2_H3_min_rise_epoch1_to_6": 0.03,
}
BUDGET = {
    "gpu_minutes_total": 435,
    "gpu_minutes_phase0": 45,
    "gpu_minutes_phase1": 85,
    "gpu_minutes_phase2": 200,
    "gpu_minutes_test": 105,
    "max_training_runs": 3,
    "max_new_csvs": 3,
}


# ---------------------------------------------------------------- frozen frame lists


def spaced_indices(total: int, count: int) -> list[int]:
    """`count` distinct indices spread evenly over 0 .. total-1: round(i * (total - 1) / (count - 1)).

    The first index is 0 and the last is total - 1, so the sample covers the whole list and is not biased
    towards its start. Integer arithmetic (round half up), so the result does not depend on float rounding.
    """
    if not 2 <= count <= total:
        raise ValueError(f"need 2 <= count <= total, got count={count}, total={total}")
    span, steps = total - 1, count - 1
    indices = [(2 * i * span + steps) // (2 * steps) for i in range(count)]
    if len(set(indices)) != count:
        raise ValueError("indices are not distinct")
    return indices


def spaced_ids(image_ids: Iterable[int], count: int) -> list[int]:
    """`count` image ids at evenly spaced positions of the sorted id list."""
    ordered = sorted(image_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("image ids are not unique")
    return [ordered[i] for i in spaced_indices(len(ordered), count)]


def ids_sha256(image_ids: Sequence[int]) -> str:
    """sha256 of the ids joined by commas (the value stored in the frozen rule)."""
    return hashlib.sha256(
        ",".join(str(i) for i in image_ids).encode("ascii")
    ).hexdigest()


# ---------------------------------------------------------------- weight averaging


def candidate_weights(cid: str) -> list[tuple[str, int, float]]:
    """(run, epoch, weight) of every checkpoint in a candidate's average; the weights sum to 1.

    Runs are averaged uniformly, so each run's epoch-e checkpoint gets epoch_weights[e - 1] / number of runs.
    Checkpoints with weight 0 are left out.
    """
    spec = GRID[cid]
    runs = spec["runs"]
    out = []
    for epoch, weight in enumerate(spec["epoch_weights"], start=1):
        if weight:
            out.extend((run, epoch, weight / len(runs)) for run in runs)
    return out


def weighted_state(states: Sequence[dict], weights: Sequence[float]) -> dict:
    """sum_i weights[i] * states[i] for floating tensors; other tensors are copied from the first state.

    Computed in float32 in the given order and stored in the first state's dtype, as `opt08_soup.soup_state`
    does for two checkpoints. All states must have the same tensor names, shapes and dtypes.
    """
    if not states or len(states) != len(weights):
        raise ValueError("need one weight per state")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {sum(weights)}")
    first = states[0]
    for other in states[1:]:
        if list(other) != list(first):
            raise ValueError("states have different tensor names")
        for key, tensor in first.items():
            if other[key].shape != tensor.shape or other[key].dtype != tensor.dtype:
                raise ValueError(f"{key}: shape or dtype differs between states")
    out = {}
    for key, tensor in first.items():
        if tensor.is_floating_point():
            mixed = weights[0] * tensor.float()
            for weight, other in zip(weights[1:], states[1:], strict=True):
                mixed = mixed + weight * other[key].float()
            out[key] = mixed.to(tensor.dtype)
        else:
            out[key] = tensor.clone()
    return out


# ---------------------------------------------------------------- BatchNorm


def bn_layers(module) -> list:
    """Every BatchNorm layer of a network, in `named_modules` order."""
    from torch import nn

    return [
        m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]


def bn_buffer_names(module) -> list[str]:
    """State-dict names of the BatchNorm buffers (running_mean, running_var, num_batches_tracked)."""
    from torch import nn

    names = []
    for name, m in module.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            names.extend(
                f"{name}.{b}" if name else b
                for b in ("running_mean", "running_var", "num_batches_tracked")
            )
    return names


def reestimate_bn(module, batches: Iterable, forward: Callable | None = None) -> dict:
    """Replace the BatchNorm running statistics by their plain average over `batches`.

    Steps, in this order:
    1. remember every module's training flag and every BatchNorm layer's momentum;
    2. put the whole network in eval mode, then only the BatchNorm layers in training mode;
    3. reset each BatchNorm layer's running_mean, running_var and num_batches_tracked and set momentum=None, so
       PyTorch keeps the cumulative average of the per-batch statistics (the same per-batch statistics the layer
       used during training);
    4. run every batch through the network in FP32 without gradients;
    5. restore every momentum and every training flag, also if a batch raised an error.

    Only BatchNorm buffers change; no parameter is touched. `forward(batch)` defaults to `module(batch)`.
    Returns the number of layers and batches. If a batch raises, or no batch is given, the original statistics
    are put back before the error is raised, so a failed call never leaves half-reset statistics behind.
    """
    import torch

    layers = bn_layers(module)
    if not layers:
        raise ValueError("the network has no BatchNorm layer")
    flags = [(m, m.training) for m in module.modules()]
    momenta = [m.momentum for m in layers]
    saved = [
        (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
        for m in layers
    ]
    count, done = 0, False
    try:
        module.eval()
        for m in layers:
            m.reset_running_stats()
            m.momentum = None
            m.train()
        with torch.no_grad():
            for batch in batches:
                batch = batch.float()
                if forward is None:
                    module(batch)
                else:
                    forward(batch)
                count += 1
        done = count > 0
    finally:
        for m, momentum in zip(layers, momenta, strict=True):
            m.momentum = momentum
        for m, flag in flags:
            m.training = flag
        if not done:
            for m, (mean, var, tracked) in zip(layers, saved, strict=True):
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(tracked)
    if count == 0:
        raise ValueError(
            "no batch was given: the BatchNorm statistics were left unchanged"
        )
    return {"bn_layers": len(layers), "batches": count}


@contextlib.contextmanager
def bn_batch_stats(module):
    """Diagnostic D1: inside the block, BatchNorm normalises with the statistics of the current batch.

    The whole network is in eval mode except the BatchNorm layers, which are in training mode with momentum 0 (the
    running statistics are not updated). On exit the three buffers of every BatchNorm layer are restored from a
    copy, so they are byte-identical to before, and every momentum and training flag is restored.
    """
    layers = bn_layers(module)
    flags = [(m, m.training) for m in module.modules()]
    momenta = [m.momentum for m in layers]
    saved = [
        (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
        for m in layers
    ]
    try:
        module.eval()
        for m in layers:
            m.momentum = 0.0
            m.train()
        yield module
    finally:
        for m, momentum, (mean, var, tracked) in zip(
            layers, momenta, saved, strict=True
        ):
            m.momentum = momentum
            m.running_mean.copy_(mean)
            m.running_var.copy_(var)
            m.num_batches_tracked.copy_(tracked)
        for m, flag in flags:
            m.training = flag


# ---------------------------------------------------------------- direct (unfused) forward


def preprocess(image, imgsz: int = IMGSZ):
    """A BGR frame -> the (1, 3, imgsz, imgsz) float32 RGB tensor in [0, 1] that RT-DETR trains and predicts on.

    Ultralytics stretches every frame to the square input for RT-DETR (dataset: resize to imgsz x imgsz;
    predictor: `LetterBox(auto=False, scale_fill=True)`), bilinear, then BGR -> RGB and / 255.
    """
    import cv2
    import numpy as np
    import torch

    if image.shape[:2] != (imgsz, imgsz):
        image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    chw = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).float() / 255.0


def direct_raw(net, image, device: str, imgsz: int = IMGSZ):
    """One full-frame pass through the *unfused* network -> `movd.infer.Raw` in frame pixels.

    The Ultralytics predictor folds BatchNorm into the convolutions, so BatchNorm cannot be switched to per-image
    statistics there. This runs the training-form network directly. In eval mode it gives the predictor's output
    up to floating-point rounding (checked against Model A's stored val_fast mAP before the diagnostics).
    The box arithmetic is `movd.infer.raw_pass`'s for the "full" view.
    """
    import torch

    from movd import infer

    height, width = image.shape[:2]
    with torch.no_grad():
        output = net(preprocess(image, imgsz).to(device))
    boxes_n = (
        output[1][0][-1].float().cpu()[0]
    )  # (Q, 4) normalised cxcywh of the last decoder layer
    scores = output[1][1][-1].sigmoid().float().cpu()[0]  # (Q, nc)
    xy, half = boxes_n[:, :2], boxes_n[:, 2:] / 2
    boxes = torch.cat([xy - half, xy + half], dim=1)
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return infer.Raw(boxes.numpy().astype("float32"), scores.numpy().astype("float32"))
