"""Inference speed benchmark shared by every detector family (CH-03).

Same method as the legacy `measure_retina_fps` cell: warm-up calls first, then timed calls, with
`torch.cuda.synchronize()` around each timed call so the GPU work really finishes inside the timer.
The benchmark is framework-agnostic: pass any callable that runs the model on one batch.
What the callable includes (preprocessing, NMS, ...) is the caller's choice; record it in `note`.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch


def device_info(device: torch.device | str) -> dict:
    device = torch.device(device)
    info = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    else:
        info["cpu"] = platform.processor() or platform.machine()
    return info


def benchmark(
    run: Callable[[Any], Any],
    batches: Sequence[Any],
    device: torch.device | str,
    batch_size: int = 1,
    warmup: int = 10,
    input_size: str | None = None,
    note: str = "",
) -> dict:
    """Time `run(batch)` for every batch after `warmup` untimed calls (cycling through `batches`).

    Returns latency per batch (mean / median / p90 / min, ms), latency per image and FPS
    (images per second over the timed calls), plus device, input size and batch size.
    """
    if not batches:
        raise ValueError("benchmark needs at least one batch")
    device = torch.device(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times = []
    with torch.inference_mode():
        for i in range(warmup):
            run(batches[i % len(batches)])
        sync()
        for batch in batches:
            start = time.perf_counter()
            run(batch)
            sync()
            times.append(time.perf_counter() - start)

    ms = np.asarray(times) * 1000.0
    num_images = len(batches) * batch_size
    return {
        "fps": num_images / float(np.sum(times)),
        "latency_ms_per_image": float(ms.sum()) / num_images,
        "latency_ms_per_batch": {
            "mean": float(ms.mean()),
            "median": float(np.median(ms)),
            "p90": float(np.percentile(ms, 90)),
            "min": float(ms.min()),
        },
        "timed_batches": len(batches),
        "warmup_batches": warmup,
        "batch_size": batch_size,
        "input_size": input_size,
        "note": note,
        **device_info(device),
    }
