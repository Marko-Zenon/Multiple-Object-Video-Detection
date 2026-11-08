"""CH-03 self-test of movd/speed.py on the GPU -> results/ch-03-evaluator/speed_selftest.json.

    python results/ch-03-evaluator/make_speed_selftest.py

Times torchvision Faster R-CNN R50-FPN v2 with RANDOM weights (no download) at the frozen input size
(min_size 756, max_size 1344, up to 500 detections) on 30 evenly spaced val_fast frames, batch 1, fp32.
The run is repeated 3 times to show the run-to-run spread. This checks the benchmark utility only; it is
not a model result (trained-model speeds come from the model chunks).
Timed: model forward (resize, normalize, backbone, RPN, RoI heads, NMS); the image is already on the GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from movd import data
from movd.speed import benchmark

REPEATS = 3
NUM_FRAMES = 30
WARMUP = 10


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    torch.manual_seed(data.SEED)
    device = torch.device("cuda")
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        num_classes=data.NUM_CLASSES + 1,
        min_size=756,
        max_size=1344,
        box_detections_per_img=500,
    )
    model.eval().to(device)

    coco = data.load_coco("val")
    images = {im["id"]: im for im in coco["images"]}
    ids = data.split_image_ids("val_fast")
    picked = [ids[i] for i in np.linspace(0, len(ids) - 1, NUM_FRAMES, dtype=int)]
    frames = []
    for image_id in picked:
        rgb = cv2.cvtColor(cv2.imread(str(data.image_path("val", images[image_id]))), cv2.COLOR_BGR2RGB)
        frames.append([torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)])
    sizes = sorted({f"{im['width']}x{im['height']}" for im in (images[i] for i in picked)})

    runs = [
        benchmark(
            model,
            frames,
            device,
            batch_size=1,
            warmup=WARMUP,
            input_size="original frames " + ", ".join(sizes) + "; model resize long side 1344 (min 756)",
            note="random-weight torchvision fasterrcnn_resnet50_fpn_v2, fp32, utility self-test",
        )
        for _ in range(REPEATS)
    ]
    fps = [r["fps"] for r in runs]
    out = {
        "description": __doc__.strip().splitlines()[0],
        "model": "fasterrcnn_resnet50_fpn_v2 (weights=None, random init, seed 42)",
        "torchvision": torchvision.__version__,
        "frames": "val_fast image ids " + ",".join(map(str, picked)),
        "fps_per_repeat": [round(f, 2) for f in fps],
        "fps_spread_percent": round(100 * (max(fps) - min(fps)) / float(np.mean(fps)), 1),
        "runs": runs,
    }
    (HERE / "speed_selftest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("fps_per_repeat", "fps_spread_percent")}), runs[0]["gpu"])


if __name__ == "__main__":
    main()
