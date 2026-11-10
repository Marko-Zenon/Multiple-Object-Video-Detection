"""OPT-03: cache Model A's raw RT-DETR output for every CH-13 view, so post-processing can be studied on the CPU.

    python scripts/opt03_cache.py --split val_fast --weights <MOVD-ch14>/outputs/final_rtdetr/train/weights/epoch0.pt --device 0

For each frame and each of the 6 CH-13 views (full, flip, tile0..tile3 at imgsz 1344) this stores what
`movd.infer.raw_pass` returns: 300 query boxes (xyxy frame pixels) and 8 class scores. Shards are written per
clip into the ignored `outputs/opt03/raw/<split>/`, so an interrupted run resumes; `--merge` (automatic at the
end) joins them into `outputs/opt03/raw/<split>.npz` with arrays image_ids (N,), boxes (N, 6, 300, 4),
scores (N, 6, 300, 8). A run record goes to `results/opt-03-modelA-diagnosis/cache/<split>.json`.

The checkpoint must have the registered Model A sha256; nothing else is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch14_infer import frames  # noqa: E402

from movd import data, infer, rtdetr  # noqa: E402

MODEL_A_SHA256 = "6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b"
PIPELINE = data.REPO_ROOT / "configs" / "runs" / "final_inference.json"
OUTPUTS = data.REPO_ROOT / "outputs" / "opt03" / "raw"
RESULTS = data.REPO_ROOT / "results" / "opt-03-modelA-diagnosis" / "cache"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def specs() -> tuple[dict, list[dict]]:
    config = json.loads(PIPELINE.read_text(encoding="utf-8"))["config"]
    return config, infer.pass_specs(config)


def merge(split: str, clips: list[str]) -> Path:
    parts = [np.load(OUTPUTS / split / f"{c}.npz") for c in sorted(clips)]
    out = OUTPUTS / f"{split}.npz"
    np.savez(
        out,
        image_ids=np.concatenate([p["image_ids"] for p in parts]),
        boxes=np.concatenate([p["boxes"] for p in parts]),
        scores=np.concatenate([p["scores"] for p in parts]),
    )
    return out


def main() -> None:
    import cv2
    import torch
    import ultralytics

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val_fast", "val_full", "test"])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument(
        "--max-clips", type=int, default=0, help="smoke test: first N clips only"
    )
    args = ap.parse_args()

    digest = sha256(args.weights)
    if digest != MODEL_A_SHA256:
        raise SystemExit(f"{args.weights}: sha256 {digest} is not Model A")
    config, pass_list = specs()
    grid, frac = config.get("tile_grid", 2), config.get("tile_frac", 0.6)
    by_clip = defaultdict(list)
    for clip, image_id, path in frames(args.split, args.dataset_root):
        by_clip[clip].append((image_id, path))
    clips = sorted(by_clip)[: args.max_clips or None]
    model = rtdetr.load_model(args.weights, "rtdetr")
    torch.zeros(1, device=f"cuda:{args.device}")
    torch.cuda.reset_peak_memory_stats(int(args.device))
    shard_dir = OUTPUTS / args.split
    shard_dir.mkdir(parents=True, exist_ok=True)
    seconds, computed = 0.0, 0
    for clip in clips:
        shard = shard_dir / f"{clip}.npz"
        if shard.exists():
            continue
        ids, boxes, scores = [], [], []
        start = time.perf_counter()
        for image_id, path in by_clip[clip]:
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"cannot read {path}")
            raws = [
                infer.raw_pass(
                    model, image, s["imgsz"], args.device, s["view"], grid, frac
                )
                for s in pass_list
            ]
            ids.append(image_id)
            boxes.append(np.stack([r.boxes for r in raws]))
            scores.append(np.stack([r.scores for r in raws]))
        seconds += time.perf_counter() - start
        computed += len(ids)
        tmp = shard.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            image_ids=np.array(ids, np.int64),
            boxes=np.stack(boxes),
            scores=np.stack(scores),
        )
        tmp.replace(shard)
        print(
            f"{args.split} {clip}: {len(ids)} frames, {seconds / 60:.1f} min",
            flush=True,
        )

    if args.max_clips:
        print("smoke run: no merge, no record")
        return
    out = merge(args.split, clips)
    RESULTS.mkdir(parents=True, exist_ok=True)
    record = {
        "split": args.split,
        "checkpoint": str(args.weights.as_posix()),
        "checkpoint_sha256": digest,
        "pipeline_config": PIPELINE.relative_to(data.REPO_ROOT).as_posix(),
        "pipeline_config_sha256": sha256(PIPELINE),
        "views": [s["view"] for s in pass_list],
        "frames": sum(len(by_clip[c]) for c in clips),
        "frames_computed_this_run": computed,
        "seconds_this_run": round(seconds, 1),
        "seconds_per_frame_6_views": round(seconds / computed, 4) if computed else None,
        "cache": out.relative_to(data.REPO_ROOT).as_posix(),
        "cache_sha256": sha256(out),
        "gpu": torch.cuda.get_device_name(int(args.device)),
        "peak_cuda_allocated_bytes_this_run": int(
            torch.cuda.max_memory_allocated(int(args.device))
        ),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "platform": platform.platform(),
        "finished_kyiv": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (RESULTS / f"{args.split}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
