"""Run a trained torchvision checkpoint (CH-04 / CH-07 format) on the test set -> COCO-results JSON (CH-05).

    python scripts/predict_test.py --checkpoint outputs/<run_id>/epoch_0006.pt --name <name>
    python scripts/predict_test.py --checkpoint ... --name <name> --resume       # continue after a stop
    python scripts/predict_test.py --checkpoint ... --name cpu_smoke --device cpu --clips clip_001 \
        --max-images 2 --allow-smoke                                               # CPU smoke test

Output (git-ignored): `outputs/ch-05-submission/<name>/`
  * `shards/<clip>.json`: one file per test clip, written atomically, so `--resume` restarts at a clip boundary;
  * `test_predictions.json`: the merged COCO-results list (`image_id` = 1-based `test_images.csv` row).
Evidence (tracked): `results/ch-05-submission/<name>/predict.json` (checkpoint sha256, config, counts, timing).

Boxes come back in original image pixels: torchvision's `GeneralizedRCNNTransform` resizes each image
(min_size/max_size from the checkpoint config), pads the batch, and `postprocess` maps the boxes back to
each image's original size (padding is only at the bottom/right, so it needs no offset). This is checked in
`tests/test_submission.py`. Test frames are read unmasked, as val frames are in `movd.tv_train`.

The checkpoint format is the one written by `movd.tv_train.train` (keys `config`, `model`, `history`, ...).
Smoke checkpoints (`smoke_steps_per_epoch` in their config) are refused unless `--allow-smoke` is given,
so an untrained model cannot slip into a real submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from movd import data  # noqa: E402
from movd.tv_train import build_model, collate_fn, save_json  # noqa: E402

OUTPUT_ROOT = Path("outputs") / "ch-05-submission"
RESULTS_ROOT = Path("results") / "ch-05-submission"


class TestFrames(Dataset):
    """`test_images.csv` rows -> (float RGB tensor in [0, 1], {"image_id": row id}), original pixels."""

    def __init__(self, rows, root=None):
        self.rows = rows
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(data.test_image_path(row, self.root)) as frame:
            rgb = np.array(frame.convert("RGB"))
        if rgb.shape[:2] != (row["height"], row["width"]):
            raise ValueError(
                f"{row['file_name']}: size {rgb.shape[1]}x{rgb.shape[0]} differs from test_images.csv"
            )
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255), {
            "image_id": row["id"]
        }


@torch.inference_mode()
def predict(model, loader, family, device, amp):
    """Same output conversion as `movd.tv_train.predict_coco` (xyxy -> xywh, RetinaNet labels + 1),
    with an explicit device so it can run on the CPU while the GPU is busy."""
    model.eval()
    predictions = []
    for images, targets in loader:
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            outputs = model([image.to(device, non_blocking=True) for image in images])
        for output, target in zip(outputs, targets):
            boxes = output["boxes"].float().cpu()
            boxes[:, 2:] -= boxes[:, :2]
            labels = output["labels"].cpu() + int(family == "retina")
            for bbox, category_id, score in zip(
                boxes.tolist(), labels.tolist(), output["scores"].float().tolist()
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device, allow_smoke: bool):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("smoke_steps_per_epoch") and not allow_smoke:
        raise SystemExit(
            f"{checkpoint_path} is a smoke checkpoint; pass --allow-smoke only for pipeline tests"
        )
    model = build_model(config, initialize_weights=False)
    model.load_state_dict(checkpoint["model"])
    history = checkpoint.get("history", [])
    info = {
        "run_id": config.get("run_id"),
        "family": config["family"],
        "epoch": history[-1]["epoch"] if history else None,
        "training_only": checkpoint.get("training_only"),
        "smoke": bool(config.get("smoke_steps_per_epoch")),
        "min_size": model.transform.min_size[0],
        "max_size": model.transform.max_size,
        "detections_per_img": getattr(model.roi_heads, "detections_per_img", None)
        if hasattr(model, "roi_heads")
        else getattr(model, "detections_per_img", None),
        "last_val_fast": history[-1].get("val_fast") if history else None,
    }
    del checkpoint
    return model.to(device), config, info


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--name", required=True, help="output folder name, e.g. <run_id>_e6"
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="disable fp16 autocast (default: on for CUDA)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep finished clip shards of the same checkpoint",
    )
    parser.add_argument(
        "--clips", nargs="*", default=None, help="only these clips (smoke tests)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="at most N frames per clip (smoke tests)",
    )
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()

    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        raise SystemExit("--name must be a single folder name")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; use --device cpu for a smoke test")
    out_dir = OUTPUT_ROOT / args.name
    shard_dir = out_dir / "shards"
    if out_dir.exists() and not args.resume:
        raise SystemExit(
            f"{out_dir} exists; pass --resume to continue it or choose a new --name"
        )

    start = time.perf_counter()
    checkpoint_sha = sha256(args.checkpoint)
    model, config, info = load_model(args.checkpoint, device, args.allow_smoke)
    amp = not args.no_amp and device.type == "cuda"
    shard_dir.mkdir(
        parents=True, exist_ok=True
    )  # only after the checkpoint passed its checks

    rows = data.load_test_images(args.dataset_root)
    clips: dict[str, list[dict]] = {}
    for row in rows:
        clips.setdefault(row["clip"], []).append(row)
    if args.clips is not None:
        unknown = set(args.clips) - set(clips)
        if unknown:
            raise SystemExit(f"unknown clips: {sorted(unknown)}")
        clips = {c: clips[c] for c in args.clips}
    if args.max_images is not None:
        clips = {c: r[: args.max_images] for c, r in clips.items()}
    partial = args.clips is not None or args.max_images is not None

    timings = {}
    for clip, clip_rows in clips.items():
        shard = shard_dir / f"{clip}.json"
        image_ids = [r["id"] for r in clip_rows]
        if shard.exists():
            saved = json.loads(shard.read_text(encoding="utf-8"))
            if (
                saved["checkpoint_sha256"] != checkpoint_sha
                or saved["image_ids"] != image_ids
            ):
                raise SystemExit(
                    f"{shard} was made by another checkpoint or frame list; use a new --name"
                )
            print(
                f"{clip}: kept finished shard ({len(saved['predictions'])} detections)",
                flush=True,
            )
            continue
        clip_start = time.perf_counter()
        loader = DataLoader(
            TestFrames(clip_rows, args.dataset_root),
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
        predictions = predict(model, loader, config["family"], device, amp)
        seconds = time.perf_counter() - clip_start
        timings[clip] = seconds
        save_json(
            shard,
            {
                "clip": clip,
                "checkpoint_sha256": checkpoint_sha,
                "image_ids": image_ids,
                "seconds": seconds,
                "predictions": predictions,
            },
        )
        print(
            f"{clip}: {len(clip_rows)} frames, {len(predictions)} detections, {seconds:.1f} s",
            flush=True,
        )

    merged = []
    for clip in clips:
        merged.extend(
            json.loads((shard_dir / f"{clip}.json").read_text(encoding="utf-8"))[
                "predictions"
            ]
        )
    save_json(out_dir / "test_predictions.json", merged)

    num_frames = sum(len(r) for r in clips.values())
    results = RESULTS_ROOT / args.name
    results.mkdir(parents=True, exist_ok=True)
    save_json(
        results / "predict.json",
        {
            "checkpoint": args.checkpoint.as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "model": info,
            "config": config,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "amp": amp,
            "batch_size": args.batch_size,
            "partial_test_set": partial,
            "clips": len(clips),
            "frames": num_frames,
            "detections": len(merged),
            "predictions_file": (out_dir / "test_predictions.json").as_posix(),
            "seconds_this_process": time.perf_counter() - start,
            "seconds_new_clips": sum(timings.values()),
            "new_clips": len(timings),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "platform": platform.platform(),
        },
    )
    print(
        f"wrote {out_dir / 'test_predictions.json'}: {num_frames} frames, {len(merged)} detections"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
