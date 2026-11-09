"""CH-14 inference with the frozen CH-13 pipeline (`configs/runs/final_inference.json`), resumable per clip.

    python scripts/ch14_infer.py val  --weights outputs/final_rtdetr/train/weights/last.pt --name final_rtdetr --device 0
    python scripts/ch14_infer.py test --weights outputs/final_rtdetr/train/weights/last.pt --name final_rtdetr --device 0

`val` (default split val_full): every frame goes through `movd.infer.predict_image` (6 passes: full, flip, 2x2
tiles at imgsz 1344, class-wise NMS 0.5, top 500). The full-frame pass also returns the plain Ultralytics result,
which is the unchanged single-pass prediction (`ultra.predict_kwargs`: conf 0.001, the CH-11 / predict_ultra
settings), so both modes come from the same forward pass. Both are scored by the project evaluator.
Outputs: shards in the ignored `outputs/ch14/<name>/<split>/shards/<clip>.json`; evidence in
`results/ch-14-final/<name>/<split>/{single,pipeline}/` (metrics.json, per_class.csv) and `run.json`.

`test`: the pipeline only, over the 3,811 `test_images.csv` frames (unmasked originals), one shard per clip, then the
merged COCO-results list `outputs/ch14/<name>/test/test_predictions.json` (`image_id` = 1-based CSV row) for the
CH-05 writer (`scripts/make_submission.py`). Evidence: `results/ch-14-final/<name>/test/run.json`.

A shard is written atomically when its clip is complete, so a stopped run restarts at a clip boundary.
`--clips` and `--max-images` are for smoke tests only; such runs are marked `smoke` in run.json.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, infer, rtdetr, ultra  # noqa: E402
from movd import evaluate as ev  # noqa: E402

RESULTS = data.REPO_ROOT / "results" / "ch-14-final"
OUTPUTS = data.REPO_ROOT / "outputs" / "ch14"
PIPELINE = data.REPO_ROOT / "configs" / "runs" / "final_inference.json"


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path: Path) -> dict:
    """Hash and training arguments of an Ultralytics checkpoint (refuses smoke checkpoints)."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("train_args", {})
    identity = {
        "checkpoint": path.resolve().as_posix(),
        "checkpoint_sha256": sha256(path),
        "checkpoint_bytes": path.stat().st_size,
        "epoch_field": ckpt.get("epoch"),
        "train_args_epochs": args.get("epochs"),
        "train_args_imgsz": args.get("imgsz"),
        "train_args_batch": args.get("batch"),
        "train_args_seed": args.get("seed"),
        "train_args_fraction": args.get("fraction"),
        "train_args_data": args.get("data"),
        "train_args_project": args.get("project"),
    }
    del ckpt
    return identity


def frames(split: str, root) -> list[tuple[str, int, Path]]:
    """(clip, image_id, path) of every frame of a val split or the test set, in a fixed order."""
    if split == "test":
        rows = data.load_test_images(root)
        return [(r["clip"], r["id"], data.test_image_path(r, root)) for r in rows]
    images = {im["id"]: im for im in data.load_coco("val", root)["images"]}
    out = []
    for image_id in data.split_image_ids(split):
        image = images[image_id]
        out.append(
            (data.clip_name(image), image_id, data.image_path("val", image, root))
        )
    return out


def run(args) -> None:
    import cv2
    import torch
    import ultralytics

    split = "test" if args.command == "test" else args.split
    smoke = bool(args.clips or args.max_images)
    out_dir = OUTPUTS / args.name / split
    shards = out_dir / "shards"
    evidence = RESULTS / args.name / split
    config = json.loads(PIPELINE.read_text(encoding="utf-8"))["config"]
    identity = checkpoint_identity(args.weights)
    if not smoke and identity["train_args_fraction"] not in (None, 1.0):
        raise SystemExit(
            f"{args.weights} is a smoke checkpoint (fraction {identity['train_args_fraction']})"
        )

    by_clip = defaultdict(list)
    for clip, image_id, path in frames(split, args.dataset_root):
        if not args.clips or clip in args.clips:
            by_clip[clip].append((image_id, path))
    if args.max_images:
        by_clip = {c: v[: args.max_images] for c, v in by_clip.items()}

    model = rtdetr.load_model(args.weights, "rtdetr")
    models = {name: model for name in {p["model"] for p in config["passes"]}}
    device = args.device
    if device != "cpu":
        torch.zeros(
            1, device=f"cuda:{device}"
        )  # initialise CUDA before resetting its counters
        torch.cuda.reset_peak_memory_stats(int(device))
    seconds, computed = 0.0, 0
    for clip in sorted(by_clip):
        shard = shards / f"{clip}.json"
        if shard.exists():
            continue
        single, pipeline = [], []
        start = time.perf_counter()
        for image_id, path in by_clip[clip]:
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"cannot read {path}")
            specs = infer.pass_specs(config)
            raws = []
            for spec in specs:
                if spec["view"] == "full" and split != "test":
                    raw, result = infer.raw_pass(
                        models[spec["model"]], image, spec["imgsz"], device, "full",
                        config.get("tile_grid", 2), config.get("tile_frac", 0.6), return_result=True,
                    )  # fmt: skip
                    single.extend(ultra.result_to_coco(result, image_id))
                else:
                    raw = infer.raw_pass(
                        models[spec["model"]], image, spec["imgsz"], device, spec["view"],
                        config.get("tile_grid", 2), config.get("tile_frac", 0.6),
                    )  # fmt: skip
                raws.append(raw)
            pipeline.extend(infer.to_coco(image_id, infer.combine(raws, config)))
        seconds += time.perf_counter() - start
        computed += len(by_clip[clip])
        save_json(
            shard,
            {
                "clip": clip,
                "frames": len(by_clip[clip]),
                "single": single,
                "pipeline": pipeline,
            },
        )
        print(
            f"{split} {clip}: {len(by_clip[clip])} frames, {len(pipeline)} dets, {seconds / 60:.1f} min",
            flush=True,
        )

    merged = {"single": [], "pipeline": []}
    for clip in sorted(by_clip):
        part = json.loads((shards / f"{clip}.json").read_text(encoding="utf-8"))
        for key, values in merged.items():
            values.extend(part[key])
    expected = sum(len(v) for v in by_clip.values())
    record = {
        **identity,
        "split": split,
        "smoke": smoke,
        "pipeline_config": str(PIPELINE.relative_to(data.REPO_ROOT).as_posix()),
        "pipeline_config_sha256": sha256(PIPELINE),
        "config": config,
        "frames": expected,
        "clips": len(by_clip),
        "frames_computed_this_run": computed,
        "inference_seconds_this_run": round(seconds, 1),
        "fps_this_run": round(computed / seconds, 3) if seconds else None,
        "gpu": torch.cuda.get_device_name(int(device)) if device != "cpu" else "cpu",
        "peak_cuda_allocated_bytes_this_run": torch.cuda.max_memory_allocated(
            int(device)
        )
        if device != "cpu"
        else None,
        "peak_cuda_reserved_bytes_this_run": torch.cuda.max_memory_reserved(int(device))
        if device != "cpu"
        else None,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "platform": platform.platform(),
        "detections": {k: len(v) for k, v in merged.items()},
    }
    if split == "test":
        ids = {p["image_id"] for p in merged["pipeline"]}
        record["images_with_detections"] = len(ids)
        if not smoke and expected != 3811:
            raise SystemExit(f"test run covers {expected} frames, expected 3811")
        predictions = out_dir / "test_predictions.json"
        save_json(predictions, merged["pipeline"])
        record["test_predictions"] = predictions.relative_to(data.REPO_ROOT).as_posix()
        record["test_predictions_sha256"] = sha256(predictions)
    else:
        for key, predictions in merged.items():
            path = out_dir / f"{key}_predictions.json"
            save_json(path, predictions)
            result = ev.evaluate(
                predictions,
                data.load_coco("val", args.dataset_root),
                [i for v in by_clip.values() for i, _ in v],
            )
            result["split"] = split
            ev.write_outputs(result, evidence / key)
            record[f"{key}_predictions_sha256"] = sha256(path)
            record[f"{key}_metrics"] = result["metrics"]
            print(f"{key}: {ev.format_summary(result)}", flush=True)
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"evidence: {evidence / 'run.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("val", "test"):
        p = sub.add_parser(name)
        p.add_argument("--weights", type=Path, required=True)
        p.add_argument("--name", required=True, help="output/evidence folder name")
        p.add_argument("--device", required=True)
        p.add_argument("--dataset-root", type=Path, default=None)
        p.add_argument("--clips", nargs="+", default=None, help="smoke only")
        p.add_argument(
            "--max-images", type=int, default=None, help="smoke only: frames per clip"
        )
        if name == "val":
            p.add_argument(
                "--split", choices=("val_full", "val_fast"), default="val_full"
            )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
