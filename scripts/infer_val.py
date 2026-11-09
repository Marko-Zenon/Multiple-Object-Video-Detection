"""CH-13 inference experiments on val: cache raw RT-DETR passes once, evaluate every post-processing variant on CPU.

Subcommands:

* `cache --model M --imgsz N --views {full,flip,tiles} --split S --device 0`: one GPU pass per view of every frame
  of a val split -> raw decoder outputs (300 or 500 query boxes + 8 class scores) in the ignored
  `outputs/ch13/raw/<model>_<imgsz>_<view>_<split>.npz`; timing, peak GPU memory and the checkpoint hash in
  `results/ch-13-inference/passes/<model>_<imgsz>_<views>_<split>.json`. An existing cache is reused.
* `eval --variant V [V ...] --split S`: a variant of `results/ch-13-inference/variants.json` from the caches ->
  COCO results (ignored `outputs/ch13/preds/`) -> the project evaluator -> `results/ch-13-inference/runs/<split>/<V>/`
  (metrics.json, per_class.csv, pipeline.json with detection counts and CPU post-processing time).
* `speed --variant V [V ...] --device 0`: the same-session end-to-end benchmark (`movd.speed.benchmark`, first 30
  val_full frames preloaded, every pass + decoding + merge) -> `results/ch-13-inference/speed/`.

`--weights-root` lets the checkpoints live elsewhere (default: this repository's `outputs/`).
The tables and figures are built by `results/ch-13-inference/make_tables.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, infer, rtdetr, ultra  # noqa: E402
from movd import evaluate as ev  # noqa: E402

RESULTS = data.REPO_ROOT / "results" / "ch-13-inference"
OUTPUTS = data.REPO_ROOT / "outputs" / "ch13"
VARIANTS = RESULTS / "variants.json"


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_variants() -> dict:
    return json.loads(VARIANTS.read_text(encoding="utf-8"))


def find_variant(name: str) -> dict:
    doc = load_variants()
    for v in doc["variants"] + doc["combinations"]:
        if v["name"] == name:
            return v
    raise KeyError(f"no variant {name!r} in {VARIANTS}")


def weights_path(model: str, root: Path | None) -> Path:
    spec = load_variants()["models"][model]
    path = (root or data.REPO_ROOT) / spec["weights"]
    return path


def checked_model(model: str, root: Path | None):
    """Open a registered checkpoint after checking its sha256 against variants.json."""
    spec = load_variants()["models"][model]
    path = weights_path(model, root)
    digest = sha256(path)
    if digest != spec["sha256"]:
        raise RuntimeError(f"{path}: sha256 {digest} != registered {spec['sha256']}")
    loaded = rtdetr.load_model(path, "rtdetr")
    if rtdetr.num_queries(loaded) != spec["queries"]:
        raise RuntimeError(
            f"{model}: {rtdetr.num_queries(loaded)} queries, expected {spec['queries']}"
        )
    return loaded, digest


def raw_path(model: str, imgsz: int, view: str, split: str) -> Path:
    return OUTPUTS / "raw" / f"{model}_{imgsz}_{view}_{split}.npz"


# ---------------------------------------------------------------- cache


def cache(args) -> None:
    import cv2
    import torch

    from movd.speed import device_info

    views = infer.view_names(args.views)
    paths = {v: raw_path(args.model, args.imgsz, v, args.split) for v in views}
    record = (
        RESULTS / "passes" / f"{args.model}_{args.imgsz}_{args.views}_{args.split}.json"
    )
    if all(p.exists() for p in paths.values()) and record.exists():
        print(f"cached: {[p.name for p in paths.values()]}")
        return
    model, digest = checked_model(args.model, args.weights_root)
    frames = ultra.val_frames(args.split, args.dataset_root)
    boxes = {v: [] for v in views}
    scores = {v: [] for v in views}
    infer.prepare(model, args.imgsz, args.device)
    if args.device != "cpu":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    from tqdm import tqdm

    gpu_seconds = 0.0
    start = time.perf_counter()
    for _, path in tqdm(
        frames,
        desc=f"{args.model} {args.imgsz} {args.views} {args.split}",
        unit="frame",
    ):
        image = cv2.imread(str(path))
        t0 = time.perf_counter()
        for v in views:
            raw = infer.raw_pass(model, image, args.imgsz, args.device, v)
            boxes[v].append(raw.boxes)
            scores[v].append(raw.scores)
        gpu_seconds += time.perf_counter() - t0
    wall = time.perf_counter() - start
    ids = np.array([i for i, _ in frames], dtype=np.int64)
    for v in views:
        paths[v].parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            paths[v],
            image_ids=ids,
            boxes=np.stack(boxes[v]),
            scores=np.stack(scores[v]),
        )
    info = device_info(args.device if args.device == "cpu" else "cuda")
    peak = {}
    if args.device != "cpu":
        peak = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    save_json(
        record,
        {
            "model": args.model,
            "checkpoint": load_variants()["models"][args.model]["weights"],
            "checkpoint_sha256": digest,
            "imgsz": args.imgsz,
            "views": views,
            "split": args.split,
            "num_frames": len(frames),
            "passes_per_frame": len(views),
            "wall_seconds": round(wall, 1),
            "pass_seconds": round(gpu_seconds, 1),
            "seconds_per_frame_all_views": round(gpu_seconds / len(frames), 4),
            "note": "pass_seconds = Ultralytics predict (resize, normalisation, forward, its own decoding) + the raw "
            "capture, for every view of each frame; excludes reading the JPEG from disk",
            "ultralytics": __import__("ultralytics").__version__,
            "finished_kyiv": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **peak,
            **info,
        },
    )
    print(f"{len(frames)} frames x {len(views)} views in {wall:.0f} s -> {record.name}")


def load_raw(
    model: str, imgsz: int, view: str, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = raw_path(model, imgsz, view, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `cache --model {model} --imgsz {imgsz} ... --split {split}`"
        )
    with np.load(path) as z:
        return z["image_ids"], z["boxes"], z["scores"]


# ---------------------------------------------------------------- eval


def predictions(config: dict, split: str) -> tuple[list[dict], dict]:
    specs = infer.pass_specs(config)
    caches = [load_raw(s["model"], s["imgsz"], s["view"], split) for s in specs]
    ids = caches[0][0]
    for c in caches[1:]:
        if not np.array_equal(c[0], ids):
            raise RuntimeError("raw caches cover different frames")
    detections, counts = [], []
    start = time.perf_counter()
    for row, image_id in enumerate(ids.tolist()):
        raws = [infer.Raw(c[1][row], c[2][row]) for c in caches]
        dets = infer.combine(raws, config)
        counts.append(len(dets))
        detections.extend(infer.to_coco(image_id, dets))
    seconds = time.perf_counter() - start
    counts = np.array(counts)
    stats = {
        "num_frames": len(ids),
        "num_detections": int(counts.sum()),
        "detections_per_frame_mean": round(float(counts.mean()), 2),
        "detections_per_frame_min": int(counts.min()),
        "detections_per_frame_max": int(counts.max()),
        "frames_at_500_cap": int((counts >= 500).sum()),
        "num_raw_passes_per_frame": len(specs),
        "cpu_postprocess_seconds": round(seconds, 2),
        "cpu_postprocess_ms_per_frame": round(1000 * seconds / len(ids), 2),
    }
    return detections, stats


def evaluate_cmd(args) -> None:
    for name in args.variant:
        variant = find_variant(name)
        detections, stats = predictions(variant["config"], args.split)
        out = OUTPUTS / "preds" / f"{args.split}_{name}.json"
        save_json(out, detections)
        result = ev.evaluate_split(detections, args.split, args.dataset_root)
        run_dir = RESULTS / "runs" / args.split / name
        ev.write_outputs(result, run_dir)
        save_json(
            run_dir / "pipeline.json",
            {
                "variant": name,
                "axis": variant["axis"],
                "split": args.split,
                "config": variant["config"],
                **stats,
            },
        )
        m = result["metrics"]
        print(
            f"{args.split} {name:16s} mAP {m['mAP']:.4f} AP50 {m['AP50']:.4f} AP75 {m['AP75']:.4f} "
            f"APs {m['AP_small']:.4f} dets/frame {stats['detections_per_frame_mean']}"
        )


# ---------------------------------------------------------------- speed


def speed_cmd(args) -> None:
    import cv2

    from movd.speed import benchmark

    names = list(args.variant)
    configs = {n: find_variant(n)["config"] for n in names}
    needed = sorted({s["model"] for c in configs.values() for s in infer.pass_specs(c)})
    models = {m: checked_model(m, args.weights_root)[0] for m in needed}
    frames = [
        cv2.imread(str(p))
        for _, p in ultra.val_frames("val_full", args.dataset_root)[:30]
    ]
    rows = {}
    for name in names:
        config = configs[name]
        result = benchmark(
            lambda frame, c=config: infer.predict_image(models, frame, c, args.device),
            frames,
            device="cuda" if args.device != "cpu" else "cpu",
            batch_size=1,
            warmup=10,
            input_size=", ".join(
                f"{s['model']}@{s['imgsz']}:{s['view']}"
                for s in infer.pass_specs(config)
            ),
            note="CH-13 end-to-end pipeline: first 30 val_full frames preloaded as BGR arrays (no disk I/O); every "
            "raw pass (Ultralytics resize, normalisation, forward) + raw capture + CPU decoding, merge and the "
            "500 cap; FP32, batch 1",
        )
        result["variant"] = name
        result["passes_per_frame"] = len(infer.pass_specs(config))
        result["test_minutes_3811_frames"] = round(3811 / result["fps"] / 60, 1)
        rows[name] = result
        save_json(RESULTS / "speed" / f"{name}.json", result)
        print(
            f"{name:16s} {result['fps']:.2f} FPS, {result['test_minutes_3811_frames']} min for 3,811 test frames"
        )
    summary = RESULTS / "speed" / "session.json"
    previous = (
        json.loads(summary.read_text(encoding="utf-8"))
        if summary.exists()
        else {"sessions": []}
    )
    previous["sessions"].append(
        {
            "finished_kyiv": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "order": names,
            "fps": {n: rows[n]["fps"] for n in names},
        }
    )
    save_json(summary, previous)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("cache", "eval", "speed"):
        p = sub.add_parser(name)
        p.add_argument("--dataset-root", type=Path, default=None)
        p.add_argument("--weights-root", type=Path, default=None)
        if name in ("cache", "speed"):
            p.add_argument("--device", required=True)
        if name == "cache":
            p.add_argument("--model", required=True)
            p.add_argument("--imgsz", type=int, required=True)
            p.add_argument("--views", choices=("full", "flip", "tiles"), default="full")
        if name in ("cache", "eval"):
            p.add_argument("--split", choices=("val_fast", "val_full"), required=True)
        if name in ("eval", "speed"):
            p.add_argument("--variant", nargs="+", required=True)
    args = parser.parse_args()
    {"cache": cache, "eval": evaluate_cmd, "speed": speed_cmd}[args.command](args)


if __name__ == "__main__":
    main()
