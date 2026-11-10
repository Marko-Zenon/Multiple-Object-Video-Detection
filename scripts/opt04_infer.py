"""OPT-04 inference: the unchanged CH-14 runner (`scripts/ch14_infer.py`) with OPT-04 folders and an optional pipeline.

    python scripts/opt04_infer.py val  --weights <ckpt> --name P_rtdetrx_e1 --split val_fast --device 0
    python scripts/opt04_infer.py test --weights <ckpt> --name P_rtdetrx_e1 --device 0
    python scripts/opt04_infer.py --pipeline configs/runs/opt04_inference_1600.json val --weights <ckpt> --name FB_... --device 0
    python scripts/opt04_infer.py speed --out <name> --model A=<ckpt> P=<ckpt> FB=<ckpt>@configs/runs/opt04_inference_1600.json --device 0

`val` / `test` are `scripts/ch14_infer.py` unchanged (CH-13 pipeline, project evaluator, resumable per-clip shards).
Only the folders change, so no earlier evidence is touched: evidence -> `results/opt-04-model-scale/<name>/<split>/`,
shards and predictions -> `outputs/opt04/<name>/<split>/`. `--pipeline` (default `configs/runs/final_inference.json`)
replaces the pipeline config; run.json records its path and sha256.

`speed`: the CH-13 / CH-14 same-session benchmark (`movd.speed.benchmark`, first 30 val_full frames preloaded, FP32,
batch 1), except that each model may name its own pipeline (`NAME=WEIGHTS@PIPELINE`), so Model A runs at 1344 and the
1600 fallback at 1600 in one session. Single-pass mode uses the full-frame pass at the pipeline's input size.
Output: `results/opt-04-model-scale/speed/<out>.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch14_infer  # noqa: E402

from movd import data, infer, rtdetr  # noqa: E402

RESULTS = data.REPO_ROOT / "results" / "opt-04-model-scale"
OUTPUTS = data.REPO_ROOT / "outputs" / "opt04"
DEFAULT_PIPELINE = data.REPO_ROOT / "configs" / "runs" / "final_inference.json"


def single_pass(pipeline: dict) -> dict:
    """The CH-13 single-pass reference config at the pipeline's (single) input size."""
    sizes = {spec["imgsz"] for spec in infer.pass_specs(pipeline)}
    if len(sizes) != 1:
        raise ValueError(f"pipeline uses several input sizes: {sorted(sizes)}")
    config = json.loads(json.dumps(ch14_infer.SINGLE_PASS))
    config["passes"][0]["imgsz"] = sizes.pop()
    return config


def parse_model_spec(spec: str) -> tuple[str, Path, Path]:
    """`NAME=WEIGHTS[@PIPELINE]` -> (name, weights, pipeline path)."""
    if "=" not in spec:
        raise ValueError(f"expected NAME=WEIGHTS[@PIPELINE], got {spec!r}")
    name, rest = spec.split("=", 1)
    weights, _, pipeline = rest.partition("@")
    pipeline_path = data.REPO_ROOT / pipeline if pipeline else DEFAULT_PIPELINE
    return name, Path(weights), pipeline_path


def speed(args) -> None:
    import cv2
    import torch

    from movd.speed import benchmark

    torch.zeros(
        1, device=f"cuda:{args.device}"
    )  # initialise CUDA before resetting its counters
    frames = [
        cv2.imread(str(p))
        for _, _, p in ch14_infer.frames("val_full", args.dataset_root)[:30]
    ]
    session = {"started_local": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": []}
    for spec in args.model:
        name, weights, pipeline_path = parse_model_spec(spec)
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))["config"]
        model = rtdetr.load_model(weights, "rtdetr")
        models = {"baseline": model}
        for mode, config in (("single", single_pass(pipeline)), ("pipeline", pipeline)):
            torch.cuda.reset_peak_memory_stats(int(args.device))
            result = benchmark(
                lambda frame, c=config, m=models: infer.predict_image(
                    m, frame, c, args.device
                ),
                frames,
                device=f"cuda:{args.device}",
                batch_size=1,
                warmup=10,
                input_size=", ".join(
                    f"{s['imgsz']}:{s['view']}" for s in infer.pass_specs(config)
                ),
                note="CH-13 method: first 30 val_full frames preloaded as BGR arrays (no disk I/O); every raw pass "
                "+ CPU decoding, merge and the 500 cap; FP32, batch 1",
            )
            result.update(
                name=name,
                weights=weights.resolve().as_posix(),
                weights_sha256=ch14_infer.sha256(weights),
                pipeline_config=pipeline_path.relative_to(data.REPO_ROOT).as_posix(),
                pipeline_config_sha256=ch14_infer.sha256(pipeline_path),
                mode=mode,
                passes_per_frame=len(infer.pass_specs(config)),
                test_minutes_3811_frames=round(3811 / result["fps"] / 60, 1),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(
                    int(args.device)
                ),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(
                    int(args.device)
                ),
            )
            session["rows"].append(result)
            print(
                f"{name:14s} {mode:8s} {result['fps']:.2f} FPS, test {result['test_minutes_3811_frames']} min, "
                f"peak {result['peak_cuda_allocated_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
        del model, models
        torch.cuda.empty_cache()
    session["finished_local"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = RESULTS / "speed" / f"{args.out}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
    print(f"speed: {out}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    known, rest = pre.parse_known_args(argv)
    if rest and rest[0] == "speed":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command")
        parser.add_argument("--model", nargs="+", required=True)
        parser.add_argument(
            "--out", required=True, help="results/opt-04-model-scale/speed/<out>.json"
        )
        parser.add_argument("--device", required=True)
        parser.add_argument("--dataset-root", type=Path, default=None)
        speed(parser.parse_args(rest))
        return
    pipeline = (
        known.pipeline
        if known.pipeline.is_absolute()
        else data.REPO_ROOT / known.pipeline
    )
    if not pipeline.exists():
        raise SystemExit(f"pipeline config {pipeline} missing")
    ch14_infer.RESULTS = RESULTS
    ch14_infer.OUTPUTS = OUTPUTS
    ch14_infer.PIPELINE = pipeline
    sys.argv = [sys.argv[0], *rest]
    ch14_infer.main()


if __name__ == "__main__":
    main()
