"""CH-12 same-session speed benchmark of the four families on one GPU, plus model size (GPU; no training).

    python results/ch-12-comparison/speed_session.py [--rounds 3]

Every timed call is a fresh process. The rounds interleave the models, so slow laptop-GPU drifts are spread over all of them.
Every call uses the shared CH-03 harness (`movd.speed.benchmark`: batch 1, the first 30 val_full frames preloaded, 10 warm-up
+ 30 timed calls).

Two timing scopes:

* `common`: the same scope for every family. FP32, the input is a preloaded BGR uint8 array in host memory, and the timed call
  includes the conversion and host-to-GPU copy, the model's own resize and normalisation, inference, NMS (none for RT-DETR) and
  box rescaling. Ultralytics models use `scripts/predict_ultra.py speed` unchanged (that is already its scope). Torchvision
  models do the same conversion (BGR -> RGB float tensor on the GPU) inside the timed call.
* `native_tv`: torchvision only. The original CH-04 scope (`movd.tv_train.measure_validation_speed`: GPU-resident tensors,
  AMP autocast as in training). It reproduces the numbers stored in results/<run>/speed.json.

Checkpoints are read where their chunks left them (read-only): CH-07/CH-08 in the main checkout's outputs/ch07 and
outputs/ch08 worktrees, CH-09/CH-10 in MOVD-ch10/outputs, CH-11 in MOVD-ch11/outputs. Each is hashed. Writes
speed_same_session/raw/*.json, speed_same_session/summary.json and model_size.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HW = ROOT.parent
MAIN = HW / "Multiple-Object-Video-Detection" / "outputs"
OUT = ROOT / "results" / "ch-12-comparison" / "speed_same_session"
# run -> (framework, checkpoint, family, imgsz)
MODELS = {
    "frcnn_baseline": (
        "tv",
        MAIN / "ch07/outputs/frcnn_baseline/epoch_0006.pt",
        "frcnn",
        1344,
    ),
    "frcnn_rpn_pre3000": (
        "tv",
        MAIN / "ch07/outputs/frcnn_rpn_pre3000/epoch_0006.pt",
        "frcnn",
        1344,
    ),
    "retina_baseline": (
        "tv",
        MAIN / "ch08/outputs/retina_baseline/epoch_0006.pt",
        "retina",
        1344,
    ),
    "retina_anchors": (
        "tv",
        MAIN / "ch08/outputs/retina_anchors/epoch_0006.pt",
        "retina",
        1344,
    ),
    "yolo_baseline_marko_3050": (
        "ultra",
        HW / "MOVD-ch10/outputs/yolo_baseline_marko_3050/train/weights/last.pt",
        "yolo11",
        1344,
    ),
    "yolo_p2": (
        "ultra",
        HW / "MOVD-ch10/outputs/yolo_p2/train/weights/last.pt",
        "yolo11",
        1344,
    ),
    "yolo_imgsz1600": (
        "ultra",
        HW / "MOVD-ch10/outputs/yolo_imgsz1600/train/weights/last.pt",
        "yolo11",
        1600,
    ),
    "rtdetr_baseline": (
        "ultra",
        HW / "MOVD-ch11/outputs/rtdetr_baseline/train/weights/last.pt",
        "rtdetr",
        1344,
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tv_model(run: str):
    import torch

    from movd import tv_train

    checkpoint = torch.load(MODELS[run][1], map_location="cpu", weights_only=False)
    config = json.loads(
        (ROOT / "results" / run / "config.json").read_text(encoding="utf-8")
    )
    model = tv_train.build_model(config, initialize_weights=False)
    model.load_state_dict(checkpoint["model"])
    return model.cuda().eval(), config


def tv_one(run: str, scope: str, out: Path) -> None:
    import cv2
    import torch

    from movd import tv_train, ultra
    from movd.speed import benchmark

    model, config = tv_model(run)
    if scope == "native_tv":
        result = tv_train.measure_validation_speed(model, config, config["amp"])
    else:
        frames = [cv2.imread(str(p)) for _, p in ultra.val_frames("val_full")[:30]]

        def forward(frame):
            tensor = (
                torch.from_numpy(frame[:, :, ::-1].copy())
                .cuda()
                .permute(2, 0, 1)
                .float()
                / 255
            )
            with torch.no_grad():
                return model([tensor])

        result = benchmark(
            forward,
            frames,
            "cuda",
            batch_size=1,
            warmup=10,
            input_size=f"{model.transform.min_size[0]}-{model.transform.max_size}",
            note="common scope: FP32; preloaded BGR uint8 host array; includes BGR->RGB, host-to-GPU copy, "
            "detector resize/normalisation, inference, NMS and box rescaling",
        )
    out.write_text(
        json.dumps({**result, "run": run, "scope": scope}, indent=2) + "\n",
        encoding="utf-8",
    )


def call(run: str, scope: str, out: Path) -> float:
    framework, weights, family, imgsz = MODELS[run]
    if framework == "tv":
        cmd = [
            sys.executable,
            __file__,
            "--tv-one",
            run,
            "--scope",
            scope,
            "--out",
            str(out),
        ]
    else:
        cmd = [
            sys.executable,
            "scripts/predict_ultra.py",
            "speed",
            "--device",
            "0",
            "--weights",
            str(weights),
            "--family",
            family,
            "--imgsz",
            str(imgsz),
            "--out",
            str(out),
        ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return json.loads(out.read_text(encoding="utf-8"))["fps"]


def model_size() -> dict:
    import torch

    from movd import rtdetr

    sizes = {}
    for run, (framework, weights, family, _) in MODELS.items():
        if framework == "tv":
            model, _ = tv_model(run)
        else:
            model = rtdetr.load_model(weights, family).model
        sizes[run] = {
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "checkpoint_bytes": weights.stat().st_size,
            "checkpoint_note": "torchvision epoch file includes optimizer and scaler state"
            if framework == "tv"
            else "Ultralytics last.pt (stripped, half-precision EMA weights)",
        }
        del model
        torch.cuda.empty_cache()
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tv-one", help=argparse.SUPPRESS)
    parser.add_argument("--scope", default="common", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.tv_one:
        tv_one(args.tv_one, args.scope, args.out)
        return
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    started = datetime.now(ZoneInfo("Europe/Kyiv")).isoformat(timespec="seconds")
    jobs = [(run, "common") for run in MODELS] + [
        (r, "native_tv") for r, m in MODELS.items() if m[0] == "tv"
    ]
    for k in range(1, args.rounds + 1):
        for run, scope in jobs:
            fps = call(run, scope, OUT / "raw" / f"{run}_{scope}_r{k}.json")
            print(f"round {k} {run} {scope}: {fps:.2f} FPS", flush=True)
    runs = {}
    for run, scope in jobs:
        raw = [
            json.loads((OUT / "raw" / f"{run}_{scope}_r{k}.json").read_text())
            for k in range(1, args.rounds + 1)
        ]
        entry = runs.setdefault(
            run,
            {
                "framework": MODELS[run][0],
                "family": MODELS[run][2],
                "imgsz": MODELS[run][3],
                "checkpoint": MODELS[run][1].relative_to(HW).as_posix(),
                "checkpoint_sha256": sha256(MODELS[run][1]),
                "gpu": raw[0]["gpu"],
            },
        )
        entry[scope] = {
            "fps_median": statistics.median(r["fps"] for r in raw),
            "fps_rounds": [r["fps"] for r in raw],
            "latency_ms_median": statistics.median(
                r["latency_ms_per_image"] for r in raw
            ),
            "note": raw[0]["note"],
        }
    summary = {
        "started_kyiv": started,
        "finished_kyiv": datetime.now(ZoneInfo("Europe/Kyiv")).isoformat(
            timespec="seconds"
        ),
        "rounds": args.rounds,
        "runs": runs,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    sizes = model_size()
    (ROOT / "results" / "ch-12-comparison" / "model_size.json").write_text(
        json.dumps(sizes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({r: round(v["common"]["fps_median"], 2) for r, v in runs.items()}))


if __name__ == "__main__":
    main()
