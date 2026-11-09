"""Same-session speed benchmark on one GPU: every checkpoint timed in interleaved rounds (GPU, after training).

    python results/ch-10-yolo-tuning/speed_session.py [--rounds 3]

Each round runs `scripts/predict_ultra.py speed` once per checkpoint (a fresh process each time, the shared CH-03
benchmark: batch 1, FP32, first 30 val_full frames preloaded, 10 warm-up + 30 timed calls, letterbox/NMS/rescale
included). Interleaving the rounds spreads slow drifts of a laptop GPU (clocks, temperature) over all models.
Writes speed_same_session/raw/<run>_r<k>.json and speed_same_session/summary.json (median FPS per run).
The CH-11 RT-DETR baseline checkpoint is read from its own worktree (read-only) when present, so CH-12 can
compare the Ultralytics families within one session.
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
OUT = ROOT / "results" / "ch-10-yolo-tuning" / "speed_same_session"
CH11_OUTPUTS = ROOT.parent / "MOVD-ch11" / "outputs"
# (run, checkpoint, family, imgsz)
MODELS = [
    (
        "yolo_baseline_marko_3050",
        ROOT / "outputs/yolo_baseline_marko_3050/train/weights/last.pt",
        "yolo11",
        1344,
    ),
    (
        "yolo_baseline_5070",
        ROOT / "outputs/yolo_baseline_5070/train/weights/last.pt",
        "yolo11",
        1344,
    ),
    ("yolo_p2", ROOT / "outputs/yolo_p2/train/weights/last.pt", "yolo11", 1344),
    (
        "yolo_imgsz1600",
        ROOT / "outputs/yolo_imgsz1600/train/weights/last.pt",
        "yolo11",
        1600,
    ),
    (
        "rtdetr_baseline",
        CH11_OUTPUTS / "rtdetr_baseline/train/weights/last.pt",
        "rtdetr",
        1344,
    ),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    models = [m for m in MODELS if m[1].exists()]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    started = datetime.now(ZoneInfo("Europe/Kyiv")).isoformat(timespec="seconds")
    for k in range(1, args.rounds + 1):
        for run, weights, family, imgsz in models:
            out = OUT / "raw" / f"{run}_r{k}.json"
            subprocess.run(
                [
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
                ],
                cwd=ROOT,
                check=True,
            )
            print(
                f"round {k} {run}: {json.loads(out.read_text())['fps']:.2f} FPS",
                flush=True,
            )
    runs = {}
    for run, weights, family, imgsz in models:
        raw = [
            json.loads((OUT / "raw" / f"{run}_r{k}.json").read_text())
            for k in range(1, args.rounds + 1)
        ]
        fps = [r["fps"] for r in raw]
        runs[run] = {
            "family": family,
            "imgsz": imgsz,
            "fps_median": statistics.median(fps),
            "fps_rounds": fps,
            "latency_ms_median": statistics.median(
                r["latency_ms_per_image"] for r in raw
            ),
            "gpu": raw[0]["gpu"],
            "checkpoint": weights.relative_to(ROOT.parent).as_posix(),
            "checkpoint_sha256": sha256(weights),
        }
    summary = {
        "started_kyiv": started,
        "finished_kyiv": datetime.now(ZoneInfo("Europe/Kyiv")).isoformat(
            timespec="seconds"
        ),
        "rounds": args.rounds,
        "benchmark": "scripts/predict_ultra.py speed (movd.speed.benchmark): batch 1, FP32, 30 preloaded val_full "
        "frames, 10 warm-up + 30 timed; YOLO: letterbox + NMS + rescale; RT-DETR: square stretch, no NMS",
        "missing": [m[0] for m in MODELS if not m[1].exists()],
        "runs": runs,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({r: round(v["fps_median"], 2) for r, v in runs.items()}))


if __name__ == "__main__":
    main()
