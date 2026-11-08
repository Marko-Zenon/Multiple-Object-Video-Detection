"""Train an Ultralytics detector (YOLO11 / RT-DETR) from a JSON run config on the masked export.

Checkpoints go to the git-ignored `outputs/<run_id>/train/weights/` (`last.pt` + `epoch<N>.pt` every epoch,
N 0-based). Evidence goes to `results/<run_id>/`: config.json, hardware.json, runtime.json and the
Ultralytics `results.csv` / `args.yaml` copies. Ultralytics' own validation is off; the per-epoch curve and
the headline metrics come from `scripts/predict_ultra.py` with our evaluator.

    python scripts/train_ultra.py --config configs/runs/yolo_baseline.json --device 0
    python scripts/train_ultra.py --config configs/runs/yolo_baseline.json --device 0 --resume

`--device` is required on purpose: nothing starts on the GPU by default.
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, ultra  # noqa: E402


def save_json(path: Path, value) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def hardware(device: str) -> dict:
    import torch
    import ultralytics

    info = {
        "device": device,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if device != "cpu":
        info["gpu"] = torch.cuda.get_device_name(int(device))
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(
            int(device)
        ).total_memory
    return info


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device", required=True, help="'0' for the first GPU or 'cpu'"
    )
    parser.add_argument(
        "--run-id", help="Use a new run directory with the same settings"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from outputs/<run_id>/train/weights/last.pt",
    )
    args = parser.parse_args()

    config = ultra.load_run_config(args.config)
    if args.run_id:
        config = ultra.validate_run_config({**config, "run_id": args.run_id})
    run_id = config["run_id"]
    output = data.REPO_ROOT / "outputs" / run_id
    results = data.REPO_ROOT / "results" / run_id
    last = output / "train" / "weights" / "last.pt"

    data_yaml = data.REPO_ROOT / config.get("data_yaml", "outputs/yolo_data/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} missing; run scripts/export_yolo.py first"
        )
    if args.resume:
        if not last.exists():
            raise FileNotFoundError(f"no checkpoint to resume: {last}")
        recorded = json.loads((results / "config.json").read_text(encoding="utf-8"))
        if recorded["run_config"] != config:
            raise ValueError("resume requires the original run configuration")
    elif results.exists() or output.exists():
        raise FileExistsError(
            f"{results} or {output} exists; use a new run_id or --resume"
        )

    from ultralytics import YOLO

    weights = data.REPO_ROOT / config["weights"]
    kwargs = ultra.train_kwargs(config, data_yaml, output, args.device)
    results.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        output.mkdir(parents=True)
        protocol = data.load_protocol()
        save_json(
            results / "config.json",
            {
                "run_config": config,
                "ultralytics_train_kwargs": kwargs,
                "weights": config["weights"],
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "protocol_version": protocol["version"],
                "train_split": "train_rubric",
                "export_manifest": json.loads(
                    (data_yaml.parent / "train_rubric_manifest.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "headline_evaluator": "movd/evaluate.py via scripts/predict_ultra.py (Ultralytics validation off)",
            },
        )
        save_json(results / "hardware.json", hardware(args.device))

    start = time.perf_counter()
    if args.resume:
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(str(weights)).train(**kwargs)
    seconds = time.perf_counter() - start

    # Small evidence copies; checkpoints stay in outputs/.
    copies = results / "ultralytics"
    copies.mkdir(exist_ok=True)
    for name in ("results.csv", "args.yaml"):
        if (output / "train" / name).exists():
            shutil.copy2(output / "train" / name, copies / name)
    runtime_path = results / "runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.exists()
        else {"train_process_seconds": []}
    )
    runtime["train_process_seconds"].append(round(seconds, 1))
    runtime["train_seconds_total"] = round(sum(runtime["train_process_seconds"]), 1)
    runtime["epoch_checkpoints"] = sorted(
        p.name for p in (output / "train" / "weights").glob("epoch*.pt")
    )
    save_json(runtime_path, runtime)
    print(
        f"training finished in {seconds / 60:.1f} min; next: python scripts/predict_ultra.py curve --run-id {run_id} --device {args.device}"
    )


if __name__ == "__main__":
    main()
