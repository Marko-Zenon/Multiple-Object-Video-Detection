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

from movd import data, rtdetr, ultra  # noqa: E402


class SmokeCheckpointStop(Exception):
    """Intentional stop after an unfinished smoke checkpoint is fully written."""


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
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Smoke only: stop after N completed epochs, preserving resumable state",
    )
    parser.add_argument(
        "--observe",
        action="store_true",
        help="Record finite batch losses, counts and CUDA peaks at each saved epoch",
    )
    args = parser.parse_args()

    config = ultra.load_run_config(args.config)
    if args.run_id:
        config = ultra.validate_run_config({**config, "run_id": args.run_id})
    if args.stop_after_epoch is not None and (
        "smoke_fraction" not in config
        or not 0 < args.stop_after_epoch < config["epochs"]
    ):
        parser.error("--stop-after-epoch requires a smoke run and 0 < N < epochs")
    # OPT-04: a full run whose config sets stop_after_epoch stops right after that epoch's checkpoint is written,
    # through the same callback as the smoke flag. It always records observations, and it is never resumed
    # (a resume would continue past the stop epoch); a crashed run is restarted under a new --run-id instead.
    config_stop = config.get("stop_after_epoch")
    if config_stop is not None:
        if args.stop_after_epoch is not None:
            parser.error("the run config already sets stop_after_epoch")
        if args.resume:
            parser.error(
                "a stop_after_epoch run is not resumed; restart it with a new --run-id"
            )
        args.observe = True
    stop_after = args.stop_after_epoch if config_stop is None else config_stop
    run_id = config["run_id"]
    output = data.REPO_ROOT / "outputs" / run_id
    results = data.REPO_ROOT / "results" / run_id
    last = output / "train" / "weights" / "last.pt"

    data_yaml = data.REPO_ROOT / config.get("data_yaml", "outputs/yolo_data/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} missing; run scripts/export_yolo.py first"
        )
    # The training list the config selects (train_split, default train_rubric), checked against data.yaml,
    # the export manifest and the frozen split file before anything is written or trained.
    manifest = ultra.train_manifest(config, data_yaml)
    train_split = config.get("train_split", ultra.DEFAULT_TRAIN_SPLIT)
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

    family = config.get("family", "yolo11")
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
                # A YAML-built model (CH-11 query variant) takes its pretrained weights from `pretrained`.
                "pretrained_sha256": hashlib.sha256(
                    (data.REPO_ROOT / kwargs["pretrained"]).read_bytes()
                ).hexdigest()
                if isinstance(kwargs.get("pretrained"), str)
                else None,
                "protocol_version": protocol["version"],
                "train_split": train_split,
                "train_split_file_sha256": hashlib.sha256(
                    (data.SPLITS_DIR / f"{train_split}.json").read_bytes()
                ).hexdigest(),
                "export_manifest": manifest,
                "headline_evaluator": "movd/evaluate.py via scripts/predict_ultra.py (Ultralytics validation off)",
            },
        )
        save_json(results / "hardware.json", hardware(args.device))

    import torch

    observe = args.observe or "smoke_fraction" in config
    observations = {"resume": args.resume, "batches": 0, "saved_epochs": []}
    model = rtdetr.load_model(last if args.resume else weights, family)

    def train_start(trainer):
        observations.update(
            start_epoch=trainer.start_epoch,
            samples=len(trainer.train_loader.dataset),
            batches_per_epoch=len(trainer.train_loader),
            optimizer_state_entries=len(trainer.optimizer.state),
            amp=bool(trainer.amp),
        )
        if args.observe and "smoke_fraction" not in config:
            expected = manifest["num_images"]
            if observations["samples"] != expected:
                raise ValueError(
                    "Training loader does not contain the full exported split"
                )
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def batch_end(trainer):
        loss = float(trainer.loss.detach())
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError("Non-finite training loss")
        observations["batches"] += 1
        # OPT-04: wall-clock of the first and latest batch, so an epoch time can be projected without setup time.
        elapsed = round(time.perf_counter() - start, 3)
        observations.setdefault("first_batch_elapsed_seconds", elapsed)
        observations["last_batch_elapsed_seconds"] = elapsed
        observations.setdefault("first_loss", loss)
        observations["last_loss"] = loss
        observations["minimum_loss"] = min(observations.get("minimum_loss", loss), loss)
        observations["maximum_loss"] = max(observations.get("maximum_loss", loss), loss)

    def model_save(trainer):
        observations["saved_epochs"].append(trainer.epoch)
        if args.observe:
            observations["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            if trainer.device.type == "cuda":
                observations["peak_cuda_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(trainer.device)
                )
                observations["peak_cuda_reserved_bytes"] = (
                    torch.cuda.max_memory_reserved(trainer.device)
                )
            save_json(results / "observations.json", observations)
        if stop_after == trainer.epoch + 1:
            raise SmokeCheckpointStop()

    # Read-only observations do not change optimizer, sampling or training budget.
    if observe:
        model.add_callback("on_train_start", train_start)
        model.add_callback("on_train_batch_end", batch_end)
        model.add_callback("on_model_save", model_save)
    start = time.perf_counter()
    status = "completed"
    try:
        if args.resume:
            model.train(resume=True)
        else:
            model.train(**kwargs)
    except SmokeCheckpointStop:
        status = (
            "checkpoint_stop"
            if config_stop is None
            else f"stopped_after_epoch_{config_stop}"
        )
    seconds = time.perf_counter() - start
    if observe and args.device != "cpu":
        observations["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(
            int(args.device)
        )
        observations["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(
            int(args.device)
        )

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
    if "smoke_fraction" in config:
        runtime.setdefault("smoke_processes", []).append(
            {"status": status, "seconds": round(seconds, 3), **observations}
        )
    elif observe:
        runtime.setdefault("training_processes", []).append(
            {"status": status, "seconds": round(seconds, 3), **observations}
        )
    runtime["train_seconds_total"] = round(sum(runtime["train_process_seconds"]), 1)
    runtime["epoch_checkpoints"] = sorted(
        p.name for p in (output / "train" / "weights").glob("epoch*.pt")
    )
    save_json(runtime_path, runtime)
    print(f"training {status} in {seconds / 60:.1f} min; evidence: {runtime_path}")


if __name__ == "__main__":
    main()
