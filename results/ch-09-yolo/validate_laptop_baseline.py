"""Audit completed CH-09 laptop evidence and trusted local checkpoints on CPU."""

import csv
import gc
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = "yolo_baseline_marko_3050"


def main():
    folder = ROOT / "results" / RUN
    output = ROOT / "outputs" / RUN
    config = json.loads((folder / "config.json").read_text())
    kwargs = config["ultralytics_train_kwargs"]
    expected = {
        "epochs": 6,
        "imgsz": 1344,
        "batch": 1,
        "workers": 0,
        "amp": True,
        "cache": False,
        "seed": 42,
        "save_period": 1,
        "mosaic": 0.0,
        "fliplr": 0.5,
    }
    for key, value in expected.items():
        assert kwargs[key] == value, (key, kwargs[key])
    assert "fraction" not in kwargs and config["export_manifest"]["num_images"] == 2800
    runtime = json.loads((folder / "runtime.json").read_text())
    processes = runtime["training_processes"]
    assert sum(p["batches"] for p in processes) == 16800
    assert processes[-1]["status"] == "completed"
    with (folder / "ultralytics/results.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in history] == list(range(1, 7))
    for row in history:
        assert all(
            math.isfinite(float(row[key]))
            for key in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
        )
    states = []
    for epoch in range(6):
        path = output / "train/weights" / f"epoch{epoch}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == epoch
        assert checkpoint["train_args"]["epochs"] == 6
        assert checkpoint["optimizer"] and checkpoint["scaler"]
        states.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "epoch": epoch,
                "optimizer_entries": len(checkpoint["optimizer"]["state"]),
                "scaler_scale": checkpoint["scaler"]["scale"],
            }
        )
        del checkpoint
        gc.collect()
    curve = json.loads((folder / "curve.json").read_text())
    assert [r["epoch"] for r in curve] == list(range(1, 7))
    assert all(0 <= row["val_fast"]["mAP"] <= 1 for row in curve)
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["num_images"] == 2846 and metrics["split"] == "val_full"
    assert metrics["max_dets"] == 500 and metrics["images_over_headline_max_dets"] == 0
    assert (
        len(metrics["per_class"]) == 8
        and sum(r["num_images"] for r in metrics["density"]) == 2846
    )
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in metrics["metrics"].values())
    timing = json.loads((folder / "speed.json").read_text())
    assert (
        timing["timed_batches"] == 30
        and timing["warmup_batches"] == 10
        and timing["batch_size"] == 1
    )
    with (folder / "failure_cases.csv").open() as stream:
        failures = list(csv.DictReader(stream))
    assert {r["case"] for r in failures} == {
        "car_van_confusion",
        "tiny_target_miss",
        "dense_frame_miss",
    }
    from PIL import Image

    for row in failures:
        with Image.open(folder / row["figure"]) as image:
            image.verify()
    audit = {
        "status": "PASS",
        "epochs_completed": 6,
        "images_seen": 16800,
        "batches": 16800,
        "checkpoint_states": states,
        "curve_epochs": 6,
        "val_full_images": 2846,
        "classes": 8,
        "failure_cases": 3,
        "transfer_status": "not transferred; recipient verification pending",
    }
    (folder / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
