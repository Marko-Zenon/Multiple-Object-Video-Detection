"""Append and execute only the CH-09 evidence subsection after acquiring the notebook lock.

Refuses duplicate integration. Never runs existing notebook cells or model code.
"""

import hashlib
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent
HEADING = "## CH-09 — YOLO11s anchor-free baseline"

CODE = """import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

# This cell only reads tracked evidence. It never loads weights or trains a model.
ch09_run = Path.cwd() / "results" / "yolo_baseline_marko_3050"
assert (ch09_run / "config.json").is_file(), "Run this notebook from the repository root"
ch09_config = json.loads((ch09_run / "config.json").read_text())
ch09_kwargs = ch09_config["ultralytics_train_kwargs"]
ch09_runtime = json.loads((ch09_run / "runtime.json").read_text())
ch09_metrics = json.loads((ch09_run / "metrics.json").read_text())
ch09_speed = json.loads((ch09_run / "speed.json").read_text())
ch09_curve = json.loads((ch09_run / "curve.json").read_text())
ch09_history = pd.read_csv(ch09_run / "ultralytics" / "results.csv")
ch09_failures = pd.read_csv(ch09_run / "failure_cases.csv")
assert [point["epoch"] for point in ch09_curve] == list(range(1, 7))
assert ch09_metrics["split"] == "val_full" and ch09_metrics["num_images"] == 2846

print("Architecture: Ultralytics YOLO11s (COCO-pretrained, eight target classes)")
display(pd.DataFrame([{
    "run": ch09_config["run_config"]["run_id"],
    "architecture": "YOLO11s",
    "weights": ch09_config["weights"],
    "train_split": ch09_config["train_split"],
    "training_frames": ch09_config["export_manifest"]["num_images"],
    "epochs": ch09_kwargs["epochs"],
    "images_seen": sum(p["batches"] * ch09_kwargs["batch"] for p in ch09_runtime["training_processes"]),
    "imgsz": ch09_kwargs["imgsz"],
    "batch": ch09_kwargs["batch"],
    "amp": ch09_kwargs["amp"],
    "workers": ch09_kwargs["workers"],
    "cache": ch09_kwargs["cache"],
    "seed": ch09_kwargs["seed"],
    "augmentation": "frozen horizontal flip only",
}]))

print("Stored per-epoch training losses (six completed epochs)")
display(ch09_history[["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss"]])
ch09_figure, ch09_axis = plt.subplots(figsize=(7, 4))
for ch09_key in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
    ch09_axis.plot(ch09_history["epoch"], ch09_history[ch09_key], marker="o", label=ch09_key)
ch09_axis.set(xlabel="Epoch", ylabel="Mean training loss", title="YOLO11s — stored training losses")
ch09_axis.legend()
ch09_axis.grid(alpha=0.3)
ch09_figure.tight_layout()
plt.show()
plt.close(ch09_figure)

print("val_fast learning curve: project evaluator, maxDets 500")
display(pd.DataFrame([{"epoch": point["epoch"], **point["val_fast"]} for point in ch09_curve]))
display(Image(filename=str(ch09_run / "figures" / "curve.png")))

print("Final checkpoint on val_full: project evaluator, original images, maxDets 500")
ch09_ap_keys = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
display(pd.DataFrame([{"metric": key, "value": ch09_metrics["metrics"][key]} for key in ch09_ap_keys]))
display(pd.read_csv(ch09_run / "per_class.csv"))
display(pd.DataFrame(ch09_metrics["density"]))

print("Recorded FP32 speed benchmark: preloaded BGR frames; excludes disk I/O")
display(pd.DataFrame([{
    "FPS": ch09_speed["fps"],
    "latency_ms_per_image": ch09_speed["latency_ms_per_image"],
    "GPU": ch09_speed["gpu"],
    "batch": ch09_speed["batch_size"],
    "timed_frames": ch09_speed["timed_batches"],
    "warmup_calls": ch09_speed["warmup_batches"],
    "scope": ch09_speed["note"],
}]))

print("Stored failure examples: score >= 0.10, same-class IoU >= 0.50")
display(ch09_failures)
for ch09_kind in ("car_van_confusion", "tiny_target_miss", "dense_frame_miss"):
    ch09_row = ch09_failures.loc[ch09_failures["case"] == ch09_kind].iloc[0]
    print(ch09_kind, "|", ch09_row["frame"], "|", ch09_row["observation"])
    display(Image(filename=str(ch09_run / ch09_row["figure"])))
"""


def main():
    notebook_path = ROOT / "homework.ipynb"
    original = json.loads(notebook_path.read_text())
    assert not any(HEADING in "".join(c["source"]) for c in original["cells"])
    board = (ROOT / "plan.md").read_text()
    lock = next(
        line for line in board.splitlines() if "**homework.ipynb lock:**" in line
    )
    assert "Marko-Zenon" in lock and "CH-09" in lock
    cells = [
        nbformat.v4.new_markdown_cell(
            HEADING + "\n\n"
            "Sources: `results/yolo_baseline_marko_3050/` (tracked metrics, configuration, history and figures). "
            "The cell below reads completed evidence only; headline metrics use the project evaluator. "
            "Ultralytics version, weights and AGPL-3.0 citation: README, CH-09 Ultralytics section."
        ),
        nbformat.v4.new_code_cell(CODE),
        nbformat.v4.new_markdown_cell(
            "### CH-09 — student discussion TODOs\n\n"
            "- Interpret the six-epoch learning curve and the final-checkpoint result.\n"
            "- Compare with the frozen-budget CH-07 evidence; distinguish GPUs and speed timing scopes.\n"
            "- Discuss small-object/per-class/density metrics and the three failure examples.\n"
            "- Record verified checkpoint receipt before closing CH-09."
        ),
    ]
    temporary = nbformat.v4.new_notebook(cells=cells)
    kernel_data = ROOT / "outputs" / "ch09_notebook_closeout_20261108"
    kernel_folder = kernel_data / "kernels" / "ch09-evidence"
    kernel_folder.mkdir(parents=True, exist_ok=True)
    (kernel_folder / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    str(ROOT / ".venv" / "bin" / "python"),
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "CH-09 evidence (.venv, CPU)",
                "language": "python",
            }
        )
    )
    os.environ["JUPYTER_PATH"] = str(kernel_data)
    client = NotebookClient(
        temporary,
        timeout=120,
        startup_timeout=30,
        kernel_name="ch09-evidence",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    appended = json.loads(nbformat.writes(temporary))["cells"]
    assert not any(
        o["output_type"] == "error"
        for c in appended
        if c["cell_type"] == "code"
        for o in c["outputs"]
    )
    updated = {**original, "cells": original["cells"] + appended}
    assert updated["cells"][: len(original["cells"])] == original["cells"]
    nbformat.validate(nbformat.from_dict(updated))
    notebook_path.write_text(json.dumps(updated, ensure_ascii=False, indent=1) + "\n")
    saved = json.loads(notebook_path.read_text())
    assert saved["cells"][: len(original["cells"])] == original["cells"]
    pngs = sum(
        "image/png" in o.get("data", {})
        for c in appended
        if c["cell_type"] == "code"
        for o in c["outputs"]
    )
    assert pngs == 5, pngs
    summary = {
        "status": "PASS",
        "original_cells_preserved": len(original["cells"]),
        "appended_cells": len(appended),
        "executed_new_code_cells": 1,
        "executed_existing_cells": 0,
        "embedded_pngs": pngs,
        "old_cells_sha256": hashlib.sha256(
            json.dumps(original["cells"], sort_keys=True).encode()
        ).hexdigest(),
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "training_or_inference_executed": False,
        "kernel_python": str(ROOT / ".venv" / "bin" / "python"),
    }
    (FOLDER / "notebook_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
