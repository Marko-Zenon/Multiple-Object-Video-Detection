"""Read-only CH-09 audit; write a new closeout report without changing run artifacts."""

import ast
import csv
import gc
import hashlib
import json
import math
import subprocess
from pathlib import Path

import nbformat
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent
RUN = ROOT / "results" / "yolo_baseline_marko_3050"
BASE = "a981042bbc5f7fc3b1df8f76f32c29a69bbd6406"


def main():
    target = FOLDER / "completion_verification.json"
    assert not target.exists(), "Preserve previous closeout reports"
    metrics = json.loads((RUN / "metrics.json").read_text())
    expected = {
        "mAP": 0.143892,
        "AP50": 0.292532,
        "AP75": 0.119226,
        "AP_small": 0.086579,
        "AP_medium": 0.143427,
        "AP_large": 0.255470,
    }
    for key, value in expected.items():
        assert round(metrics["metrics"][key], 6) == value
    speed = json.loads((RUN / "speed.json").read_text())
    assert round(speed["fps"], 2) == 36.37
    assert metrics["num_images"] == 2846 and metrics["max_dets"] == 500
    assert metrics["split"] == "val_full"
    config = json.loads((RUN / "config.json").read_text())
    kwargs = config["ultralytics_train_kwargs"]
    assert (
        kwargs["epochs"],
        kwargs["imgsz"],
        kwargs["batch"],
        kwargs["workers"],
        kwargs["seed"],
    ) == (6, 1344, 1, 0, 42)
    assert kwargs["amp"] and not kwargs["cache"] and "fraction" not in kwargs
    assert config["export_manifest"]["num_images"] == 2800
    with (RUN / "ultralytics/results.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in history] == list(range(1, 7))
    assert all(
        math.isfinite(float(row[key]))
        for row in history
        for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss")
    )
    curve = json.loads((RUN / "curve.json").read_text())
    assert [row["epoch"] for row in curve] == list(range(1, 7))
    manifest = json.loads((RUN / "artifact_manifest.json").read_text())
    for row in manifest["files"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"], (
                path
            )
    assert "ultralytics==8.4.207" in (ROOT / "requirements.txt").read_text()
    readme = (ROOT / "README.md").read_text()
    assert "AGPL-3.0" in readme and "Jocher" in readme
    adapter = json.loads((ROOT / "results/ch-09-yolo/adapter_check.json").read_text())
    assert (
        adapter["gt_round_trip"]["passed"]
        and adapter["gt_round_trip"]["metrics"]["mAP"] >= 0.99
    )
    export = json.loads(
        (
            ROOT / "results/ch-09-yolo/marko_baseline_export/export_check.json"
        ).read_text()
    )
    assert export["checks"]["train_rubric"]["label_counts_match_coco_targets"]
    assert export["checks"]["train_rubric"]["masked_frames"] == 2161
    for image in (RUN / "figures").glob("*.png"):
        with Image.open(image) as decoded:
            decoded.verify()
    before = json.loads(
        subprocess.check_output(
            ["git", "show", f"{BASE}:homework.ipynb"], cwd=ROOT, text=True
        )
    )
    after = json.loads((ROOT / "homework.ipynb").read_text())
    assert after["cells"][:90] == before["cells"] and len(after["cells"]) == 93
    assert {k: v for k, v in before.items() if k != "cells"} == {
        k: v for k, v in after.items() if k != "cells"
    }
    nbformat.validate(nbformat.from_dict(after))
    new = after["cells"][90:]
    code = [cell for cell in new if cell["cell_type"] == "code"]
    assert len(code) == 1 and code[0]["execution_count"] == 1
    assert not any(output["output_type"] == "error" for output in code[0]["outputs"])
    assert (
        sum("image/png" in output.get("data", {}) for output in code[0]["outputs"]) == 5
    )
    tree = ast.parse("".join(code[0]["source"]))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"train", "predict", "fit"}
        for node in ast.walk(tree)
    )
    protected = [
        "index.html",
        "assets",
        ".github",
        ".nojekyll",
        "configs/protocol.json",
        "movd/data.py",
        "movd/evaluate.py",
    ]
    assert not subprocess.check_output(
        ["git", "diff", "origin/develop", "--", *protected], cwd=ROOT, text=True
    )
    states = []
    for path in sorted(
        (ROOT / "outputs/yolo_baseline_marko_3050/train/weights").glob("*.pt")
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        is_epoch = path.stem.startswith("epoch")
        if is_epoch:
            assert checkpoint["epoch"] == int(path.stem[5:])
            assert (
                checkpoint["optimizer"] and len(checkpoint["optimizer"]["state"]) == 255
            )
            assert checkpoint["scaler"]
        else:
            assert checkpoint["optimizer"] is None
        states.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "epoch": checkpoint["epoch"],
                "resume_state_retained": is_epoch,
            }
        )
        del checkpoint
        gc.collect()
    report = {
        "status": "PASS",
        "technical_acceptance_criteria": {
            "1_pinned_cited": "PASS",
            "2_export_policy_checked": "PASS",
            "3_adapter_verified": "PASS",
            "4_baseline_project_curve": "PASS",
            "5_no_export_commit": "pending full PR path audit",
        },
        "metrics": metrics["metrics"],
        "fps": speed["fps"],
        "original_manifest_files_verified": manifest["count"],
        "original_manifest_bytes": manifest["total_bytes"],
        "notebook_existing_cells_preserved": 90,
        "notebook_new_cells": 3,
        "notebook_executed_new_cells": 1,
        "notebook_figures": 5,
        "checkpoint_states": states,
        "training_or_inference_executed": False,
        "transfer_status": "PENDING: no agreed destination or recipient verification",
        "merged": False,
    }
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS: original metrics/manifest, notebook prefix and all eight checkpoint states verified"
    )


if __name__ == "__main__":
    main()
