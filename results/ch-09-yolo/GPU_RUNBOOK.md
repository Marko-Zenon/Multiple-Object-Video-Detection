# CH-09 GPU runbook (remaining steps)

Everything below needs the RTX 5070. Do not start it while another chunk's GPU job is running.
The CPU preparation (the export, the adapter check, CLIs, configs, tests) is done. See `README.md` in this folder.

Run from the CH-09 worktree (`../MOVD-ch09`, branch `ch-09-yolo`) with the shared `.venv`. The dataset is not
inside the worktree, so every command that reads images takes the dataset root:

```bash
cd ../MOVD-ch09
PY=../Multiple-Object-Video-Detection/.venv/Scripts/python.exe
DS=../Multiple-Object-Video-Detection/multiple-object-video-detection-hw-2-ucu-ai-2026
```

Already present (git-ignored, not committed): `outputs/yolo_data/` (masked `train_rubric` and the `val_fast`
copy, `data.yaml` with an absolute path to this worktree), and `outputs/weights/yolo11{s,m}.pt`.
If `outputs/yolo_data/` is missing, recreate it on the CPU first with `$PY scripts/export_yolo.py --dataset-root $DS`
(about 90 s, about 1.9 GB).

## 1. GPU smoke (about 50 iterations per epoch, 2 epochs, about 200 images)

```bash
nvidia-smi                                                          # GPU idle? No other training process?
$PY scripts/train_ultra.py --config configs/runs/yolo_smoke_s.json --device 0
$PY scripts/train_ultra.py --config configs/runs/yolo_smoke_s.json --device 0 --resume   # must say "nothing to resume" or continue cleanly
$PY scripts/predict_ultra.py curve --run-id yolo_smoke_s --device 0 --dataset-root $DS
$PY scripts/train_ultra.py --config configs/runs/yolo_smoke_m.json --device 0             # only if smoke_s leaves memory headroom
```

Check: no OOM, peak memory (`nvidia-smi` or the Ultralytics `GPU_mem` column) below about 7.5 GB, `epoch0.pt` and `epoch1.pt`
in `outputs/yolo_smoke_*/train/weights/`, and `results/yolo_smoke_*/curve.json` with 2 rows. Then fix `weights` and
`batch_size` in `configs/runs/yolo_baseline.json` (and its `note`). If batch 4 OOMs, use 2 and record it.
The smoke runs are kept as evidence and are never reported as results.

## 2. Full baseline (train_rubric x 6 epochs = 16,800 images, imgsz 1344, seed 42)

```bash
$PY scripts/train_ultra.py --config configs/runs/yolo_baseline.json --device 0
# after an interruption:
$PY scripts/train_ultra.py --config configs/runs/yolo_baseline.json --device 0 --resume
```

Resume restarts from `outputs/yolo_baseline/train/weights/last.pt`. The config must be unchanged (checked against
`results/yolo_baseline/config.json`). Ultralytics' own validation is off. It still validates once on the final
epoch with its own metric, which is never reported.

## 3. val_fast curve and val_full evaluation (project evaluator)

```bash
$PY scripts/predict_ultra.py curve --run-id yolo_baseline --device 0 --dataset-root $DS   # 6 rows, resumable
$PY scripts/predict_ultra.py final --run-id yolo_baseline --device 0 --dataset-root $DS   # val_full metrics + speed
```

## 4. Speed

`final` writes `results/yolo_baseline/speed.json` (`movd.speed.benchmark`, 30 preloaded val frames, batch 1,
10 warm-up calls, FP32, imgsz 1344, including letterbox and NMS). Run it with the GPU otherwise idle. For an FP16 figure,
add `--half` to a separate run and record it as a separate number.

## 5. Failure analysis (CPU, from the saved predictions)

```bash
$PY scripts/failures_ultra.py --run-id yolo_baseline --dataset-root $DS
```

Writes `results/yolo_baseline/failure_cases.csv` and `results/yolo_baseline/figures/{car_van_confusion,tiny_target_miss,dense_frame_miss}.png`
with the CH-07 rule (score >= 0.10, same-class IoU >= 0.50), so the two families are comparable. The interpretation is written by the students.

## 6. Expected evidence (`results/yolo_baseline/`)

| File | Written by |
|---|---|
| `config.json` (run config, Ultralytics kwargs, weights sha256, protocol version, export manifest) | `train_ultra.py` |
| `hardware.json`, `runtime.json` (wall-clock per process, epoch checkpoints) | `train_ultra.py` |
| `ultralytics/results.csv`, `ultralytics/args.yaml` | `train_ultra.py` |
| `curve.json`, `figures/curve.png` (val_fast mAP / AP50 / AP_small per epoch) | `predict_ultra.py curve` |
| `metrics.json`, `per_class.csv` (val_full, maxDets 500 headline) | `predict_ultra.py final` |
| `speed.json` | `predict_ultra.py final` |
| `failure_cases.csv`, `figures/<kind>.png` | `failures_ultra.py` |

Ignored (never committed): `outputs/yolo_baseline/` (checkpoints, `val_full_predictions.json`).

## 7. Notebook section (under the 🔒 lock, after the evidence exists; no training in the notebook)

Draft for one self-contained subsection, "Anchor-free: YOLO11 baseline (CH-09)". It only loads `results/`:

```python
import json
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

run = Path("results/yolo_baseline")
config = json.loads((run / "config.json").read_text())
curve = json.loads((run / "curve.json").read_text())
metrics = json.loads((run / "metrics.json").read_text())
speed = json.loads((run / "speed.json").read_text())
print(config["weights"], config["ultralytics_train_kwargs"]["imgsz"], config["ultralytics_train_kwargs"]["batch"])
display(pd.DataFrame([{"epoch": r["epoch"], **r["val_fast"]} for r in curve])[["epoch", "mAP", "AP50", "AP_small"]])
display(Image(str(run / "figures" / "curve.png")))
# metrics.json / speed.json: print mAP, AP50, AP75, AP_S/M/L and FPS (check the key names in the files first)
display(pd.read_csv(run / "per_class.csv"))
display(pd.read_csv(run / "failure_cases.csv"))
for kind in ("car_van_confusion", "tiny_target_miss", "dense_frame_miss"):
    display(Image(str(run / "figures" / f"{kind}.png")))
```

Below it, a markdown cell with TODO prompts for the students: the comparison with CH-07 Faster R-CNN at the same budget, small-object AP, and the three failure cases.

## 8. Close the chunk

```bash
$PY -m pytest -q tests/test_ultra.py && $PY -m pytest -q tests
$PY -m ruff check movd scripts tests && $PY -m ruff format --check movd scripts tests
git merge origin/develop          # keep every other board row's latest state
```

Fill the CH-09 evidence log in `plan.md`, set the row to DONE, flip CH-10 and CH-11 to READY if their start dependencies are DONE
(both only need CH-09), open the PR `ch-09-yolo -> develop` (title `CH-09 ...`), merge it, and confirm the board on `origin/develop`.
