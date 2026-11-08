# UCU AI Homework Report Template

Your report is published as a website in UCU colors. The page is generated from `report.md`, and it shows the date of your repository's last push.

## How to fill

1. Click **Use this template** to create your own **public** repository. If you fork instead, also open the **Actions** tab and enable workflows.
2. Enable GitHub Pages (see below).
3. Edit **`report.md`**:
   - Fill in the header block at the top (`title`, `author`, `course`, `date`).
   - Write the report in Markdown. You can use tables, code blocks, and LaTeX math.
   - Put images in `images/` and reference them as `![caption](images/plot.png)`.
4. Commit and push to `main`. GitHub Actions deploys the site automatically.
5. Submit the published website link.

## Enabling GitHub Pages

This is a one-time step. GitHub does not copy it to your repository.

1. Open **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Open the **Actions** tab, select the failed **Deploy report** run, and click **Re-run jobs** if necessary.

## Local preview

```bash
python -m http.server 8000
```

Open http://localhost:8000.

Opening `index.html` directly from disk will not work.

## ⚠️ Website Template Rules

You may change **only** `report.md` and the `images/` folder when editing the report website.

**Any change to the core website (`index.html`, `assets/`, `.github/`) will annul the grade for the homework.**

This includes changes to how the last-push date is displayed.

---

# Multiple Object Video Detection

**Course:** UCU AI 2026 — Computer Vision

This repository contains our implementation of the Multiple Object Video Detection assignment using the VisDrone dataset.

The assignment involves training and comparing four object detection architectures, investigating different training configurations, and developing an improved final detection pipeline.

## 1. Project Structure

The implementation is organized primarily in a single Jupyter Notebook.

```text
Multiple-Object-Video-Detection/
├── homework.ipynb
├── multiple-object-video-detection-hw-2-ucu-ai-2026/
├── outputs/
├── images/
├── README.md
├── requirements.txt
├── report.md
├── index.html
├── assets/
└── .github/
```

- `homework.ipynb` — main notebook containing dataset exploration, preprocessing, model training, experiments, evaluation, and visualizations.
- `multiple-object-video-detection-hw-2-ucu-ai-2026/` — local VisDrone dataset, excluded from Git.
- `outputs/` — trained model weights and generated experiment artifacts, excluded from Git.
- `images/` — figures intended for the final academic report.
- `requirements.txt` — Python dependencies.
- `report.md` — separate technical report published using GitHub Pages.

The main implementation is written directly in the notebook rather than distributed across multiple Python modules.

## 2. Dataset

We use an eight-class VisDrone video-frame dataset containing COCO annotations, YOLO labels, and ignored-region annotations.

| Split | Sequences | Frames | Target Objects |
|---|---:|---:|---:|
| Train | 56 | 24,201 | 1,105,516 |
| Validation | 7 | 2,846 | 114,132 |
| Test | 48 | 3,811 | Unlabeled |

COCO annotations are used as the main source of bounding boxes, categories, and image metadata.

The dataset includes:

- Object category IDs
- Bounding box coordinates
- Image and video identifiers
- Tracking information
- Crowd / ignored annotations

The supplied training and validation splits are preserved.

## 3. Environment Setup

The experiments were developed on Ubuntu using Python 3.12 and an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM.

Main dependencies include:

- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- OpenCV
- pycocotools
- JupyterLab

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the main notebook:

```bash
python -m jupyterlab homework.ipynb
```

Select the Python environment containing the required packages.

The notebook should be executed sequentially because later experiments reuse the dataset loaders, configuration variables, and training/evaluation functions defined in earlier sections.

Running the complete notebook will retrain the models. Previously saved checkpoints are stored locally in `outputs/`.

### Setup: dataset placement and CUDA environment

**1. Dataset.** Put `multiple-object-video-detection-hw-2-ucu-ai-2026.zip` (10,611,162,657 bytes) in the repo root.
The ZIP has **no top-level folder** (`train/`, `val/`, `test/`, `test_images.csv`), so always extract it into the named
target directory. Never extract it into the repo root. Both the ZIP and the target directory are git-ignored. Do not commit,
move or delete the ZIP.

```bash
# Linux / macOS / Git Bash on Windows (verified on Windows with Git Bash unzip, about 4 min)
unzip -q multiple-object-video-detection-hw-2-ucu-ai-2026.zip -d multiple-object-video-detection-hw-2-ucu-ai-2026
```

```powershell
# Windows PowerShell (built-in tar.exe, same target)
New-Item -ItemType Directory multiple-object-video-detection-hw-2-ucu-ai-2026
tar -xf multiple-object-video-detection-hw-2-ucu-ai-2026.zip -C multiple-object-video-detection-hw-2-ucu-ai-2026
```

Expected layout: `multiple-object-video-detection-hw-2-ucu-ai-2026/{train,val,test,test_images.csv}` (about 11 GB).
Verify it with the standard-library check, which prints the counts and exits 0 on success:

```bash
python scripts/check_dataset.py            # add --root <path> if the dataset lives elsewhere
```

**2. Python environment.** Use Python 3.12 in `.venv`. PyTorch must be the CUDA build, installed **before** the rest. Plain PyPI
gives a CPU-only torch on Windows, and RTX 50xx (sm_120) GPUs need the CUDA 12.8+ builds. CUDA 13.0 is verified here.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv -r requirements.txt
```

Without uv: `python -m venv .venv`, activate it, then run the same two installs with `python -m pip install …`.

**3. Verify the GPU.**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Use `.venv/Scripts/python` on Windows and `.venv/bin/python` on Linux, or activate the venv first.

## 4. Implemented Stages

### Stage 1 — Data Understanding and Preprocessing

The first stage explores the dataset and prepares it for object detection.

Implemented components:

- Dataset overview and statistics
- Object class distribution
- Bounding box size analysis
- Small, medium, and large object distributions
- Scene density analysis
- Ground-truth bounding box visualization
- Inspection of crowd and ignored annotations
- Train/validation sequence separation checks
- Image resizing with bounding box transformations
- Basic data augmentation

The analysis is used to motivate subsequent model configurations, particularly for small objects and crowded scenes.

### Stage 2 — Faster R-CNN

We implemented a two-stage anchor-based detector using pretrained Faster R-CNN with a ResNet-50 FPN backbone.

Four configurations were investigated:

1. **Baseline** — standard Faster R-CNN configuration.
2. **RPN Proposal Tuning** — modified proposal counts before and after NMS.
3. **RoI Sampling Tuning** — modified RoI batch size and positive sampling fraction.
4. **Loss Modification** — modified Smooth L1 regression loss behavior in the RPN.

The training implementation includes:

- A custom PyTorch Dataset
- DataLoader and batching
- Pretrained model initialization
- AdamW optimization
- Automatic mixed precision
- Gradient clipping
- Training loss monitoring
- Model checkpoint saving
- COCO evaluation

Validation metrics include mAP, AP50, AP75, AP_small, AP_medium, and AP_large.

### Stage 3 — RetinaNet

We implemented a one-stage anchor-based detector using pretrained RetinaNet with a ResNet-50 FPN backbone.

Three configurations were investigated:

1. **Baseline** — standard pretrained RetinaNet.
2. **Smaller Anchors** — reduced anchor sizes motivated by the object size analysis.
3. **Focal Loss Tuning** — modified Focal Loss gamma from 2.0 to 1.5.

All configurations use the same training subset and validation protocol as Faster R-CNN.

The notebook includes training curves, validation metrics, model comparisons, inference speed evaluation, and visualizations of predicted bounding boxes.

## 5. Training and Evaluation Protocol

The initial experiments use a limited computational budget.

| Parameter | Configuration |
|---|---|
| Training subset | 560 frames |
| Sampling | 10 frames per training sequence |
| Validation set | 2,846 frames |
| Epochs | 3 |
| Batch size | 1 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Model input size | 512–896 |
| GPU | RTX 3050 Laptop 6 GB |
| Main evaluation | COCO bbox mAP |

Faster R-CNN uses model class labels 1–8 with background label 0.

The current RetinaNet implementation retains the shared 1–8 label mapping with an additional unused output index. This mapping should be reviewed before finalizing the submission.

The models are evaluated using COCO-style detection metrics and the original validation annotations, including crowd information.

These are short experimental runs on a small training subset. Full-dataset training and model convergence have not yet been established.

## 6. Current Results

The notebook contains the measured results for all completed Faster R-CNN and RetinaNet experiments.

Based on the initial comparisons:

- RoI and RPN tuning produced improvements in different Faster R-CNN metrics.
- Smaller anchors improved RetinaNet's small-object AP in the observed run but did not improve its overall mAP.
- Changing Focal Loss gamma improved RetinaNet's overall mAP in the observed run.
- Faster R-CNN generally achieved stronger detection accuracy than RetinaNet under the tested configurations.

These observations are preliminary because the models were trained on

<!-- CH-09 section (Ultralytics). The truncated sentence above belongs to section 6 and is left for its authors. -->

## Ultralytics data path, licence and citation (CH-09)

The anchor-free (YOLO11) and query-based (RT-DETR) families use **Ultralytics `ultralytics==8.4.207`**
(pinned in `requirements.txt`), installed into the same environment as torch 2.14.1+cu130.

- **Licence:** Ultralytics is distributed under the **GNU AGPL-3.0** (https://ultralytics.com/license).
  This repository uses it unmodified as a library for coursework. The pretrained checkpoints it downloads and
  the models fine-tuned from them are covered by the same licence. They are not committed (`outputs/` is git-ignored).
- **Weights:** `yolo11s.pt` (sha256 `85a76fe8…d502d5`) and `yolo11m.pt` (sha256 `d5ffc1a6…305b95`), COCO-pretrained,
  from https://github.com/ultralytics/assets/releases/tag/v8.4.0, downloaded to `outputs/weights/`.
- **Citation:** Jocher, G., Qiu, J. *Ultralytics YOLO11*, version 11.0.0, 2024. https://github.com/ultralytics/ultralytics
  (AGPL-3.0). RT-DETR: Zhao, Y. et al. *DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024.

Pipeline (CPU steps first, GPU steps only with an explicit `--device`):

```bash
python scripts/export_yolo.py --dataset-root <dataset>          # masked train_rubric + val_fast copy -> outputs/yolo_data/ (~1.9 GB)
python scripts/check_ultra_adapter.py --dataset-root <dataset>  # GT -> Ultralytics Results -> evaluator, mAP must be >= 0.99
python scripts/train_ultra.py --config configs/runs/yolo_baseline.json --device 0 [--resume]
python scripts/predict_ultra.py curve --run-id yolo_baseline --device 0 --dataset-root <dataset>   # val_fast mAP per epoch
python scripts/predict_ultra.py final --run-id yolo_baseline --device 0 --dataset-root <dataset>   # val_full metrics + speed
python scripts/failures_ultra.py --run-id yolo_baseline --dataset-root <dataset>                     # CPU: failure-case figures from saved val predictions
```

Exact GPU steps, expected evidence files and the notebook draft: `results/ch-09-yolo/GPU_RUNBOOK.md`.

The shipped `images/`, `labels/` and `data.yaml` are not used for training because they do not handle ignore regions.
`movd/ultra.py` applies the CH-02 policy instead (crowd never a target; ignore regions filled with the ImageNet mean except
inside labelled target boxes). Every reported number comes from `movd/evaluate.py` on the original val images, never from
Ultralytics' own validator.

<!-- CH-11 section (RT-DETR). -->

## Query-based detector: RT-DETR (CH-11)

- **Model:** Ultralytics RT-DETR-l (`ultralytics==8.4.207`, AGPL-3.0, see the CH-09 section). Official COCO-pretrained
  `rtdetr-l.pt` from https://github.com/ultralytics/assets/releases/tag/v8.4.0 (66,511,432 bytes, sha256
  `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`), downloaded to the git-ignored `outputs/weights/`.
- **Citation:** Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., Chen, J. *DETRs Beat YOLOs on Real-time
  Object Detection*, CVPR 2024. Carion, N. et al. *End-to-End Object Detection with Transformers* (DETR), ECCV 2020.
- **Code:** the CH-09 Ultralytics path unchanged (masked export, training arguments, adapter with the 500 cap, project
  evaluator). `movd/rtdetr.py` opens RT-DETR checkpoints with `ultralytics.RTDETR` (run-config `"family": "rtdetr"`).
  RT-DETR has no NMS and returns at most one box per decoder query. `configs/models/rtdetr-l-q500.yaml` is the packaged
  `rtdetr-l.yaml` with 500 instead of 300 queries.

```bash
python scripts/export_yolo.py                                    # once, CPU (~1.5 min, 1.9 GB in outputs/yolo_data/)
bash results/ch-11-rtdetr/run_ch11.sh python                     # baseline, 500 queries, lr0 1e-4: train -> val_fast curve -> val_full + speed (resumable)
python results/ch-11-rtdetr/make_evidence.py                     # CPU: comparison tables and figures in results/ch-11-rtdetr/
```

Single steps: `python scripts/train_ultra.py --config configs/runs/rtdetr_baseline.json --device 0 --observe [--resume]`,
`python scripts/predict_ultra.py curve|final --run-id rtdetr_baseline --device 0`,
`python scripts/predict_ultra.py speed --weights <ckpt> --family yolo11 --out <json> --device 0`.
