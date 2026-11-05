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