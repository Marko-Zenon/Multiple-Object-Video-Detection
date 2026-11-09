# plan.md — live task board and execution plan

> **Official Kaggle Close:** Tue 2026-11-10 **21:55 UTC / 23:55 Europe/Kyiv (EET, UTC+02:00)**, verified from the official Overview Close tooltip on 2026-11-08 (G1 RESOLVED). The student-reported 23:59 display conflicts with this current official value and is retained in G1 as unconfirmed. **Freeze target:** Tue 21:30 Kyiv (unchanged).
> **The live version of this file is the one on `origin/develop`.** Fetch before reading it. Rules are in `CLAUDE.md`.
> **To start work:** pick a row with `🟢 READY` and `Start now? = 🟢 YES`, then run the atomic claim (§9.2) **before** coding.
> **States:** ✅ DONE · 🟢 READY · 🔵 IN_PROGRESS · 🟡 BLOCKED · 🟠 PARTIAL / PAUSED · 🔴 FAILED / NEEDS_DECISION.
> **Depends on:** `start:` IDs must be DONE before a claim; `finish:` IDs must be DONE before this chunk can be DONE; `X or Y` = either.

## 1. Live task board

| ID | Chunk | State | Start now? | Owner | Depends on | Blocked by / reason | Branch / PR | Last update |
|---|---|---|---|---|---|---|---|---|
| CH-01 | Repo hygiene & local dataset/env setup | ✅ DONE | ⛔ NO — completed | Badabuba | — | — (M-DATA + M-GPU satisfied on Nazar's RTX 5070) | ch-01-repo-env-setup / [PR #1](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/1) | 2026-11-07 16:15 Kyiv |
| CH-02 | Protocol freeze, Kaggle facts, data module | ✅ DONE | ⛔ NO — completed | Badabuba | — | — (G3, G4, G2a, G2c resolved; G1 → CH-16, G2b → CH-05 upload) | ch-02-protocol-data / [PR #2](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/2) | 2026-11-07 17:00 Kyiv |
| CH-03 | Evaluator, speed benchmark, tests | ✅ DONE | ⛔ NO — completed | Badabuba | start: CH-02 | — (worker: Claude; `movd/evaluate.py` + `scripts/evaluate.py`, headline maxDets 500 per image) | ch-03-evaluator / [PR #3](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/3) | 2026-11-07 17:18 Kyiv |
| CH-04 | Torchvision training core | ✅ DONE | ⛔ NO — completed | Badabuba | start: CH-02 · finish: CH-03 | worker: Codex; CH-03 evaluator and speed integration verified | ch-04-torchvision-training / [PR #4](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/4) | 2026-11-07 17:43 Kyiv |
| CH-05 | Test inference, submission writer/validator, first Kaggle upload | ✅ DONE | ⛔ NO — completed | Badabuba | start: CH-02 · finish: CH-03, CH-04 or CH-09 | — (worker: Claude; CH-07 frcnn_rpn_pre3000 e6 submitted: Kaggle public 0.2011538266, val_full mAP 0.1893; real round-trip diff 0; G2b = 5/day, G8 resolved; CH-09 stays paused) | ch-05-submission / [PR #7](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/7) | 2026-11-08 18:10 Kyiv |
| CH-06 | Corrected EDA & preprocessing evidence (notebook) | ✅ DONE | ⛔ NO — completed | Badabuba | — | — (worker: Claude; Stage 1 re-run from a clean kernel, 0 errors; notebook lock released) | ch-06-eda / [PR #5](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/5) | 2026-11-07 19:20 Kyiv |
| CH-07 | Two-stage: Faster R-CNN corrected + tuning + failure cases | ✅ DONE | ⛔ NO — completed | Badabuba | start: CH-04 | worker: Codex; selected frcnn_rpn_pre3000; CH-07 notebook section executed; GPU and notebook lock released | ch-07-frcnn / [PR #6](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/6) | 2026-11-08 00:33 Kyiv |
| CH-08 | One-stage anchor-based: RetinaNet + anchors + focal-loss experiment | ✅ DONE | ⛔ NO — completed | Badabuba | start: CH-04 | worker: Codex; selected retina_anchors by final val_fast mAP; GPU and notebook lock released | ch-08-retina / [PR #11](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/11) | 2026-11-08 23:45 Kyiv |
| CH-09 | Ultralytics export + YOLO11 anchor-free baseline | ✅ DONE | ⛔ NO — completed | Marko-Zenon (closeout: Badabuba) | start: CH-02 · finish: CH-03 | — (YOLO11s val_full mAP 0.1439 / AP50 0.2925; 36.37 FPS on RTX 3050, not comparable with RTX 5070 numbers; Telegram artifact ZIP recipient-verified: 163/163 + 145/145 SHA-256, predictions re-evaluated exactly; notebook merged structurally, 95 cells; lock released) | ch-09-yolo / [PR #10](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/10) | 2026-11-09 00:45 Kyiv |
| CH-10 | Anchor-free tuning + small/dense analysis | 🟢 READY | 🟢 YES | — | start: CH-09 | — (CH-09 ✅: export, adapter, YOLO11s baseline and curve merged) | — | 2026-11-09 00:45 Kyiv |
| CH-11 | Query-based RT-DETR baseline + tuning + learning curve | ✅ DONE | ⛔ NO — completed | Badabuba | start: CH-09 | — (worker: Claude; RT-DETR-l val_full mAP 0.2577 / AP50 0.4775 at the frozen budget; tuned q500 0.2389, lr0 1e-4 0.2179, confounded lr1e4 kept and marked; learning curve vs YOLO11s/FRCNN/RetinaNet + analysis in results/ch-11-rtdetr/; G9 resolved, 3.3 GB peak at batch 1; RTX 5070 released; notebook lock released; CH-10 not touched) | ch-11-rtdetr / [PR #12](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/12) | 2026-11-09 07:00 Kyiv |
| CH-12 | Cross-family comparison & final-config decision | 🟡 BLOCKED | ⛔ NO — CH-10 | — | start: CH-07, CH-08, CH-10, CH-11 | CH-07, CH-08, CH-11 ✅, CH-09 baseline merged; still needs CH-10 | — | 2026-11-09 07:00 Kyiv |
| CH-13 | Final-pipeline inference experiments on val | 🟡 BLOCKED | ⛔ NO — CH-12 | — | start: CH-12 | needs chosen final family | — | 2026-11-07 15:00 Kyiv |
| CH-14 | Final training, test predictions, Kaggle submissions & selection | 🟡 BLOCKED | ⛔ NO — CH-12 | — | start: CH-05, CH-12 · finish: CH-13 | CH-05 ✅ (submission tooling ready); still needs CH-12 (frozen final config) | — | 2026-11-08 18:10 Kyiv |
| CH-15 | Report scaffold & Pages deployment check | ✅ DONE | ⛔ NO — completed | Marko-Zenon | gate: G5 | PR #8 and M1 #9 merged; deployment 42507083884 succeeded at fec11cb; public scaffold verified | ch-15-report / [PR #8](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/8), [M1 #9](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/9) | 2026-11-08 20:02 Kyiv |
| CH-16 | Reproducibility, evidence pack, rubric audit, freeze | 🟡 BLOCKED | ⛔ NO — CH-12, CH-14 | — | start: CH-01, CH-06, CH-12, CH-14, CH-15 · finish gates: G1, G10 | CH-15 ✅; still needs CH-12 and CH-14; G1 resolved; finish gate G10 remains open | — | 2026-11-08 20:02 Kyiv |

🔒 **homework.ipynb lock:** FREE

### Gates

| ID | Gate | Status | Owner | Resolved by / where |
|---|---|---|---|---|
| G1 | Exact official Kaggle close time in UTC and Europe/Kyiv. Operational | RESOLVED | Marko-Zenon | Verified 2026-11-08 on official [Overview](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026/overview): rendered Close → 2 days to go span has native title `Tue Nov 10 2026 23:55:00 GMT+0200 (Eastern European Standard Time)`; same title appears in the competition header. Thus competition Close = 2026-11-10 21:55 UTC / 2026-11-10 23:55 Europe/Kyiv (EET, UTC+02:00), checked with zoneinfo. Student-reported 2026-11-10 23:59 is preserved as conflicting/unconfirmed, not the current verified cutoff. Rules inspected; no separate absolute final-selection cutoff exposed, so do not infer a later selection window. Public HTML/API attempts returned 404, but the rendered public page loaded and supplied explicit-offset evidence. Freeze remains 21:30 Kyiv. CH-16 operational finish gate resolved |
| G2a | Internal evaluation protocol: metric, IoU thresholds, ignore handling, max dets. Technical | RESOLVED | Badabuba | CH-02: rubric-required standard COCO (pycocotools, AP@0.50:0.05:0.95, AP50/75, AP_S/M/L), GT with `iscrowd=1` kept, maxDets 500. Sources: the assignment + Kaggle Data text (ignored-region detections not penalized; ≤ 500 dets scored). The hidden Kaggle scorer code was not exposed and is not claimed identical |
| G2b | Kaggle daily submission limit. Operational | RESOLVED | Badabuba | 5 submissions per day, verified by Badabuba on the authenticated Kaggle Submit page 2026-11-08 (a limit-increase request is pending, not granted; re-check if it changes). First upload used 1 of 5 that day |
| G2c | Number of final selections | RESOLVED | Badabuba | CH-02: 2, chosen by internal evaluation, not only the public LB (Kaggle Rules, official Kaggle Overview/Rules/Data text copied by Badabuba 2026-11-07) |
| G3 | Submission header/format | RESOLVED | Badabuba | CH-02: `image_id,PredictionString`, one row per test image, groups `class score x y width height` (class 0–7, top-left, original px), empty string valid (Kaggle Data text). `sample_submission.csv` is missing from the official ZIP (77,434 entries), so a canonical copy is generated by `scripts/make_sample_submission.py` (git-ignored) |
| G4 | Ignore-file format and semantics (relation to COCO `iscrowd`) | RESOLVED | Badabuba | CH-02: `ignore/*.txt` = normalized `cx cy w h`, no class (VisDrone orig 0 + 11); each region is in the COCO JSON once per class with `iscrowd=1, ignore=1`; 100 % match on train + val (`results/ch-02-protocol/data_facts.json`) |
| G5 | GitHub Pages enabled (Settings → Pages → Source: GitHub Actions); admin only | RESOLVED | Marko-Zenon | Enabled and verified by Marko-Zenon on 2026-11-08. Deploy report #2 re-run succeeded (21s, commit 0330c83, main). Pages URL: https://marko-zenon.github.io/Multiple-Object-Video-Detection/ |
| G6 | Nazar's commit email linked to GitHub `Badabuba` | RESOLVED | Badabuba | CH-01: GitHub API maps `mykolaichuk.pn@ucu.edu.ua` commits to login `Badabuba` |
| G7 | PyTorch CUDA wheel with sm_120 support works on the RTX 5070 | RESOLVED | CH-01 owner | CH-01: torch 2.14.1+cu130, CUDA conv + NMS verified |
| G8 | At least one student can upload to Kaggle via the web UI (no tokens in Git) | RESOLVED | Badabuba | Badabuba uploaded the CH-05 submission via the web UI on 2026-11-08 (team Nachalnyky); accepted, public 0.2011538266 |
| G9 | RT-DETR trains reliably on 8 GB, otherwise a recorded fallback decision | RESOLVED | CH-11 owner | CH-11: rtdetr-l trains reliably on the 8 GB RTX 5070 at batch 1 (3.3-3.4 GB peak, 4 full 6-epoch runs, resume tested); no fallback needed |
| G10 | Students wrote the report prose and conclusions | OPEN | Badabuba + Marko-Zenon | finish gate of CH-16 |

## 2. Current verified state (inspection 2026-11-07, read-only)

- **Git/GitHub:** public repo `Marko-Zenon/Multiple-Object-Video-Detection`. `main` = `0330c83`, 2 commits, both by Marko.
  The GitHub contributor list shows only `Marko-Zenon`. Nazar (`Badabuba`) has WRITE access, not admin. No PRs. `develop` was created by this setup.
- **Pages:** not enabled (API 404). Both "Deploy report" runs failed on the workflow's "Pages not enabled" guard.
  The workflow deploys **only on push to `main`**. `index.html`, `assets/`, `.github/` must never be edited (template rule).
- **Notebook** `homework.ipynb`: 58 executed cells (Marko, Ubuntu, RTX 3050 6 GB). It contains EDA, Faster R-CNN ×4 and RetinaNet ×3.
  Legacy protocol "v0":
  - 560 training frames (10/clip), 3 epochs, AdamW 1e-4, batch 1, input 512–896 px, one seed.
  - Detections capped at 100 twice (`box_detections_per_img` and COCOeval `maxDets`).
  - Crowd/ignored objects dropped from targets, so they become background.
  - The `ignore/` files are never read.

  Legacy val results (v0):

  | Run | mAP | AP50 | AP_small |
  |---|---:|---:|---:|
  | Faster R-CNN baseline | 0.1273 | 0.2596 | 0.0522 |
  | Faster R-CNN RPN-tuned | 0.1287 | 0.2668 | 0.0760 |
  | Faster R-CNN RoI-tuned | 0.1064 | 0.2275 | 0.0567 |
  | Faster R-CNN RPN-loss β 1/3 | 0.1173 | 0.2450 | 0.0633 |
  | RetinaNet baseline | 0.1075 | 0.2157 | 0.0418 |
  | RetinaNet small anchors | 0.1025 | 0.2247 | 0.0460 |
  | RetinaNet focal γ 1.5 | 0.1163 | 0.2240 | 0.0403 |

  RetinaNet speed ≈ 17 FPS @ 896 on the 3050. The failure-case cell holds figures only, with no written analysis.
  Faster R-CNN has no speed or failure-case results.
- **Dataset facts from the notebook:**
  - Train: 24,201 images and 1,616,260 annotations, of which 1,105,516 are targets and 510,744 have `iscrowd = ignore = 1`.
  - Val: 2,846 images, 146,092 annotations, 114,132 targets, 31,960 crowd.
  - All crowd annotations carry one of the 8 class IDs.
  - Target sizes: small 24.9 %, medium 63.5 %, large 11.6 %. Targets per train frame: mean 45.7, max 180.
  - Frames are about 1904×1071 (some 1344×756).
- **Configs:** `configs/data.json` and `configs/experiments.json` do **not** describe what the notebook ran.
  `requirements.txt` lacks opencv-python, tqdm and ultralytics. README §6 is truncated mid-sentence.
- **ZIP:** `./multiple-object-video-detection-hw-2-ucu-ai-2026.zip`, 10,611,162,657 B (10.98 GB unpacked), 77,434 entries, git-ignored.
  - **No top-level folder**: the entries are `train/`, `val/`, `test/`, `test_images.csv`.
  - Clips: 56 train, 7 val, 48 test (`clip_001…clip_048`). `ignore/` txt files exist for 50/56 train clips and 5/7 val clips.
  - `data.yaml` is present for train and val.
  - **`sample_submission.csv` is not in the ZIP.**
- **Machines:**
  - Nazar: RTX 5070 Laptop 8 GB, Windows 11, 229 GB free disk. No torch installed yet; uv + CPython 3.12 available. No Kaggle CLI or credentials.
  - Marko: RTX 3050 Laptop 6 GB, Ubuntu, dataset likely already extracted (verify at claim time).
- **Kaggle:** public baseline `baseline.csv` = **0.22163**. The metric, submission limit and sample format are unknown (G2, G3).

## 3. Mandatory scope and non-goals

| Rubric item | Chunk(s) |
|---|---|
| Class distribution, object size distribution, objects per frame, sequence differences, GT visualization | CH-06 |
| Clip-level train/val split, justified subsampling | CH-02 (splits), CH-06 (justification) |
| Resizing, normalization, augmentation | CH-02 (policy), CH-04 / CH-09 (implementation), CH-06 (evidence) |
| Correct ignored-region handling | CH-02 (policy), CH-03 (evaluation), CH-04 / CH-09 (training) |
| Two-stage anchor-based: pretrained baseline, tuning, mAP/AP50/AP_small/speed, ≥ 3 failure cases | CH-07 |
| One-stage anchor-based: baseline, anchors/focal tuning, comparison with two-stage | CH-08 |
| One-stage anchor-free: baseline + tuned, assignment/loss/resolution/high-res head, size-specific & dense analysis | CH-09, CH-10 |
| Query-based: baseline + tuned, comparable budget, mAP learning curve vs CNN, convergence/small/dense analysis | CH-11 |
| ≥ 2 meaningful experiments beyond baselines, hyperparameter tuning, a defensible loss experiment | CH-07, CH-08 (loss experiment), CH-10, CH-11 |
| Controlled comparisons, consistent validation, failed experiments kept | CH-02, CH-03, all model chunks, CH-12 |
| Final pipeline justified by results; stronger data/inference strategy; separate intermediate comparisons | CH-12, CH-13, CH-14 |
| Valid test inference (original pixels, classes 0–7, `class score x y w h` groups, one row per image, ≤ 500 dets), validator | CH-05, CH-14 |
| ≥ 1 Kaggle score > baseline 0.22163; final selection by internal validation | CH-14 (first upload in CH-05) |
| Reproducible code & dependencies, stored evidence, run instructions | CH-01, CH-16 (all chunks contribute) |
| Student-authored report, public repo, working GitHub Pages | CH-15, CH-16, G10 |
| Meaningful contributions from both teammates, clean frozen repo | CH-16 (G6), the claim protocol |

**Non-goals:** temporal models, tracking, optical flow, Grad-CAM, attention visualization, W&B bonus work, custom
architectures, and any other bonus task. Do not rewrite the legacy notebook.

## 4. Frozen protocol and interfaces

All values below were frozen by CH-02 on 2026-11-07 (machine-readable copy: `configs/protocol.json`). Changing a frozen value requires a §11 change-log entry
that a student has approved.

| Item | Value |
|---|---|
| Dataset root | `./multiple-object-video-detection-hw-2-ucu-ai-2026/` with `train/`, `val/`, `test/` and `test_images.csv` directly inside |
| Split | Provided clip-level split: 56 train clips / 7 val clips (0 overlap, verified in the legacy notebook). Test = 48 clips |
| `val_full` | All 2,846 val frames. Every reported headline number uses it |
| `val_fast` | `configs/splits/val_fast.json`: 714 frames = every 4th frame of each of the 7 val clips, from the first frame. Used only for per-epoch curves |
| `train_rubric` | `configs/splits/train_rubric.json`: 2,800 frames = 50 evenly spaced frames (`np.linspace`, legacy rule) from each of the 56 train clips. Used for all four-family comparisons. Deterministic (`python scripts/make_splits.py --check`) |
| Rubric compute budget | `train_rubric` × **6 epochs = 16,800 images seen**, input **long side 1344 px** (aspect kept; torchvision `max_size=1344, min_size=756`, Ultralytics `imgsz=1344`), batch size recorded per run. Measured on the RTX 5070 (AMP, bs 2, no data loading): Faster R-CNN v2 8.38 img/s → ≈ 33 min, RetinaNet v2 10.19 img/s; ≈ 40 % headroom to the 60 min target. Same list, images seen and long side for all compared runs; report wall-clock and GPU too |
| Seed | 42 (logged per run) |
| Label map | COCO `category_id` 1–8 (person, bicycle, car, van, truck, tricycle, bus, motor) ↔ submission/YOLO class 0–7 = `category_id − 1`. Confirmed: Kaggle Data text (class 0–7) and the provided YOLO labels (24,339 boxes, 0 mismatches). Merged VisDrone ids: 1+2→person, 3→bicycle, 4→car, 5→van, 6→truck, 7+8→tricycle, 9→bus, 10→motor |
| Ignore-region policy | **Training:** crowd annotations are never targets; ignore regions are filled with the ImageNet mean (124, 116, 104) except pixels inside labelled target boxes (`movd.data.mask_ignore_regions`, figure `results/ch-02-protocol/figures/ignore_masking_check.png`). **Evaluation:** the provided COCO GT unchanged, `iscrowd=1` copies kept (one per class), so pycocotools ignores detections in those regions for every class. Matches the Kaggle Data text: detections inside ignored regions are not penalized |
| Prediction interchange format | COCO results JSON: list of `{"image_id": int, "category_id": 1–8, "bbox": [x, y, w, h] in original image pixels, "score": float}`. Val: COCO image id. Test: `image_id` = 1-based row number in `test_images.csv` (`movd.data.load_test_images()[i]["id"]`); the submission uses its string `image_key` (e.g. `clip_001_0000001`) |
| Evaluator settings | `movd/evaluate.py` (CH-03): rubric-required standard COCO bbox via pycocotools. **Primary: mAP = AP@IoU 0.50:0.05:0.95**, plus AP50, AP75, AP_S/M/L (COCO area ranges), per-class AP. **maxDets 500** headline (Kaggle scores ≤ 500 dets/image); maxDets 100 only for the legacy-comparability note. Headline split `val_full` |
| Submission format | Header `image_id,PredictionString`; exactly one row per `test_images.csv` image (3,811), in its order; groups `class score x y width height` (class 0–7, top-left x y, original pixels), space-separated; empty string valid; ≤ 500 groups scored. Local canonical sample: `scripts/make_sample_submission.py` (the official ZIP lacks the file) |
| Kaggle | Baseline 0.22163 (public LB). **2 final selections**, chosen by internal evaluation (Rules). Ignored-region detections not penalized; ≤ 500 dets/image scored (Data). Hidden scorer code not exposed. **Unknown:** daily limit (G2b, before CH-05's upload), exact UTC close time resolved by G1: 2026-11-10 21:55 UTC / 23:55 Kyiv (official Overview Close tooltip, verified 2026-11-08) |
| Framework decision | Two-stage and one-stage anchor-based: torchvision (Faster R-CNN R50-FPN v2, RetinaNet R50-FPN v2). Anchor-free: Ultralytics **YOLO11**. Query-based: Ultralytics **RT-DETR** (`rtdetr-l` unless the smoke test forces otherwise). Exact package and weight versions are pinned and cited. AGPL-3.0 notice in the README |

## 5. Dependencies

Dependency kinds:
- **start:** must be DONE before a claim.
- **finish:** must be DONE before this chunk can be DONE.
- **Machine prerequisites** (not chunks, checked at claim time):
  - M-ZIP: the ZIP is present.
  - M-DATA: the dataset is extracted at the dataset root with the expected counts.
  - M-GPU: a working CUDA PyTorch environment.
  - 🔒: needs the notebook lock.
- **Gates:** see §1.

| ID | start | finish | machine | gates |
|---|---|---|---|---|
| CH-01 | — | — | M-ZIP, M-GPU (set up here) | G6 (check), G7 |
| CH-02 | — | — | M-DATA | G2a, G2c, G3, G4 (resolved here); G1, G2b re-scoped as operational |
| CH-03 | CH-02 | — | M-DATA | — |
| CH-04 | CH-02 | CH-03 | M-DATA, M-GPU | — |
| CH-05 | CH-02 | CH-03, CH-04 or CH-09 | M-DATA, M-GPU | G2b, G8 (before the first upload) |
| CH-06 | — | — | M-DATA, 🔒 | — |
| CH-07 | CH-04 | — | M-DATA, M-GPU, 🔒 short | — |
| CH-08 | CH-04 | — | M-DATA, M-GPU, 🔒 short | — |
| CH-09 | CH-02 | CH-03 | M-DATA, M-GPU | — |
| CH-10 | CH-09 | — | M-DATA, M-GPU, 🔒 short | — |
| CH-11 | CH-09 | — | M-DATA, M-GPU (8 GB preferred), 🔒 short | G9 |
| CH-12 | CH-07, CH-08, CH-10, CH-11 | — | CPU, 🔒 short | — |
| CH-13 | CH-12 | — | M-DATA, M-GPU | — |
| CH-14 | CH-05, CH-12 | CH-13 | M-DATA, M-GPU | G8 |
| CH-15 | — | — | CPU | G5 (start gate) |
| CH-16 | CH-01, CH-06, CH-12, CH-14, CH-15 | — | CPU (clean env) | G1, G6, G10 (finish gates) |

DAG (start edges `→`, finish edges `⇢`). It has no cycles:

```text
CH-01 ─────────────────────────────────────────────────────────────┐
CH-06 ─────────────────────────────────────────────────────────────┤
CH-02 → CH-03            (CH-03 ⇢ CH-04, CH-05, CH-09)             │
CH-02 → CH-04 → CH-07 ─┐                                           │
          └───→ CH-08 ─┤                                           ├→ CH-16
CH-02 → CH-09 → CH-10 ─┼→ CH-12 → CH-13 ⇢ CH-14 ────────────────────┤
          └───→ CH-11 ─┘     └──────────→ CH-14 ────────────────────┤
CH-02 → CH-05 ────────────────────────────→ CH-14  (CH-04 or CH-09 ⇢ CH-05)
G5 → CH-15 ────────────────────────────────────────────────────────┘
```

## 6. Critical path

- **Score path:** CH-02 → CH-09 → CH-10 → CH-12 → CH-14 → CH-16. The final model is most likely a YOLO11 or RT-DETR variant,
  but CH-12 decides that from evidence.
- **Rubric-completeness path:** CH-02 → CH-04 → CH-07 / CH-08, and CH-09 → CH-11, all feeding into CH-12.
- **Decision gate for the final config:** CH-12 freezes `configs/runs/final.json`. CH-14 trains from it. CH-13 (inference-side
  settings) runs in parallel with CH-14 training and is a finish dependency of CH-14.

## 7. Parallel waves and schedule (Kyiv time)

| Wave | Unlocks when | Chunks available | Suggested split (anyone may claim any READY row) |
|---|---|---|---|
| W0 | now (Sat 7, 15:00) | CH-01, CH-02, CH-06; CH-15 once G5 is resolved | Nazar: CH-01, then CH-02. Marko: G5 (2 min), CH-06 (lock), CH-15 |
| W1 | CH-02 DONE (target Sat ~22:00) | CH-03, CH-04, CH-05, CH-09 | Marko: CH-03, then CH-09. Nazar: CH-04, then CH-05 (**first Kaggle upload by Sun 20:00**) |
| W2 | CH-04 / CH-09 DONE (Sun) | CH-07, CH-08, CH-10, CH-11 | Sun night: CH-07 on the 5070, CH-08 on the 3050. Mon: CH-10 (3050), CH-11 (5070, 8 GB) |
| W3 | CH-07, CH-08, CH-10, CH-11 DONE (target Mon 17:00) | CH-12 | either; **decision point Mon 19:00** |
| W4 | CH-12 DONE | CH-14 (training) in parallel with CH-13 | **Final training starts no later than Mon 22:00** (5070). CH-13 on the 3050 |
| W5 | CH-14 DONE | CH-16 | Tue |

Latest safe points:
- **Mon 19:00:** if a family chunk is not DONE, CH-12 goes to 🔴 NEEDS_DECISION. A student decides whether to proceed with a
  recorded deviation.
- **Mon 22:00:** final training must have started.
- **Tue 08:00:** no new training jobs after this.
- **Tue 12:00:** experiments stop.
- **Tue 16:00:** test predictions, uploads and the final selection are done.
- **Tue 19:00:** students have finished the report prose (drafting starts Mon evening from the evidence in `results/`).
- **Tue 21:30:** freeze (`develop → main`).
- **Tue 22:30:** Pages and repo verification done. The remaining time is buffer up to the verified official Kaggle Close at 23:55 Kyiv / 21:55 UTC (G1); the internal freeze remains 21:30 Kyiv.

CPU and documentation work fills GPU waits: CH-06, CH-12, CH-15, README sections, evidence logs.
The estimated GPU need is about 18–22 GPU-hours across two GPUs. That is **feasible but not guaranteed**.

Fallbacks:
- OOM: lower batch or input size, use AMP, freeze early backbone stages.
- Ultralytics or torch problems on the 5070: run on the 3050, or move the anchor-free work to torchvision FCOS (🔴 NEEDS_DECISION).
- RT-DETR unreliable on 8 GB: follow the G9 options in CH-11.
- Final training late: submit the best CH-10 / CH-11 checkpoint with the CH-13 settings.

## 8. Branch model, file ownership, merge conflicts

| Branch | Purpose | Written by |
|---|---|---|
| `main` | Stable milestones only; GitHub Pages source; default branch (unchanged) | only `develop → main` milestone PRs |
| `develop` | Integration branch and live board | task-PR merges plus tiny `coord(CH-XX): …` commits |
| `ch-XX-<slug>` | One task branch per chunk, from the latest `origin/develop` | the chunk owner only |
| `claim/CH-XX-<login>` | Local, throwaway branch used only for the claim push (§9.2) | the claimer |

Milestone PRs into `main`:
- **M1:** after CH-15, so Pages can be verified.
- **M2:** optional, after the first valid submission (CH-05).
- **M3:** final freeze in CH-16.

No task branch is based on `main`. Nothing is pushed directly to `main`. There are no force pushes and no history rewriting.
Marko may protect `main` later; that is not a blocker.

File ownership:
- New files belong to the chunk that creates them.
- `movd/data.py` belongs to CH-02 and `movd/evaluate.py` to CH-03. Later changes to either go through a PR reviewed by the other student.
- `requirements.txt` and `README.md`: edit only your own lines or section.
- `plan.md`: edit only your own row and your own chunk spec or evidence log, and never reflow other rows.
- `homework.ipynb`: only under the 🔒 lock.
- Figures go in `results/…/figures/`. `images/` and `report.md` are written only by CH-15, CH-16 and the students.
- `index.html`, `assets/`, `.github/` and `.nojekyll` are never edited.

Conflict resolution:
- `plan.md`: keep the newest version of every other row. Your own row takes your PR's final state.
- Notebook conflicts should not happen under the lock. If one occurs, stop and report; do not hand-merge notebook JSON.

## 9. Progress protocol

### 9.1 Legal state transitions

- `🟡 BLOCKED → 🟢 READY`: only when every start dependency is ✅ DONE and every start gate is resolved.
- `🟢 READY → 🔵 IN_PROGRESS`: only through the atomic claim (§9.2).
- `🔵 IN_PROGRESS → ✅ DONE`: only when every acceptance criterion passes, the finish dependencies are DONE, and the work is
  merged into `develop`. Code that has only been written, or a training run that has only finished, never counts.
- `🔵 IN_PROGRESS → 🟡 BLOCKED`: a real blocker is found. Push the update immediately with the exact chunk ID or gate.
- `🔵 IN_PROGRESS → 🟠 PARTIAL / PAUSED`: useful incomplete work is pushed and recorded, and work stops.
- `🟠 PARTIAL / PAUSED → 🔵 IN_PROGRESS`: only through an explicit claim by the recorded owner, or a recorded handoff.
- `🔴 FAILED / NEEDS_DECISION → 🟢 READY`: only after the failure is recorded and a student approves a revised approach.
- **No destructive Git:** `reset --hard`, force push, `clean -f` and history rewriting are never used automatically. Only a human
  may run them, explicitly.
- **Unblock recalculation:** whoever marks a chunk DONE re-checks every row that lists it. A row flips to 🟢 READY with
  `🟢 YES` only if all of its start dependencies are DONE and its gates are resolved. If a still-blocked row is now waiting
  on fewer IDs, update its `Start now?` text. Rows that are IN_PROGRESS and had it as a finish dependency get a short note.
- **`Start now?` values:**
  - `🟢 YES` only for READY.
  - `⛔ NO — CH-XX[, CH-YY]` or `⛔ NO — gate Gn` for BLOCKED.
  - `⛔ NO — owned by <login>` for IN_PROGRESS.
  - `⛔ NO — completed` for DONE.
  - `⛔ NO — decision required` for PARTIAL / PAUSED and FAILED / NEEDS_DECISION.

### 9.2 Atomic claim (mandatory for Nazar, Marko, Claude, Codex)

1. `git fetch origin`, `git switch develop`, `git pull --ff-only`. The working tree must be clean; otherwise stop.
2. Read the latest `CLAUDE.md` and this board.
3. Confirm the row is `🟢 READY` with `Start now? = 🟢 YES` and Owner `—`, all start dependencies are DONE, the gates are resolved,
   and your machine meets the prerequisites. If anything is ambiguous or inconsistent, stop and ask.
4. `git switch -c claim/CH-XX-<login> origin/develop`, then edit **only that row**:
   - State → `🔵 IN_PROGRESS`
   - Start now? → `⛔ NO — owned by <login>`
   - Owner → `<login>`
   - Branch / PR → `ch-XX-<slug>`
   - Last update → now
   - In the evidence log, set "Claimed".
   - If you need the notebook lock now, take it in the same commit.
5. `git commit -m "coord(CH-XX): claim <title> for <login>"`, then `git push origin claim/CH-XX-<login>:develop`.
   This is a normal fast-forward push, never a force push.
6. **The successful push is the lock.** A local edit or local commit is not a claim.
7. Then `git switch develop`, `git pull --ff-only`, `git switch -c ch-XX-<slug>`, `git push -u origin ch-XX-<slug>`, and start working.
   The local claim branch may be removed with `git branch -d claim/CH-XX-<login>`. This safe form refuses to delete unmerged work.

If the push is rejected, never use destructive recovery:
- Never force-push. Never run `git reset --hard`, `git checkout -- .` or `git clean`, and never delete the claim branch automatically.
  Preserve all local work.
- Run `git fetch origin` and inspect: `git log --oneline claim/CH-XX-<login>..origin/develop` and
  `git diff claim/CH-XX-<login> origin/develop -- plan.md`.
- **Same chunk or same row changed on `origin/develop`:** stop and report the collision (who claimed it, which commit) to the user or
  teammate. Leave the claim branch untouched and do not start the chunk. A human decides what happens next, for example claiming
  another READY row and later deleting the stale claim branch as their own explicit action.
- **Only unrelated changes:** run `git merge --no-edit origin/develop` on the claim branch.
  - If the merge conflicts, run `git merge --abort`, then stop and report.
  - If it succeeds, re-read the row and confirm it is still READY + YES with no owner, then retry the push.
  - Retry at most twice, then stop and report.
- Local `develop` was never modified, so `git pull --ff-only` on it always stays clean.

### 9.3 While working

- Work on the branch recorded in the board. Commits are `CH-XX: …`; PR titles start with `CH-XX`.
- One active chunk per person. A second CPU/documentation chunk is allowed only while the first waits on a long GPU or external
  operation, and it must be recorded in both rows.
- **Heartbeat:** about every 4 h of active work and before any long break, push `coord(CH-XX): heartbeat` to `develop`.
  It updates only your row's `Last update` and, if useful, a short note in "Blocked by / reason". Make it with the same claim-branch
  pattern as §9.2.
- Never hold the 🔒 notebook lock during a GPU run.
- Push all safe work to the task branch before stopping.

### 9.4 Blocked, paused, failed, stale

- **Blocked or paused:** push a board update immediately. Include the state, the exact blocker (chunk ID or gate), the branch,
  the preserved artifacts and the next required action. Record the details in the evidence log.
- **Failed approach:** set `🔴 FAILED / NEEDS_DECISION` and keep all evidence. A student approves the revised approach before the row
  goes back to READY.
- **Stale row** (no update for about 6 h): do not take it over. Contact the owner or user, record the handoff decision in the evidence
  log, and only then change the owner.

### 9.5 Completion

1. Run every validation and acceptance check in the spec.
2. Fill the evidence log (§9.7) with real commands, metrics, artifacts, runtime, hardware and deviations.
3. `git merge origin/develop` into the task branch. Resolve `plan.md` keeping every other row's latest state.
4. In the same PR: set the row to `✅ DONE` / `⛔ NO — completed`, with the PR link and timestamp, and apply the unblock recalculation (§9.1).
5. Open the PR into `develop`. The author may self-merge after the checks pass. Shared modules (`movd/data.py`, `movd/evaluate.py`,
   `movd/submission.py`) get a quick review from the other student when they are available.
6. After the merge, check the board on `origin/develop`. Fix any drift with `coord(CH-XX): post-merge board fix`.
7. Never mark DONE while the work exists only on an unmerged task branch.

### 9.6 Session start after a break

1. `git fetch origin`, then inspect `origin/develop`.
2. Read the board first: the 🔵, 🟠, 🟡 and 🔴 rows, the timestamps, the branches, and `gh pr list --base develop`.
3. Choose only a row that is `🟢 READY` with `Start now? = 🟢 YES`.
4. Claim it with §9.2 before changing any code.

### 9.7 Evidence log template (one per chunk, in §10)

```text
Owner: — | Claimed: — | Last update: — | Branch: — | Commits: — | PR: —
Files changed: —
Validation commands: —
Metrics / results: —
Artifacts: —
Runtime / hardware: —
Blockers / deviations: —
Completion or handoff note: —
```

## 10. Chunk specifications

Owner and state live **only on the board (§1)**. Specs do not duplicate them, so they cannot drift apart. "Files" lists the expected
changes. Anything else needs a recorded deviation.

---

### CH-01 — Repo hygiene & local dataset/env setup

- **Goal:** Make the executing machine ready (dataset in the right place plus a CUDA environment) and make the repo safe against accidental data commits.
- **Why required:**
  - Every data or GPU task needs the dataset at the path the code expects.
  - The ZIP has no top-level folder, so naive extraction would dump `train/ val/ test/` into the repo root, where those paths are not ignored.
  - `requirements.txt` is incomplete, and the RTX 5070 needs an sm_120-capable PyTorch build.
- **Reuse / preserve:**
  - The existing `.gitignore` already ignores `*.zip` and `/multiple-object-video-detection-hw-2-ucu-ai-2026/`.
  - The notebook constant `PROJECT_ROOT / "multiple-object-video-detection-hw-2-ucu-ai-2026"`.
  - The README §2 dataset table.
- **Scope:**
  1. Check free disk space (≥ 25 GB) and that the target directory does not exist yet.
  2. Add `.gitignore` guards (`/train/`, `/val/`, `/test/`, `/test_images.csv`, `/sample_submission.csv`, `/submissions/`) and commit them **before** extracting.
  3. Extract with `unzip -q multiple-object-video-detection-hw-2-ucu-ai-2026.zip -d multiple-object-video-detection-hw-2-ucu-ai-2026`
     (or the Windows equivalent with the same target).
  4. Add `scripts/check_dataset.py` (standard library only). It verifies the structure and counts: 24,201 / 2,846 / 3,811 images,
     56 / 7 / 48 clips, the label and ignore directories, that both COCO JSON files load, and that `test_images.csv` exists.
     It is the reusable **M-DATA** check for every machine.
  5. Create `.venv` on Python 3.12 (uv) and install a CUDA PyTorch build that supports sm_120. Verify that `torch.cuda.is_available()`
     is true and that a small conv runs on the GPU (G7).
  6. Complete `requirements.txt` with verified pins: torch/torchvision (with an index-URL note), opencv-python, tqdm, pytest.
  7. Add a README "Setup" section: dataset placement and extraction for Windows and Linux, the venv, and the verification command.
  8. Check G6: Nazar's commit email is linked to `Badabuba`.
- **Non-goals:** EDA, training, notebook edits, deleting or moving the ZIP, results sections in the README.
- **Start deps:** — · **Finish deps:** — · **Machine prerequisites:** M-ZIP · **Gates:** G6 (check), G7 (resolve).
- **Parallel:** yes, with CH-02 (on another machine), CH-06 and CH-15.
- **Owner / State:** see the board.
- **Files expected to change:** `.gitignore`, `requirements.txt`, `README.md` (Setup section only), `scripts/check_dataset.py`.
- **Conflict risk:** low. `requirements.txt` and `README.md` are shared, so edit only the relevant lines or section.
- **Implementation outline:** disk check → ignore-guard commit → extract → `check_dataset.py` → venv + torch → requirements → README → G6.
- **Validation:**
  - `python scripts/check_dataset.py` exits 0 and prints the counts.
  - `git check-ignore -v` on the ZIP, the dataset directory and a sample image.
  - `git status --porcelain` shows no data files.
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
  - The ZIP size is still 10,611,162,657 B.
- **Evidence & artifacts:** the command outputs pasted into the evidence log. No data is committed.
- **Acceptance criteria:**
  1. The dataset sits at `./multiple-object-video-detection-hw-2-ucu-ai-2026/{train,val,test,test_images.csv}`, with no double nesting and no root-level `train/`.
  2. `check_dataset.py` passes.
  3. Git ignores everything under the dataset directory and the ZIP, and `git status` lists no data files.
  4. Torch sees CUDA on the target GPU, or G7 is recorded as failed with a fallback.
  5. `requirements.txt` installs cleanly in a fresh venv.
  6. The README Setup section exists.
  7. The ZIP is unchanged.
  8. G6 status is recorded.
- **Effort:** about 1.5 h. **Runtime / GPU / disk:** about 10–20 min to unzip, +11 GB dataset, about 4 GB venv.
- **Likely blockers:** the sm_120 wheel (G7); Windows Application Control blocking some Python executables (one uv-managed `python3.12.exe` was already blocked on Nazar's machine); disk I/O time.
- **Handoff / unblocks:** satisfies M-DATA and M-GPU on that machine. CH-01 is a start dependency of CH-16 only. No row is unblocked directly.
- **Evidence log:**
```text
Owner: Badabuba | Claimed: 2026-11-07 15:55 Kyiv (claim commit bc244c5) | Last update: 2026-11-07 16:15 Kyiv | Branch: ch-01-repo-env-setup | Commits: e7bac7a, 2d5bde4 + board commit | PR: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/1
Files changed: .gitignore, scripts/check_dataset.py (new), requirements.txt, README.md (new "Setup" subsection under §3), plan.md (CH-01 row, G6/G7, CH-16 Start now?, this log)
Validation commands:
  Pre-checks: ZIP 10611162657 B; disk 228 GB free; dataset dir absent. Guard commit e7bac7a pushed BEFORE extraction.
  unzip -q <zip> -d multiple-object-video-detection-hw-2-ucu-ai-2026  -> exit 0, real 3m47s; no root-level train/ val/ test/
  python scripts/check_dataset.py -> PASSED, exit 0 (32 OK, 3.3 s): train 56 clips/24201 img/24201 labels/ignore 50 clips/
    COCO 24201 img + 1616260 ann; val 7/2846/2846/ignore 5 clips/COCO 2846 + 146092 ann; categories = 8 expected names;
    test 48 clips/3811 img; test_images.csv 3811 rows, 0 missing files; no double nesting; no stray root dirs
  git check-ignore -v: zip -> .gitignore:14 *.zip; dataset dir, sample jpg, nested test_images.csv -> .gitignore:2;
    .venv/ -> :18; root train/ -> :34; root test_images.csv -> :37
  git status --porcelain -> only README.md, requirements.txt, scripts/ (no data files)
  python -c "import torch; ..." -> 2.14.1+cu130 True NVIDIA GeForce RTX 5070 Laptop GPU
    cuda 13.0, cudnn 92400, capability (12, 0), arch_list includes sm_120; Conv2d on GPU max |diff| vs CPU 7.2e-07;
    torchvision.ops.nms on CUDA OK
  Fresh .venv (uv 0.12.21, CPython 3.12.14): cu130 torch install, then uv pip install -r requirements.txt -> OK;
    uv pip check -> "All installed packages are compatible" (115 pkgs); torch still +cu130 afterwards
  G6: gh api repos/.../commits/<sha> -> author.email mykolaichuk.pn@ucu.edu.ua, login Badabuba (bc244c5, e7bac7a)
  ZIP size after extraction: 10611162657 B (unchanged)
Metrics / results: none (setup chunk)
Artifacts: none committed. Local only: ./multiple-object-video-detection-hw-2-ucu-ai-2026/ (~11 GB), .venv/
Runtime / hardware: ~20 min wall-clock; Windows 11, RTX 5070 Laptop 8 GB (8151 MiB), driver 610.74
Blockers / deviations: requirements pins added: opencv-python==5.0.0.93, tqdm==4.70.1, pytest==9.2.0 (latest on PyPI 2026-11-07);
  existing torch==2.14.1 / torchvision==0.29.1 pins kept, served from the cu130 index (cu128/cu129 have no 2.14.1 Windows wheels).
  The PowerShell tar.exe extraction command in the README is documented but was not run (Git Bash unzip was used).
  The venv uses the uv-managed CPython 3.12.14 by full path; ~/.local/bin/python3.12.exe is blocked by Application Control.
  The claim commit row timestamp said 13:54 (UTC, Git Bash TZ issue); the real claim time was 15:55 Kyiv.
Completion or handoff note: M-DATA and M-GPU are satisfied on Nazar's machine. Other machines run `python scripts/check_dataset.py`.
  Unblock recalculation: CH-01 is a start dep of CH-16 only; CH-16 stays BLOCKED on CH-06, CH-12, CH-14, CH-15.
```

---

### CH-02 — Protocol freeze, Kaggle facts, data module

- **Goal:** Freeze every shared decision that later chunks rely on, and provide `movd/data.py`.
- **Why required:**
  - Without a frozen protocol the four detector families are not comparable.
  - The legacy run mishandled ignored regions and detection caps.
  - The Kaggle facts are still unknown.
- **Reuse / preserve:**
  - The legacy notebook's COCO loading, the label mapping logic (`category_ids`), and the clip-level overlap check.
  - Its subsampling idea (evenly spaced frames per clip).
- **Scope:**
  1. Record G1–G3 in §4, sourced from the Kaggle page by a student: the UTC close time; the metric with IoU thresholds, ignore handling and max detections; the daily limit; the number of final selections; and the sample-submission header and format. Put a downloaded `sample_submission.csv` in the dataset directory, where it is git-ignored.
  2. Inspect the `ignore/` txt format, its coordinate convention, and how it relates to COCO `iscrowd`/`ignore` annotations (G4). Inspect the `test_images.csv` columns and the `image_id` mapping, `data.yaml`, the YOLO label format, and the image sizes per clip.
  3. Decide and document the **ignore-region policy**:
     - the training treatment (masking is the leading candidate, but the decision follows the verified semantics);
     - the evaluation rule, matching G2.
  4. Confirm the label map and fill §4.
  5. Create the fixed split files `configs/splits/{train_rubric,val_fast,val_full}.json` (image-id lists), clip-stratified and seeded.
  6. Fix the rubric compute budget in `configs/protocol.json`: number of frames, epochs and input long side, with the estimate behind it.
  7. Write `movd/data.py`: paths, COCO and ignore loaders, the label map, split loading, and the ignore policy implementation (for example a masking function), plus `tests/test_data.py`.
  8. Move `configs/data.json` and `configs/experiments.json` to `configs/legacy/`, with a `README.md` note that they do not match the v0 notebook run.
- **Non-goals:** training, the evaluator (CH-03), notebook edits, EDA plots beyond quick checks.
- **Start deps:** — · **Finish deps:** — · **Machine prerequisites:** M-DATA · **Gates:** resolves G1, G2, G3, G4.
  G2 and G3 need a student to read Kaggle; at most one follow-up prompt is expected for this.
- **Parallel:** yes, with CH-01, CH-06 and CH-15.
- **Owner / State:** see the board.
- **Files expected to change:**
  - `movd/__init__.py`, `movd/data.py`
  - `configs/protocol.json`, `configs/splits/*.json`, `configs/legacy/*`
  - `tests/test_data.py`
  - `plan.md` §4 and the gates table
- **Conflict risk:** medium. This chunk defines the shared module and §4. Nobody else edits §4 while it is IN_PROGRESS.
- **Implementation outline:** Kaggle facts → data inspection → policy decision (with a short written rationale in the evidence log) → splits → budget → module + tests → legacy config move.
- **Validation:**
  - `python -m pytest tests/test_data.py`.
  - The split files are deterministic: regenerating them gives the same hash.
  - The split files show no train/val clip overlap.
  - The masking (or chosen treatment) is checked visually on 2 frames, with the figure in `results/ch-02-protocol/figures/`.
- **Evidence & artifacts:** `configs/protocol.json`, the split files, `results/ch-02-protocol/` (notes and figures), and the filled §4.
- **Acceptance criteria:**
  1. G1–G4 are resolved, with sources in the gates table and §4.
  2. The ignore-region policy is written for both training and evaluation, with evidence.
  3. No PENDING remains in §4 except items explicitly deferred to a named chunk.
  4. The split files and protocol JSON are committed.
  5. The `movd/data.py` tests pass.
  6. The legacy configs are moved with a note.
- **Effort:** about 3 h. **Runtime / GPU / disk:** CPU only, small.
- **Likely blockers:** Kaggle page access (a student must read it); ambiguous ignore semantics (record the evidence and choose the safer rule).
- **Handoff / unblocks:** CH-03, CH-04, CH-05 and CH-09 become READY (their start dependency).
- **Evidence log:**
```text
Owner: Badabuba | Claimed: 2026-11-07 16:34 Kyiv | Last update: 2026-11-07 17:00 Kyiv | Branch: ch-02-protocol-data | Commits: e0fa838, 0b59367, 77f8725, 9c4aab4 (+ PR-link commit) | PR: #2 (https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/2)
Files changed: movd/__init__.py, movd/data.py; tests/test_data.py, tests/test_sample_submission.py; scripts/make_splits.py,
  scripts/make_sample_submission.py; configs/protocol.json, configs/splits/{train_rubric,val_fast,val_full}.json;
  configs/legacy/{data.json,experiments.json} (git mv), configs/legacy/README.md; results/ch-02-protocol/{inspect_dataset.py,
  data_facts.json,make_mask_figure.py,figures/ignore_masking_check.png}; plan.md (§1 rows, gates, §4, §5, §11, this log).
Validation commands:
  python -m pytest tests -q                       -> 38 passed (30 test_data + 8 test_sample_submission), 7.8 s
  python scripts/make_splits.py --check           -> [OK] x3, canonical sha256 train_rubric 58acdbb35f1b6fed…,
                                                     val_fast e4eaf32da5896462…, val_full 1a09ab8ac02b1e2b… (= written)
  python scripts/make_sample_submission.py [--check] -> [OK] header image_id,PredictionString, 3811 rows, ids unique,
                                                     same order as test_images.csv, none missing/extra, all empty,
                                                     sha256 aee0b5dcd332c05c3658a9e34a0dee1c1eb5f2e3cb14da40b61c7574d56f37b2
  python results/ch-02-protocol/inspect_dataset.py  -> data_facts.json (12.6 s)
  python results/ch-02-protocol/make_mask_figure.py -> figures/ignore_masking_check.png (1 val + 1 train frame, checked visually)
  python scripts/check_dataset.py                 -> PASSED (M-DATA)
Metrics / results (verified facts):
  - ignore/*.txt: one line per region, normalized "cx cy w h", no class; = VisDrone orig 0 (ignored regions) + 11 (others).
    COCO JSON repeats each region once per category 1–8 (iscrowd=1, ignore=1): train 63,843 regions x 8 = 510,744,
    val 3,995 x 8 = 31,960; 100 % of COCO regions matched in the txt files (train and val).
  - Targets with >= 50 % of their area in an ignore region: train 1.07 % (11,883), val 1.06 % (1,208) -> masking keeps target pixels.
  - Label map: merged VisDrone 1+2->1, 3->2, 4->3, 5->4, 6->5, 7+8->6, 9->7, 10->8; YOLO class = category_id-1 (24,339 boxes, 0 mismatch).
  - Splits: train_rubric 2,800 (50/clip x 56), val_fast 714 (every 4th, 7 clips), val_full 2,846; 0 train/val clip overlap.
  - Budget: 6 epochs x 2,800 = 16,800 images, long side 1344. RTX 5070 synthetic train step (AMP, bs 2): Faster R-CNN v2
    8.38 img/s (3.33 GiB), RetinaNet v2 10.19 img/s at 1344; 6.09 / 7.17 img/s at 1600 -> about 33 min for Faster R-CNN.
  - Test: 3,811 frames, 48 clips; image_id is a string (clip_001_0000001); sizes include 680x382 and 960x540.
Confirmed official facts (Kaggle Overview/Rules/Data text copied by Badabuba, 2026-11-07):
  - Detections inside ignored regions are not penalized; COCO ignored annotations use iscrowd=1; <= 500 dets/image scored.
  - Submission: image_id,PredictionString; one row per test image; "class score x y width height", class 0–7, top-left,
    original pixels; empty PredictionString valid.
  - 2 final selections, chosen by internal evaluation, not only the public LB.
  Frozen internal metric (rubric-required standard COCO, pycocotools): mAP = AP@0.50:0.05:0.95, plus AP50, AP75, AP_S/M/L,
  per-class AP, maxDets 500 headline, 100 for the legacy note only, GT incl. iscrowd=1 regions, on val_full.
Unresolved operational facts (not on the supplied pages, NOT recorded as values): exact UTC close time (G1, finish gate of
  CH-16); daily submission limit (G2b, before CH-05's first upload); hidden Kaggle scorer code (not exposed; the internal
  protocol is not claimed identical to it).
Artifacts: configs/protocol.json, configs/splits/*.json, results/ch-02-protocol/ (data_facts.json, figure, scripts).
  Local only (git-ignored): multiple-object-video-detection-hw-2-ucu-ai-2026/sample_submission.csv.
Runtime / hardware: about 30 min of agent work; CPU inspection < 1 min per script; GPU benchmark about 2 min.
  Nazar's laptop: RTX 5070 Laptop 8 GB, Windows 11, Python 3.12 .venv, torch 2.14.1+cu130, torchvision 0.29.1+cu130.
Blockers / deviations:
  - sample_submission.csv is listed in the official Data description but missing from the "Download All" ZIP
    (77,434 entries; only CSV is test_images.csv). The local copy is generated from the documented schema and
    test_images.csv order by scripts/make_sample_submission.py, not downloaded; it is git-ignored and never committed.
  - Acceptance 1 ("G1–G4 resolved") is met with a re-scope approved by Badabuba (§11): G2 split into G2a/G2b/G2c; G1 and G2b
    are operational gates carried by CH-16 and CH-05, since the official pages do not state them.
  - Throughput is a synthetic estimate without data loading; Ultralytics was not installed, so YOLO11/RT-DETR time is not
    measured here (RT-DETR: gate G9 in CH-11).
  - Added scripts/make_splits.py, scripts/make_sample_submission.py, tests/test_sample_submission.py and results/ch-02-protocol/
    scripts beyond the expected file list (generators and evidence for the validation steps).
Completion or handoff note: CH-03, CH-04, CH-05 and CH-09 set READY (start dep CH-02 DONE). CH-03 implements the evaluation
  rule above in movd/evaluate.py; CH-04/CH-09 use movd.data.training_sample / mask_ignore_regions for training data.
```

---

### CH-03 — Evaluator, speed benchmark, tests

- **Goal:** Provide one framework-agnostic evaluator and a speed benchmark that every headline number uses.
- **Why required:**
  - The legacy evaluation capped detections at 100 and ignored the `ignore/` regions.
  - Results from torchvision and Ultralytics must be comparable.
- **Reuse / preserve:** the legacy `evaluate_model` (pycocotools usage, metric names) and `measure_retina_fps` (warm-up plus CUDA sync).
- **Scope:**
  1. `movd/evaluate.py` and `scripts/evaluate.py`. Input: a COCO-results JSON (§4) plus a split name.
  2. Settings: maxDets 500 as the headline, with 100 also reported.
  3. Outputs: mAP, AP50, AP75, AP_S/M/L, per-class AP, and AP by density bin (objects per frame), written to `metrics.json` and `per_class.csv`.
  4. Apply the evaluation side of the ignore-region policy frozen by CH-02.
  5. `movd/speed.py`: a benchmark (warm-up, sync, latency, FPS, recorded GPU name, input size and batch size).
  6. `tests/test_evaluate.py` on synthetic data:
     - perfect predictions → mAP 1.0;
     - a detection inside an ignore region is not penalized, per the policy;
     - more than 100 correct detections in one image are all counted;
     - class-ID shifting is caught.
- **Non-goals:** training and inference code, re-running legacy experiments. Re-evaluating one legacy checkpoint is optional if one is available.
- **Start deps:** CH-02 · **Finish deps:** — · **Machine prerequisites:** M-DATA · **Gates:** —.
- **Parallel:** yes, with CH-04, CH-05 and CH-09, which start once CH-02 is DONE but need CH-03 to finish.
- **Owner / State:** see the board.
- **Files expected to change:** `movd/evaluate.py`, `movd/speed.py`, `scripts/evaluate.py`, `tests/test_evaluate.py`, `results/ch-03-evaluator/`.
- **Conflict risk:** low, since these are new files. Shared-module rule afterwards.
- **Implementation outline:** wrap pycocotools COCOeval with configurable `maxDets` → ignore-rule hook → per-class and density breakdown → CLI → tests → speed utility.
- **Validation:**
  - `python -m pytest tests/test_evaluate.py`.
  - Evaluating GT-as-predictions on `val_fast` gives about 1.0, written to `results/ch-03-evaluator/sanity.json`.
- **Evidence & artifacts:** test output and `results/ch-03-evaluator/sanity.json`.
- **Acceptance criteria:**
  1. All tests pass.
  2. The sanity check is ≥ 0.99 mAP on `val_fast`.
  3. The CLI documents its usage in `--help`.
  4. maxDets 500 is the default.
  5. The ignore rule follows the CH-02 policy, which is cited in the module docstring.
- **Effort:** about 3 h. **Runtime / GPU / disk:** CPU. Full val evaluation takes about 1–2 min.
- **Likely blockers:** COCOeval slowness with 500 detections per image (acceptable); details of the ignore rule.
- **Handoff / unblocks:** satisfies the finish dependency of CH-04, CH-05 and CH-09.
- **Evidence log:**
```text
Owner: Badabuba (worker: Claude) | Claimed: 2026-11-07 17:06 Kyiv (claim commit 9e72a61; its board row said 15:06, a UTC clock slip) | Last update: 2026-11-07 17:18 Kyiv | Branch: ch-03-evaluator | Commits: 9e72a61 (claim), 90e5b92 (implementation + evidence), merge of origin/develop, board commits | PR: [#3](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/3)
Files changed: movd/evaluate.py, movd/speed.py, scripts/evaluate.py, tests/test_evaluate.py (new); results/ch-03-evaluator/{make_sanity.py, sanity.json, make_speed_selftest.py, speed_selftest.json, cli_help.txt, pytest.txt, cli_example_val_fast/metrics.json, cli_example_val_fast/per_class.csv}; plan.md (this row, CH-04/05/09 notes, this log)
Validation commands: python -m pytest tests/ (59 passed: 38 existing + 21 new; output in results/ch-03-evaluator/pytest.txt) · python -m pytest tests/test_evaluate.py (21 passed) · python results/ch-03-evaluator/make_sanity.py · python results/ch-03-evaluator/make_speed_selftest.py · python scripts/evaluate.py --help · python scripts/evaluate.py <GT-as-preds json> --split val_fast --out results/ch-03-evaluator/cli_example_val_fast · ruff check + ruff format on the new files
Metrics / results: GT-as-predictions mAP@500: val_fast 1.0000, val_full 1.0000 (acceptance ≥ 0.99 ✓). Same at the legacy per-image cap 100: 0.9839 on both (density bin ≥100 targets/frame: 189 val frames, mAP 0.80), so the v0 cap alone costs recall on dense frames. Ignore rule on real data (val_fast): adding a score-1.0 detection on each of the 3,995 val ignore regions (classes cycling 1–8) leaves mAP at 1.0000; with the crowd annotations removed from GT the same detections drop mAP to 0.8667 (control). In-range label shift (c → c % 8 + 1): mAP 0.0004; 0–7 class ids are rejected with an error. Speed self-test (random-weight Faster R-CNN v2, fp32, bs 1, long side 1344, 30 val_fast frames, 10 warm-up): 8.51 / 8.44 / 8.28 FPS over 3 repeats (2.8 % spread), ≈ 117 ms/image — a utility check, not a model result
Artifacts: results/ch-03-evaluator/ (above). GT-as-predictions JSON in git-ignored outputs/ch-03-evaluator/
Runtime / hardware: CPU evaluation: val_fast ≈ 13 s, val_full ≈ 53 s (two COCOeval passes, caps 500 and 100). Speed self-test on NVIDIA GeForce RTX 5070 Laptop GPU 8 GB, torch 2.14.1+cu130, torchvision 0.29.1, CUDA 13.0, Python 3.12.14, pycocotools 2.0.11, Windows 11
Blockers / deviations: (1) Clarification, not a protocol change: pycocotools' `maxDets` counts detections per image per class, so `maxDets=500` alone would let up to 8 × 500 detections per image through. The §4 value (500, "Kaggle scores ≤ 500 dets/image") is implemented as "keep each image's top-500 by score over all classes, then COCOeval maxDets=[500]" (`movd.evaluate.cap_per_image`); the legacy 100 uses the same per-image rule, like torchvision's `box_detections_per_img=100`. Documented in the module docstring and tested. (2) Density bins are fixed at [0,20), [20,50), [50,100), [≥100] target objects per frame (val: 977 / 1,062 / 618 / 189 frames). (3) No legacy checkpoint re-evaluated (optional; none in the repo). (4) Worked in a separate git worktree because the CH-04 session uses the main checkout concurrently
Completion or handoff note: Use `movd.evaluate.evaluate_split(predictions, "val_full")` (or the CLI) for every headline number; write predictions in the §4 COCO-results format with category_id 1–8. `movd.speed.benchmark(run, batches, device, batch_size=..., input_size=...)` for speed; state in `note` what the timed call includes. Unblocks the CH-03 finish dependency of CH-04, CH-05 and CH-09. Shared-module rule now applies to movd/evaluate.py (§8)
```

---

### CH-04 — Torchvision training core

- **Goal:** Provide a reusable, resumable, Windows-safe trainer for Faster R-CNN and RetinaNet under the frozen protocol.
- **Why required:**
  - The notebook-defined Dataset with `num_workers > 0` breaks on Windows.
  - The legacy runs were not resumable, capped detections at 100, and treated ignored objects as background.
- **Reuse / preserve:** the legacy `VisDroneDataset`, `collate_fn`, `train_detector` (AMP, gradient clipping), the model constructors and head replacement, and `create_retinanet`.
- **Scope:**
  1. `movd/tv_train.py` and `scripts/train_tv.py --config configs/runs/<run>.json`.
  2. An importable Dataset that applies the ignore-region policy frozen by CH-02, horizontal flip and the chosen resize.
  3. Model factories for Faster R-CNN and RetinaNet, with configurable anchors, RPN/RoI settings, focal α/γ and `detections_per_img=500`.
  4. Per-epoch checkpoints in `outputs/<run_id>/` and `--resume`.
  5. A per-epoch `val_fast` evaluation through CH-03, written as a curve.
  6. A final `val_full` evaluation and speed measurement written to `results/<run_id>/`.
  7. A timing smoke that confirms or records the rubric budget.
- **Non-goals:** the actual rubric experiments (CH-07, CH-08), the Ultralytics code.
- **Start deps:** CH-02 · **Finish deps:** CH-03 · **Machine prerequisites:** M-DATA, M-GPU · **Gates:** —.
- **Parallel:** yes, with CH-03, CH-05 and CH-09.
- **Owner / State:** see the board.
- **Files expected to change:**
  - `movd/tv_train.py`, `scripts/train_tv.py`
  - `configs/runs/smoke_frcnn.json`, `configs/runs/smoke_retina.json`
  - `tests/test_tv_train.py`
  - `results/smoke_*/`
  - `requirements.txt` (own lines)
- **Conflict risk:** low.
- **Implementation outline:** port the notebook code into the module → config schema → checkpoint/resume → evaluation hook → smoke runs → kill-and-resume test.
- **Validation:**
  - A 50-iteration smoke run for both models.
  - Interrupting after epoch 1 and resuming continues from epoch 2, with the same step count logged.
  - `pytest tests/test_tv_train.py` (dataset target shapes; the ignore policy is applied).
- **Evidence & artifacts:** `results/smoke_frcnn/`, `results/smoke_retina/` (config, curve, metrics, speed, hardware) and the timing note.
- **Acceptance criteria:**
  1. Both smoke runs complete on the GPU with finite losses.
  2. The resume test passes.
  3. Models output up to 500 detections.
  4. `results/` artifacts are written in the documented layout.
  5. The rubric budget is confirmed or adjusted, and any adjustment is logged in §11.
- **Effort:** about 3 h. **Runtime / GPU / disk:** about 1 h GPU, a few GB of checkpoints in `outputs/`.
- **Likely blockers:** Windows DataLoader quirks, OOM at higher resolution (use AMP or a smaller size).
- **Handoff / unblocks:** CH-07 and CH-08 become READY. Satisfies the "CH-04 or CH-09" finish dependency of CH-05.
- **Evidence log:**
```text
Owner: Badabuba (worker: Codex) | Claimed: 2026-11-07 17:06 Kyiv | Last update: 2026-11-07 17:43 Kyiv | Branch: ch-04-torchvision-training | Commits: ac92cf2, ec7967b, bfa5206 | PR: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/4
Files changed: movd/tv_train.py, scripts/train_tv.py, tests/test_tv_train.py, configs/runs/smoke_{frcnn,retina}.json, results/smoke_{frcnn,retina}{,_integrated}/, plan.md
Validation commands: python scripts/check_dataset.py; python scripts/make_splits.py --check; python -m pytest -q --basetemp outputs/ch04-pytest-integrated (62 passed); ruff check movd/tv_train.py scripts/train_tv.py tests/test_tv_train.py; python -u scripts/train_tv.py --training-only --config configs/runs/smoke_{frcnn,retina}.json; kill Faster R-CNN after epoch 1 and repeat with --resume; python -u scripts/train_tv.py --config configs/runs/smoke_{frcnn,retina}.json --run-id smoke_{frcnn,retina}_integrated; verify curve counts, finite losses/AP, 500 detection cap, val_full frame count and speed JSON
Metrics / results: Both families completed training-only and integrated 50-step smokes with finite losses. Forced Faster R-CNN restart resumed epoch 2 from step 25 and completed at step 50. Separate four-frame stress check returned 500 detections/frame per family, valid original-pixel boxes and category IDs. Integrated runs wrote two val_fast AP entries and val_full AP for all 2,846 frames, with zero frames over 500 detections. Batch-1 GPU-resident benchmark: 19.39 / 22.07 FPS (Faster R-CNN / RetinaNet). Random-weight smoke AP is pipeline evidence.
Artifacts: results/smoke_{frcnn,retina}/{config,curve,metrics,speed,hardware,runtime,prediction_check}.json; results/smoke_frcnn/smoke_notes.md; results/smoke_{frcnn,retina}_integrated/{config,curve,metrics,speed,hardware,runtime}.json; local ignored outputs/smoke_*/epoch_*.pt
Runtime / hardware: RTX 5070 Laptop GPU 8 GB, Windows, torch 2.14.1+cu130, torchvision 0.29.1+cu130, AMP, batch 2, two spawned workers. Integrated Faster R-CNN: 36.51 s train / 580.02 s total; RetinaNet: 34.94 s train / 675.39 s total (100 images each).
Blockers / deviations: CH-03 merged in PR #3. CUDA RoI Align backward warns about nondeterminism. Revised rubric runtime forecast ~119 / ~116 min, including validation, based on short smokes. Frozen 16,800-image budget, input resize and one-hour target retained. Forecast is approximate and includes startup overhead; no full-budget run was required.
Completion or handoff note: User approved parallel workers under Badabuba: Claude owned CH-03, Codex owned CH-04. CH-03 integration passed. CH-07 and CH-08 start dependency satisfied; CH-05 finish dependencies satisfied, with upload gates remaining.
```

---

### CH-05 — Test inference, submission writer/validator, first Kaggle upload

- **Goal:** Have a proven, validated path from any model to an accepted Kaggle submission early.
- **Why required:**
  - Test inference and submission files do not exist yet.
  - Format errors found late would be fatal.
- **Reuse / preserve:** the label map and policy from CH-02; the COCO-results interchange format from §4.
- **Scope:**
  1. `movd/submission.py`: write rows from COCO-results predictions. Original-pixel `x y w h`, class = `category_id − 1`, `score`, sorted by score, ≤ 500 per image, exactly one row per `test_images.csv` entry (empty predictions allowed), header per G3.
  2. Validator: checks columns, row count, ID set, group arity, class range, score range, finite non-negative boxes inside the image bounds (tolerance documented) and ≤ 500 groups.
  3. CLIs: `scripts/predict_test.py` (torchvision or Ultralytics checkpoint → COCO-results JSON), `scripts/make_submission.py` and `scripts/validate_submission.py`.
  4. Val round-trip: write a val "submission" → parse it back → evaluate. The result must equal direct evaluation.
  5. First upload: a student uploads through the web UI. Log it in `results/submissions.csv` (file sha256, model, val mAP, Kaggle public score, date, uploader).
- **Non-goals:** improving the model, final selection (CH-14).
- **Start deps:** CH-02 · **Finish deps:** CH-03, and CH-04 or CH-09 (a trained checkpoint) · **Machine prerequisites:** M-DATA, M-GPU · **Gates:** G8.
- **Parallel:** yes, with CH-03, CH-04, CH-07, CH-08 and CH-09.
- **Owner / State:** see the board.
- **Files expected to change:**
  - `movd/submission.py`
  - `scripts/{predict_test,make_submission,validate_submission}.py`
  - `tests/test_submission.py`
  - `results/submissions.csv`, `results/ch-05-submission/`
- **Conflict risk:** low. `results/submissions.csv` is append-only afterwards.
- **Implementation outline:** writer → validator → tests (including malformed files) → round-trip → test inference → upload → log.
- **Validation:**
  - `pytest tests/test_submission.py`.
  - `python scripts/validate_submission.py submissions/<file>.csv` exits 0.
  - The round-trip difference is ≤ 1e-6.
- **Evidence & artifacts:** the validator output, the round-trip numbers, the `results/submissions.csv` row, and a screenshot or text note of the Kaggle score. The submission CSV itself stays in the ignored `submissions/` folder.
- **Acceptance criteria:**
  1. The tests pass, including rejection of malformed inputs.
  2. The round-trip matches.
  3. At least one Kaggle upload is accepted and its score is logged.
  4. No submission CSV is committed.
- **Effort:** about 3 h. **Runtime / GPU / disk:** about 20 min GPU for test inference (3,811 images).
- **Likely blockers:** the unknown header (G3), Kaggle access (G8), the daily limit.
- **Handoff / unblocks:** start dependency of CH-14. Milestone M2 is optional.
- **Evidence log:**
```text
Owner: Badabuba (worker: Claude) | Claimed: 2026-11-07 21:30 Kyiv | Resumed: 2026-11-08 17:00 Kyiv | Last update: 2026-11-08 18:10 Kyiv | Branch: ch-05-submission (worktree ../MOVD-ch05) | Commits: 4cc0423 (claim), 9408fdb, b22e484 + 270ede9 (resume), 1082623, 596298a, cafed6c (heartbeat), this completion commit | PR: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/7
Files changed: movd/submission.py; scripts/{predict_test,predict_val,make_submission,validate_submission,submission_roundtrip}.py; tests/test_submission.py; results/submissions.csv (first row); results/ch-05-submission/ (README, round-trip, val check, test predict/submission/validation evidence); plan.md (CH-05, CH-09, CH-14 rows, G2b, G8, this log). No change to movd/data.py, movd/evaluate.py, movd/tv_train.py, homework.ipynb or any CH-07/CH-08/CH-09 file.
Checkpoint: CH-07 selected frcnn_rpn_pre3000 epoch 6, local ignored outputs/frcnn_rpn_pre3000/epoch_0006.pt (in the CH-07 worktree outputs/ch07/), sha256 522f867dea8de585b0d6b72a3f2750a7cd2470ce165225d2f9514f278804eb9f. Verified on CPU: loads, finite weights, checkpoint config == configs/runs/frcnn_rpn_pre3000.json, history epochs 1-6, 16,800 images seen, not smoke; final val_fast 0.19255 = CH-07 evidence.
Validation commands:
  nvidia-smi before each GPU job -> no other compute process
  python scripts/predict_val.py --checkpoint <ckpt> --name frcnn_rpn_pre3000_e6 -> val_full mAP 0.189329 vs CH-07 0.189335; max |diff| 7.9e-5 (AP_large); 164,999 vs 165,000 dets (GPU fp16/CUDA nondeterminism between two passes of one checkpoint)
  python scripts/submission_roundtrip.py --predictions <those val preds> --name frcnn_rpn_pre3000_e6 -> FAIL, max |diff| 4.59e-6 (AP50) with 8/5 decimals; kept as evidence. Probe: 12 box decimals 1.5e-6, 17 digits 0 (matches exactly on an IoU threshold)
  writer switched to lossless repr floats; same round-trip (name frcnn_rpn_pre3000_e6_lossless) -> PASSED, max |diff| 0; synthetic_seed42_lossless -> PASSED, 0
  python scripts/predict_test.py --checkpoint <ckpt> --name frcnn_rpn_pre3000_e6_test -> 48 clips (per-clip shards, resumable), 3,811 frames, 261,940 dets, 373 s on the RTX 5070
  python scripts/make_submission.py ... -> 3,811 rows, 0 empty, 0 rejected, 0 over the cap; validate_submission.py [OK]; independent stdlib check (no movd code) [OK]: same order as test_images.csv, no missing/extra ids, 6 tokens per group, classes 0-7, finite positive boxes inside the image, max 160 dets/image; boxes reach each of the 6 test resolutions' full width/height (original pixels)
  pytest -q tests -> 103 passed, 9 skipped (dataset tests; no dataset in the worktree); pytest tests/test_data.py tests/test_sample_submission.py in the main checkout -> 38 passed; check_dataset.py -> PASSED; make_splits.py --check -> 3/3 hashes match; ruff format --check + ruff check on the 7 CH-05 files -> clean except RUF100 on `# noqa: E402` (repo convention)
Metrics / results: submission frcnn_rpn_pre3000_e6_test.csv, 24,181,776 bytes, sha256 152b8ad1e8c608330bed086772dd8ff35380d41c49b085cbc16be303c0c625b0. Kaggle public score 0.2011538266 (accepted; team Nachalnyky, rank 46 at upload, 1 submission), below the 0.2216324580 public baseline. val_full mAP 0.1893 (CH-07 metrics.json). Logged in results/submissions.csv.
Artifacts: tracked results/ch-05-submission/{roundtrip_*.json, frcnn_rpn_pre3000_e6/val_check.json, frcnn_rpn_pre3000_e6_test/{predict,submission}.json, validation.txt}. Ignored: submissions/*.csv, outputs/ch-05-submission/ (val and test predictions, shards, log). DO_NOT_UPLOAD_* files never uploaded.
Runtime / hardware: RTX 5070 Laptop GPU 8 GB, Windows, AMP, batch 2; val_full regeneration ~5 min (run twice: first attempt crashed when saving because the output folder was missing, fixed), test inference 373 s.
Blockers / deviations: G2b resolved = 5/day (increase request pending, not granted). Upload time not recorded exactly (Kaggle showed "4m ago" on the student's screenshot; logged as shortly before 18:02 Kyiv). Writer format changed from 8/5 decimals to lossless repr (~90 instead of ~52 chars per detection). Added scripts/predict_val.py (not in the original file list) for the real-checkpoint round-trip. configs/protocol.json kaggle.daily_submission_limit still says UNKNOWN (CH-02-owned file; G2b is recorded here and in the gate table).
Earlier CPU phase (2026-11-07, before CH-07 was DONE; kept verbatim):
  Files changed: movd/submission.py; scripts/{predict_test,make_submission,validate_submission,submission_roundtrip}.py; tests/test_submission.py; results/submissions.csv (header only); results/ch-05-submission/{README.md, roundtrip_synthetic_seed42.json, DO_NOT_UPLOAD_synthetic_test/, DO_NOT_UPLOAD_cpu_smoke_frcnn/}. No change to movd/data.py, movd/evaluate.py, movd/tv_train.py, homework.ipynb or any CH-07 file.
  Validation commands (all CPU, CUDA_VISIBLE_DEVICES=""):
    pytest -q tests/test_submission.py -> 49 passed (7 s); pytest -q tests -> 102 passed, 9 skipped (skips = dataset tests, no dataset in the worktree); pytest tests/test_data.py tests/test_sample_submission.py in the main checkout with the dataset -> 38 passed
    python scripts/check_dataset.py --root <dataset> -> PASSED (32 OK); python scripts/make_splits.py --check -> 3/3 split hashes match
    ruff format --check + ruff check on the 6 new files -> clean except RUF100 on `# noqa: E402` (same convention as existing scripts)
    python scripts/submission_roundtrip.py --synthetic --name synthetic_seed42 -> PASSED, max |diff| = 0 on all 6 headline metrics (val_full, 176,140 synthetic detections, 2 min 20 s)
    python scripts/make_submission.py outputs/ch-05-submission/DO_NOT_UPLOAD_synthetic_test/test_predictions.json -> 3,811 rows, 241,353 dets, 77 empty rows, 560 zero-area rejected, 5,784 over the cap; validate_submission.py [OK] sha256 0cf65fc6…; independent pandas check (no movd code) OK
    python scripts/validate_submission.py <dataset>/sample_submission.csv -> [OK] sha256 aee0b5dc… (matches CH-02)
    python scripts/predict_test.py --checkpoint outputs/smoke_frcnn_integrated/epoch_0002.pt --device cpu --clips clip_001 clip_011 --max-images 2 --allow-smoke -> 4 frames, 599 dets, about 5 s per clip on CPU; refused without --allow-smoke; --resume kept both shards; boxes reach x=1710 (1920 px frame) and x=680 (680 px frame), so they are in original pixels; make_submission + validate [OK]
  Metrics / results: no model metric (no trained checkpoint used). Round-trip difference 0 at score 8 / box 5 decimals; it was 2e-4 / 3.4e-6 / 1.4e-5 at 2 / 3 / 4 box decimals (borderline IoU flips), hence the chosen precision.
  Artifacts: results/ch-05-submission/ (small JSON, tracked). Ignored: submissions/DO_NOT_UPLOAD_*.csv, outputs/ch-05-submission/DO_NOT_UPLOAD_*/ (synthetic/smoke, never upload).
  Runtime / hardware: CPU only on Nazar's laptop while CH-07 trained on the RTX 5070; the GPU was never used by CH-05.
  Blockers / deviations: acceptance 3 (accepted Kaggle upload + logged score) open. It needs (a) a validated trained checkpoint explicitly named by CH-07 evidence, (b) a free GPU for 3,811-frame inference (plan estimate about 20 min), (c) G2b (daily limit, still UNKNOWN, not guessed) and G8 (student web upload). The val round-trip was run with synthetic predictions; rerun it on the chosen checkpoint's val predictions before upload. Deviation: predict_test.py has its own 15-line predict loop with a device argument (tv_train.predict_coco is CUDA-only); a test asserts identical output to predict_coco.
  Coordination 2026-11-07 23:46 Kyiv: paused waiting on CH-07 DONE, a free GPU and the student upload. Nazar approved Claude doing CPU-only CH-09 prep meanwhile (recorded in both rows); Codex exclusively owns the GPU until CH-07 is DONE; Claude must not edit CH-07 files, checkpoints, results or notebook section. Order after CH-07: CH-05 inference first, then CH-09 GPU work, never concurrently.
  Completion or handoff note: next steps after CH-07 is DONE: predict_test.py on its selected checkpoint -> make_submission.py -> validate_submission.py -> student checks G2b and uploads via the web UI -> append a row to results/submissions.csv -> PR ch-05-submission -> develop, mark DONE.
Completion or handoff note: acceptance 1-4 pass. CH-14 now waits only on CH-12. CH-09 remains PARTIAL / PAUSED (CPU prep at 64f08e6); resuming it needs an explicit claim. CH-08 (Codex) untouched.
```

---

### CH-06 — Corrected EDA & preprocessing evidence (notebook)

- **Goal:** Complete and correct Stage 1 so it covers every data-understanding rubric item and justifies the preprocessing decisions.
- **Why required:**
  - Ignored annotations and regions are under-analyzed.
  - Sequence differences, resolution effects and the reason for subsampling are missing.
- **Reuse / preserve:** the existing §3–§4 cells and outputs (class distribution, sizes, density, GT visualization, split check, resize and augmentation demos). Extend them rather than rewrite them.
- **Scope:** new or extended notebook cells covering:
  1. Class distribution including ignored (crowd) annotations, per split.
  2. Size distribution at the original resolution and after the candidate training resizes, including the share of objects under 8 or 16 px.
  3. Objects per frame including ignored objects.
  4. A per-sequence table: resolution, mean and max density, class mix, share of ignored.
  5. Ignore-region visualization.
  6. Frame-to-frame redundancy (for example GT overlap between consecutive frames), giving the subsampling rationale.
  7. A written summary of the preprocessing decisions, linked to the CH-02 policy (or marked provisional if CH-02 is not DONE).
  8. Figures exported to `results/ch-06-eda/figures/`.
- **Non-goals:** changing the shared modules or §4, training.
- **Start deps:** — · **Finish deps:** — · **Soft (not a dependency):** CH-02. Use `movd/data.py` if it is merged, otherwise notebook-local code · **Machine prerequisites:** M-DATA, 🔒 notebook lock · **Gates:** —.
- **Parallel:** yes, with any non-notebook chunk.
- **Owner / State:** see the board.
- **Files expected to change:** `homework.ipynb` (under the lock), `results/ch-06-eda/` (tables CSV, figures).
- **Conflict risk:** high for the notebook, which the lock prevents. Release the lock between sessions.
- **Implementation outline:** take the lock → add cells after the existing §3/§4 subsections → run only the new cells → export figures and CSVs → release the lock.
- **Validation:** the new cells run top-to-bottom from the notebook start without errors; the figures and CSVs exist.
- **Evidence & artifacts:** `results/ch-06-eda/{tables,figures}`.
- **Acceptance criteria:**
  1. Every Stage-1 rubric row in §3 has a cell and figure or table.
  2. The legacy outputs are intact.
  3. The written decisions are present.
  4. The lock is released.
- **Effort:** about 3 h. **Runtime / GPU / disk:** CPU, a few minutes.
- **Likely blockers:** memory when loading the 280 MB train JSON (fine on both machines); notebook lock contention.
- **Handoff / unblocks:** start dependency of CH-16. Its evidence feeds the report.
- **Evidence log:**
```text
Owner: Badabuba (worker: Claude) | Claimed: 2026-11-07 19:00 Kyiv | Last update: 2026-11-07 19:20 Kyiv | Branch: ch-06-eda | Commits: 4be64d9 (work), 4b46ac3 (merge origin/develop), c0c0246 (completion) | PR: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/5
Files changed: homework.ipynb (+30 cells: 13 code, 17 markdown; all 58 legacy cells unchanged), results/ch-06-eda/ (10 figures, 17 CSV tables, key_numbers.json, run_stage1.py), plan.md (own row, CH-16 recalculation, lock, this log)
Validation commands:
  python scripts/check_dataset.py                      -> PASSED
  python -m pytest -q tests                            -> 62 passed (plain `pytest` from the repo root also collects the ignored outputs/ch07-claim worktree copy and fails on duplicate module names; not a CH-06 issue)
  python scripts/make_splits.py --check                -> OK (val_fast, val_full, train_rubric hashes match)
  python results/ch-06-eda/run_stage1.py               -> Stage 1: 52 cells (26 code) executed without errors in 45 s (clean kernel, .venv)
  notebook diff check (scripted): 58/58 legacy cells keep identical source; the re-executed legacy Stage 1 cells reproduce Marko's text outputs exactly; the 24 legacy code cells of sections 5-6 are byte-identical (not re-executed: training never runs in the notebook); every number in the CH-06 markdown occurs in a cell output, a CH-06 table or configs/protocol.json
Metrics / results (train / val; key_numbers.json):
  classes (targets only): car 45.71 %, person 29.73 %, bus 0.87 % (9,653) in train; imbalance 52.3:1 train, 190.6:1 val (264 buses). Legacy cell 3.2 added 63,843 / 3,995 crowd copies to every class (legacy bus 73,496 = 4.55 %)
  sizes sqrt(w*h): COCO small 24.91 % / 19.20 %, < 16 px 2.16 % / 0.33 %; after the frozen 1344 resize: < 32 px 50.21 % / 44.34 %, < 16 px 8.26 % / 7.92 % (legacy 896: 77.24 % / 75.11 % and 28.72 % / 22.98 %)
  occlusion partial 40.32 % / 52.41 %, heavy 5.33 % / 6.64 %; truncated 4.29 % / 5.33 %
  density: targets/frame mean 45.68 / 40.10, max 180 / 116; frames > 100 targets 7.65 % / 6.43 %; no frame > 500
  ignore: 63,843 / 3,995 regions (8 COCO copies each, 100 % matched to ignore/*.txt), in 74.75 % / 50.39 % of frames, mean 3.23 % / 3.21 % of those frames
  resolutions: long side 2688-3840 px in 16.4 % train / 54.8 % val / 0 % test frames; test has 8 clips (600 frames) at 680x382 / 960x540, smaller than any train frame (upsampled x1.976 / x1.4 by the frozen resize)
  redundancy (same track ids): 1 frame apart 99.5 % shared (median IoU 0.897 / 0.873); 8 apart 96.6 % / 96.5 % (IoU 0.440 / 0.348). train_rubric = 11.6 % of train frames (median gap 8) and holds 98.6 % of the 7,342 train tracks
Artifacts: results/ch-06-eda/{figures,tables}, results/ch-06-eda/key_numbers.json, results/ch-06-eda/run_stage1.py
Runtime / hardware: CPU only, Stage 1 runs in about 45 s (Windows 11, Nazar's laptop, .venv Python 3.12); no GPU, no training
Blockers / deviations: (1) the notebook is executed top to bottom for Stage 1 only (cells before "## 5. Faster R-CNN"); sections 5-6 are the legacy v0 training runs and keep Marko's stored outputs, marked with a banner, because training never runs in the notebook (CLAUDE.md section 7) and the chunk forbids training. (2) legacy heading numbering (1, 3, 4, ...) left as is. (3) the 4.5 summary is a draft from the computed tables; the students write the report prose (G10). (4) per-sequence "ignore area %" is the summed ignore-box area / frame area, as in CH-02 (overlaps counted twice). (5) redundancy IoU is in image coordinates, so it includes camera motion
Completion or handoff note: all 4 acceptance criteria met (every Stage-1 rubric row of section 3 has a cell plus a figure or table; legacy outputs intact; written decisions in notebook section 4.5; lock released). Unblock recalculation: CH-16 is still blocked by CH-12, CH-14, CH-15. Manual review: Nazar and Marko should read 4.5 and the legacy notes before the defence
```

---

### CH-07 — Two-stage: Faster R-CNN corrected + tuning + failure cases

- **Goal:** Produce a corrected two-stage baseline, controlled tuning, speed and a written failure analysis.
- **Why required:** these are rubric items. The legacy results used a flawed protocol and have no speed or failure analysis.
- **Reuse / preserve:**
  - CH-04's trainer.
  - The legacy settings as hypotheses: RPN 3000/1000/2000/1000 helped AP_small, while RoI 256/0.5 hurt.
  - The legacy numbers are reported as "v0", not overwritten.
- **Scope:**
  1. Baseline under the rubric budget.
  2. At least 2 controlled tuning runs, one variable each: for example input resolution, RPN proposals, anchor sizes.
  3. mAP, AP50, AP_small and FPS on `val_full`.
  4. At least 3 failure cases (for example dense crowds, tiny objects, class confusion van/car, ignored-region effects), each with a figure and a written cause.
  5. A note comparing v0 to the corrected protocol.
  6. A short notebook section that loads `results/` (🔒 short).
- **Non-goals:** RetinaNet, new loss experiments (CH-08 owns the designated loss experiment).
- **Start deps:** CH-04 · **Finish deps:** — · **Machine prerequisites:** M-DATA, M-GPU, 🔒 short · **Gates:** —.
- **Parallel:** yes, with CH-08, CH-10 and CH-11 (run on another GPU).
- **Owner / State:** see the board.
- **Files expected to change:** `configs/runs/frcnn_*.json`, `results/frcnn_*/`, `results/ch-07-frcnn/figures/`, `homework.ipynb` (short, under the lock).
- **Conflict risk:** low, apart from the notebook (lock).
- **Implementation outline:** configs → baseline → tuning runs (resumable) → failure-case figures → analysis → notebook section.
- **Validation:** every run has its full `results/<run_id>/` set; the tables regenerate from the files.
- **Evidence & artifacts:** the run directories, the comparison CSV and the failure figures.
- **Acceptance criteria:**
  1. Baseline plus 2 or more tuned runs exist under an identical budget.
  2. mAP, AP50, AP_small and FPS are reported.
  3. At least 3 failure cases are analyzed in writing.
  4. Negative results are kept.
  5. The notebook section exists.
- **Effort:** about 3 h. **Runtime / GPU / disk:** about 2.5 h GPU, a few GB of checkpoints.
- **Likely blockers:** OOM at higher resolution on 6 GB; time overrun (cap at 3 runs).
- **Handoff / unblocks:** start dependency of CH-12.
- **Evidence log:**
```text
Owner: Badabuba (worker: Codex) | Claimed: 2026-11-07 21:07 Kyiv | Last update: 2026-11-08 00:33 Kyiv | Branch: ch-07-frcnn | Commits: ec9f3dd, ca9dd33, 652c6a5, 2301262 | PR: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/6
Files changed: configs/runs/frcnn_{baseline,rpn_pre3000,roi_pos50}.json; results/frcnn_*; results/ch-07-frcnn; homework.ipynb (two appended cells); plan.md
Validation commands: python scripts/check_dataset.py; python scripts/make_splits.py --check; python -m pytest -q --basetemp outputs/ch07-pytest-final (62 passed); ruff check results/ch-07-frcnn/make_evidence.py movd/tv_train.py scripts/train_tv.py tests/test_tv_train.py; python scripts/train_tv.py --config outputs/<run>_smoke.json --training-only (three); python scripts/train_tv.py --config configs/runs/<run>.json (three); python results/ch-07-frcnn/make_evidence.py and --tables-only (byte-identical CSV); nbformat validation and clean-kernel nbclient execution of only the two new cells, all 88 prior cells and outputs preserved; git diff --check
Metrics / results: Baseline / RPN pre3000 / RoI pos50: final val_fast mAP 0.18959 / 0.19255 / 0.18327; val_full mAP 0.18733 / 0.18934 / 0.18136, AP50 0.38078 / 0.38467 / 0.36430, AP_small 0.09126 / 0.10829 / 0.10035; 18.69 / 14.85 / 17.48 FPS. Selected RPN by final val_fast mAP. All runs saw 16,800 images over six epochs; every val_full result uses 2,846 frames and global maxDets 500.
Artifacts: results/ch-07-frcnn/{protocol.md,comparison.csv,failure_analysis.md,failure_cases.csv,figures/,make_evidence.py}; results/frcnn_{baseline,rpn_pre3000,roi_pos50}{,_smoke}/{config,curve,metrics,hardware,runtime,speed}.json; local ignored outputs/<run>/epoch_*.pt. Config and weight SHA-256 recorded in protocol.md.
Runtime / hardware: RTX 5070 Laptop GPU 8 GB, Windows, torch 2.14.1+cu130, torchvision 0.29.1+cu130, AMP, batch two, two workers. Full runs 3663.15 / 3969.77 / 3868.59 s wall (191.69 min combined, 151.32 min training); smokes 88.83 s wall; one additional val_fast prediction pass for figure generation.
Blockers / deviations: Short-smoke forecast was about six hours; observed full-run wall was about 3 h 12 min. One seed, CUDA RoI Align backward warns about nondeterminism. Whole-repo Ruff reported 16 existing findings outside CH-07 before integration of CH-06; scoped Ruff passed. RoI negative overall AP and RPN large-object/speed regressions retained. Legacy v0 kept separately.
Completion or handoff note: User approved parallel CH-06 (Claude CPU/EDA/notebook) and CH-07 (Codex GPU/Faster R-CNN) work under Badabuba. Acceptance 5 was deferred until CH-06 merged and released the lock; CH-07 then claimed, appended and validated its own section and releases the lock with this PR. CH-12 still needs CH-08, CH-10 and CH-11. GPU training is finished.
```

---

### CH-08 — One-stage anchor-based: RetinaNet + anchors + focal-loss experiment

- **Goal:** Produce a corrected RetinaNet baseline, an anchor experiment, and **the project's designated loss experiment**, all compared with Faster R-CNN.
- **Why required:** rubric items, plus the "defensible loss-related experiment". The legacy γ = 1.5 result is a single, noisy, flawed-protocol run.
- **Reuse / preserve:** CH-04's trainer; legacy `create_retinanet`, `focal_loss_gamma_15` and the anchor code. Legacy numbers are reported as v0.
- **Scope:**
  1. Baseline under the rubric budget.
  2. Data-driven anchors, chosen from the CH-06/CH-02 size statistics.
  3. Focal-loss experiment: change one variable (γ or α), state the hypothesis in advance, and keep everything else fixed.
  4. A comparison table with CH-07 at equal budget, including FPS.
  5. A short notebook section (🔒 short).
- **Non-goals:** a full hyperparameter sweep, other losses.
- **Start deps:** CH-04 · **Finish deps:** — · **Machine prerequisites:** M-DATA, M-GPU, 🔒 short · **Gates:** —.
- **Parallel:** yes, with CH-07, CH-10 and CH-11.
- **Owner / State:** see the board.
- **Files expected to change:** `configs/runs/retina_*.json`, `results/retina_*/`, `results/ch-08-retina/figures/`, `homework.ipynb` (short, under the lock).
- **Conflict risk:** low, apart from the notebook (lock).
- **Implementation outline:** configs → baseline → anchors run → focal run → comparison → notebook.
- **Validation:** the complete `results/` sets; the comparison CSV regenerates.
- **Evidence & artifacts:** the run directories, `results/ch-08-retina/comparison.csv` and figures.
- **Acceptance criteria:**
  1. Three runs (baseline, anchors, focal) exist under an identical budget.
  2. The loss experiment's hypothesis and result are written.
  3. The comparison with CH-07, with FPS, is present.
  4. Negative results are kept.
- **Effort:** about 3 h. **Runtime / GPU / disk:** about 2.5 h GPU.
- **Likely blockers:** OOM on 6 GB at higher resolution.
- **Handoff / unblocks:** start dependency of CH-12.
- **Evidence log:**
```text
Owner: Badabuba (worker: Codex) | Claimed: 2026-11-08 16:59 Kyiv | Last update: 2026-11-08 23:45 Kyiv | Branch: ch-08-retina | Commits: 2306005, 3d70633, 21041a5 | PR: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/11
Files changed: configs/runs/retina_{baseline,anchors,focal_gamma15}.json; tests/test_retina_preflight.py; results/retina_*; results/ch-08-retina; homework.ipynb (two cells); plan.md
Validation commands: python scripts/check_dataset.py; python scripts/make_splits.py --check; python -m pytest -q --basetemp outputs/ch08-execution-tests (116 passed, one pre-existing collection warning); scoped Ruff; official weights CPU strict reload; python scripts/train_tv.py --config outputs/<run>_smoke.json --training-only then --resume (three forced interruptions); python scripts/train_tv.py --config configs/runs/<run>.json (three); python results/ch-08-retina/make_comparison.py (byte-identical regeneration); nbclient clean-kernel and nbformat; git diff --check
Metrics / results: Baseline / anchors / focal gamma 1.5: final val_fast mAP 0.16434 / 0.17053 / 0.15554; val_full mAP 0.16233 / 0.16925 / 0.15242; AP_small 0.07801 / 0.07541 / 0.08319; 12.67 / 19.71 / 24.22 FPS. Selected anchors. CH-07 selected Faster R-CNN: 0.18934 mAP and 14.85 FPS. All full runs 16,800 images with 2,846-frame final evaluation and maxDets 500.
Artifacts: results/ch-08-retina/{preflight.md,results.md,anchor_stats.json,comparison.csv,per_class.csv,density.csv,figures/,make_comparison.py}; results/retina_{baseline,anchors,focal_gamma15}{,_smoke}/; ignored outputs/<run>/epoch_*.pt
Runtime / hardware: RTX 5070 Laptop GPU, Windows, torch 2.14.1+cu130, torchvision 0.29.1+cu130, AMP, batch 2, two workers. Full runs 3996.66 / 3464.18 / 2641.30 s wall; 168.37 min combined.
Blockers / deviations: None at completion. One seed, speed includes model processing on GPU-resident inputs. Negative focal overall AP and anchor AP_small results retained. Three smokes used two 25-step epochs each for forced restart verification.
Completion or handoff note: GPU free after CH-08; CH-09 current ownership and review state preserved. CH-12 still needs CH-10 and CH-11. Notebook lock released with this PR.
```

---

### CH-09 — Ultralytics export + YOLO11 anchor-free baseline

- **Goal:** Build the Ultralytics data path and the evaluator adapter, and train the YOLO11 anchor-free baseline with a learning curve.
- **Why required:**
  - The anchor-free family is a rubric item.
  - The same pipeline serves RT-DETR (CH-11) and is the likely final-model path.
- **Reuse / preserve:**
  - The shipped `images/`+`labels/` layout and `data.yaml` serve as a format reference only. The shipped files alone do not handle ignore regions.
  - The CH-02 splits and policy, and the CH-03 evaluator.
- **Scope:**
  1. Pin `ultralytics==<version>` in requirements. Cite it and add an AGPL-3.0 note to the README.
  2. `movd/ultra.py` export: build YOLO-format train data for `train_rubric` from the COCO JSON, applying the **ignore-region policy frozen by CH-02**, into `outputs/yolo_data/` (ignored). Val images stay untouched.
  3. Adapter: Ultralytics predict (`conf=0.001`, `max_det=500`, original-pixel boxes) → COCO-results JSON → CH-03 evaluator.
  4. A smoke test to pick the model size (for example `yolo11s` or `yolo11m`) and the batch size for 8 GB / 6 GB.
  5. A baseline under the rubric budget with per-epoch checkpoints (`save_period=1`).
  6. A `val_fast` mAP-per-epoch curve computed with **our** evaluator.
- **Non-goals:** tuning (CH-10), RT-DETR (CH-11).
- **Start deps:** CH-02 · **Finish deps:** CH-03 · **Machine prerequisites:** M-DATA, M-GPU · **Gates:** —.
- **Parallel:** yes, with CH-03, CH-04 and CH-05.
- **Owner / State:** see the board.
- **Files expected to change:**
  - `movd/ultra.py`, `scripts/export_yolo.py`, `scripts/train_ultra.py`, `scripts/predict_ultra.py`
  - `configs/runs/yolo_*.json`, `results/yolo_*/`
  - `requirements.txt` (own lines), `README.md` (licence/citation lines)
- **Conflict risk:** low.
- **Implementation outline:** pin → export (with a visual check of the policy on 2 frames) → adapter + round-trip check against GT → smoke → baseline → curve.
- **Validation:**
  - Exported label counts match the COCO targets for the subset.
  - The adapter on GT-derived boxes gives mAP ≥ 0.99.
  - The baseline `results/` set is complete.
- **Evidence & artifacts:** `results/yolo_baseline/` (config, curve, metrics, speed) and the export check figures in `results/ch-09-yolo/figures/`.
- **Acceptance criteria:**
  1. The version is pinned and cited.
  2. The export applies the frozen policy and is checked.
  3. The adapter is verified.
  4. A baseline with a per-epoch curve from our evaluator exists.
  5. No exported data is committed.
- **Effort:** about 3 h. **Runtime / GPU / disk:** about 1.5 h GPU, up to about 3 GB of export.
- **Likely blockers:** Ultralytics/torch compatibility on the 5070; Windows dataloader workers (set `workers` low).
- **Handoff / unblocks:** CH-10 and CH-11 become READY. Satisfies the "CH-04 or CH-09" finish dependency of CH-05.
- **Evidence log:**
```text
Owner: Marko-Zenon (GPU execution; handed off by Badabuba/Claude) | Claimed originally: 2026-11-07 23:46 Kyiv by Badabuba | Handoff/resume: 2026-11-08 21:21 Kyiv | Last update: 2026-11-08 23:15 Kyiv | Branch: ch-09-yolo | Commits preserved: 55196b9, 64f08e6, 8e4888192057bdbff490d0382c7f935eb50fe73f | PR: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/10 (draft; transfer/acceptance pending)
Files changed: movd/ultra.py; scripts/{export_yolo,check_ultra_adapter,train_ultra,predict_ultra}.py; tests/test_ultra.py; configs/runs/{yolo_baseline,yolo_smoke_s,yolo_smoke_m,DO_NOT_USE_cpu_pipeline_check}.json; requirements.txt (ultralytics==8.4.207); README.md (new CH-09 section only: AGPL-3.0, weights sha256, citation, commands); results/ch-09-yolo/; results/DO_NOT_USE_cpu_pipeline_check/. No change to movd/data.py, movd/evaluate.py, homework.ipynb, CH-05 or CH-07 files.
Validation commands (all CPU, CUDA_VISIBLE_DEVICES=-1; note: an empty value does not hide the GPU on Windows):
  uv pip install ultralytics==8.4.207 into the shared .venv: dry-run first, purely additive (6 new packages, no existing version changed); albumentations not installed, so no hidden Ultralytics augmentation
  pytest -q tests/test_ultra.py -> 20 passed; pytest -q tests -> 72 passed, 10 skipped (dataset tests, no dataset in the worktree); ruff check (RUF100 on `# noqa: E402` kept, repo convention) + ruff format --check clean
  python scripts/export_yolo.py --dataset-root <dataset> -> train_rubric 2,800 images, 137,756 labels = 137,756 COCO targets (0 degenerate, 0 duplicates), classes 0-7 all present, max label box error 0.002 px, 2,161 masked / 639 byte-copied frames, 7,680 ignore regions; pixel check on 40 masked frames: fill MAE <= 1.14, kept-target MAE <= 1.45 (JPEG q95); val_fast 714 byte copies, 28,685 labels; 90 s
  python scripts/check_ultra_adapter.py --model-preflight outputs/weights/yolo11s.pt -> GT -> YOLO lines -> Ultralytics Results -> adapter -> 500 cap -> movd.evaluate on val_full: mAP 1.0000 (114,132 dets); CPU preflight of COCO-pretrained yolo11s at imgsz 1344 on 3 val frames: boxes in original pixels, ids 1-8, evaluator accepts (80-class model filtered to classes 0-7 for this format check only; unfiltered, the adapter correctly rejects class 38)
  python scripts/train_ultra.py --config configs/runs/DO_NOT_USE_cpu_pipeline_check.json --device cpu (8 images, imgsz 320) + predict_ultra.py curve -> epoch0/epoch1 checkpoints, resumable curve, 0 corrupt labels, label cache only under outputs/yolo_data (dataset folder untouched). Found: nbs 64 accumulation -> batch 4 gives about 44 optimizer steps/epoch (recorded in yolo_baseline.json)
Metrics / results: no model metric yet (no GPU training).
Artifacts: results/ch-09-yolo/{export_check.json, adapter_check.json, figures/export_check.png, README.md}; results/DO_NOT_USE_cpu_pipeline_check/ (plumbing only). Ignored: outputs/yolo_data/ (1.92 GB incl. Ultralytics label caches), outputs/weights/yolo11{s,m}.pt (sha256 85a76fe8…, d5ffc1a6…).
Runtime / hardware: CPU only on Nazar's laptop (Ryzen 9 8940HX) while CH-07/Codex used the RTX 5070 (85-98 %, 7.7 GB). No CUDA context was created by CH-09.
Blockers / deviations: acceptance 4 (baseline + per-epoch curve) waits on CH-07 DONE + idle GPU, then CH-05 test inference first. Deviation: masked frames are re-encoded as JPEG q95 (torchvision masks in memory); fill error <= 1.14/255 measured. Commit 64f08e6 message says 21 tests; the correct count is 20.
Handoff authorization: Marko-Zenon explicitly reports Badabuba's direct authorization to transfer CH-09 GPU execution to Marko on his separate Ubuntu RTX 3050 6 GB, while Badabuba runs CH-08 on his RTX 5070. Recorded paused→IN_PROGRESS handoff follows CLAUDE sections 3/4 and plan section 9.1; no takeover inferred from inactivity. Push of this coordination commit to develop is the lock before installation/implementation. No CH-08/10/11 state or notebook lock changes. Current session stops before full baseline training.
Laptop GPU execution update (2026-11-08 21:37 Kyiv, Marko-Zenon): handoff lock 7e3b440 successfully pushed to origin/develop before installation/training; task commit 2b1a4233c2f9ae71eae5f20eac37375dd5814b15 successfully pushed to origin/ch-09-yolo, preserving 8e48881 and integrating develop. State PARTIAL / PAUSED for review; no completion PR or DONE. Earlier CPU-only metrics/blocker statements above describe the historical Nazar session, not this laptop run.
Validation: check_dataset.py PASS (train 24,201/56, val 2,846/7, test 3,811/48); CUDA + CUDA NMS PASS, RTX 3050 Laptop 6 GB, torch 2.14.1+cu130 / torchvision 0.29.1+cu130 / CUDA 13.0 unchanged. Added only 13 missing pinned packages including ultralytics 8.4.207 and pytest 9.2.0; zero installed versions replaced. pip check has the same pre-existing unrelated inherited environment errors, no new conflicts. YOLO export PASS (2,800/137,756 targets train; 714/28,685 val_fast; class IDs 0-7, max coordinate error 0.00288 px, 2,161 masked frames); two-frame visual check PASS. Original shared export evidence/configs preserved; new evidence in results/ch-09-yolo/marko_export/.
Smoke config configs/runs/yolo_smoke_marko_3050.json: official YOLO11s, imgsz1344, batch1, AMP true, workers0, cachefalse, seed42, frozen flip-only; fraction0.018 = 50 first sorted frames, two epochs. Exactly 50 initial + 50 resumed batches (100 total, not optimizer steps); finite scalar loss ranges 4.321056-6.448348 and 4.053787-6.448348. Initial epoch 6.65s (~7.52 batches/s); CLI times 17.978s and 77.083s including initialization/final automatic Ultralytics validation. Internal validation metrics are not project headline metrics.
Recovery PASS: graceful stop after epoch0 on_model_save; preserved resumable last.pt copy (94,823,715 bytes, SHA256 d92352123faf120db632f5b35d768f141948daadacfc041fa149452b23d13db1), epoch0 with 255 optimizer entries and scaler state. Native --resume started epoch1 with 255 optimizer state entries, trained 50 batches, wrote epoch1.pt and completed. This tests epoch-boundary recovery, not mid-batch/power-loss recovery. Five checkpoints retained under ignored outputs/yolo_smoke_marko_3050/train/weights/.
Resources: peak CUDA allocated 1,376,612,352 bytes / reserved 1,574,961,152; sampled total device 1,799 MiB, process RSS 2,997,063,680 bytes, minimum available system RAM 1,929,363,456. No OOM or 768 MiB RAM guard activation. Nearly-full 4 GiB swap remains a risk; first 50 frames from one clip do not verify worst-case dense-frame memory safety. Post-run free disk 81,412,046,848 bytes. Unit/integration tests test_ultra.py 39 PASS; full CPU suite 149 PASS / 2 skipped (CUDA hidden, optional sample absent). Ruff --ignore RUF100 and format check PASS; git diff --check and staged-scope checks PASS; protected site/notebook/protocol/shared smoke and baseline configs unchanged.
Artifacts/transfer: results/yolo_smoke_marko_3050/ contains config/hardware/runtime/resume source/resources/tests/dataset/dependency evidence and SHA256 manifests (52 files, 345,539,113 bytes, all hashes locally verified); full logs and resource samples in ignored outputs/ch09_marko_setup_20261108/. Official pretrained SHA256 85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5. After an authorized destination is supplied, transfer every manifested file via SSH rsync or agreed private cloud into a fresh directory and verify sha256sum -c on Badabuba's machine. No external transfer initiated; no weights/exports/raw predictions committed.
Next action: student reviews memory/results before explicit owner resume and laptop baseline batch/worker selection. Shared yolo_baseline.json remains six epochs at 1344, batch4/workers2; it was not changed or launched. A recorded laptop batch1/workers0/cachefalse config is recommended, keeping the full 2,800-frame six-epoch budget. Baseline, project-evaluator curve/val_full, speed/failure evidence and notebook integration remain pending. No CH-08/10/11 state changes, no notebook or protected-file edits, no full baseline.
Full-baseline resume claim (2026-11-08 21:50 Kyiv): recorded owner Marko-Zenon explicitly reviewed the successful smoke and authorizes full baseline/evaluations on his separate RTX 3050. PARTIAL / PAUSED -> IN_PROGRESS through this pushed coordination update, before implementation/training. Existing handoff 7e3b440, CPU 8e48881 and smoke d84bf00 preserved. No DONE, PR merge, Kaggle upload or external artifact transfer authorized by this claim. Pre-claim RAM about 6.0 GiB available, no sampled swap-in/out; disk 76 GiB free; GPU prerequisites rechecked before launch.
Full-baseline completion evidence (2026-11-08 22:57 Kyiv, Marko-Zenon; paused for review, NOT DONE): resume claim a20a01b was successfully pushed and verified before implementation/training. Training source 80781c4427ef3fd56a71a5fa0bb22956a9635418; full evidence 4c8e41bf9bb838044ed9245d4f5b690090fd542a and CSV-normalization follow-up 2d3542e02d61eb88fd3ca995f405395fcf9cd26e successfully pushed to origin/ch-09-yolo. CPU 8e48881, original smoke d84bf00 and all previous checkpoints/results retained. Earlier pending-baseline statements above are historical and superseded by this measured run.
Exact config: configs/runs/yolo_baseline_marko_3050.json; official YOLO11s; full 2,800 train_rubric images x 6 epochs = 16,800 images and 16,800 batch1 post-backward batches; imgsz1344, AMP true, workers0, cachefalse, seed42, frozen flip-only. Unique run_id and approved batch4->1/workers2->0/cachefalse differences from the unchanged shared baseline. Native optimizer=auto selected AdamW lr0.000833; nbs64 accumulation retained, batch count is not optimizer-update count. No split/resolution/image-budget deviation; no full-run interruption or extra training.
Training PASS: exactly six completed epochs and epoch0.pt through epoch5.pt. Mean box/cls/DFL per epoch: [1.60578/1.18525/1.16181; 1.48538/0.93850/1.11623; 1.40437/0.83341/1.08214; 1.33393/0.75662/1.05468; 1.25737/0.68628/1.02389; 1.17565/0.62598/0.99509]. All 16,800 scalar batch losses finite, range1.53873467-12.23427010. Each saved epoch loads on CPU, expected epoch, 255 optimizer entries and scaler state (scale512); last/best are completed stripped checkpoints. Prior smoke already verified real native epoch-boundary resume. Full-run checkpoint sizes/paths/hashes are in validation.json and artifact_manifest.json under results/yolo_baseline_marko_3050/.
Project evaluator PASS: six val_fast mAP points [0.17641914,0.18454328,0.15629284,0.16096419,0.14795409,0.14405777] (curve.json stores exact values; no smoothing/selection). Final last.pt evaluated on all 2,846 original val_full images: mAP0.1438917717, AP50 0.2925317447, AP75 0.1192264113, AP_small0.0865789889, AP_medium0.1434272936, AP_large0.2554700319. 635,095 predictions; zero out-of-split/over-cap images; conf0.001/NMS IoU0.7/maxDets500, original-pixel boxes and frozen ignore policy. Eight per-class APs and four density bins stored in metrics.json/per_class.csv. Ultralytics automatic final validation is not used for headline numbers.
Speed/failure/comparison PASS: FP32, batch1, first30 val_full frames preloaded as BGR,10 warmups, synchronized benchmark: 36.37445682 FPS /27.49181947 ms/image, letterbox/normalization/NMS/rescaling included, disk I/O excluded. Three failure cases/figures generated and visually checked under CH-07 score>=0.10 and same-class IoU>=0.50 rule. frcnn_comparison.csv cites original stored project metrics for three FRCNN runs plus YOLO: YOLO final mAP0.14389177 < FRCNN baseline0.18732811 and RPN0.18933547. FRCNN speed is RTX5070/GPU-resident inputs; YOLO speed RTX3050/CPU BGR preprocessing+transfer, so no architecture-only speedup claim or final-family decision.
Resources: training CLI2002.990s (33.38min), monitored train2007.217s, curve718.295s, final575.680s. Training peak CUDA allocated1,454,161,408/reserved1,625,292,800 bytes; sampled total GPU1849MiB; training RSS2,981,535,744 bytes, minimum available RAM3,520,086,016 bytes. Final evaluation peak RSS3,161,444,352 bytes. No OOM, guard stop, material mitigation or dependency installation. Heavy pre-existing swap use increased during evaluation; settings unchanged. All full samples/logs preserved in ignored outputs/ch09_marko_baseline_20261108/.
Validation PASS: dataset/CUDA/NMS/official pretrained SHA256/export recheck; CH-09 unit/integration39 passed prelaunch; final CPU suite149 passed/2 skipped; separate CUDA benchmark integration1 passed. Evidence/checkpoint audit PASS; scoped CH-09 Ruff/format10 files PASS with existing RUF100 convention ignored; final aggregate git diff --check PASS after CSV line endings normalized without changing parsed data. Broader repo lint is NOT clean: pre-existing UP035 movd/data.py:37 and PLW1510 tests/test_data.py:177; six unchanged files need formatting (lint_summary.json). Confirmed all flagged files byte-unchanged from d84bf00; unrelated fixes not made. Protected site/notebook/shared configs/protocol/other experiments untouched; no Kaggle upload or completion PR.
Artifacts/transfer: results/yolo_baseline_marko_3050/ contains exact config/hardware/runtime, all six curve points, final project metrics/per-class/density, speed, figures, source-backed FRCNN comparison, validation/test/lint evidence, manifests and file list. Ignored outputs/yolo_baseline_marko_3050/ retains all eight full-run checkpoints (six epoch files about94.8MB each; epoch5 94,959,395 bytes; last/best19,284,881 bytes each), all six raw val_fast predictions and val_full predictions. Current artifact manifest locally verifies145 files /1,339,757,445 bytes, including preserved laptop smoke artifacts. No large artifacts committed or transferred. After an authorized SSH/private-cloud destination is supplied, copy listed files plus manifest into a fresh directory and require Badabuba's sha256sum -c verification before claiming transfer complete.
Remaining action/review: baseline+project evaluation acceptance evidence exists; student acceptance/transfer review and verified recipient transfer remain open. The runbook notebook subsection is pending (no lock taken, no notebook edited). Implementation remains unmerged. Keep CH-09 PARTIAL / PAUSED with Marko-Zenon until reviewed; do not mark DONE, merge PR, claim CH-10/11 or start tuning. CH-08/10/11 rows and notebook lock unchanged. User authorization stops at full baseline/evaluation/evidence, not automatic completion.
Notebook/PR preparation resume (2026-11-08 23:02 Kyiv): explicit Marko-Zenon authorization resumes PARTIAL / PAUSED -> IN_PROGRESS. Existing ownership/handoff retained, branch ch-09-yolo. Verified all 145 original manifest entries (1,339,757,445 bytes), six reported AP metrics and FPS before any edits. Notebook lock was FREE and acquired in this coordination commit; only a successful push authorizes editing. Scope: append evidence-only self-contained CH-09 cells, preserve all existing cells/outputs, verify transfer inventory and prepare an unmerged PR; no training or external transfer. Do not mark DONE before acceptance/notebook/recipient transfer verification.
Notebook integration / lock release (2026-11-08 23:08 Kyiv): lock acquired through pushed 556ccf3, then latest develop merged into ch-09-yolo before editing. Notebook commit 7150f82 successfully pushed. Appended 3 CH-09 cells; only 1 new self-contained evidence cell executed in isolated .venv CPU kernel; 5 embedded PNGs (training losses, project learning curve, 3 failures). All 90 prior cells/outputs and notebook metadata preserved exactly; no legacy cell/model code executed. Six AP metrics, actual config, speed, per-class/density and failure tables load from tracked results. Notebook validation PASS; original145-entry SHA256 manifest still verifies. Lock released FREE in this coordination commit. CH-09 remains IN_PROGRESS for tests/transfer inventory/PR preparation; not DONE; recipient transfer remains pending.
Notebook + draft PR preparation complete (2026-11-08 23:15 Kyiv; NOT DONE): resume/lock claim 556ccf3 successfully pushed, notebook7150f8228156bdab1a3672da3dc9080e7c347dcb successfully pushed, lock release7759022 successfully pushed and verified FREE. Closeout/transfer evidence f26c35b9369652c31da6987e129a244cdd4ce2b9 successfully pushed. Draft PR https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/10 targets develop from ch-09-yolo; OPEN/isDraft=true, mergeStateStatus CLEAN, no GitHub status checks reported. It was attached to the task. No merge requested or performed.
Acceptance audit: (1) ultralytics8.4.207 pinned/cited with AGPL notice PASS; (2) frozen-ignore export checked, 2,161 masked train frames, counts/alignment/pixels PASS; (3) stored GT adapter val_full mAP1.0 >=0.99 and adapter unit/integration tests PASS; (4) six-epoch 16,800-image baseline plus all six project-evaluator curve points and full metrics/speed/failures PASS; (5) full develop-to-task path/credential-pattern audit and staged scope checks show no datasets/exports/weights/raw predictions/submission CSVs/protected files/unrelated experiments in Git PASS. Headline mAP0.1438917717, AP50 0.2925317447, AP75 0.1192264113, AP_small0.0865789889, AP_medium0.1434272936, AP_large0.2554700319 and FP32 36.37445682FPS match stored files. Original run artifacts and145-entry manifest unchanged and fully verified; no retraining/inference/new experiment in this session.
Notebook requirement PASS: append-only3 cells, all90 previous cells/outputs and metadata exactly preserved; only1 self-contained new CPU evidence cell executed, 5 embedded figures (training losses, project curve,3 failure examples). Displays actual YOLO11s/config, training losses, all6 APs, FPS/latency, per-class/density/failure tables; student discussion remains TODOs. Notebook is93 cells; no other contributor section touched. Initial kernel startup stalled; only that helper was terminated and a fresh isolated managed .venv CPU kernel succeeded. User's pre-existing kernel untouched. Source/structural/execution evidence in results/ch-09-yolo/notebook_closeout/; notebook SHA256 7618edccb2492a2a8546015a1145808c22eb1b4f7c16a6e163c5ed99276cb5c4.
Tests/checks: final full CPU pytest149 passed/2 skipped in25.77s (CH-09 tests included); scoped new Python Ruff/format PASS, original broader lint findings retained; full diff --check PASS; four original figures decode and five notebook PNGs verified. All8 baseline checkpoints loaded on CPU again: each epoch0-5 retains expected epoch,255 optimizer entries and scaler; last/best stripped as completed checkpoints. Existing actual smoke recovery evidence retained. No GPU job, no model code execution, no package installation or unrelated fix.
Supplemental transfer inventory: results/ch-09-yolo/notebook_closeout/transfer_manifest.json + .sha256 + transfer_files.txt,163 files /1,351,399,719 bytes, all hashes locally verified. Includes every full-run epoch/last/best checkpoint and resume state, all6 raw val_fast predictions + raw val_full, full logs/resource/config/package/hardware metadata, metrics/figures, preserved smoke artifacts, notebook and closeout evidence. Original145-file manifest and payload retained unchanged. TRANSFER NOT INITIATED: no agreed destination/permission or recipient verification. TRANSFER.md provides rsync-over-SSH templates for a fresh agreed destination and recipient sha256sum -c check; Badabuba must confirm every file/count/size before completion is claimed.
Remaining blockers / next actions: student accepts notebook/technical evidence; agree and authorize a secure destination; transfer the complete listed payload and inventories; record Badabuba's successful SHA256 receipt; then complete the repository review/merge/DONE procedure. Until then remain PARTIAL / PAUSED under Marko-Zenon, draft PR unmerged, CH-10/11 unchanged/blocked. Protected site infrastructure, shared frozen protocol/configs, previous experiments and other task states untouched. Notebook lock was released at7759022; no further notebook edits after release. No new chunk started.
Closeout by Badabuba (worker: Claude), 2026-11-09 00:20-00:45 Kyiv, recorded handoff e2f5889 (notebook lock + the recorded next actions; Marko-Zenon stays owner and author of the CH-09 work). (1) Recipient verification: Marko's Telegram ZIP CH-09_YOLO11_Marko_Zenon_Artifacts.zip, 755,266,717 bytes, SHA-256 f95fb52db46961eebe9669c7de86b5418d3ba9e3ae95df9b1de5ba532f65b3f0; CRC test pass, 167 entries, no absolute/traversal/symlink/encrypted entries; original left untouched; extracted to a fresh timestamped local folder outside the repo; no script from the ZIP was executed. transfer_manifest.sha256 163/163 and artifact_manifest.sha256 145/145 pass. The two older smoke manifests mismatch only on LAPTOP_RUNBOOK.md and scripts/train_ultra.py; both equal their d84bf00 hashes in Git and were edited later. Every code/config/result file in the ZIP is byte-identical to PR head 9976e8f. A local VERIFIED.json stays with the extraction (machine-specific, not committed). (2) Protocol: all eight checkpoints load through Ultralytics on CPU: epochs 0-5 with 255 optimizer entries + scaler, last/best stripped; names person..motor = classes 0-7; train_args imgsz 1344, epochs 6, batch 1, AMP, seed 42, flip-only; the loader saw 2,800 train / 714 val_fast images. (3) Evaluator: movd/evaluate.py on the supplied raw predictions reproduces val_full mAP 0.1438917717, AP50 0.2925317447, AP75, AP_S/M/L with max abs diff 0.0 (tolerance 1e-6), identical per-class and density tables, 635,095 predictions, 0 images over the 500 cap, category ids 1-8; all six val_fast curve points reproduce exactly. (4) Artifacts present: 6 epoch + last + best checkpoints, 6 val_fast + val_full predictions, train/curve/final logs and resource samples, curve + 3 failure figures + export figure, failure_cases.csv, configs and manifests. (5) Integration: merged origin/develop (CH-08) into ch-09-yolo; homework.ipynb resolved structurally (90 shared + 2 CH-08 + 3 CH-09 cells = 95; all cells, outputs and metadata preserved; nbformat valid); the CH-08 and CH-09 cells re-executed from a clean kernel with 0 errors (outputs not written back); added only a markdown caveat that 36.37 FPS is RTX 3050 and not directly comparable with RTX 5070 throughput. Checks after integration: check_dataset.py PASS; make_splits.py --check PASS; pytest 155 passed (dataset present); scoped Ruff check + format on the 13 CH-09 Python files PASS; git diff --check clean; no checkpoints, predictions, exports, secrets or site files in the diff. All five acceptance criteria PASS -> DONE; CH-10 and CH-11 flipped to READY; notebook lock released.
Historical completion outline (remaining work, not executed): after CH-05 inference: GPU smoke yolo_smoke_s / yolo_smoke_m (train_ultra.py --device 0, then --resume, then predict_ultra.py curve) -> fix model size + batch in yolo_baseline.json -> baseline train -> predict_ultra.py curve + final -> PR ch-09-yolo -> develop, mark DONE, flip CH-10/CH-11 to READY.
```

---

### CH-10 — Anchor-free tuning + small/dense analysis

- **Goal:** Tune the anchor-free detector and analyze its behaviour by object size and scene density.
- **Why required:** rubric item ("assignment, loss, resolution or high-resolution head; size-specific and dense-scene analysis").
- **Reuse / preserve:** the CH-09 pipeline and baseline.
- **Scope:**
  1. At least 2 controlled variants. A higher input size is required.
  2. A **P2 / high-resolution head only after a compatibility smoke test passes**, with partial weight transfer documented. Otherwise vary an assignment or loss setting (for example a box/cls loss gain).
  3. Size-bucket AP and AP-vs-density-bin tables.
  4. A written analysis and a short notebook section (🔒 short).
- **Non-goals:** final long training, TTA or tiling (CH-13).
- **Start deps:** CH-09 · **Finish deps:** — · **Machine prerequisites:** M-DATA, M-GPU, 🔒 short · **Gates:** —.
- **Parallel:** yes, with CH-07, CH-08 and CH-11.
- **Owner / State:** see the board.
- **Files expected to change:** `configs/runs/yolo_*.json`, `results/yolo_*/`, `results/ch-10-yolo-tuning/figures/`, `homework.ipynb` (short, under the lock).
- **Conflict risk:** low.
- **Implementation outline:** variant configs → runs → breakdown tables → analysis → notebook.
- **Validation:** the complete `results/` sets; the tables regenerate.
- **Evidence & artifacts:** the run directories, the size and density tables, figures.
- **Acceptance criteria:**
  1. Two or more variants exist under an identical budget, apart from the variable under test.
  2. The size and density tables are present.
  3. The written analysis is present.
  4. The P2 decision is documented, whichever way it went.
- **Effort:** about 2.5 h. **Runtime / GPU / disk:** about 2 h GPU.
- **Likely blockers:** OOM at high input size; P2 config incompatibility.
- **Handoff / unblocks:** start dependency of CH-12. Gives the best anchor-free checkpoint for CH-13.
- **Evidence log:**
```text
Owner: — | Claimed: — | Last update: — | Branch: — | Commits: — | PR: —
Files changed: —
Validation commands: —
Metrics / results: —
Artifacts: —
Runtime / hardware: —
Blockers / deviations: —
Completion or handoff note: —
```

---

### CH-11 — Query-based RT-DETR baseline + tuning + learning curve

- **Goal:** Train a query-based detector (baseline plus a tuned variant) under a comparable budget, and compare its learning curve with a CNN.
- **Why required:** rubric item: comparable compute, mAP learning curve against a CNN, and analysis of convergence, small objects and dense scenes.
- **Reuse / preserve:** the CH-09 export, adapter and evaluator; the CH-09 YOLO curve as the CNN comparator (and the CH-07 curve if available).
- **Scope:**
  1. An 8 GB smoke test of Ultralytics RT-DETR (`rtdetr-l`) covering batch size, input size, AMP and memory. If it is unreliable, set 🔴 NEEDS_DECISION with options: external GPU (Kaggle or Colab) or an approved alternative implementation (G9).
  2. A baseline under the same images-seen budget and input size as the YOLO comparator, with per-epoch checkpoints.
  3. At least 1 tuned variant, for example the number of queries, input size or the learning-rate schedule.
  4. A mAP-vs-epoch curve (our evaluator) against YOLO.
  5. Analysis of convergence, small objects and dense scenes, plus a short notebook section (🔒 short).
- **Non-goals:** other DETR variants, final training.
- **Start deps:** CH-09 · **Finish deps:** — · **Machine prerequisites:** M-DATA, M-GPU (8 GB preferred), 🔒 short · **Gates:** G9.
- **Parallel:** yes, with CH-07, CH-08 and CH-10.
- **Owner / State:** see the board.
- **Files expected to change:** `configs/runs/rtdetr_*.json`, `results/rtdetr_*/`, `results/ch-11-rtdetr/figures/`, `homework.ipynb` (short, under the lock).
- **Conflict risk:** low.
- **Implementation outline:** smoke → baseline → tuned run → curves → analysis → notebook.
- **Validation:** the complete `results/` sets; the curve plot regenerates from the curve files.
- **Evidence & artifacts:** the run directories and `results/ch-11-rtdetr/figures/learning_curve.png`.
- **Acceptance criteria:**
  1. G9 is resolved.
  2. Baseline plus 1 or more tuned runs exist under the same budget as the comparator.
  3. The learning-curve comparison is present.
  4. The convergence, small-object and dense-scene analysis is written.
- **Effort:** about 3 h. **Runtime / GPU / disk:** about 3 h GPU on 8 GB.
- **Likely blockers:** memory, slow convergence (an expected finding, record it).
- **Handoff / unblocks:** start dependency of CH-12.
- **Evidence log:**
```text
Owner: Badabuba (worker: Claude) | Claimed: 2026-11-09 00:50 Kyiv | Last update: 2026-11-09 07:00 Kyiv | Branch: ch-11-rtdetr | Commits: 5c4d566, f08836a, c60bd54, 17a452c, a270db4, e84fb53, f648831 (+ develop merges, this closeout) | PR: #12
Files changed: movd/rtdetr.py (new), movd/ultra.py (family field), scripts/train_ultra.py, scripts/predict_ultra.py (family, speed subcommand, RT-DETR speed note), configs/models/rtdetr-l-q500.yaml, configs/runs/rtdetr_*.json (3 smoke + baseline, q500, lr1e4, lr1e4_wb0), tests/test_rtdetr.py (12 tests), results/rtdetr_{baseline,q500,lr1e4,lr1e4_wb0,smoke_b1,smoke_b2,smoke_q500}/, results/ch-11-rtdetr/ (runner, evidence generator, path relativizer, notebook-cell script, tables, figures, query_saturation.json, speed_recheck/, yolo11s_speed_rtx5070.json, export_check/, analysis.md), README.md (CH-11 section), homework.ipynb (3 cells appended under the lock, 95 existing cells unchanged).
Validation commands: pytest -q (166 passed, 1 skipped: CUDA-only test with the GPU hidden); python scripts/check_dataset.py (PASSED); python scripts/make_splits.py --check (val_fast 714 / val_full 2846, hashes match); ruff check --ignore RUF100 + ruff format --check on the CH-11 files (pass); python results/ch-11-rtdetr/make_evidence.py --tables-only (byte-identical tables); CH-11 notebook cells executed from a clean kernel (0 errors, 5 figures); git diff --check (clean).
Metrics / results: val_full (project evaluator, maxDets 500): rtdetr_baseline mAP 0.2577 / AP50 0.4775 / AP75 0.2392 / AP_S 0.1296 / AP_M 0.2246 / AP_L 0.4538; rtdetr_q500 0.2389; rtdetr_lr1e4_wb0 0.2179; rtdetr_lr1e4 (confounded) 0.1560. Comparators: YOLO11s 0.1439, Faster R-CNN RPN pre3000 0.1893, RetinaNet anchors 0.1693. val_fast best: baseline 0.2691 at epoch 3. Densest bin (>=100 targets): 0.368 at maxDets 500 vs 0.315 at 100. Every val_full frame returns exactly the query cap (300/500); 1.13 confident (>=0.3) detections per target.
Artifacts: tracked: results/rtdetr_*/ and results/ch-11-rtdetr/ (figures/learning_curve.png, figures/size_density.png). Not tracked (outputs/, Nazar's machine): per-epoch checkpoints epoch0-5 + last/best for the 4 full runs, raw val_full/val_fast predictions, logs in outputs/ch11_logs/; pretrained rtdetr-l.pt sha256 6de60b10…e73f.
Runtime / hardware: Nazar's RTX 5070 Laptop GPU 8 GB, torch 2.14.1+cu130, ultralytics 8.4.207. Training 4,018-4,289 s per full run (about 4 img/s), plus about 18 min curve + val_full each; total GPU time about 6 h incl. smoke (00:55-06:50 Kyiv plus smokes). Speed (same session): RT-DETR-l 10.6 FPS, YOLO11s 64.2 FPS (CH-09 checkpoint on the RTX 5070; 36.37 FPS on the RTX 3050); absolute FPS varied 20-25% between sessions.
Blockers / deviations: none on protocol (imgsz 1344, 6 epochs, splits and ignore policy unchanged). Batch 1 chosen from the smoke (batch 2 peaked at 6.3 GB, same throughput). rtdetr_lr1e4 is confounded: an explicit optimizer skips Ultralytics' auto branch that sets warmup_bias_lr 0.0; kept and marked, and the clean rtdetr_lr1e4_wb0 was added. Notebook lock line on develop said "taken 07:05"; actual time was about 06:49 Kyiv.
Completion or handoff note: G9 resolved; CH-12 still needs CH-10, which was not touched. Report prose and conclusions stay with the students (TODOs in analysis.md section 10 and in the notebook). No Kaggle submission was made. RTX 5070 released.
```

---

### CH-12 — Cross-family comparison & final-config decision

- **Goal:** Compare all four families on consistent validation, and freeze an evidence-based final training configuration.
- **Why required:** the final pipeline must be justified by earlier results. CH-12 is the decision gate for CH-14.
- **Reuse / preserve:** all the `results/` directories from CH-07, CH-08, CH-10 and CH-11.
- **Scope:**
  1. `scripts/aggregate_results.py` → `results/summary.csv` plus comparison plots (accuracy, AP_small, FPS).
  2. A written justification for the final family and the training config (data amount, input size, epochs), each claim linked to a result file.
  3. Freeze `configs/runs/final.json`, including its expected runtime.
  4. A short notebook section (🔒 short).
- **Non-goals:** new training runs.
- **Start deps:** CH-07, CH-08, CH-10, CH-11 · **Finish deps:** — · **Machine prerequisites:** CPU, 🔒 short · **Gates:** —.
- **Parallel:** yes, with CH-06 and CH-15 work.
- **Owner / State:** see the board.
- **Files expected to change:** `scripts/aggregate_results.py`, `results/summary.csv`, `results/ch-12-comparison/`, `configs/runs/final.json`, `homework.ipynb` (short, under the lock).
- **Conflict risk:** low.
- **Implementation outline:** aggregate → plots → decision note → `final.json` with a runtime estimate.
- **Validation:** `python scripts/aggregate_results.py` regenerates `summary.csv` deterministically.
- **Evidence & artifacts:** `summary.csv`, plots, the decision note.
- **Acceptance criteria:**
  1. All four families appear with the same metrics.
  2. The decision note cites evidence.
  3. `final.json` is committed, with a runtime estimate that fits the schedule.
- **Effort:** about 2 h. **Runtime / GPU / disk:** CPU.
- **Likely blockers:** a missing family by the Mon 19:00 decision point (§7). A student then decides how to proceed.
- **Handoff / unblocks:** CH-13 becomes READY, and CH-14 becomes READY once CH-05 is DONE.
- **Evidence log:**
```text
Owner: — | Claimed: — | Last update: — | Branch: — | Commits: — | PR: —
Files changed: —
Validation commands: —
Metrics / results: —
Artifacts: —
Runtime / hardware: —
Blockers / deviations: —
Completion or handoff note: —
```

---

### CH-13 — Final-pipeline inference experiments on val

- **Goal:** Measure the inference-side improvements one at a time and freeze the final inference settings.
- **Why required:** the rubric asks for a stronger inference strategy with separate intermediate comparisons.
- **Reuse / preserve:** the best checkpoint of the chosen family (CH-10 or CH-11) as one fixed reference; the CH-03 evaluator.
- **Scope:** on `val_full`, each change measured separately against the reference:
  1. Inference input size.
  2. Tiling or slicing (with merge and NMS).
  3. Confidence threshold, NMS IoU and `max_det` (≤ 500).
  4. TTA.
  5. An ensemble (for example WBF), **only if** it gains on val.

  Freeze the settings in `configs/runs/final_inference.json`.
- **Non-goals:** retraining.
- **Start deps:** CH-12 · **Finish deps:** — · **Machine prerequisites:** M-DATA, M-GPU · **Gates:** —.
- **Parallel:** yes, with CH-14 training (on the other GPU).
- **Owner / State:** see the board.
- **Files expected to change:** `movd/infer.py` (tiling, TTA helpers), `scripts/infer_val.py`, `results/ch-13-inference/`, `configs/runs/final_inference.json`.
- **Conflict risk:** low.
- **Implementation outline:** reference → one experiment per change → comparison table → freeze.
- **Validation:** every row of the table has its own `metrics.json`; the same reference is used throughout.
- **Evidence & artifacts:** `results/ch-13-inference/comparison.csv` and figures in `results/ch-13-inference/figures/`.
- **Acceptance criteria:**
  1. Each change is measured separately.
  2. The table includes failed or neutral changes.
  3. The settings are frozen, along with their speed cost.
- **Effort:** about 2.5 h. **Runtime / GPU / disk:** about 2 h GPU.
- **Likely blockers:** inference time with tiling; an ensemble needing two models in memory.
- **Handoff / unblocks:** finish dependency of CH-14.
- **Evidence log:**
```text
Owner: — | Claimed: — | Last update: — | Branch: — | Commits: — | PR: —
Files changed: —
Validation commands: —
Metrics / results: —
Artifacts: —
Runtime / hardware: —
Blockers / deviations: —
Completion or handoff note: —
```

---

### CH-14 — Final training, test predictions, Kaggle submissions & selection

- **Goal:** Train the final model, produce validated test predictions, beat the Kaggle baseline, and select the final submissions using internal validation.
- **Why required:** these are the core deliverable and score.
- **Reuse / preserve:** `configs/runs/final.json` (CH-12), `final_inference.json` (CH-13), and the CH-05 tooling.
- **Scope:**
  1. Resumable final training per `final.json`.
  2. Compare the final model with the rubric-budget model on `val_full`; this is the data-strategy gain.
  3. Test inference with the CH-13 settings, then the validator, then upload by a student.
  4. Log every upload in `results/submissions.csv`.
  5. Choose the final selection(s) mainly by internal val (G2: number of selections), and record the rationale.
- **Non-goals:** new architecture experiments after the Tue 08:00 cut-off.
- **Start deps:** CH-05, CH-12 · **Finish deps:** CH-13 · **Machine prerequisites:** M-DATA, M-GPU · **Gates:** G8.
- **Parallel:** training runs in parallel with CH-13.
- **Owner / State:** see the board.
- **Files expected to change:** `results/final_*/`, `results/submissions.csv`, `results/ch-14-final/`.
- **Conflict risk:** low. `results/submissions.csv` is append-only.
- **Implementation outline:** launch training by Mon 22:00 → monitor and resume → evaluate → test predictions → validate → upload → select.
- **Validation:** validator exit 0 for every uploaded file; val metrics come from our evaluator.
- **Evidence & artifacts:** the run directory, the submissions log with sha256 and scores, the selection note.
- **Acceptance criteria:**
  1. The final training is complete (or a documented fallback checkpoint is used).
  2. At least 1 Kaggle score is above 0.22163.
  3. Every uploaded file passed the validator.
  4. The selection follows internal val and is recorded.
- **Effort:** about 2 h hands-on. **Runtime / GPU / disk:** 6–10 h GPU, about 20–40 min test inference.
- **Likely blockers:** the training does not finish in time (fallback in §7); the Kaggle daily limit.
- **Handoff / unblocks:** start dependency of CH-16.
- **Evidence log:**
```text
Owner: — | Claimed: — | Last update: — | Branch: — | Commits: — | PR: —
Files changed: —
Validation commands: —
Metrics / results: —
Artifacts: —
Runtime / hardware: —
Blockers / deviations: —
Completion or handoff note: —
```

---

### CH-15 — Report scaffold & Pages deployment check

- **Goal:** Prove the GitHub Pages pipeline early and give the students a rubric-mapped report skeleton.
- **Why required:**
  - Pages is currently broken (not enabled).
  - Fixing it late is risky.
  - The students need the structure early to write their prose.
- **Reuse / preserve:** the template's `report.md` front-matter format and site (never edit the site files).
- **Scope:**
  1. `report.md`: front matter (title, authors, course, date) and section headings mapped to the §3 rubric. Under each heading, **TODO prompts for the students (no prose)**. Add image conventions (`images/<chunk>-<name>.png`, copied from `results/…/figures/`).
  2. Milestone PR M1 `develop → main`.
  3. Verify that the "Deploy report" run is green and the Pages URL loads. Record the URL.
- **Non-goals:** writing the report content; editing `index.html`, `assets/`, `.github/`.
- **Start deps:** — · **Finish deps:** — · **Machine prerequisites:** CPU · **Gates:** G5 (start gate, Marko admin).
- **Parallel:** yes, with anything.
- **Owner / State:** see the board.
- **Files expected to change:** `report.md`, optionally `images/` (placeholder or test figure only).
- **Conflict risk:** low.
- **Implementation outline:** G5 confirmed → skeleton → PR into `develop` → M1 PR to `main` → check the Actions run and the URL.
- **Validation:** `gh run list --workflow "Deploy report" -L 1` shows success; the Pages URL returns 200 and shows the skeleton.
- **Evidence & artifacts:** the run ID, the URL, the commit.
- **Acceptance criteria:**
  1. Pages is live from `main`.
  2. The skeleton is merged.
  3. No site files are touched.
  4. No report prose was written by an AI.
- **Effort:** about 1.5 h. **Runtime / GPU / disk:** CPU.
- **Likely blockers:** G5 (admin); the Actions environment approval.
- **Handoff / unblocks:** start dependency of CH-16.
- **Evidence log:**
```text
Owner: Marko-Zenon | Claimed: 2026-11-08 18:29 Kyiv (pushed c3873d2) | Last update: 2026-11-08 20:02 Kyiv | Branch: ch-15-report; claim/CH-15-closeout-Marko-Zenon (coordination only) | Commits: 27e4110, 278ce28, e83af53; task merge aafc526; M1 merge fec11cb; post-merge board fix (git log origin/develop) | PR: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/8 ; M1: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/9
Files changed: report.md; images/ch-15-ignore-masking-check.png; plan.md (CH-15 row and this log only).
Validation commands / observations:
  git fetch origin; git switch develop; git pull --ff-only; git switch -c ch-15-report origin/develop -> success; clean start; claim verified on origin/develop.
  git merge --no-edit origin/develop -> already up to date. Public GitHub REST /pulls?state=open&base=develop -> [] at inspection.
  Initial session: gh was not authenticated. Publication follow-up: gh auth status -> Marko-Zenon authenticated; gh pr list --base develop -> no open PR before CH-15 publication.
  git fetch origin; git switch develop; git pull --ff-only; git switch ch-15-report; git merge --no-edit origin/develop -> already up to date, no conflicts; original 27e4110 preserved.
  .venv/bin/python /tmp/validate_ch15.py -> PASS metadata, 21 local links, table columns, exact six AP metrics/FPS/latency per run, eight per-class AP rows, six curve epochs/run, image decode/byte identity, allowed file list and CH-15-only board changes.
  gh run list --workflow "Deploy report" -L 1 --json databaseId,conclusion,headSha,url -> run 42058475925 success, main 0330c83; curl public Pages URL -> HTTP 200.
  git push -u origin ch-15-report -> success; remote ref confirms 27e4110. gh pr create --base develop --head ch-15-report --title "CH-15: report scaffold and verified evidence inventory" --body-file /tmp/ch15-pr-body.md -> PR #8.
  Pytest availability rechecked -> No module named pytest; previous desktop/mobile preview retained (report/image unchanged).
  Public REST /actions/workflows/deploy.yml/runs?per_page=1 -> run 42058475925, attempt 2, success, main SHA 0330c836df45f3ba8199a801cbb7671f7fa1a264.
  https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/actions/runs/42058475925
  https://marko-zenon.github.io/Multiple-Object-Video-Detection/ -> HTTP 200; browser confirms old template, not this unmerged scaffold.
  python -m http.server 8000 --bind 127.0.0.1 -> local browser renders title/authors, 16 h2 headings, 7 tables, loaded image, 0 broken TOC anchors.
  Browser at 390x844 -> image fits, page width 375 <= viewport 390; seven tables use overflow-x:auto; viewport reset afterwards.
  .venv/bin/python inline stdlib/Pillow audit -> PASS frontmatter keys, 21 existing local links/tracked sources, table column counts, image decode/byte identity, three complete six-epoch artifact sets (16800 images seen each).
  Notebook read-only AST inspection -> 90 cells, 48 with outputs; no code-cell syntax errors, imports use available repo movd paths (runtime dependencies not executed).
  .venv/bin/python -m pytest tests/test_evaluate.py -q -> NOT RUN: local environment has no pytest; no dependency installation or GPU work performed.
  git diff --check -> PASS. Protected infrastructure, notebook, code/config/results, README and requirements diff -> empty.
  M1 follow-up (2026-11-08 19:35 Kyiv): git fetch origin; gh pr view 8 --json url,state,mergedAt,mergeCommit,baseRefName,headRefName -> state OPEN, mergedAt null, mergeCommit null, base develop. No actual merge evidence exists yet.
  Fetched origin/develop = c3873d2ba7fb133865d6ef36eef9e4a8db251ec7; original CH-15 report remains absent there. git switch develop; git pull --ff-only; git switch ch-15-report; git merge --no-edit origin/develop -> already up to date; clean initial tree.
  plan.md section 8 and CH-15 scope explicitly require M1 develop -> main AFTER the task scaffold reaches develop. gh pr list --base main --state open -> no existing milestone PR. M1 not created because PR #8 is still open; creating it now would omit the scaffold.
  gh run list --workflow "Deploy report" -L 1 -> 42058475925 success, head 0330c836df45f3ba8199a801cbb7671f7fa1a264. Public /report.md HTTP 200 still has Homework 1 / John Doe template metadata, not the CH-15 scaffold.
  Protected website/model/config/result/notebook diff against origin/develop -> empty. This follow-up changes only this CH-15 evidence log; git diff --check passes.
  FINAL CLOSEOUT (2026-11-08 20:02 Kyiv; supersedes earlier pending/open statements below):
  git fetch origin; gh pr view 8 -> MERGED 2026-11-08T17:38:32Z, merge aafc526fe9d07be3c2374d2d23c3f3a53d96a486 on develop.
  gh pr view 9 -> MERGED 2026-11-08T17:51:59Z, merge fec11cbc0150fd410356f2973d5939750ea5e4f8 on main.
  gh run list --workflow "Deploy report" -L 2 --json databaseId,conclusion,headSha,url,status -> 42507083884, completed/success at main fec11cbc0150fd410356f2973d5939750ea5e4f8.
  Deployment: https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/actions/runs/42507083884
  Public https://marko-zenon.github.io/Multiple-Object-Video-Detection/ -> HTTP 200. Live /report.md and /images/ch-15-ignore-masking-check.png fetched successfully; byte-identical to origin/main. Image decode passes. main/develop report + figure identical; all 21 report-local source links exist in main's tree.
  Browser verifies Homework 2 title, all 16 h2 headings (including evidence inventory), seven rendered tables, loaded test image, zero broken contents anchors, and student-authored discussion/conclusions TODOs.
  git merge-base --is-ancestor aafc526 origin/develop and origin/main -> both pass; scaffold merged on both branches.
  git diff 0330c83 origin/main --name-only -- index.html assets/ .github/ .nojekyll -> empty; protected infrastructure and last-push mechanism unchanged.
  Acceptance 1 PASS: Pages live from main, successful run matches M1 merge SHA. Acceptance 2 PASS: scaffold merged into develop via #8 and main via #9. Acceptance 3 PASS: protected site files untouched. Acceptance 4 PASS: student prose/discussion/conclusions remain TODOs; additions are prompts and source-backed evidence tables/inventory, not student conclusions.
  CH-16 recalculation: start CH-01, CH-06, CH-15 DONE; CH-12 and CH-14 not DONE, so retain BLOCKED and remove only CH-15 from its pending list. G1/G10 are unchanged finish gates. No other task states changed.
  Closeout scope: plan.md only (CH-15 row/log, CH-16 dependency row); no code, notebook, model, experiment, report prose or protected-file edits. git diff --check and exact row/log scope audit pass; no training or notebook execution.
Metrics / results: Three val_full Faster R-CNN runs, all six AP metrics + eight per-class APs and FPS/latency transcribed from tracked metrics/speed JSONs. RPN selected mAP 0.1893354654; 14.847445 FPS / 67.351653 ms, GPU-resident detector timing (decode/transfer excluded). First Kaggle accepted score 0.2011538266 < baseline 0.22163, source results/submissions.csv. Smoke/legacy runs excluded from headline tables.
Artifacts: report.md contains inventory and source links; one pipeline test image only, from results/ch-02-protocol/figures/ignore_masking_check.png; SHA256 7629c7435bea3f61c771fc597adc88f3e4c78205aea51e8c072f2229f9a355aa. Final figure selection/copy belongs to CH-16. No new chart or extra documentation artifact.
Runtime / hardware: CPU documentation/static inspection and browser preview on Marko's Ubuntu; no training, inference or notebook execution. Stored run hardware is RTX 5070 Laptop on Windows, not this session's measurement.
Read-only audit, confirmed: CH-08 full RetinaNet, CH-09/10 YOLO11, CH-11 RT-DETR, CH-12 cross-family decision, CH-13 inference ablations, CH-14 final config/selection and baseline-beating upload missing from this develop snapshot. FRCNN has complete stored speed/failures; other-family full-run evidence absent. README still describes notebook-centric legacy 560-frame/3-epoch protocol, ends mid-sentence and says push main; use CLAUDE/plan instead. requirements.txt on develop lacks ultralytics and README lacks its license notice (CH-09 work is paused/unmerged); local pytest absent. configs/protocol.json retains stale UNKNOWN daily limit despite board G2b resolved. G1/G9/G10 remain open.
Read-only audit, unverified risks: full clean-env reproducibility and notebook runtime dependencies not tested; legacy cells retrain if executed; hidden Kaggle scorer equivalence unknown. Static code agrees with frozen 1-8/0-7 label conversion, train mask and crowd-preserving evaluator; no confirmed new split/label/ignore/evaluation defect found. Constant-fill masking is a heuristic, not explicit loss suppression. Single seed and non-end-to-end speed limit comparison claims.
Blockers / deviations: None for CH-15. Initial authentication and pending-merge/deployment blockers above are historical and resolved by PR #8, M1 #9 and successful run 42507083884. Final figure pack and student report writing remain CH-16/G10 work; those are not CH-15 scaffold blockers. No claim that the final report or remaining model chunks are complete.
Completion or handoff note: DONE after all four CH-15 acceptance criteria passed and implementation was merged into develop/main. Closeout uses a normal fast-forward board-only coord push from the throwaway branch to develop, followed by fetch/live-board verification. CH-16 remains BLOCKED by CH-12/CH-14. No new chunk claimed. No currently READY unowned technical chunk exists; next recommended target is CH-10 after Badabuba completes CH-09 and it becomes READY, subject to machine checks and an atomic claim. Do not take over Badabuba's CH-08 or paused CH-09 without a recorded handoff.

```

---

### CH-16 — Reproducibility, evidence pack, rubric audit, freeze

- **Goal:** Deliver a clean, reproducible, verified repository and freeze it before the deadline.
- **Why required:** final-delivery rubric items: reproducibility, evidence, the public repo, Pages, contributions and the freeze.
- **Reuse / preserve:** everything. No experiment results are changed.
- **Scope:**
  1. Clean-environment check (fresh venv from `requirements.txt`; `check_dataset.py`; one evaluation command reproduces a stored metric).
  2. README: exact run commands for every stage, the results index and citations. Fix the truncated §6.
  3. Evidence pack:
     - copy the selected figures from `results/` into `images/`;
     - build the evidence tables;
     - run a **traceability check**: every number in `report.md` maps to a `results/` file.
  4. Contributor check: both students have attributed technical commits (G6).
  5. Rubric audit: §12 all ✓ or justified.
  6. Students confirm G10, the report prose they wrote.
  7. Final PR M3 `develop → main` and tag `v1.0-final`.
  8. Verify the Pages deployment.
- **Non-goals:** new experiments; writing report conclusions.
- **Start deps:** CH-01, CH-06, CH-12, CH-14, CH-15 · **Finish deps:** — · **Machine prerequisites:** CPU, clean env · **Gates:** G6, G10 (finish).
- **Parallel:** no. It is last.
- **Owner / State:** see the board.
- **Files expected to change:** `README.md`, `images/`, `report.md` (formatting and traceability fixes only, with student approval), `plan.md`.
- **Conflict risk:** medium. Announce a freeze on the board before starting and ask others to stop pushing.
- **Implementation outline:** freeze announcement → clean env → README → evidence pack → traceability → audit → M3 → tag → Pages check.
- **Validation:**
  - Clean-env commands succeed.
  - The traceability script reports 0 unmatched numbers.
  - `gh api repos/Marko-Zenon/Multiple-Object-Video-Detection/contributors` lists both students.
  - The Pages URL returns 200.
- **Evidence & artifacts:** the audit checklist, the run IDs, the tag.
- **Acceptance criteria:**
  1. Every §12 item is ✓ or justified.
  2. M3 is merged and tagged by 21:30.
  3. Pages shows the final report.
  4. Both contributors are visible.
  5. G10 is confirmed by the students.
- **Effort:** about 3 h. **Runtime / GPU / disk:** CPU.
- **Likely blockers:** late chunks; contributor email attribution (G6).
- **Handoff / unblocks:** project complete.
- **Evidence log:**
```text
Owner: — | Claimed: — | Last update: — | Branch: — | Commits: — | PR: —
Files changed: —
Validation commands: —
Metrics / results: —
Artifacts: —
Runtime / hardware: —
Blockers / deviations: —
Completion or handoff note: —
```

## 11. Change log

| Date (Kyiv) | Change | By | Approved by |
|---|---|---|---|
| 2026-11-07 15:00 | Initial coordination setup: `develop` branch, `CLAUDE.md`, `plan.md` with 16 chunks | Badabuba (via Claude) | Badabuba |
| 2026-11-07 17:00 | CH-02 freezes §4. G2 split into G2a (internal metric, resolved), G2b (daily limit, operational, before CH-05's upload) and G2c (2 final selections, resolved). G1 (UTC close time) re-scoped as an operational finish gate of CH-16. Neither value is on the supplied official pages; neither blocks the technical protocol | Badabuba (via Claude) | Badabuba |
| 2026-11-07 | CH-04 runtime forecast revised from the one-hour planning target: ~119 min Faster R-CNN, ~116 min RetinaNet on the RTX 5070. Extrapolates 100-image train time to 16,800, adds six measured val_fast passes plus observed final validation, speed and overhead. Frozen protocol unchanged. | Badabuba (Codex) | — |

## 12. Definition of done and rubric audit checklist

The project is done when every box is ✓ (or has a written justification), M3 is merged into `main` and tagged, and Pages shows
the final report before the verified Kaggle Close at 23:55 Europe/Kyiv / 21:55 UTC on 2026-11-10 (G1). The earlier student-reported 23:59 display conflicts with the official Close tooltip and is not used as the cutoff.

- [ ] Data understanding: class, size and density distributions, sequence differences, GT visualization (CH-06)
- [ ] Clip-level split and justified subsampling; resizing, normalization, augmentation documented (CH-02, CH-06)
- [ ] Ignore-region policy verified, frozen and applied in training and evaluation (CH-02, CH-03, CH-04, CH-09)
- [ ] Two-stage: baseline, tuning, mAP/AP50/AP_small/speed, ≥ 3 failure cases (CH-07)
- [ ] One-stage anchor-based: baseline, anchors/focal tuning, comparison with two-stage (CH-08)
- [ ] Anchor-free: baseline + tuned, size-specific and dense-scene analysis (CH-09, CH-10)
- [ ] Query-based: baseline + tuned, comparable budget, learning curve vs CNN, analysis (CH-11)
- [ ] ≥ 2 meaningful experiments beyond baselines, including a defensible loss experiment; failed runs kept (CH-07, CH-08, CH-10, CH-11)
- [ ] Consistent validation: all headline numbers from the project evaluator on `val_full` (CH-03, all model chunks)
- [ ] Final pipeline justified by results, with separate intermediate comparisons (CH-12, CH-13, CH-14)
- [ ] Valid test submission: original pixels, classes 0–7, `class score x y w h`, one row per image, ≤ 500, validator passed (CH-05, CH-14)
- [ ] ≥ 1 Kaggle score > 0.22163; final selection by internal validation (CH-14)
- [ ] Reproducible code, pinned dependencies, exact run instructions, clean-env check (CH-01, CH-16)
- [ ] Evidence stored in `results/`; every report number traceable (CH-16)
- [ ] Report prose and conclusions written by the students (G10)
- [ ] Public repo; GitHub Pages live from `main` (CH-15, CH-16)
- [ ] Meaningful, attributed technical commits from both Badabuba and Marko-Zenon (G6, CH-16)
- [ ] No dataset, ZIP, checkpoints, predictions, submissions or secrets in Git; site files untouched (CH-16)
- [ ] Repository frozen and tagged before the deadline (CH-16)
