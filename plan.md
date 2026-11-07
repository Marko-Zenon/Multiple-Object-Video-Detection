# plan.md — live task board and execution plan

> **Deadline:** Tue 2026-11-10 23:55 Kyiv (gate G1: confirm Kaggle's UTC close time). **Freeze target:** Tue 21:30.
> **The live version of this file is the one on `origin/develop`.** Fetch before reading it. Rules are in `CLAUDE.md`.
> **To start work:** pick a row with `🟢 READY` and `Start now? = 🟢 YES`, then run the atomic claim (§9.2) **before** coding.
> **States:** ✅ DONE · 🟢 READY · 🔵 IN_PROGRESS · 🟡 BLOCKED · 🟠 PARTIAL / PAUSED · 🔴 FAILED / NEEDS_DECISION.
> **Depends on:** `start:` IDs must be DONE before a claim; `finish:` IDs must be DONE before this chunk can be DONE; `X or Y` = either.

## 1. Live task board

| ID | Chunk | State | Start now? | Owner | Depends on | Blocked by / reason | Branch / PR | Last update |
|---|---|---|---|---|---|---|---|---|
| CH-01 | Repo hygiene & local dataset/env setup | ✅ DONE | ⛔ NO — completed | Badabuba | — | — (M-DATA + M-GPU satisfied on Nazar's RTX 5070) | ch-01-repo-env-setup / [PR #1](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/1) | 2026-11-07 16:15 Kyiv |
| CH-02 | Protocol freeze, Kaggle facts, data module | 🔵 IN_PROGRESS | ⛔ NO — owned by Badabuba | Badabuba | — | waiting on student Kaggle read for G1–G3 (close time UTC, metric, limits, sample_submission.csv); all other scope pushed to branch | ch-02-protocol-data | 2026-11-07 17:05 Kyiv |
| CH-03 | Evaluator, speed benchmark, tests | 🟡 BLOCKED | ⛔ NO — CH-02 | — | start: CH-02 | label map, ignore policy, Kaggle metric come from CH-02 | — | 2026-11-07 15:00 Kyiv |
| CH-04 | Torchvision training core | 🟡 BLOCKED | ⛔ NO — CH-02 | — | start: CH-02 · finish: CH-03 | needs frozen protocol & data module | — | 2026-11-07 15:00 Kyiv |
| CH-05 | Test inference, submission writer/validator, first Kaggle upload | 🟡 BLOCKED | ⛔ NO — CH-02 | — | start: CH-02 · finish: CH-03, CH-04 or CH-09 | needs label map & submission format (G3) | — | 2026-11-07 15:00 Kyiv |
| CH-06 | Corrected EDA & preprocessing evidence (notebook) | 🟢 READY | 🟢 YES | — | — | — (machine needs M-DATA; take notebook lock) | — | 2026-11-07 15:00 Kyiv |
| CH-07 | Two-stage: Faster R-CNN corrected + tuning + failure cases | 🟡 BLOCKED | ⛔ NO — CH-04 | — | start: CH-04 | needs training core | — | 2026-11-07 15:00 Kyiv |
| CH-08 | One-stage anchor-based: RetinaNet + anchors + focal-loss experiment | 🟡 BLOCKED | ⛔ NO — CH-04 | — | start: CH-04 | needs training core | — | 2026-11-07 15:00 Kyiv |
| CH-09 | Ultralytics export + YOLO11 anchor-free baseline | 🟡 BLOCKED | ⛔ NO — CH-02 | — | start: CH-02 · finish: CH-03 | needs frozen protocol & ignore policy | — | 2026-11-07 15:00 Kyiv |
| CH-10 | Anchor-free tuning + small/dense analysis | 🟡 BLOCKED | ⛔ NO — CH-09 | — | start: CH-09 | needs YOLO baseline & export | — | 2026-11-07 15:00 Kyiv |
| CH-11 | Query-based RT-DETR baseline + tuning + learning curve | 🟡 BLOCKED | ⛔ NO — CH-09 | — | start: CH-09 | needs export, adapter & YOLO curve | — | 2026-11-07 15:00 Kyiv |
| CH-12 | Cross-family comparison & final-config decision | 🟡 BLOCKED | ⛔ NO — CH-07, CH-08, CH-10, CH-11 | — | start: CH-07, CH-08, CH-10, CH-11 | needs all four families | — | 2026-11-07 15:00 Kyiv |
| CH-13 | Final-pipeline inference experiments on val | 🟡 BLOCKED | ⛔ NO — CH-12 | — | start: CH-12 | needs chosen final family | — | 2026-11-07 15:00 Kyiv |
| CH-14 | Final training, test predictions, Kaggle submissions & selection | 🟡 BLOCKED | ⛔ NO — CH-05, CH-12 | — | start: CH-05, CH-12 · finish: CH-13 | needs submission tooling & frozen final config | — | 2026-11-07 15:00 Kyiv |
| CH-15 | Report scaffold & Pages deployment check | 🟡 BLOCKED | ⛔ NO — gate G5 | — | gate: G5 | Marko must enable Pages (Settings → Pages → Source: GitHub Actions) | — | 2026-11-07 15:00 Kyiv |
| CH-16 | Reproducibility, evidence pack, rubric audit, freeze | 🟡 BLOCKED | ⛔ NO — CH-06, CH-12, CH-14, CH-15 | — | start: CH-01, CH-06, CH-12, CH-14, CH-15 · finish gate: G10 | final-delivery chunk | — | 2026-11-07 15:00 Kyiv |

🔒 **homework.ipynb lock:** FREE

### Gates

| ID | Gate | Status | Owner | Resolved by / where |
|---|---|---|---|---|
| G1 | Deadline 2026-11-10 23:55 Kyiv; confirm Kaggle's UTC close time | OPEN | either student | CH-02 §4 |
| G2 | Kaggle metric definition (IoU thresholds, ignore handling, max dets), daily submission limit, number of final selections | OPEN | either student | CH-02 §4 |
| G3 | Sample submission header/format (`sample_submission.csv` is not in the ZIP; download from Kaggle Data tab) | OPEN | either student | CH-02 §4 |
| G4 | Ignore-file format and semantics (relation to COCO `iscrowd`) | OPEN | CH-02 owner | CH-02 §4 |
| G5 | GitHub Pages enabled (Settings → Pages → Source: GitHub Actions); admin only | OPEN | Marko-Zenon | start gate of CH-15 |
| G6 | Nazar's commit email linked to GitHub `Badabuba` | RESOLVED | Badabuba | CH-01: GitHub API maps `mykolaichuk.pn@ucu.edu.ua` commits to login `Badabuba` |
| G7 | PyTorch CUDA wheel with sm_120 support works on the RTX 5070 | RESOLVED | CH-01 owner | CH-01: torch 2.14.1+cu130, CUDA conv + NMS verified |
| G8 | At least one student can upload to Kaggle via the web UI (no tokens in Git) | OPEN | students | needed by CH-05 |
| G9 | RT-DETR trains reliably on 8 GB, otherwise a recorded fallback decision | OPEN | CH-11 owner | CH-11 |
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

Values marked **PENDING CH-02** are filled in by CH-02 and then frozen. Changing a frozen value requires a §11 change-log entry
that a student has approved.

| Item | Value |
|---|---|
| Dataset root | `./multiple-object-video-detection-hw-2-ucu-ai-2026/` with `train/`, `val/`, `test/` and `test_images.csv` directly inside |
| Split | Provided clip-level split: 56 train clips / 7 val clips (0 overlap, verified in the legacy notebook). Test = 48 clips |
| `val_full` | All 2,846 val frames. Every reported headline number uses it |
| `val_fast` | PENDING CH-02 (fixed list, about every 4th val frame, all 7 clips). Used only for per-epoch curves |
| `train_rubric` | PENDING CH-02 (fixed, clip-stratified frame subsample, used for all four-family comparisons) |
| Rubric compute budget | PENDING CH-02. Rule: the same `train_rubric` list, the same number of images seen (epochs × frames) and the same input long side for all compared runs. Report wall-clock and GPU too. Target ≤ 60 min per run on the RTX 5070 |
| Seed | 42 (logged per run) |
| Label map | COCO `category_id` 1–8 (person, bicycle, car, van, truck, tricycle, bus, motor) ↔ submission class 0–7 = `category_id − 1`. **PENDING CH-02** confirmation against G3 and YOLO labels |
| Ignore-region policy | **PENDING CH-02** (G2, G4). It covers (a) the training treatment and (b) the evaluation rule. Masking ignored regions in training images is the leading candidate; the evaluation rule must match the Kaggle metric |
| Prediction interchange format | COCO results JSON: list of `{"image_id": int, "category_id": 1–8, "bbox": [x, y, w, h] in original image pixels, "score": float}`. For test images, `image_id` follows the `test_images.csv` mapping (PENDING CH-02) |
| Evaluator settings | `movd/evaluate.py` (CH-03): COCO bbox, maxDets 500 headline (100 also reported for legacy comparison), mAP, AP50, AP75, AP_S/M/L, per-class AP |
| Submission format | One row per `test_images.csv` image. The predictions column holds space-separated groups `class score x y width height`. ≤ 500 groups. Header PENDING CH-02 (G3) |
| Kaggle | Baseline 0.22163 (public LB). Metric, daily limit and number of final selections PENDING CH-02 (G2) |
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
| CH-02 | — | — | M-DATA | G1, G2, G3, G4 (resolved here) |
| CH-03 | CH-02 | — | M-DATA | — |
| CH-04 | CH-02 | CH-03 | M-DATA, M-GPU | — |
| CH-05 | CH-02 | CH-03, CH-04 or CH-09 | M-DATA, M-GPU | G8 |
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
| CH-16 | CH-01, CH-06, CH-12, CH-14, CH-15 | — | CPU (clean env) | G6, G10 (finish gates) |

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
- **Tue 22:30:** Pages and repo verification done. The remaining time is buffer up to 23:55.

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
Owner: Badabuba | Claimed: 2026-11-07 16:34 Kyiv | Last update: 2026-11-07 16:34 Kyiv | Branch: ch-02-protocol-data | Commits: — | PR: —
Files changed: —
Validation commands: —
Metrics / results: —
Artifacts: —
Runtime / hardware: —
Blockers / deviations: —
Completion or handoff note: —
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

## 12. Definition of done and rubric audit checklist

The project is done when every box is ✓ (or has a written justification), M3 is merged into `main` and tagged, and Pages shows
the final report before 23:55 on 2026-11-10.

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
