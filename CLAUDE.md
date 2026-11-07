# CLAUDE.md — working rules for this repository

These rules apply to every agent: Nazar (`Badabuba`), Marko (`Marko-Zenon`), Claude sessions and Codex sessions.
The tasks, the live board and the detailed specs are in `plan.md`. This file only says how to work.

## 1. Purpose and sources of truth

- **Project:** Multiple Object Video Detection (VisDrone-VID, 8 classes), UCU AI 2026, homework 2. It covers four detector
  families, a final pipeline, a Kaggle submission and a report on GitHub Pages. Deadline: **Tue 2026-11-10 23:55 Kyiv**.
- **`origin/develop` is the live source of truth** for task state. Never act on a stale local `plan.md`.
  Always fetch and read the board on the latest `origin/develop` first.
- Technical claims must come from code plus stored artifacts in `results/`. Do not fabricate metrics, runs,
  leaderboard scores or conclusions. If something is unknown, write "unknown" or name the open gate from `plan.md`.

## 2. Session start (mandatory, also after any break)

1. `git fetch origin`, `git switch develop`, `git pull --ff-only`. If the working tree is not clean, stop and report.
2. Read this file and `plan.md` §1: the board, the notebook lock and the gates.
3. Review `🔵 IN_PROGRESS`, `🟠 PARTIAL / PAUSED` and `🟡 BLOCKED` rows, their timestamps and branches, and open PRs (`gh pr list --base develop`).
4. Read the full spec of the chunk you were asked to do (`plan.md` §10).
5. Check its machine prerequisites. M-DATA can be checked with `python scripts/check_dataset.py` once CH-01 has added that script.

## 3. Claiming work (atomic claim, full procedure in `plan.md` §9.2)

- Claim only a row whose state is `🟢 READY` **and** whose `Start now?` is `🟢 YES`.
- **Push the claim to `origin/develop` before writing any implementation code.** The successful push is the lock.
  A local edit or a local commit is not a claim. A failed claim push means you do not own the chunk.
- The claim is made on a throwaway branch: `git switch -c claim/CH-XX-<login> origin/develop`. Edit only that row.
  Commit `coord(CH-XX): claim <title> for <login>`, then `git push origin claim/CH-XX-<login>:develop`.
- If the push is rejected, follow `plan.md` §9.2. Preserve all work and inspect `origin/develop`.
  If the same chunk was claimed, **stop and report**. If the changes are unrelated, resync with
  `git merge --no-edit origin/develop`; on conflict run `git merge --abort`, then stop and report.
- Never work on another owner's `IN_PROGRESS` or `PARTIAL / PAUSED` chunk. Never take over a stale row
  (no update for about 6 h) without contacting the owner or user and recording the handoff.
- **Stop and ask** if the state, ownership, dependencies or gates are ambiguous or inconsistent.
- One active chunk per person. A second CPU/documentation chunk is allowed only while the first waits on a long GPU
  job or an external action. Record this in both rows.
- No silent scope expansion. Bonus work (tracking, temporal models, optical flow, Grad-CAM, attention maps, W&B,
  custom architectures) is out of scope.

## 4. Legal state transitions (also in `plan.md` §9.1)

- `🟡 BLOCKED → 🟢 READY`: only when every **start** dependency is `✅ DONE` and every start gate is resolved.
- `🟢 READY → 🔵 IN_PROGRESS`: only through the atomic claim.
- `🔵 IN_PROGRESS → ✅ DONE`: only when all acceptance criteria pass, the finish dependencies are DONE, and the work is
  **merged into `develop`**. Written code or a finished training run is never enough.
- `🔵 IN_PROGRESS → 🟡 BLOCKED`: a real blocker is found. Push the board update immediately, naming the blocker and the next action.
- `🔵 IN_PROGRESS → 🟠 PARTIAL / PAUSED`: useful incomplete work is pushed and recorded, and work stops.
- `🟠 PARTIAL / PAUSED → 🔵 IN_PROGRESS`: only through an explicit claim by the recorded owner, or a recorded handoff.
- `🔴 FAILED / NEEDS_DECISION → 🟢 READY`: only after the failure is recorded and a student approves a revised approach.
- Never mark a row READY while a hard (start) dependency is not DONE.
- **Unblock recalculation:** when you mark a chunk DONE, re-check every row that depends on it. Flip a row to READY only if all of
  its start dependencies are DONE and its gates are resolved.

## 5. Git workflow

- `main` holds stable milestones only and is the Pages source. `develop` is the integration branch and live board.
  `ch-XX-<slug>` is one task branch per chunk, created from the latest `origin/develop`.
- Task PRs go into `develop`, never into `main`. Milestones reach `main` only through `develop → main` PRs (`plan.md` §8).
- Never push implementation work to `main`.
- Never force-push, rebase or amend pushed commits, `reset --hard`, `clean -f`, or use `--no-verify`.
  Destructive commands are never run automatically; only a human may run them, explicitly.
- Keep a task branch current with `git merge origin/develop`. When resolving `plan.md` conflicts, keep every other
  row's latest state.
- Commit messages: `CH-XX: <what>` for work, `coord(CH-XX): <what>` for board-only commits. PR titles start with `CH-XX`.
- Each commit is authored by the student whose session made it, under their GitHub-linked email. Never commit as
  someone else. Do not add a `Co-Authored-By` naming the other student unless they contributed to that commit
  and explicitly approved it.
- Both students must end up with meaningful, attributed technical commits.

## 6. Progress hygiene

- Push a heartbeat (`coord(CH-XX): heartbeat …`, updating only your row's `Last update` and a short note) about every 4 h
  of active work, and before any long break.
- Blocked or paused? Push the board update right away: the state, the exact blocker (chunk ID or gate), the branch,
  the preserved artifacts and the next action.
- Completion: run every acceptance check, fill the chunk's evidence log, merge `origin/develop` into the task branch,
  set the row to DONE and recalculate dependents in the same PR, merge the PR into `develop`, then confirm the board on `origin/develop`.

## 7. Shared files, the notebook lock and hard prohibitions

- **`homework.ipynb` lock:** edit the notebook only while the lock line under the board names you, and take the lock
  with a pushed `coord` commit. Pull, edit briefly, commit, push and release in the same session.
  Never hold the lock during a GPU run.
  Keep the legacy cells and their outputs. New sections are self-contained (their own imports) and load numbers from `results/`.
  Training never runs inside the notebook.
- `plan.md`: edit only your own board row and your own chunk spec or evidence log. Keep board rows on one line each.
- `requirements.txt`, `README.md`: edit only your own lines or section. `movd/data.py` (owned by CH-02) and `movd/evaluate.py`
  (owned by CH-03) change after their chunk only through a reviewed PR.
- **Figures** go to `results/<run_id or ch-XX-slug>/figures/`, never to `images/`. Only CH-15, CH-16 and the students write
  `images/`, copying selected figures from `results/` and keeping the source path.
- **Never edit** `index.html`, `assets/`, `.github/` or `.nojekyll`. The template rule says this annuls the grade. For the website,
  only `report.md` and `images/` may change.
- **Never commit** the dataset, the ZIP, `outputs/`, checkpoints (`*.pt`, `*.pth`), raw predictions, submission CSVs,
  YOLO-format exports, caches, `kaggle.json`, tokens or any secret.
- Never move, delete or re-extract the dataset ZIP without explicit user approval.

## 8. Technical rules

- Code layout: `movd/` holds the importable package (needed for Windows DataLoader workers), `scripts/` the CLIs,
  `configs/` the protocol, splits and run configs, `tests/` the pytest tests, and `results/<run_id>/` the small tracked evidence.
  `outputs/` (ignored) holds checkpoints, raw predictions and exports.
- Use relative `pathlib` paths from the repo root. The dataset lives at `./multiple-object-video-detection-hw-2-ucu-ai-2026/`.
- Use the **frozen protocol** in `plan.md` §4 (filled by CH-02): the provided clip-level train/val split, the fixed `val_fast`
  and `val_full` lists, the label map, the rubric compute budget and seed 42.
- Use **the ignore-region policy frozen by CH-02**, for both training and evaluation. Until CH-02 is DONE there is no frozen
  policy, and no chunk may hard-code one.
- All headline numbers come from **the project evaluator** (`movd/evaluate.py`, CH-03), whatever the framework.
  Ultralytics' or torchvision's internal metrics are never headline numbers.
- Before any expensive run, do a smoke test of about 50 iterations. GPU jobs must be resumable (per-epoch checkpoints in `outputs/`).
- Every run writes `results/<run_id>/`: config, metrics, curve, speed, hardware and wall-clock, with figures in `figures/`.
  Failed and negative runs are kept and marked, never deleted or overwritten.
- Validate every submission with the project validator before upload. Log every upload and its Kaggle score in
  `results/submissions.csv`.
- Write readable code that each student can explain at an individual defense. Preserve valid existing work instead of rewriting it.

## 9. Integrity and the report

- AI assistance with code is allowed. **The report prose and conclusions are written by the students.** Claude/Codex may
  produce evidence tables, figures, outlines with TODO prompts, and number-traceability and consistency checks.
  They must not write the report's conclusions as if a student wrote them.
- Cite torchvision, Ultralytics (AGPL-3.0), pycocotools, the pretrained weights and the original papers. Record exact
  package and weight versions.

## 10. Definition of done

A chunk is done only when it meets the state rules in §4. The project is done when every item of the checklist in `plan.md` §12
is ✓ or explicitly justified, the final `develop → main` merge is made, and the freeze happens before the deadline.
