# OPT-07 x4_border_confirmation — OPT-03 e2 tile-border drop, confirmed on val_full (frozen before evaluation)

Written and committed **before** x4 was evaluated on val_full or test. Not changed after results are seen.

## What is confirmed

OPT-03 candidate **e2** exactly as frozen in OPT-03 (`results/opt-03-modelA-diagnosis/selection_rule.md`, 528167a):
best of 6 on val_fast (0.2906 vs Model A 0.2892, +0.0014), below the OPT-03 +0.003 gate, never evaluated on val_full.
OPT-07 gives it its single val_full evaluation under the OPT-07 gate. No new parameter, no re-selection.

## Exact deterministic rule

CH-13 merge of Model A's 6 views (full, flip, tile0..tile3 at imgsz 1344; decode top-300, conf 0.001) with one added
step: in each tile view, every box with x0 <= tile_x0 + 0.005 tile_w (if tile_x0 > 0), y0 <= tile_y0 + 0.005 tile_h
(if tile_y0 > 0), x1 >= tile_x1 - 0.005 tile_w (if tile_x1 < W) or y1 >= tile_y1 - 0.005 tile_h (if tile_y1 < H) is
dropped; then class-wise NMS 0.5 and the per-frame top 500. Full and flip views are never changed. Code:
`movd/opt03.py::edge_filter_combine` through `scripts/opt03_edge.py::run_candidate` (both unchanged), called by
`scripts/opt07_loop.py run x4_border_confirmation`.

Frozen parameters (`movd/opt07.py` EXPERIMENTS; asserted equal to `opt03_edge.CANDIDATES["e2"]`):

    {"factor": 0.0, "kind": "opt03_edge", "margin": 0.005}

## Inputs and preconditions

* Model A raw per-view cache `outputs/opt03/raw/val_full.npz` (git-ignored). 4 of 7 val_full clip shards exist from
  OPT-03 (`MOVD-opt03/outputs/opt03/raw/val_full/`, copied, not moved); the 3 missing clips (268, 305, 339; 1,437
  frames) are filled with the unchanged `scripts/opt03_cache.py` on the RTX 5070 (checkpoint sha256 checked by the
  script; the OPT-07 GPU limit of 90 min total applies). A test cache (3,811 frames) is built only if x4 is promoted.
* **Identity precondition** (`scripts/opt07_loop.py identity --split val_full`): the cache merged with factor 1 must
  equal Model A's stored val_full predictions (sha256 `fccda2b3…036c`) exactly (same per-image multiset, max abs
  difference 0). If it does not, x4 is not evaluated and is recorded as not run. Same check on test before any CSV.

## Evaluation and rejection criteria

Evaluated **once** on val_full with the project evaluator against Model A on the same subsets. Rejected if any of
P1–P7 in [`../progress.md`](../progress.md) fails: P1 val_full mAP < A + 0.0005; P2 width <= 1920 subset mAP < A +
0.0005; P3 AP_small loss > 0.003; P4 density >= 100 loss > 0.005; P5 fewer than 4 of 7 clips non-negative; P6
without-4K mAP < A + 0.0005; P7 non-deterministic predictions. If promoted: test raw cache, test identity, one
lossless test CSV, both validators, never uploaded.
