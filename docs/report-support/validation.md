# Report draft validation and handoff

Evidence base: `81ddfee79671ae4b38669007ef5070af2ab3f9a0` (`origin/develop` at worktree creation). Branch `codex/report-deadline`; independent documentation task authorized by Marko, with no task claim or board edit. Only report.md and docs/report-support/ are permitted changes.

## Numerical traceability

- [Metric-cell registry](metric_trace.json): each new AP table cell points to its committed metrics.json field, rounded to four decimals. Existing Faster R-CNN AP, speed/latency and per-class tables are retained and independently checked.
- Dataset counts, class shares, sizes, track overlap and subsampling coverage: results/ch-06-eda/key_numbers.json (split_overview, class_targets, size_by_split, size_after_resize, redundancy, subsampling). Per-sequence observations: results/ch-06-eda/tables/per_sequence.csv.
- Budget, seed, class map, cap, IoU grid: configs/protocol.json; actual batch/resize/AMP/augmentation: results/<run>/config.json, plus movd/rtdetr.py and the CH-11 speed scope. The actual RT-DETR square stretch is called out rather than repeating the stale letterbox assumption in the frozen config text.
- Evaluator controls: results/ch-03-evaluator/sanity.json, checks.{gt_as_predictions_val_full,gt_plus_ignore_region_dets_val_fast,control_crowd_removed_val_fast,label_shift_val_fast}.
- Faster R-CNN changes and selection: results/ch-07-frcnn/protocol.md; three metrics.json files. RetinaNet changes/selection: results/ch-08-retina/{preflight.md,results.md,comparison.csv} and run configs/metrics. No statistical-significance claim is made.
- YOLO final checkpoint and epoch-two peak: results/yolo_baseline_marko_3050/{README.md,curve.json,metrics.json}. Speed 36.37 / 27.49: sibling speed.json, explicitly RTX 3050.
- RT-DETR baseline and valid negative variants: results/rtdetr_{baseline,q500,lr1e4_wb0}/metrics.json. Confounded run: results/rtdetr_lr1e4/README.md, never used as isolated tuning evidence. Curve superiority is checked for each matched epoch against every completed CNN run in results/ch-11-rtdetr/curves.csv.
- Same-session RTX 5070 speeds 64.21 / 10.60: results/ch-11-rtdetr/speed_recheck/{yolo_baseline_marko_3050,rtdetr_baseline}.json. The comparison excludes different-session torchvision timing scopes.
- Submission values and counts: results/submissions.csv; results/ch-05-submission/frcnn_rpn_pre3000_e6_test/validation.txt. Library versions/weight identifiers: recorded hardware/config/README files, requirements.txt.
- Deadline: [official-page observation and normalization](deadline.md). No authenticated account verification or separate later selection deadline is claimed.

## Checks

Run from repository root with an existing Python environment containing Pillow:

```bash
python docs/report-support/validate_report.py
git diff --check
```

The validator is read-only: no model imports, GPU access, artifact regeneration or notebook execution. It verifies metric cells, source values, matched-epoch curves, local Markdown links, image decoding/identity, table columns, metadata, timezone conversion and the documentation-only working diff. It also supports a clean committed branch; review the full PR diff separately for its allowed paths.

The previous session's /tmp/validate_ch15.py is a local historical helper, not a tracked repository tool. Its report/link/numerical checks are rerun separately; its old assertion requiring plan.md and a new image is intentionally inapplicable to this documentation-only task. No tracked report/link validation CLI exists in this evidence base. The new self-contained validator preserves and extends those checks.

Validation PASS on 2026-11-09: 72 newly traced metric cells, all retained Faster R-CNN AP/per-class/speed cells, EDA/protocol/speed/convergence controls, 59 internal links, image decoding and source identity, table shape, front matter, timezone conversion, and documentation-only scope. Historical CH-15 report/link/numerical checks also PASS (obsolete board-edit assertion excluded). Local preview through the unchanged Pages renderer shows 16 section headings, eight tables, both images loaded, and zero broken contents anchors. No model execution or GPU work. Remote reference/paper links are source citations; internal paths and rendered images are checked locally. No claim of exhaustive external-URL availability is made.

## Merge hold and remaining work

Keep the PR draft and do not merge until Claude's CH-12 work is merged, this branch is updated from develop, conflicts are resolved, and numerical/link checks are rerun. The current RT-DETR validation lead is provisional, not final selection.

Only report TODOs: CH-10 final tuning/size/density results; CH-12 finalist selection; CH-13 inference tuning; CH-14 final training/submission/selection; final student discussion and conclusions in Nazar and Marko's own words. G10 is not satisfied by this AI-assisted working draft. The final clean-environment audit and Pages deployment remain later project checks, not completed by this PR.
