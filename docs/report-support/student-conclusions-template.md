# Student Discussion and Conclusions — Fillable Template

Use this worksheet to draft your own analysis for [report sections 13–14](../../report.md#13-discussion--student-authored) and review the factual limitations in section 12. It does not replace the report or confirm student-authorship gate G10.

**How to use:** replace each bracketed placeholder with your own words; keep a nearby evidence link for every measured claim. Distinguish measured outcomes from possible explanations. Do not copy historical recommendations as current decisions. Remove the prompts only after both students review the text.

**Evidence snapshot:** `origin/develop` at `f89973be7f71b8ce6da86255c80e1839982fa736`, inspected 12 November 2026. Use merged evidence from this snapshot; refresh the sources before final submission. The separate OPT-10 branch is not accepted experiment evidence for this template.

## 1. Discussion of Results

### 1.1 Overall comparison of detector families

*Which detector families were compared under the shared budget? Which differences matter practically? What does the measured ranking support, and what does it leave untested?*

[STUDENTS: Write your interpretation of the cross-family comparison. Separate the fixed-resolution comparison from the approved higher-resolution experiment and later final-pipeline optimization.]

Evidence: [cross-family decision](../../results/ch-12-comparison/decision.md), [selection record](../../results/ch-12-comparison/selection.json), [frozen protocol](../../configs/protocol.json).

### 1.2 Accuracy versus inference speed

*Which trade-off would matter for the intended use? How do additional inference passes change that trade-off?*

[STUDENTS: Explain the measured accuracy/speed trade-off using measurements from the same device, session and timing scope. Keep Marko's RTX 3050 timing separate from RTX 5070 comparisons. Do not assign a speed to a later optimized pipeline without its own measurement.]

Evidence: [common-scope RTX 5070 benchmark](../../results/ch-12-comparison/speed_same_session/summary.json), [original RTX 3050 timing](../../results/yolo_baseline_marko_3050/speed.json), [inference-tuning analysis and timing sources](../../results/ch-13-inference/analysis.md).

### 1.3 Small objects and densely packed scenes

*Which size buckets, classes or clips remain difficult? Do the qualitative examples support the quantitative breakdown?*

[STUDENTS: Interpret the small-object and density results. Explain whether an overall gain also helps the difficult subsets; avoid treating density as an isolated cause when class and scene mix differ.]

Evidence: [EDA numbers](../../results/ch-06-eda/key_numbers.json), [size/density comparison](../../results/ch-11-rtdetr/density.csv), [YOLO small/dense analysis](../../results/ch-10-yolo-tuning/analysis.md), [Faster R-CNN failure cases](../../results/ch-07-frcnn/failure_analysis.md).

### 1.4 Architecture and tuning experiments

*Which hypothesis did each change test? Was the change isolated? Did the result support that hypothesis within this budget?*

[STUDENTS: Discuss the changes you consider most informative: proposals and sampling, anchors and focal loss, resolution and P2, queries and learning rate, or inference-time composition. Link the measured outcome of each chosen experiment.]

Evidence: [Faster R-CNN protocol](../../results/ch-07-frcnn/protocol.md), [RetinaNet results](../../results/ch-08-retina/results.md), [YOLO tuning](../../results/ch-10-yolo-tuning/analysis.md), [RT-DETR analysis](../../results/ch-11-rtdetr/analysis.md), [inference selection rule](../../results/ch-13-inference/selection_rule.md).

### 1.5 Unexpected and unsuccessful results

*Which expectation did a result contradict? Which competing explanations remain possible? What additional controlled experiment would distinguish them?*

[STUDENTS: Choose a few informative negative results and explain them in your own words. Keep observations separate from untested causes. Treat the earlier confounded RT-DETR learning-rate run as confounded, not as isolated tuning evidence.]

Evidence: [confounded-run record](../../results/rtdetr_lr1e4/README.md), [final-training comparison](../../results/ch-14-final/selection.json), [authorized early-checkpoint extension](../../results/ch-14b-rtdetr-short-finetune/selection.json), [optional experiment index below](#6-reproducibility-and-evidence).

### 1.6 Reliability of the evaluation

*What was validated about the evaluator? What cannot be established from these experiments? How independent were tuning and final evaluation?*

[STUDENTS: Explain the strengths and limits of the evidence: clip-level splits, ignore handling, checkpoint selection rules, repeated validation use and the distinction between internal AP and Kaggle public scores.]

Evidence: [evaluator sanity checks](../../results/ch-03-evaluator/sanity.json), [ignore-policy verification](../../results/ch-02-protocol/data_facts.json), [discovery/confirmation split](../../results/opt-08-overnight/split.json), [robustness checks](../../results/opt-08-overnight/robustness/summary.json), [submission log](../../results/submissions.csv).

## 2. Personal Observations

Each student completes their own subsection. Verify attribution using the evidence logs and commit history; this template does not assign a division of work.

### 2.1 Marko-Zenon's observations

**My contribution.** *Which implementation, execution, verification or documentation work did you personally perform?*

[MARKO: Describe your contribution in your own words and link the relevant commits or evidence.]

**My experimental observations.** *Which result or failure did you inspect directly?*

[MARKO: Write your own observation and identify its source.]

**What surprised me.** *Which expectation changed after you examined the evidence?*

[MARKO: Write your own reflection, or explain why no result was unexpected.]

**What I learned.** *Which technical lesson can you explain and defend from your own work?*

[MARKO: Write your own lesson and distinguish it from a general claim about all detectors.]

### 2.2 Nazar's observations

**My contribution.** *Which implementation, execution, verification or documentation work did you personally perform?*

[NAZAR: Describe your contribution in your own words and link the relevant commits or evidence.]

**My experimental observations.** *Which result or failure did you inspect directly?*

[NAZAR: Write your own observation and identify its source.]

**What surprised me.** *Which expectation changed after you examined the evidence?*

[NAZAR: Write your own reflection, or explain why no result was unexpected.]

**What I learned.** *Which technical lesson can you explain and defend from your own work?*

[NAZAR: Write your own lesson and distinguish it from a general claim about all detectors.]

Contribution sources: [task and evidence logs](../../plan.md), [CH-09 execution record](../../results/yolo_baseline_marko_3050/README.md), [CH-09 integration and transfer record](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/10). Add specific commit links for other contributions when writing.

## 3. Limitations

Do not repeat the existing limitations paragraph verbatim. Decide which limitations materially constrain your conclusions and refer back to report section 12.

- **Dataset:** *How do small targets, occlusion, class imbalance, correlated frames and resolution/scene shifts affect the interpretation?* [STUDENTS: State the limitations supported by EDA and failure evidence.]
- **Hardware and budget:** *Which conclusions depend on the short training schedule, resizing geometry or benchmark scope?* [STUDENTS: Explain the relevant resource constraints without comparing unlike FPS measurements.]
- **Experimental uncertainty:** *Which runs share a seed, which have a replicate, and what uncertainty remains unmeasured?* [STUDENTS: Describe the actual replication evidence; do not invent confidence intervals or statistical significance.]
- **Evaluation:** *How much validation reuse occurred? What remains unknown about the hidden scorer and generalization?* [STUDENTS: State the limits of internal validation, public scores and subset robustness checks.]

Evidence: [existing report limitations](../../report.md#12-limitations-and-reproducibility-ch-16), [protocol](../../configs/protocol.json), [YOLO replicate record](../../results/ch-10-yolo-tuning/selection.json), [optional optimization limitations](../../results/opt-08-overnight/README.md).

## 4. Final Conclusions

### 4.1 Main project outcome

[STUDENTS: State the final accepted pipeline, its purpose and the strongest supporting evidence in your own words. Confirm the final selection and CH-14 closeout before calling the project complete.]

### 4.2 What worked and what did not

[STUDENTS: Summarize the most useful successful and unsuccessful hypotheses. State the tested conditions; do not infer a general architectural law from this dataset and budget.]

### 4.3 Most important technical lessons

[STUDENTS: Agree on the lessons you can both defend, supported by specific experiments or verification records.]

### 4.4 Possible improvements and future work

[STUDENTS: Propose a small number of motivated next experiments. Explain what uncertainty each would address and label all proposals as untested.]

Final-outcome sources: [accepted final-training evidence](../../results/ch-14-final/README.md), [accepted extension](../../results/ch-14b-rtdetr-short-finetune/README.md), [optional optimization evidence](../../results/opt-08-overnight/summary.csv), [latest committed submission log](../../results/submissions.csv).

## 5. Joint Review Before Integration

- [ ] Each student wrote and reviewed their own observations; contributions are accurately attributed.
- [ ] Discussion and conclusions reflect both students' interpretation, with evidence beside factual claims.
- [ ] Fixed-budget family comparison, later training and optional optimization are clearly separated.
- [ ] Benchmark device, timing scope and session are stated wherever speed is compared.
- [ ] Current selections are reconciled with the submission log; user-reported Kaggle evidence is identified as such.
- [ ] Unsuccessful/confounded experiments and uncertainty are described honestly.
- [ ] CH-14 decisions, CH-16 audit and the extended-deadline gates are checked against the latest board.
- [ ] Both students approve the final prose before it is placed into the report; G10 is confirmed separately through the repository procedure.

## 6. Reproducibility and Evidence

Consult the existing sources rather than duplicating their tables here. Check the snapshot date of each narrative README: later committed records may supersede its upload or selection recommendations.

| Writing topic | Evidence to consult |
|---|---|
| Dataset, preprocessing and evaluator | [EDA](../../results/ch-06-eda/key_numbers.json), [protocol](../../configs/protocol.json), [sanity checks](../../results/ch-03-evaluator/sanity.json) |
| Training curves and failure figures | [cross-family learning curve](../../results/ch-11-rtdetr/figures/learning_curve.png), [Faster R-CNN failures](../../results/ch-07-frcnn/failure_cases.csv), [YOLO failures](../../results/yolo_baseline_marko_3050/failure_cases.csv), [RT-DETR failures](../../results/rtdetr_baseline/failure_cases.csv) |
| Accuracy and speed comparison | [CH-12 decision](../../results/ch-12-comparison/decision.md), [same-session timing](../../results/ch-12-comparison/speed_same_session/summary.json), [CH-13 comparison](../../results/ch-13-inference/comparison.csv) |
| Final training and authorized extension | [CH-14 comparison](../../results/ch-14-final/comparison.csv), [CH-14 decision](../../results/ch-14-final/selection.json), [CH-14b comparison](../../results/ch-14b-rtdetr-short-finetune/comparison_stageA_epoch1.csv), [CH-14b decision](../../results/ch-14b-rtdetr-short-finetune/selection.json) |
| Earlier optional experiments | [OPT-02](../../results/opt-02-rtdetr-ensemble/README.md), [OPT-03](../../results/opt-03-modelA-diagnosis/README.md), [OPT-04](../../results/opt-04-model-scale/README.md), [OPT-05](../../results/opt-05-small-specialist/README.md), [OPT-06A](../../results/opt-06a-c4-valfull/README.md) |
| Later optional experiments | [OPT-07](../../results/opt-07-overnight/README.md), [OPT-08](../../results/opt-08-overnight/README.md), [OPT-09](../../results/opt-09-e6-tuning/README.md) |
| Reproduction and traceability | [runbook](../../results/ch-09-yolo/GPU_RUNBOOK.md), [final training config](../../configs/runs/final_rtdetr.json), [inference config](../../configs/runs/final_inference.json), [OPT-08 rules and configuration references](../../results/opt-08-overnight/TASK.md), [report validation record](validation.md) |
| Submission provenance and remaining gates | [append-only submission log](../../results/submissions.csv), [live board and acceptance criteria](../../plan.md) |

### Snapshot notes for the authors

These are coordination facts, not student conclusions:

- CH-01–CH-13 and CH-15 are marked DONE. CH-14 code/evidence PRs [#18](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/18) and [#19](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/19) are merged, but the live board still marks CH-14 PARTIAL / PAUSED. CH-16 is BLOCKED and its final audit is not recorded as complete.
- The committed submission log records OPT-08 E6 and E4 as accepted and selected for Kaggle final submission **as reported by the student**. This is not independent authenticated verification or a CH-14 closeout. The accidental validation-file upload remains explicitly excluded as a candidate; its acceptance-criterion treatment still needs the students' recorded decision.
- Report sections 2–9 contain dataset/protocol, family metrics, curves and comparison evidence; section 10 contains CH-13 inference evidence. Sections 10–11 still have historical CH-14 placeholders and lack the later accepted submission/optimization results. Sections 13–14 remain student-authored TODOs. The defense guide also predates the later final-training/optimization evidence. Do not treat these historical placeholders as the latest remote status.
- G10 remains open. G11 (exact extended competition deadline) and G12 (daily submission-limit recheck) also remain open. The earlier deadline in the report is superseded by the extension recorded in the board's change log; do not reuse it as the current cutoff.
- `main` remains at `fec11cbc0150fd410356f2973d5939750ea5e4f8`. The latest successful [Pages deployment](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/actions/runs/42507083884) serves that milestone; the public report's bytes were checked against `origin/main:report.md`. Later completed work on `develop` is not yet published there.
- The existing report validator stops at its historical byte-identity assertion for `results/submissions.csv` against evidence commit `5ec9d1f14a9f9aa9e16a4e56c3113dbf3aad3633`: the log has legitimate later committed additions. Refreshing the final report/traceability audit is separate work; this template does not change or bypass that validator.

Before integrating completed prose, refresh this snapshot, reconcile the historical report placeholders with accepted evidence, and review the final audit/deployment requirements together.
