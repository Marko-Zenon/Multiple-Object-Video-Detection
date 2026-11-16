# Independent preflight audit — 2026-11-09

Accepted-code/evidence base: `5ec9d1f14a9f9aa9e16a4e56c3113dbf3aad3633` (merged CH-13 PR #17). Latest coordination recheck: `164a525878f6515c53fe4d3c38b81b5361d643fb` (CH-14 claimed by Badabuba); the only intervening upstream change is plan.md, so CPU checks still cover the current code. This is independent documentation/CPU validation authorized by Marko. **It is not a CH-16 claim, completion record or repository freeze.** CH-14 is reserved for Nazar and was neither claimed nor modified. The board remains untouched.

## State and integration

- CH-09, CH-10, CH-11, CH-12, CH-13 and CH-15: DONE on the inspected develop board. Merged task PRs: #10, #14, #12, #15, #17; scaffold #8/M1 #9.
- CH-14: initially READY at the accepted-code snapshot; now IN_PROGRESS, owned by Badabuba on the final coordination recheck. Both documentation branches preserve that upstream update. Do not take it over or change its state.
- CH-16: BLOCKED on CH-14. G10 student-authored report sign-off remains open. It cannot be claimed or marked DONE now.
- Documentation PRs #13 and #16 are updated from develop using ordinary merges; no history rewrite. At inspection both were draft, without review, acceptance comments or approvals. plan.md §9.5 explicitly allows author self-merge after checks; these documentation-only PRs do not touch shared modules requiring peer review. The CH-12/update gate is satisfied. Integration is permitted under the current user authorization after final validation; student G10 sign-off remains a final-delivery gate, not a claim that these factual drafts are student conclusions.

## Checks performed

| Check | Result and scope |
|---|---|
| CPU tests | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests --basetemp /tmp/movd-finalization-pytest`: **178 passed, 14 skipped**, 20.87 s. Shared data/evaluation, submission, Ultralytics, P2, aggregation and inference tests included. Worktree has no dataset/weights; GPU was hidden. Skips are not GPU validation. |
| Real dataset | `python scripts/check_dataset.py --root <existing dataset>`: PASS, train 24,201/56 clips, val 2,846/7, test 3,811/48; annotations, labels, ignore structure and CSV image paths checked. No extraction, modification or relocation. |
| Stored-prediction reproduction | Current project evaluator on the original CH-09 val_full prediction JSON, GPU hidden, outputs only under /tmp: mAP **0.14389177166432673**, maximum difference across six AP metrics **0.0**, per-class and density records identical. No model loaded or inference run. Existing environment only; not fresh-env CH-16 reproduction. |
| Frozen splits | `python scripts/make_splits.py --check --root <existing dataset>`: PASS; train_rubric 2,800 / val_fast 714 / val_full 2,846, regenerated in memory and committed SHA-256 identical. |
| Final-plan sample list | Read-only every-fourth train-frame computation: 6,072 IDs; SHA-256 `68bc424a08eeb68460bab09cbdbf90b53f8e4e9b80eb3b1d1e3c8770d2add42f` matches final.json. No export/manifest/config written; no CH-14 execution. |
| Notebook | nbformat schema PASS; 104 cells, 53 with outputs; unique nonempty IDs, zero stored error outputs; ordinary Python-cell AST syntax PASS. No cells executed, no notebook edited, no lock taken. This does not prove a complete clean-kernel run. |
| CH-09 local source manifests | Original primary checkout: **145/145** artifact and **163/163** supplemental transfer entries exist and match SHA-256. This verifies the frozen transfer snapshot, not that old source files should replace newer develop files. |
| Preserved CH-09 ZIP | 755,266,717 bytes, SHA-256 `f95fb52db46961eebe9669c7de86b5418d3ba9e3ae95df9b1de5ba532f65b3f0`, unchanged. Read only; no repack/upload. CRC was verified during original packaging, not repeated here. |
| Other checkpoint provenance | CH-12 summary records checkpoint hashes and correspondence with earlier receipts; CH-13 `validated_with.sha256` matches the CH-12 RT-DETR hash `da5d36258918e7fdddcb9cfe6e87f6a78ef8ee1a04e65deb3c0eedb42fb6e872`. Those checkpoints are on Nazar's machine; no claim of local rehash or new inference. |
| CH-13 identity/round-trip | Stored reference equals CH-11 metrics. Initial strict round-trip is **failed** at about 9.9e-6 vs 1e-6 because of clipping; applying equal clipping yields exactly 0.0 difference. Tests and docs preserve both facts. |
| Report | Read-only validator verifies every AP table cell from its linked JSON, legacy tables, EDA/protocol/selection/speed controls, common-session FPS, inference settings, links, images, metadata and documentation-only working scope. No results are regenerated. |
| Git safety | No tracked weights, ZIP, NPZ, dataset, outputs/, submissions/ payload or kaggle.json; no matches for checked credential/private-key patterns. The only tracked file above 10 MiB is the existing notebook (17,733,398 bytes). Pattern scan is not a full secret audit. |
| Attribution | GitHub contributors include **Badabuba** and **Marko-Zenon**. New commits use Marko's configured identity, no Co-Authored-By. |
| Pages | Latest Deploy report run **42507083884** succeeded at main `fec11cbc0150fd410356f2973d5939750ea5e4f8`; public `/report.md` returned HTTP 200. This is the old M1 scaffold, not deployment of these unmerged documentation drafts. |

## Environment and reproducibility limitations

Existing `.venv` was reused; no packages installed or changed. `pip check` is **not clean**: inherited unrelated system-site packages report missing Flask/Flask-SQLAlchemy, arrow, proton-vpn-local-agent, cffi, pandas-stubs and referencing, plus invalid pybind11 metadata. The historical CH-09 environment log already records unrelated inherited conflicts. A fresh project-only environment is still required for CH-16; passing CPU tests is not a clean-environment reproduction claim.

README still has legacy notebook-centric instructions, v0 training settings, a truncated sentence in section 6 and direct-to-main template wording. Use the live CLAUDE/plan workflow and committed runbooks, not those historical sections. The blocked CH-16 owner must reconcile README with the module/CLI pipeline and final evidence. This audit records the discrepancy without taking the blocked chunk or altering shared instructions.

The final.json runtime projection predates the chosen six-pass CH-13 pipeline: its old test-time estimate is not the CH-13 estimate. Nazar must budget from the actual selected pipeline and final checkpoint; no config was changed here.

## Required handoff before final delivery

1. **Nazar / CH-14 only:** preserve the now-recorded ownership; smoke/resume the final data path, train/evaluate or document the accepted fallback; keep all checkpoint provenance. Implement/verify the train_final manifest handling noted in final.json if still needed. Compare final data strategy with the rubric checkpoint using val_full.
2. **Nazar / CH-14 only:** apply frozen final_inference settings to the actual submitted model, validate each test CSV, record its SHA-256 and accepted Kaggle score. Obtain a score above the exact recorded public baseline 0.2216324580 and document final two selections primarily from internal validation. The first score 0.2011538266 does not satisfy this requirement.
3. **Both students:** review report facts and defense guide; write the final discussion/conclusions in their own words and explicitly confirm G10. Review the integrated factual documentation and complete G10 before final delivery; no personal observations or conclusions are fabricated.
4. **CH-16 after eligibility:** atomic claim, fresh-environment reproduction, complete rubric/traceability audit, README reconciliation, selected figure pack, final develop-to-main M3 and tag `v1.0-final`, successful matching Pages deployment. No final milestone or tag is created by this preflight.
5. Respect verified Close **2026-11-10 23:55 Europe/Kyiv / 21:55 UTC** and the earlier repository freeze target. No late pushes, backdating or history manipulation.

Independent checks do not close CH-14, CH-16, G10 or the baseline-beating submission requirement. Raw predictions/checkpoints and final-training results remain external to Git and under their owners' control.

## Publication outcome

The author-self-merge rule and upstream CH-12/update gate were checked, but automatic approval review rejected the attempted `gh pr ready 13` operation because the earlier draft-only instruction was considered unresolved. That command did not execute. No ready/merge action was attempted by another route. PRs #13 and #16 remain draft pending explicit user confirmation; title/body updates were authorized and applied. This approval blocker is separate from student G10 and CH-16 dependencies.

### Explicit publication authorization — follow-up

Marko explicitly authorized converting PRs #13 and #16 from Draft to Ready for Review and merging both into develop, superseding the earlier draft-only restriction. The historical approval blocker above is therefore resolved. A fresh develop fetch and documentation/numerical/link validations passed; GitHub reported both PRs CLEAN/MERGEABLE, with no checks or review decision pending. Use the PR event records for the resulting merge commits. This permission does not cover CH-14 execution or prematurely claiming CH-16/G10.
