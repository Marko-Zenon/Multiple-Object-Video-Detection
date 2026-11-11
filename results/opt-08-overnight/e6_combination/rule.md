# OPT-08 e6_combination: E6 — combination of the independently passed components (1 candidate)

Written and committed **before** E6 was evaluated on any split (TASK.md E6 rule, fixed at 0e73d20). Not changed after
results are seen.

## Components (each passed S1, S2, S3 and U1-U8 on its own against x2)

* weights: E4 `s1` soup w080 (0.8 x Model A + 0.2 x epoch1.pt, `soup_build.json` sha256 `2d60d3fa…d939`);
* border: E1's x4 step (OPT-03 e2: tile-view boxes within 0.005 tile side of an inner tile edge dropped before the
  CH-13 merge), applied to the soup's raw views with the unchanged `opt03.edge_filter_combine(factor 0, margin 0.005)`;
* temporal rule: E3 `b3` (`five_frame_mean`), the only passed replacement of x2 (val_full +0.0006 vs x2; E2 failed S2).

E6 = weights -> border -> b3, in that order (TASK.md). Rejected E2 is not used. E5 is not included (not evaluated
when this rule was written). Single candidate: S1 is a gate (discovery >= x2 + 0.0003), then S2 and S3 / U1-U8 as
TASK.md, U7 = fresh-process CPU rerun (the soup's GPU determinism is already shown in `../e4_checkpoint_soup/gpu_recheck_w080.json`).

Frozen parameters (`movd/opt08.py` EXPERIMENTS):

    k1 {"factor": 0.0, "kind": "combo", "margin": 0.005, "mode": "five_frame_mean", "tag": "w080"}

Comparison baseline: OPT-07 x2 (val_full `6c3a929f…2c03`, public 0.2805168313). CPU only (the soup raw caches of E4).
