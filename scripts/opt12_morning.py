"""OPT-12 morning pipeline: selection, caches, gates, CSVs, validators, hand-off (started by scripts/opt12_night.py).

Frozen rule: results/opt-12-regularized-finetune/selection_rule.md and movd/opt12.py. No network, no upload.

    python scripts/opt12_morning.py run                 # orchestrator (GPU queue + CPU queue, artifact-driven)
    python scripts/opt12_morning.py baseline            # c1 rebuilt from its caches: must equal the protected c1
    python scripts/opt12_morning.py select              # GPU: single-view val_fast curve, AVG3, N-star, W(alpha)
    python scripts/opt12_morning.py cache TAG STAGE     # GPU: 6-view raw cache (discovery / confirmation / test)
    python scripts/opt12_morning.py evalcand TAG        # CPU: c1 pipeline on val_full + subsets + frozen gate
    python scripts/opt12_morning.py ens                 # CPU: C3 grid on discovery, gate on val_full
    python scripts/opt12_morning.py csv NAME            # CPU: test predictions -> CSV -> both validators
    python scripts/opt12_morning.py rebuild NAME        # CPU: fresh-process rebuild, sha256 must match
    python scripts/opt12_morning.py buildw070           # CPU: fallback soup 0.70 x Model A epoch0 + 0.30 x epoch1
    python scripts/opt12_morning.py handoff             # HANDOFF.md from the stored records

Tags: w075 (c1, existing caches, read only), nstar, wstar, w070, salvage. Candidates: C1 = nstar, C2 = wstar (or salvage),
C3 = ensemble(c1, nstar), C4 = w070. The c1 pipeline (E6 parameters with the c1 weights) is used unchanged.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from movd import opt12  # noqa: E402
from movd.opt12 import EVID, OUT, SUB, event, load_json, now, save_json, sha256, stamp  # noqa: E402

PY = sys.executable
HW2 = REPO.parent
RUN_ID = "opt12_reg_ft"
WEIGHTS_DIR = REPO / "outputs" / RUN_ID / "train" / "weights"
SOUP_DIR = OUT / "soup"
REG = EVID / "registry.json"  # tag -> {"path", "sha256"}
SEL = EVID / "selection.json"
BASE = EVID / "baseline_c1.json"
C1_PREDS = {"val_full": OUT / "preds" / "w075" / "val_full.json", "test": OUT / "preds" / "w075" / "test.json"}
W075_RAW = {"discovery": HW2 / "MOVD-opt09/outputs/opt09/soup/raw/w075/discovery",
            "confirmation": HW2 / "MOVD-opt09/outputs/opt09/soup/raw/w075/confirmation",
            "test": HW2 / "MOVD-opt11/outputs/opt11/raw/w075/test"}
# results/opt-09-e6-tuning/c_final/c1/val_full/metrics.json (exact). results/submissions.csv lists 0.29739905 for c1,
# which disagrees in the 7th decimal with this stored metric (transcription error there; recorded in the OPT-12 README).
C1_VAL_FULL_REF = 0.2973993546311055
CSV_NAMES = {"C1": "opt12_c1_nstar_test.csv", "C2": "opt12_c2_wiseft_test.csv", "C3": "opt12_c3_ens_c1_nstar_test.csv",
             "C4": "opt12_c4_w070_test.csv"}
CAND_TAG = {"C1": "nstar", "C2": "wstar", "C4": "w070"}
STAGE_MIN = {"discovery": 11, "confirmation": 13, "test": 29}  # measured 6-view cache minutes (OPT-11 logs) + margin


# ---------------------------------------------------------------- registry and patched OPT-09 code


def registry() -> dict:
    return load_json(REG) if REG.exists() else {}


def register(tag: str, path: Path, extra: dict | None = None) -> dict:
    reg = registry()
    reg[tag] = {"path": Path(path).resolve().as_posix(), "sha256": sha256(path), **(extra or {})}
    save_json(REG, reg)
    return reg[tag]


def weights_of(tag: str) -> Path:
    if tag == "w075":
        path, expected = opt12.W075
    else:
        rec = registry()[tag]
        path, expected = Path(rec["path"]), rec["sha256"]
    if sha256(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from the registry")
    return Path(path)


def raw_dir(tag: str, stage: str) -> Path:
    if tag == "w075":
        return W075_RAW[stage]
    return OUT / "raw" / tag / stage


def patched():
    import opt09_loop as ol
    import opt09_soup as s9

    s9.weights_path = weights_of
    s9.raw_dir = raw_dir
    if not getattr(ol, "_opt12", False):
        ol.merged = functools.partial(ol.merged, workers=5)
        ol._opt12 = True
    return ol, s9


def params(tag: str) -> dict:
    from movd import opt09

    return {**opt09.E6, "soup": tag}


def n_shards(tag: str, stage: str) -> tuple[int, int]:
    import opt08_soup as s8

    clips = sorted(s8.stage_frames(stage))
    d = raw_dir(tag, stage)
    return sum((d / f"{c}.npz").exists() for c in clips), len(clips)


def complete(tag: str, stage: str) -> bool:
    have, total = n_shards(tag, stage)
    return have == total


# ---------------------------------------------------------------- evaluation helpers


def eval_subsets(preds_path: Path) -> dict:
    import opt08_loop as o8

    subs = o8.subsets()
    res = o8.eval_paths([(preds_path, k, ids) for k, ids in subs.items()])
    return {k: res[k]["metrics"] for k in subs}


def deltas_vs_c1(cand: dict, base: dict) -> tuple[dict, dict, int]:
    keys = ("val_full", "width_le_1920", "without_4k_clip", "discovery", "confirmation")
    d = {k: cand[k]["mAP"] - base[k]["mAP"] for k in keys}
    clips = {k: cand[k]["mAP"] - base[k]["mAP"] for k in cand if k.startswith("clip:")}
    return d, clips, sum(v >= 0 for v in clips.values())


def single_view(weights: Path) -> dict:
    import predict_ultra as pu

    from movd import evaluate as ev

    dets, seconds = pu.predict(weights, "val_fast", "0", 1344, False, None, "rtdetr")
    res = ev.evaluate_split(dets, "val_fast")
    return {"mAP": res["metrics"]["mAP"], "metrics": res["metrics"], "predict_seconds": round(seconds, 1)}


# ---------------------------------------------------------------- commands


def cmd_baseline(_a) -> None:
    """c1 rebuilt from its existing caches with the OPT-12 code path; must equal the protected c1 exactly."""
    from movd import submission

    ol, _ = patched()
    val = ol.pipeline(params("w075"), "val_full")
    save_json(C1_PREDS["val_full"], val)
    ref = HW2 / "MOVD-opt09/outputs/opt09/c_final/c1/val_full.json"
    same_val = sha256(C1_PREDS["val_full"]) == sha256(ref) or load_json(ref) == val
    del val
    metrics = eval_subsets(C1_PREDS["val_full"])
    test = ol.pipeline(params("w075"), "test")
    save_json(C1_PREDS["test"], test)
    tmp_csv = OUT / "rebuild" / "c1_rebuilt.csv"
    rows, _ = submission.build_rows(test, submission.test_image_table())
    submission.write_csv(tmp_csv, rows)
    csv_sha = sha256(tmp_csv)
    rec = {"val_full_predictions_equal_opt09_c1": same_val, "val_full_mAP": metrics["val_full"]["mAP"],
           "val_full_mAP_registered": C1_VAL_FULL_REF,
           "val_full_matches_registered": metrics["val_full"]["mAP"] == C1_VAL_FULL_REF,
           "csv_sha256": csv_sha, "csv_equals_protected_c1": csv_sha == opt12.PROTECTED["c1_csv"][1],
           "subsets": metrics, "at_kyiv": stamp()}
    rec["identity"] = rec["val_full_matches_registered"] and rec["csv_equals_protected_c1"]
    save_json(BASE, rec)
    print(json.dumps({k: rec[k] for k in ("val_full_mAP", "csv_equals_protected_c1", "identity")}))
    if not rec["identity"]:
        raise SystemExit("c1 rebuild differs: morning path not proven")


def epoch_ckpts() -> list[Path]:
    return sorted(WEIGHTS_DIR.glob("epoch*.pt"), key=lambda p: int(p.stem[5:]))


def cmd_select(_a) -> None:
    """GPU: single-view val_fast (project evaluator) for every epoch, w075 and AVG3; N-star; W(alpha); screens."""
    from movd.opt12_soup import build_soup

    sel = load_json(SEL) if SEL.exists() else {"scores": {}, "started": stamp()}

    def score(key: str, path: Path) -> float:
        if key not in sel["scores"]:
            r = single_view(path)
            sel["scores"][key] = {"mAP": r["mAP"], "path": path.as_posix(), "AP_small": r["metrics"]["AP_small"]}
            save_json(SEL, sel)
            event("single_view", key=key, mAP=r["mAP"])
        return sel["scores"][key]["mAP"]

    w075 = score("w075", weights_of("w075"))
    ckpts = epoch_ckpts()
    if not ckpts:
        raise SystemExit("no epoch checkpoints")
    curve = {int(p.stem[5:]) + 1: score(f"epoch{int(p.stem[5:]) + 1}", p) for p in ckpts}
    last_ep = max(curve)
    best_ep = max(curve, key=lambda e: curve[e])
    avg_eps = sorted(curve)[-3:]
    avg_path = SOUP_DIR / "avg3.pt"
    if "avg3" not in registry():
        n = len(avg_eps)
        build_soup([WEIGHTS_DIR / f"epoch{e - 1}.pt" for e in avg_eps], [1.0 / n] * n, avg_path,
                   {"kind": "avg3", "epochs_1based": avg_eps})
        register("avg3", avg_path, {"epochs_1based": avg_eps})
    avg3 = score("avg3", weights_of("avg3"))
    scores = {"avg3": avg3, "last": curve[last_ep], "best": curve[best_ep]}
    choice = opt12.choose_nstar(scores)
    nstar_path = {"avg3": weights_of("avg3"), "last": WEIGHTS_DIR / f"epoch{last_ep - 1}.pt",
                  "best": WEIGHTS_DIR / f"epoch{best_ep - 1}.pt"}[choice]
    if "nstar" not in registry():
        register("nstar", nstar_path, {"choice": choice})
    nscore = scores[choice]
    alphas = {}
    for a in opt12.ALPHAS:
        tag = f"w{int(a * 100):03d}n"
        p = SOUP_DIR / f"{tag}.pt"
        if tag not in registry():
            build_soup([weights_of("w075"), weights_of("nstar")], [1 - a, a], p,
                       {"kind": "wiseft", "alpha": a, "w075_sha256": opt12.W075[1],
                        "nstar_sha256": registry()["nstar"]["sha256"]})
            register(tag, p, {"alpha": a})
        alphas[a] = score(tag, weights_of(tag))
    a_star = opt12.choose_alpha(alphas)
    wtag = f"w{int(a_star * 100):03d}n"
    if "wstar" not in registry():
        rec = registry()[wtag]
        register("wstar", Path(rec["path"]), {"alpha": a_star, "source_tag": wtag})
    sel.update({
        "curve_single_view_val_fast": curve, "w075": w075, "last_epoch": last_ep, "best_epoch": best_ep,
        "avg3_epochs": avg_eps, "nstar_scores": scores, "nstar_choice": choice, "nstar_mAP": nscore,
        "nstar_path": nstar_path.as_posix(), "nstar_screen": opt12.nstar_screen(nscore, w075),
        "alpha_scores": {str(k): v for k, v in alphas.items()}, "alpha_star": a_star,
        "wstar_mAP": alphas[a_star], "wstar_screen": opt12.wstar_screen(alphas[a_star], w075),
        "rule": "selection_rule.md (N-star margin 0.003, screen -0.010; alpha tie 0.001; W-star screen +0.002)",
        "decided_at_kyiv": stamp(), "done": True})
    save_json(SEL, sel)
    event("selection", nstar=choice, nstar_mAP=nscore, w075=w075, nstar_screen=sel["nstar_screen"],
          alpha_star=a_star, wstar_mAP=alphas[a_star], wstar_screen=sel["wstar_screen"])


def cmd_select_salvage(_a) -> None:
    """Fallback salvage: W(0.25) of w075 with the best-monitor epoch, screened like W-star."""
    from movd.opt12_soup import build_soup

    obs = load_json(REPO / "results" / RUN_ID / "observations.json")
    eps = [e for e in obs["epochs"] if e.get("checkpoint_exists") and e["monitor_mAP50_95"] == e["monitor_mAP50_95"]]
    if not eps:
        raise SystemExit("no finite epoch for salvage")
    best = max(eps, key=lambda e: e["monitor_mAP50_95"])
    p = SOUP_DIR / "salvage.pt"
    build_soup([weights_of("w075"), Path(best["checkpoint"])], [0.75, 0.25], p,
               {"kind": "salvage", "alpha": 0.25, "epoch_1based": best["epoch"]})
    register("salvage", p, {"alpha": 0.25, "epoch_1based": best["epoch"]})
    w = single_view(weights_of("w075"))["mAP"]
    s = single_view(weights_of("salvage"))["mAP"]
    save_json(EVID / "salvage_selection.json", {"w075": w, "salvage": s, "screen": opt12.wstar_screen(s, w),
                                                 "epoch_1based": best["epoch"], "at_kyiv": stamp()})


def cmd_buildw070(_a) -> None:
    from movd.opt12_soup import build_soup

    if sha256(opt12.MODEL_A[0]) != opt12.MODEL_A[1]:
        raise SystemExit("Model A changed")
    p = SOUP_DIR / "w070.pt"
    info = build_soup([opt12.MODEL_A[0], opt12.MODEL_A_E2], [opt12.W070, 1 - opt12.W070], p,
                      {"kind": "w070", "model_A_sha256": opt12.MODEL_A[1], "epoch1_sha256": sha256(opt12.MODEL_A_E2)})
    register("w070", p, {"weight_model_A": opt12.W070, **info})


def cmd_cache(a) -> None:
    _, s9 = patched()
    res = s9.cache(a.tag, a.stage, out_dir=raw_dir(a.tag, a.stage))
    print(json.dumps(res), flush=True)


def gate_name(tag: str) -> str:
    return {"nstar": "C1", "wstar": "C2", "salvage": "C2", "w070": "C4"}[tag]


def cmd_evalcand(a) -> None:
    ol, _ = patched()
    tag = a.tag
    t0 = time.perf_counter()
    preds = ol.pipeline(params(tag), "val_full")
    path = OUT / "preds" / tag / "val_full.json"
    save_json(path, preds)
    del preds
    base = load_json(BASE)["subsets"]
    cand = eval_subsets(path)
    d, clips, nn = deltas_vs_c1(cand, base)
    g = opt12.gate(gate_name(tag), d, nn)
    rec = {"tag": tag, "registry": registry().get(tag), "params": params(tag), "subsets": cand, "deltas_vs_c1": d,
           "clip_deltas_vs_c1": clips, **g, "val_full_predictions_sha256": sha256(path),
           "seconds": round(time.perf_counter() - t0, 1), "at_kyiv": stamp()}
    save_json(EVID / "cand" / tag / "gate.json", rec)
    event("gate", candidate=g["gate"], tag=tag, passed=g["passed"], deltas=d, clips_non_negative=nn)


def cmd_ens(_a) -> None:
    import opt08_loop as o8

    from movd.ensemble import ensemble

    subs = o8.subsets()
    base = load_json(BASE)["subsets"]
    c1v = load_json(C1_PREDS["val_full"])
    nv = load_json(OUT / "preds" / "nstar" / "val_full.json")
    nstar_gate = load_json(EVID / "cand" / "nstar" / "gate.json")
    disc = set(subs["discovery"])
    c1d = [p for p in c1v if p["image_id"] in disc]
    nd = [p for p in nv if p["image_id"] in disc]
    jobs = []
    for i, (w, iou) in enumerate(opt12.C3_GRID):
        path = OUT / "ens" / f"g{i}_discovery.json"
        save_json(path, ensemble([c1d, nd], list(w), iou))
        jobs.append((path, f"g{i}", subs["discovery"]))
    res = o8.eval_paths(jobs)
    grid = {f"g{i}": {"weights": list(w), "iou": iou, "discovery_mAP": res[f"g{i}"]["metrics"]["mAP"]}
            for i, (w, iou) in enumerate(opt12.C3_GRID)}
    best = max(grid, key=lambda k: grid[k]["discovery_mAP"])
    w, iou = grid[best]["weights"], grid[best]["iou"]
    path = OUT / "preds" / "ens" / "val_full.json"
    save_json(path, ensemble([c1v, nv], w, iou))
    del c1v, nv, c1d, nd
    cand = eval_subsets(path)
    d, clips, nn = deltas_vs_c1(cand, base)
    g = opt12.gate("C3", d, nn)
    nstar_ok = nstar_gate["deltas_vs_c1"]["val_full"] >= opt12.C3_NSTAR_MIN
    g["checks"]["nstar_val_full_ge_c1_minus_0.010"] = nstar_ok
    g["passed"] = g["passed"] and nstar_ok
    rec = {"grid_discovery": grid, "chosen": best, "weights_c1_nstar": w, "iou": iou, "subsets": cand,
           "deltas_vs_c1": d, "clip_deltas_vs_c1": clips, **g, "at_kyiv": stamp()}
    save_json(EVID / "cand" / "ens" / "gate.json", rec)
    event("gate", candidate="C3", passed=g["passed"], chosen=best, deltas=d, clips_non_negative=nn)


NSTAR_TEST = OUT / "preds" / "nstar" / "test.json"


def nstar_test_preds() -> list[dict]:
    """N-star test predictions through the c1 pipeline (computed once, shared by C1 and C3)."""
    if NSTAR_TEST.exists():
        return load_json(NSTAR_TEST)
    ol, _ = patched()
    preds = ol.pipeline(params("nstar"), "test")
    save_json(NSTAR_TEST, preds)
    return preds


def cmd_predtest(_a) -> None:
    nstar_test_preds()
    event("nstar_test_predictions", sha256=sha256(NSTAR_TEST))


def test_predictions(name: str) -> tuple[list[dict], str, dict]:
    if name == "C3":
        from movd.ensemble import ensemble

        g = load_json(EVID / "cand" / "ens" / "gate.json")
        preds = ensemble([load_json(C1_PREDS["test"]), nstar_test_preds()], g["weights_c1_nstar"], g["iou"])
        return preds, registry()["nstar"]["sha256"], {
            "composition": f"ensemble(c1, N-star) weights {g['weights_c1_nstar']} class-wise NMS {g['iou']} top 500",
            "c1_checkpoint_sha256": opt12.W075[1], "c1_test_predictions_sha256": sha256(C1_PREDS["test"])}
    if name == "C1":
        tag = "nstar"
        preds = nstar_test_preds()
    else:
        tag = CAND_TAG[name]
        if name == "C2" and not (EVID / "cand" / "wstar" / "gate.json").exists():
            tag = "salvage"
        ol, _ = patched()
        preds = ol.pipeline(params(tag), "test")
    return preds, registry()[tag]["sha256"], {"tag": tag, "params": params(tag), "checkpoint": registry()[tag]["path"]}


def write_and_validate(name: str, preds: list[dict], ckpt_sha: str, extra: dict) -> dict:
    from movd import submission

    tdir = EVID / "test" / name
    pred_path = OUT / "test" / name / "test_predictions.json"
    save_json(pred_path, preds)
    csv_path = SUB / CSV_NAMES[name]
    images = submission.test_image_table()
    rows, stats = submission.build_rows(preds, images)
    submission.write_csv(csv_path, rows)
    problems, vsum = submission.validate_rows(submission.read_csv(csv_path), images)
    run = {"candidate": name, **extra, "csv": csv_path.resolve().as_posix(), "csv_bytes": csv_path.stat().st_size,
           "csv_sha256": sha256(csv_path), "test_predictions": pred_path.resolve().as_posix(),
           "test_predictions_sha256": sha256(pred_path), "checkpoint_sha256": ckpt_sha, "writer": stats,
           "project_validator_inline": {"valid": not problems, "problems": problems[:20], **vsum},
           "uploaded": False, "written_at_kyiv": stamp()}
    save_json(tdir / "run.json", run)
    v1 = subprocess.run([PY, "scripts/validate_submission.py", str(csv_path)], cwd=REPO, capture_output=True,
                        text=True, check=False)
    (tdir / "project_validator.txt").write_text(v1.stdout + v1.stderr, encoding="utf-8")
    v2 = subprocess.run([PY, "scripts/check_submission_independent.py", str(csv_path), "--predictions", str(pred_path),
                         "--run-json", str(tdir / "run.json"), "--expected-checkpoint-sha256", ckpt_sha,
                         "--out", str(tdir / "independent_validator.json")],
                        cwd=REPO, capture_output=True, text=True, check=False)
    (tdir / "independent_validator.txt").write_text(v2.stdout + v2.stderr, encoding="utf-8")
    run["validators_exit"] = {"project": v1.returncode, "independent": v2.returncode}
    run["valid"] = (not problems and v1.returncode == 0 and v2.returncode == 0 and vsum.get("data_rows") == 3811
                    and vsum.get("max_detections_in_a_row", 0) <= 500)
    save_json(tdir / "run.json", run)
    event("csv", candidate=name, csv=run["csv"], sha256=run["csv_sha256"], valid=run["valid"],
          detections=stats.get("detections"))
    if not run["valid"]:
        os.replace(tdir / "run.json", tdir / "run_FAILED_validation.json")
        raise SystemExit(1)
    return run


def cmd_csv(a) -> None:
    preds, sha, extra = test_predictions(a.name)
    write_and_validate(a.name, preds, sha, extra)


def cmd_rebuild(a) -> None:
    from movd import submission

    preds, _, _ = test_predictions(a.name)
    rows, _ = submission.build_rows(preds, submission.test_image_table())
    p = OUT / "rebuild" / CSV_NAMES[a.name]
    submission.write_csv(p, rows)
    run = load_json(EVID / "test" / a.name / "run.json")
    same = sha256(p) == run["csv_sha256"]
    save_json(EVID / "test" / a.name / "determinism.json", {"rebuild_sha256": sha256(p), "csv_sha256": run["csv_sha256"],
                                                            "identical": same, "at_kyiv": stamp()})
    p.unlink(missing_ok=True)
    event("rebuild", candidate=a.name, identical=same)


# ---------------------------------------------------------------- advisory low-resolution check (T-LR)

LOWRES_CLIPS = ("uav0000086_00000_v", "uav0000182_00000_v", "uav0000305_00000_v", "uav0000339_00001_v")
TLR = EVID / "tlr.json"


def degrade(image, factor: int = 2, quality: int = 90):
    """Down-scale by `factor` (INTER_AREA) and re-encode as JPEG, like a low-resolution test clip."""
    import cv2

    h, w = image.shape[:2]
    small = cv2.resize(image, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError("jpeg encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def rescale_raw(boxes, sx: float, sy: float):
    out = boxes.copy()
    out[..., [0, 2]] *= sx
    out[..., [1, 3]] *= sy
    return out


def cmd_cache_lowres(a) -> None:
    """GPU: the 6-view raw cache on 2x down-scaled frames of the four width <= 1920 val clips, boxes in original px."""
    import cv2
    import numpy as np
    import opt08_soup as s8
    import torch

    from movd import infer, rtdetr

    cfg = s8.config()
    specs = infer.pass_specs(cfg)
    grid, frac = cfg.get("tile_grid", 2), cfg.get("tile_frac", 0.6)
    by_clip = {**s8.stage_frames("discovery"), **s8.stage_frames("confirmation")}
    model = rtdetr.load_model(weights_of(a.tag), "rtdetr")
    torch.zeros(1, device="cuda:0")
    out_dir = OUT / "raw" / a.tag / "lowres"
    out_dir.mkdir(parents=True, exist_ok=True)
    for clip in LOWRES_CLIPS:
        shard = out_dir / f"{clip}.npz"
        if shard.exists():
            continue
        ids, boxes, scores = [], [], []
        for image_id, path in by_clip[clip]:
            image = cv2.imread(str(path))
            small = degrade(image)
            sx, sy = image.shape[1] / small.shape[1], image.shape[0] / small.shape[0]
            raws = [infer.raw_pass(model, small, s["imgsz"], "0", s["view"], grid, frac) for s in specs]
            ids.append(image_id)
            boxes.append(rescale_raw(np.stack([r.boxes for r in raws]), sx, sy))
            scores.append(np.stack([r.scores for r in raws]))
        tmp = shard.with_suffix(".tmp.npz")
        np.savez(tmp, image_ids=np.array(ids, np.int64), boxes=np.stack(boxes), scores=np.stack(scores))
        tmp.replace(shard)
        print(f"lowres {a.tag} {clip}: {len(ids)} frames", flush=True)


def lowres_preds(tag: str) -> list[dict]:
    import opt08_loop as o8
    import opt08_soup as s8

    from movd import opt09

    p = params(tag)
    wh = s8.sizes("val")
    pre = []
    for clip in LOWRES_CLIPS:
        pre.extend(s8.merge_shard(OUT / "raw" / tag / "lowres" / f"{clip}.npz", wh, p["factor"], p["margin"]))
    return opt09.five_frame_boost(pre, o8.neighbours("val"), p["iou"], p["strength"])


def cmd_eval_lowres(_a) -> None:
    import opt08_loop as o8

    from movd.ensemble import ensemble

    ids = o8.subsets()["width_le_1920"]
    tags = [t for t in ("w075", "nstar", "wstar") if (OUT / "raw" / t / "lowres" / f"{LOWRES_CLIPS[-1]}.npz").exists()]
    preds = {t: lowres_preds(t) for t in tags}
    jobs = []
    for t, pr in preds.items():
        path = OUT / "lowres" / f"{t}.json"
        save_json(path, pr)
        jobs.append((path, t, ids))
    ens_gate = EVID / "cand" / "ens" / "gate.json"
    if "nstar" in preds and ens_gate.exists():
        g = load_json(ens_gate)
        path = OUT / "lowres" / "ens.json"
        save_json(path, ensemble([preds["w075"], preds["nstar"]], g["weights_c1_nstar"], g["iou"]))
        jobs.append((path, "ens", ids))
    res = {k: v["metrics"]["mAP"] for k, v in o8.eval_paths(jobs).items()}
    name = {"nstar": "C1", "wstar": "C2", "ens": "C3"}
    rec = {"mAP_width_le_1920_downscaled_2x": res,
           "delta_vs_c1": {name[k]: v - res["w075"] for k, v in res.items() if k in name},
           "demote_below": opt12.TLR_DEMOTE, "advisory_only": True, "at_kyiv": stamp()}
    save_json(TLR, rec)
    event("tlr", **rec["delta_vs_c1"])


# ---------------------------------------------------------------- hand-off


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def cmd_handoff(_a) -> None:
    st = opt12.get_state()
    lines = [f"# OPT-12 hand-off ({stamp()} Kyiv)", "",
             ("Nothing was uploaded. c1 (public 0.2960489045) and c2 (0.2959983429) stay the selected, upload-ready "
              "fallbacks; their CSVs are unchanged (see protected_*.json)."), ""]
    sel = load_json(SEL) if SEL.exists() else None
    tr = st.get("decisions", {}).get("train_result")
    lines.append(f"Training result: `{tr}`. Reference R (Ultralytics monitor of w075) = "
                 f"{st.get('decisions', {}).get('reference_R')}.")
    obs_p = REPO / "results" / RUN_ID / "observations.json"
    if obs_p.exists():
        # the trainer's final evaluation of best.pt also fires on_fit_epoch_end; it has no epoch checkpoint
        eps = [e for e in load_json(obs_p).get("epochs", []) if e.get("checkpoint_exists")]
        lines += ["", "| Epoch | monitor mAP50-95 | epoch s | BN stats frozen |", "|---|---|---|---|"]
        lines += [f"| {e['epoch']} | {e['monitor_mAP50_95']:.4f} | {e['epoch_seconds_incl_val']} | {e['bn_hash_equal_start']} |"
                  for e in eps]
    if sel and sel.get("done"):
        lines += ["", f"Selection (single view, val_fast, project evaluator): w075 {sel['w075']:.4f}; curve "
                  + ", ".join(f"e{k} {v:.4f}" for k, v in sel["curve_single_view_val_fast"].items())
                  + f"; AVG3 {sel['nstar_scores']['avg3']:.4f}. N-star = {sel['nstar_choice']} ({sel['nstar_mAP']:.4f}, "
                  f"screen {'pass' if sel['nstar_screen'] else 'FAIL'}). alpha-star = {sel['alpha_star']} "
                  f"(W {sel['wstar_mAP']:.4f}, screen {'pass' if sel['wstar_screen'] else 'FAIL'})."]
    lines += ["", "## Candidates (deltas vs c1 on val; gates frozen in selection_rule.md)", "",
              "| Candidate | val_full | w<=1920 | w/o 4K | discovery | confirmation | clips >= 0 | gate | CSV |",
              "|---|---|---|---|---|---|---|---|---|"]
    passed, rows_csv = {}, []
    for name, gdir in (("C1", "nstar"), ("C2", "wstar"), ("C2", "salvage"), ("C3", "ens"), ("C4", "w070")):
        gp = EVID / "cand" / gdir / "gate.json"
        if not gp.exists():
            continue
        g = load_json(gp)
        d = g["deltas_vs_c1"]
        run_p = EVID / "test" / name / "run.json"
        run = load_json(run_p) if run_p.exists() else None
        passed[name] = g["passed"] and run is not None and run.get("valid")
        lines.append(f"| {name} ({gdir}) | {fmt(d['val_full'])} | {fmt(d['width_le_1920'])} | {fmt(d['without_4k_clip'])} | "
                     f"{fmt(d['discovery'])} | {fmt(d['confirmation'])} | {g['clips_non_negative']}/7 | "
                     f"{'PASS' if g['passed'] else 'fail'} | {'ready' if run and run.get('valid') else '-'} |")
        if run and run.get("valid"):
            det = EVID / "test" / name / "determinism.json"
            rows_csv.append(f"- **{name}**: `{run['csv']}`  \n  sha256 `{run['csv_sha256']}`, "
                            f"{run['writer'].get('detections')} detections, 3,811 rows; validators project "
                            f"{run['validators_exit']['project']} / independent {run['validators_exit']['independent']}"
                            + (f"; rebuild identical: {load_json(det)['identical']}" if det.exists() else ""))
    lines += ["", "## Validated CSVs", ""] + (rows_csv or ["None of the new candidates passed its gate and validators."])
    tlr = load_json(TLR)["delta_vs_c1"] if TLR.exists() else None
    if tlr:
        lines += ["", "Advisory low-resolution check (width<=1920 val clips down-scaled 2x, delta vs c1; "
                  "demote below -0.005, never a gate): " + ", ".join(f"{k} {v:+.4f}" for k, v in tlr.items())]
    order = opt12.upload_order(passed, sel.get("alpha_star") if sel else None, tlr)
    lines += ["", "## Recommended manual upload order", ""]
    if order:
        lines += [f"{i + 1}. {c}" for i, c in enumerate(order)]
        lines += ["", ("Sequential rule: upload C1 first if present. If public(C1) >= 0.2930, continue with C3 then C2. "
                       "If public(C1) < 0.2930, skip C3 and upload C2 only if alpha-star <= 0.25. Keep c1 selected "
                       "until a new public score beats 0.2960489045; the two final selections remain the students' "
                       "decision.")]
    else:
        lines += ["No new CSV. Keep c1 + c2 as the final selections."]
    (EVID / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------- orchestrator


def child(args: list[str], log: str, kill_at, gpu: bool) -> int:
    if gpu:
        while not opt12.acquire_lock(args[0], os.getpid()):
            event("lock_wait", job=args)
            time.sleep(30)
            if now() >= kill_at:
                return -8
        waited = 0
        while bad := opt12.foreign_gpu_python(set()):
            if waited >= 900 or now() >= kill_at:
                opt12.release_lock()
                event("refused_gpu_busy", foreign=bad)
                return -7
            event("wait_foreign_gpu", foreign=bad)
            time.sleep(30)
            waited += 30
    out = (OUT / "logs" / f"{log}.out").open("a", encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU children never open a CUDA context
    p = subprocess.Popen([PY, str(Path(__file__)), *args], cwd=REPO, stdout=out, stderr=subprocess.STDOUT, env=env)
    if gpu:
        opt12.set_lock_pid(p.pid, log)
    opt12.update_state("jobs", log, {"status": "running", "pid": p.pid, "started": stamp()})
    event("child_start", job=log, pid=p.pid, gpu=gpu)
    while p.poll() is None:
        if now() >= kill_at:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, check=False)
            p.wait()
            event("killed_at_cutoff", job=log)
            break
        time.sleep(10)
    if gpu:
        opt12.release_lock()
    opt12.update_state("jobs", log, {"status": "ended", "rc": p.returncode, "ended": stamp()})
    event("child_end", job=log, rc=p.returncode)
    return p.returncode


def fits(minutes: float) -> bool:
    t = now()
    return t <= opt12.CLOCK["last_gpu_start"] and t.timestamp() + 60 * minutes <= opt12.CLOCK["gpu_end"].timestamp()


def gpu_cache(tag: str, stage: str) -> bool:
    if complete(tag, stage):
        return True
    if not fits(STAGE_MIN[stage]):
        event("gpu_skip_time", tag=tag, stage=stage)
        return False
    rc = child(["cache", tag, stage], f"cache_{tag}_{stage}", opt12.CLOCK["gpu_end"], gpu=True)
    return rc == 0 and complete(tag, stage)


def gate_passed(gdir: str) -> bool | None:
    p = EVID / "cand" / gdir / "gate.json"
    return load_json(p)["passed"] if p.exists() else None


def gpu_thread(primary: bool, done: threading.Event) -> None:
    try:
        if primary:
            if not (SEL.exists() and load_json(SEL).get("done")):
                if fits(30):
                    child(["select"], "select", opt12.CLOCK["gpu_end"], gpu=True)
            sel = load_json(SEL) if SEL.exists() else {}
            if sel.get("nstar_screen"):
                for stage in ("discovery", "confirmation", "test"):
                    if not gpu_cache("nstar", stage):
                        break
            if sel.get("wstar_screen"):
                for stage in ("discovery", "confirmation", "test"):
                    if not gpu_cache("wstar", stage):
                        break
        else:
            obs_ok = (REPO / "results" / RUN_ID / "observations.json").exists() and epoch_ckpts()
            if obs_ok and not (EVID / "salvage_selection.json").exists() and fits(10):
                child(["select_salvage"], "select_salvage", opt12.CLOCK["gpu_end"], gpu=True)
            sp = EVID / "salvage_selection.json"
            if sp.exists() and load_json(sp)["screen"]:
                for stage in ("discovery", "confirmation", "test"):
                    if not gpu_cache("salvage", stage):
                        break
        # C4 fallback soup: only started before 09:30 (weak expected benefit; student time policy)
        if now() <= opt12.at("09:30"):
            if "w070" not in registry():
                child(["buildw070"], "buildw070", opt12.CLOCK["gpu_end"], gpu=False)
            if "w070" in registry():
                if gpu_cache("w070", "discovery") and gpu_cache("w070", "confirmation"):
                    while (gate_passed("w070") is None and "eval_w070" not in FAILED
                           and now() < opt12.CLOCK["no_new_experiment"]):
                        time.sleep(30)
                    if gate_passed("w070"):
                        gpu_cache("w070", "test")
        # Advisory T-LR (never a gate): only before 09:30
        sel = load_json(SEL) if SEL.exists() else {}
        if primary and sel.get("nstar_screen") and now() <= opt12.at("09:30"):
            for t in ["w075", "nstar"] + (["wstar"] if sel.get("wstar_screen") else []):
                if not (OUT / "raw" / t / "lowres" / f"{LOWRES_CLIPS[-1]}.npz").exists() and fits(13):
                    child(["cache_lowres", t], f"cache_lowres_{t}", opt12.CLOCK["gpu_end"], gpu=True)
            save_json(OUT / "lowres_caches_done.json", {"at": stamp()})
    finally:
        done.set()


FAILED: set[str] = set()  # CPU tasks that exited non-zero (shared with the GPU thread)


def cpu_tasks(primary: bool) -> list[tuple[str, list[str], callable]]:
    def ready_eval(tag):
        return lambda: complete(tag, "discovery") and complete(tag, "confirmation") and BASE.exists()

    def ready_csv(name, gdir, tag):
        return lambda: gate_passed(gdir) is True and complete(tag, "test")

    t = [("baseline", ["baseline"], lambda: True)]
    if primary:
        t += [("eval_nstar", ["evalcand", "nstar"], ready_eval("nstar")),
              ("predtest_nstar", ["predtest"], lambda: complete("nstar", "test")),
              ("csv_C1",["csv", "C1"], ready_csv("C1", "nstar", "nstar")),
              ("ens", ["ens"], lambda: (EVID / "cand" / "nstar" / "gate.json").exists()),
              ("csv_C3", ["csv", "C3"], lambda: gate_passed("ens") is True and complete("nstar", "test")),
              ("eval_wstar", ["evalcand", "wstar"], ready_eval("wstar")),
              ("csv_C2", ["csv", "C2"], ready_csv("C2", "wstar", "wstar"))]
    else:
        t += [("eval_salvage", ["evalcand", "salvage"], ready_eval("salvage")),
              ("csv_C2", ["csv", "C2"], ready_csv("C2", "salvage", "salvage"))]
    t += [("eval_w070", ["evalcand", "w070"], ready_eval("w070")),
          ("csv_C4", ["csv", "C4"], ready_csv("C4", "w070", "w070")),
          ("eval_lowres", ["eval_lowres"], lambda: (OUT / "lowres_caches_done.json").exists()
           and (OUT / "raw" / "w075" / "lowres" / f"{LOWRES_CLIPS[-1]}.npz").exists())]
    return t


def task_done(name: str) -> bool:
    files = {"baseline": BASE, "eval_nstar": EVID / "cand/nstar/gate.json", "ens": EVID / "cand/ens/gate.json",
             "eval_wstar": EVID / "cand/wstar/gate.json", "eval_salvage": EVID / "cand/salvage/gate.json",
             "eval_w070": EVID / "cand/w070/gate.json"}
    if name == "predtest_nstar":
        return NSTAR_TEST.exists()
    if name == "eval_lowres":
        return TLR.exists()
    if name in files:
        return files[name].exists()
    if name.startswith("csv_"):
        return (EVID / "test" / name[4:] / "run.json").exists()
    return False


def cmd_run(_a) -> None:
    opt12.keep_awake()
    tr = opt12.get_state().get("decisions", {}).get("train_result")
    primary = tr in ("done", "partial", "killed") and bool(epoch_ckpts())
    if tr == "partial" and len(epoch_ckpts()) < 3:
        primary = bool(epoch_ckpts())
    opt12.update_state("decisions", "morning_path", "primary" if primary else "fallback")
    event("morning_start", train_result=tr, primary=primary, epochs=len(epoch_ckpts()))
    done = threading.Event()
    th = threading.Thread(target=gpu_thread, args=(primary, done), daemon=True)
    th.start()
    failed, interim = FAILED, False
    tasks = cpu_tasks(primary)
    rebuilt = set()
    while True:
        progressed = False
        for name, args, ready in tasks:
            if task_done(name) or name in failed or not ready():
                continue
            if now() >= opt12.CLOCK["hard_handoff"]:
                break
            rc = child(args, name, opt12.at("11:58"), gpu=False)
            if rc != 0:
                failed.add(name)
            progressed = True
            if name.startswith("csv_") and rc == 0:
                child(["handoff"], "handoff", opt12.at("11:59"), gpu=False)
            break
        if not interim and now() >= opt12.CLOCK["interim_handoff"]:
            child(["handoff"], "handoff", opt12.at("11:59"), gpu=False)
            interim = True
        if not progressed:
            if done.is_set():
                pending = [n for n, _, r in tasks if not task_done(n) and n not in failed and r()]
                if not pending:
                    break
            # idle CPU: determinism rebuilds of finished CSVs
            for name in ("C1", "C3", "C2", "C4"):
                if name not in rebuilt and (EVID / "test" / name / "run.json").exists() and \
                        now() < opt12.at("11:30"):
                    rebuilt.add(name)
                    child(["rebuild", name], f"rebuild_{name}", opt12.at("11:50"), gpu=False)
                    progressed = True
                    break
            if not progressed:
                time.sleep(20)
        if now() >= opt12.CLOCK["hard_handoff"]:
            break
    child(["handoff"], "handoff", opt12.at("11:59"), gpu=False)
    opt12.update_state("decisions", "morning_done", stamp())
    event("morning_done", failed=sorted(failed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "baseline", "select", "select_salvage", "ens", "handoff", "buildw070", "predtest", "eval_lowres"):
        sub.add_parser(c)
    p = sub.add_parser("cache_lowres")
    p.add_argument("tag")
    p = sub.add_parser("cache")
    p.add_argument("tag")
    p.add_argument("stage", choices=("discovery", "confirmation", "test"))
    p = sub.add_parser("evalcand")
    p.add_argument("tag")
    for c in ("csv", "rebuild"):
        p = sub.add_parser(c)
        p.add_argument("name", choices=tuple(CSV_NAMES))
    a = ap.parse_args()
    {"run": cmd_run, "baseline": cmd_baseline, "select": cmd_select, "select_salvage": cmd_select_salvage,
     "ens": cmd_ens, "handoff": cmd_handoff, "buildw070": cmd_buildw070, "cache": cmd_cache,
     "evalcand": cmd_evalcand, "csv": cmd_csv, "rebuild": cmd_rebuild, "predtest": cmd_predtest,
     "cache_lowres": cmd_cache_lowres, "eval_lowres": cmd_eval_lowres}[a.cmd](a)


if __name__ == "__main__":
    main()
