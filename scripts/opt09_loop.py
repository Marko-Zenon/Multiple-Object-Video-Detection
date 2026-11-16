"""OPT-09: one focused local search around the protected E6 pipeline (CPU stages; GPU only via opt09_soup.py).

    python scripts/opt09_loop.py verify          # protected E6 hashes + E6 val_full rebuilt by the OPT-09 code path
    python scripts/opt09_loop.py baseline        # E6 on every subset once (must equal the OPT-08 E6 record)
    python scripts/opt09_loop.py screen G        # S1: every frozen candidate of grid G on the discovery clips
    python scripts/opt09_loop.py confirm G       # S2: the S1 selection of G on the confirmation clips, once
    python scripts/opt09_loop.py plan-c          # Phase C candidates from the passed components (fixed rule)
    python scripts/opt09_loop.py full C          # val_full + gate subsets once for a Phase C candidate, gate G1-G9
    python scripts/opt09_loop.py determinism C   # fresh process: val_full + test rebuilt, sha256 compared
    python scripts/opt09_loop.py test C          # only if eligible: lossless test CSV + both validators

Grids: movd/opt09.py GRIDS; rules: results/opt-09-e6-tuning/selection_rule.md (committed before `screen`).
The clip split, subsets and evaluator are OPT-08's / the project's, unchanged. Nothing is uploaded.
Large files go to git-ignored outputs/opt09/ and submissions/.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import opt07_loop as o7
import opt08_loop as o8
import opt08_soup as s8
import opt09_soup as s9

from movd import data, opt09, submission
from movd import evaluate as ev

HW2 = REPO.parent
EVID = REPO / "results" / "opt-09-e6-tuning"
OUT = REPO / "outputs" / "opt09"
SUB = REPO / "submissions"
OPT08 = HW2 / "MOVD-opt08"
RULE = EVID / "selection_rule.md"
MAX_NEW_CSVS = 2

# Protected E6 artifacts (OPT-08 e6_combination k1; read only, never written) and the other public CSVs.
PROTECTED = {
    "model_A_epoch0": (s8.A[0], s8.A[1]),
    "adjacent_epoch1": (s8.B[0], s8.B[1]),
    "E6_soup_w080": s9.E6_SOUP,
    "E6_val_full_predictions": (
        OPT08 / "outputs/opt08/e6_combination/k1/val_full.json",
        "bdb2806edad09fa093d5b59369780a8dbbb6fea4758522c72c86b8c8e98cad52",
    ),
    "E6_test_predictions": (
        OPT08 / "outputs/opt08/e6_combination/k1/test/test_predictions.json",
        "ef3f5c2ff42598696dfcd44aa1431c0d422598203bbfce8f0370c333c14cb0aa",
    ),
    "E6_csv": (
        OPT08 / "submissions/opt08_e6_combination_k1_test.csv",
        "35a225c905ac56ed86532314046821d10085f73781666ee843fb4c6d28127c35",
    ),
    "E4_csv": (
        OPT08 / "submissions/opt08_e4_checkpoint_soup_s1_test.csv",
        "7dc9b318e354d4839b5dffae8f5973dd7da234fdbeb3c5f84057694878852014",
    ),
    "x2_csv": o8.X2["csv"],
    "x4_csv": o8.X4["csv"],
    "pipeline_config": (
        REPO / "configs/runs/final_inference.json",
        None,  # must equal origin/develop (frozen-file check)
    ),
}
E6_RECORD = REPO / "results/opt-08-overnight/e6_combination"

STAGE = {"S1_min_delta_discovery": 0.0003, "S2_min_delta_confirmation_exclusive": 0.0}
GATE = {
    "G1_min_delta_mAP": 0.0010,
    "G2_min_delta_mAP_width_le_1920": 0.0005,
    "G3_min_clips_non_negative": 5,
    "G4_min_delta_mAP_without_4k_clip": 0.0010,
    "G5_max_AP_small_loss": 0.002,
    "G6_max_density_ge100_loss": 0.002,
}
FROZEN_FILES = (
    "movd/evaluate.py",
    "movd/submission.py",
    "movd/data.py",
    "movd/infer.py",
    "movd/opt03.py",
    "movd/opt07.py",
    "movd/opt08.py",
    "configs",
    "scripts/validate_submission.py",
    "scripts/check_submission_independent.py",
)
GRID_ORDER = ("a1_border_margin", "a2_persistence", "b_soup_weight")

sha256, save_json, load_json, rel, write_rows = (
    o7.sha256,
    o7.save_json,
    o7.load_json,
    o7.rel,
    o7.write_rows,
)


# ---------------------------------------------------------------- pipeline


def _merge_job(shard: str, split: str, factor: float, margin: float) -> list[dict]:
    return s8.merge_shard(Path(shard), s8.sizes(split), factor, margin)


def merged(params: dict, stages: list[str], workers: int = 7) -> list[dict]:
    """Soup raw caches -> CH-13 merge with the border rule; same order as opt08_soup.merged (stage, sorted clip)."""
    split = "test" if stages == ["test"] else "val"
    shards = []
    for stage in stages:
        for clip in sorted(s8.stage_frames(stage)):
            shard = s9.raw_dir(params["soup"], stage) / f"{clip}.npz"
            if not shard.exists():
                raise SystemExit(
                    f"{shard} missing: cache {params['soup']} {stage} first"
                )
            shards.append(str(shard))
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        parts = pool.map(
            _merge_job,
            shards,
            [split] * len(shards),
            [params["factor"]] * len(shards),
            [params["margin"]] * len(shards),
        )
        return [p for part in parts for p in part]


def pipeline(params: dict, split: str, stages: list[str] | None = None) -> list[dict]:
    """Candidate predictions: soup -> border -> five-frame boost. Val stages default to the full val set."""
    if params["soup"] != opt09.E6["soup"]:
        s9.weights_path(
            params["soup"]
        )  # hash check of the checkpoint behind the raw cache
    stages = ["test"] if split == "test" else stages or ["discovery", "confirmation"]
    pre = merged(params, stages)
    return opt09.five_frame_boost(
        pre, o8.neighbours(split), params["iou"], params["strength"]
    )


# ---------------------------------------------------------------- helpers


def frozen_unchanged() -> bool:
    r = subprocess.run(
        ["git", "diff", "--quiet", "origin/develop", "--", *FROZEN_FILES],
        cwd=REPO,
        check=False,
    )
    return r.returncode == 0


def rule_committed() -> str:
    rel_rule = RULE.relative_to(REPO).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_rule, "movd/opt09.py"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", rel_rule, "movd/opt09.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if tracked.returncode or dirty:
        raise SystemExit(
            "selection_rule.md or movd/opt09.py is not committed unchanged"
        )
    text = RULE.read_text(encoding="utf-8")
    for grid, cands in opt09.GRIDS.items():
        for cid, change in cands.items():
            if f"{grid}/{cid} {json.dumps(change, sort_keys=True)}" not in text:
                raise SystemExit(f"selection_rule.md does not state {grid}/{cid}")
    return o8.commit_of(RULE)


def baseline() -> dict:
    return load_json(EVID / "baseline" / "subsets.json")["subsets"]


def eval_preds(preds: list[dict], path: Path, jobs: dict[str, list[int]]) -> dict:
    save_json(path, preds)
    results = o8.eval_paths([(path, k, ids) for k, ids in jobs.items()])
    return {k: results[k] for k in jobs}


# ---------------------------------------------------------------- commands


def cmd_verify(_args) -> None:
    record = {}
    for k, (path, expected) in PROTECTED.items():
        actual = sha256(path)
        record[k] = {
            "path": rel(path),
            "sha256": actual,
            "matches": expected is None or actual == expected,
        }
    shards = {}
    for stage in ("discovery", "confirmation", "test"):
        d = s9.raw_dir("w080", stage)
        shards[stage] = {p.name: sha256(p) for p in sorted(d.glob("*.npz"))}
    record["E6_raw_shards"] = {"dir": rel(s9.raw_dir("w080", "")), "sha256": shards}
    gt_path = data.split_dir("val") / "annotations_coco.json"
    record["val_annotations"] = {"path": rel(gt_path), "sha256": sha256(gt_path)}
    record["frozen_files_unchanged"] = frozen_unchanged()
    # the OPT-09 code path must rebuild E6's val_full predictions byte-identically
    t0 = time.perf_counter()
    rebuilt = OUT / "e6_rebuild" / "val_full.json"
    save_json(rebuilt, pipeline(opt09.E6, "val_full"))
    record["E6_val_full_rebuilt_sha256"] = sha256(rebuilt)
    record["E6_val_full_rebuilt_identical"] = (
        record["E6_val_full_rebuilt_sha256"] == PROTECTED["E6_val_full_predictions"][1]
    )
    record["rebuild_seconds"] = round(time.perf_counter() - t0, 1)
    record["ok"] = (
        all(v["matches"] for k, v in record.items() if k in PROTECTED)
        and record["frozen_files_unchanged"]
        and record["E6_val_full_rebuilt_identical"]
        and all(len(v) for v in shards.values())
    )
    record["checked_at_kyiv"] = o7.kyiv_now()
    save_json(EVID / "identity.json", record)
    print(
        json.dumps({k: v for k, v in record.items() if k != "E6_raw_shards"}, indent=2)
    )
    if not record["ok"]:
        raise SystemExit("IDENTITY FAILED: stop")


def cmd_baseline(_args) -> None:
    target = EVID / "baseline" / "subsets.json"
    if target.exists():
        raise SystemExit("baseline already evaluated")
    path, expected = PROTECTED["E6_val_full_predictions"]
    o8.checked(path, expected)
    start = time.perf_counter()
    jobs = o8.subsets()
    results = o8.eval_paths([(path, k, ids) for k, ids in jobs.items()])
    results = {k: results[k] for k in jobs}
    reg = load_json(E6_RECORD / "gate.json")["candidate_metrics"]
    reg_disc = load_json(E6_RECORD / "screen.json")["candidates"]["k1"]["discovery"]
    reg_conf = load_json(E6_RECORD / "confirm.json")["confirmation"]
    same = {
        "val_full": all(
            results["val_full"]["metrics"][k] == v
            for k, v in reg["val_full"].items()
            if k in results["val_full"]["metrics"]
        ),
        "width_le_1920": all(
            results["width_le_1920"]["metrics"][k] == v
            for k, v in reg["width_le_1920"].items()
            if k in results["width_le_1920"]["metrics"]
        ),
        "discovery": results["discovery"]["metrics"] == reg_disc,
        "confirmation": results["confirmation"]["metrics"] == reg_conf,
    }
    ev.write_outputs(results["val_full"], EVID / "baseline" / "val_full")
    save_json(
        target,
        {
            "predictions": rel(path),
            "predictions_sha256": expected,
            "matches_opt08_e6_record": same,
            "subsets": o7.compact(results),
            "wall_seconds": round(time.perf_counter() - start, 1),
        },
    )
    print(
        "E6 val_full mAP",
        results["val_full"]["metrics"]["mAP"],
        "matches OPT-08:",
        same,
    )
    if not all(same.values()):
        raise SystemExit("baseline does not reproduce the OPT-08 E6 record: stop")


def cmd_screen(args) -> None:
    grid = args.name
    commit = rule_committed()
    out_dir = EVID / grid
    if (out_dir / "screen.json").exists():
        raise SystemExit(f"{grid} already screened once")
    base = baseline()["discovery"]
    ids = o8.subsets()["discovery"]
    start = time.perf_counter()
    jobs, meta = [], {}
    for cid in opt09.GRIDS[grid]:
        params = opt09.candidate(grid, cid)
        t0 = time.perf_counter()
        preds = pipeline(params, "val_full", ["discovery"])
        path = OUT / grid / cid / "discovery.json"
        save_json(path, preds)
        meta[cid] = {
            "params": params,
            "changed": opt09.changed_keys(params),
            "discovery_predictions_sha256": sha256(path),
            "num_predictions": len(preds),
            "transform_seconds": round(time.perf_counter() - t0, 1),
        }
        jobs.append((path, cid, ids))
        del preds
    results = o8.eval_paths(jobs)
    rows = []
    for cid, info in meta.items():
        m = results[cid]["metrics"]
        info["discovery"] = m
        info["delta_discovery_mAP"] = m["mAP"] - base["mAP"]
        rows.append(
            {
                "candidate": cid,
                "changed": "+".join(info["changed"]),
                **{k: info["params"][k] for k in opt09.E6},
                "E6_mAP": base["mAP"],
                "mAP": m["mAP"],
                "delta": info["delta_discovery_mAP"],
                "AP_small": m["AP_small"],
                "delta_AP_small": m["AP_small"] - base["AP_small"],
            }
        )
    order = list(meta)
    best = max(meta, key=lambda c: (meta[c]["delta_discovery_mAP"], -order.index(c)))
    passes = meta[best]["delta_discovery_mAP"] >= STAGE["S1_min_delta_discovery"]
    write_rows(out_dir / "screen.csv", rows)
    save_json(
        out_dir / "screen.json",
        {
            "grid": grid,
            "rule_commit": commit,
            "stage_rule": STAGE,
            "baseline_E6_discovery": base,
            "candidates": meta,
            "selected": best,
            "passes_S1": passes,
            "wall_seconds": round(time.perf_counter() - start, 1),
            "evaluated_at_kyiv": o7.kyiv_now(),
            "decision": "S1 pass: confirmation next"
            if passes
            else "rejected at S1 (discovery)",
        },
    )
    print(
        json.dumps(
            {c: round(meta[c]["delta_discovery_mAP"], 6) for c in meta}, indent=2
        )
    )
    print("selected", best, "S1 pass" if passes else "S1 FAIL")


def cmd_confirm(args) -> None:
    grid = args.name
    out_dir = EVID / grid
    if (out_dir / "confirm.json").exists():
        raise SystemExit(f"{grid} already confirmed once")
    s1 = load_json(out_dir / "screen.json")
    if not s1["passes_S1"]:
        raise SystemExit(f"{grid} failed S1: no confirmation (rule)")
    cid = s1["selected"]
    params = opt09.candidate(grid, cid)
    base = baseline()["confirmation"]
    ids = o8.subsets()["confirmation"]
    path = OUT / grid / cid / "confirmation.json"
    save_json(path, pipeline(params, "val_full", ["confirmation"]))
    m = o8.eval_paths([(path, cid, ids)], workers=1)[cid]["metrics"]
    delta = m["mAP"] - base["mAP"]
    passes = delta > STAGE["S2_min_delta_confirmation_exclusive"]
    save_json(
        out_dir / "confirm.json",
        {
            "grid": grid,
            "selected": cid,
            "params": params,
            "baseline_E6_confirmation": base,
            "confirmation": m,
            "delta_confirmation_mAP": delta,
            "confirmation_predictions_sha256": sha256(path),
            "passes_S2": passes,
            "evaluated_at_kyiv": o7.kyiv_now(),
            "decision": "component passes (S1 and S2)"
            if passes
            else "rejected at S2 (confirmation)",
        },
    )
    print(cid, "confirmation delta", delta, "S2 pass" if passes else "S2 FAIL")


def component_status() -> dict[str, dict]:
    out = {}
    for grid in GRID_ORDER:
        s1_path = EVID / grid / "screen.json"
        if not s1_path.exists():
            raise SystemExit(f"{grid} not screened: Phase C needs every phase finished")
        s1 = load_json(s1_path)
        s2_path = EVID / grid / "confirm.json"
        s2 = load_json(s2_path) if s2_path.exists() else None
        if s1["passes_S1"] and s2 is None:
            raise SystemExit(f"{grid} passed S1 but is not confirmed yet")
        cid = s1["selected"]
        out[grid] = {
            "selected": cid,
            "change": opt09.GRIDS[grid][cid],
            "delta_discovery": s1["candidates"][cid]["delta_discovery_mAP"],
            "delta_confirmation": s2["delta_confirmation_mAP"] if s2 else None,
            "passes": bool(s1["passes_S1"] and s2 and s2["passes_S2"]),
        }
    return out


def cmd_plan_c(_args) -> None:
    target = EVID / "c_final" / "plan.json"
    if target.exists():
        raise SystemExit("Phase C plan already fixed")
    status = component_status()
    passed = [g for g in GRID_ORDER if status[g]["passes"]]
    cands = {}
    if passed:
        strongest = max(
            passed, key=lambda g: (status[g]["delta_discovery"], -GRID_ORDER.index(g))
        )
        cands["c1"] = {
            "components": [strongest],
            "params": {**opt09.E6, **status[strongest]["change"]},
        }
        if len(passed) >= 2:
            params = dict(opt09.E6)
            for g in passed:
                params.update(status[g]["change"])
            cands["c2"] = {"components": passed, "params": params}
    record = {
        "rule": "selection_rule.md Phase C: c1 = passed component with the largest discovery delta (ties: earlier phase); "
        "c2 = all passed components together, only if at least two passed",
        "components": status,
        "candidates": cands,
        "decision": "Phase C candidates fixed"
        if cands
        else "NEGATIVE: no component passed S1 and S2; no Phase C",
        "fixed_at_kyiv": o7.kyiv_now(),
    }
    save_json(target, record)
    print(json.dumps(record, indent=2))


def phase_c(cid: str) -> dict:
    plan = load_json(EVID / "c_final" / "plan.json")
    if cid not in plan["candidates"]:
        raise SystemExit(f"{cid} is not a Phase C candidate")
    return plan["candidates"][cid]["params"]


def gate(base: dict, cand: dict, per_class_rows: int) -> dict:
    clip_delta = {
        k[5:]: cand[k]["mAP"] - base[k]["mAP"] for k in cand if k.startswith("clip:")
    }
    d = {
        "delta_mAP": cand["val_full"]["mAP"] - base["val_full"]["mAP"],
        "delta_mAP_width_le_1920": cand["width_le_1920"]["mAP"]
        - base["width_le_1920"]["mAP"],
        "clips_non_negative": sum(v >= 0 for v in clip_delta.values()),
        "delta_mAP_without_4k_clip": cand["without_4k_clip"]["mAP"]
        - base["without_4k_clip"]["mAP"],
        "delta_AP_small": cand["val_full"]["AP_small"] - base["val_full"]["AP_small"],
        "delta_density_ge100": cand["val_full"]["density_ge100"]
        - base["val_full"]["density_ge100"],
        "delta_discovery": cand["discovery"]["mAP"] - base["discovery"]["mAP"],
        "delta_confirmation": cand["confirmation"]["mAP"] - base["confirmation"]["mAP"],
    }
    checks = {
        "G1 val_full mAP >= E6 + 0.0010": d["delta_mAP"] >= GATE["G1_min_delta_mAP"],
        "G2 width<=1920 mAP >= E6 + 0.0005": d["delta_mAP_width_le_1920"]
        >= GATE["G2_min_delta_mAP_width_le_1920"],
        "G3 >= 5 of 7 clips non-negative": d["clips_non_negative"]
        >= GATE["G3_min_clips_non_negative"],
        "G4 without-4K mAP >= E6 + 0.0010": d["delta_mAP_without_4k_clip"]
        >= GATE["G4_min_delta_mAP_without_4k_clip"],
        "G5 AP_small loss <= 0.002": d["delta_AP_small"]
        >= -GATE["G5_max_AP_small_loss"],
        "G6 density>=100 loss <= 0.002": d["delta_density_ge100"]
        >= -GATE["G6_max_density_ge100_loss"],
        "G7 all 8 class results reported": per_class_rows == 8,
        "G9 evaluator / protocol / CSV format unchanged": frozen_unchanged(),
        "stage consistency: discovery and confirmation both above E6": d[
            "delta_discovery"
        ]
        > 0
        and d["delta_confirmation"] > 0,
    }
    return {
        "deltas": d,
        "clip_delta_mAP": clip_delta,
        "checks": checks,
        "passes_G1_G7_G9": all(checks.values()),
    }


def cmd_full(args) -> None:
    cid = args.name
    out_dir = EVID / "c_final" / cid
    if (out_dir / "gate.json").exists():
        raise SystemExit(f"{cid} already evaluated on val_full once")
    params = phase_c(cid)
    base = baseline()
    start = time.perf_counter()
    path = OUT / "c_final" / cid / "val_full.json"
    results = eval_preds(pipeline(params, "val_full"), path, o8.subsets())
    ev.write_outputs(results["val_full"], out_dir / "val_full")
    cand = o7.compact(results)
    b_full = load_json(EVID / "baseline" / "val_full" / "metrics.json")
    per_class = [
        {
            "class": a["name"],
            "E6_AP": a["AP"],
            "cand_AP": c["AP"],
            "delta": c["AP"] - a["AP"],
        }
        for a, c in zip(
            b_full["per_class"], results["val_full"]["per_class"], strict=True
        )
    ]
    write_rows(out_dir / "per_class.csv", per_class)
    write_rows(
        out_dir / "subsets.csv",
        [
            {
                "subset": k,
                "E6_mAP": base[k]["mAP"],
                "cand_mAP": cand[k]["mAP"],
                "delta": cand[k]["mAP"] - base[k]["mAP"],
                "frames": cand[k]["num_images"],
                "E6_predictions": base[k]["num_predictions"],
                "cand_predictions": cand[k]["num_predictions"],
            }
            for k in cand
        ],
    )
    write_rows(
        out_dir / "size_density.csv",
        [
            {
                "metric": k,
                "E6": b_full["metrics"][k],
                "cand": results["val_full"]["metrics"][k],
                "delta": results["val_full"]["metrics"][k] - b_full["metrics"][k],
            }
            for k in ("AP50", "AP75", "AP_small", "AP_medium", "AP_large")
        ]
        + [
            {
                "metric": f"density {a['bin']}",
                "E6": a["mAP"],
                "cand": c["mAP"],
                "delta": c["mAP"] - a["mAP"],
            }
            for a, c in zip(
                b_full["density"], results["val_full"]["density"], strict=True
            )
        ],
    )
    g = gate(base, cand, len(per_class))
    record = {
        "candidate": cid,
        "params": params,
        "predictions": rel(path),
        "predictions_sha256": sha256(path),
        "E6": {
            k: base[k]
            for k in (
                "val_full",
                "width_le_1920",
                "without_4k_clip",
                "discovery",
                "confirmation",
            )
        },
        "candidate_metrics": {
            k: cand[k]
            for k in (
                "val_full",
                "width_le_1920",
                "without_4k_clip",
                "discovery",
                "confirmation",
            )
        },
        "gate": GATE,
        **g,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "evaluated_at_kyiv": o7.kyiv_now(),
        "decision": "G1-G7, G9 pass: determinism (G8) next"
        if g["passes_G1_G7_G9"]
        else "rejected (promotion gate)",
    }
    save_json(out_dir / "gate.json", record)
    print(
        json.dumps(
            {k: record[k] for k in ("deltas", "clip_delta_mAP", "checks", "decision")},
            indent=2,
        )
    )


def cmd_determinism(args) -> None:
    cid = args.name
    params = phase_c(cid)
    record = load_json(EVID / "c_final" / cid / "gate.json")
    if not record["passes_G1_G7_G9"]:
        raise SystemExit("failed the promotion gate: no determinism run needed (rule)")
    tmp = OUT / "c_final" / cid / "val_full_rerun.json"
    save_json(tmp, pipeline(params, "val_full"))
    test_path = OUT / "c_final" / cid / "test_predictions_run1.json"
    save_json(test_path, pipeline(params, "test"))
    out = {
        "val_full_sha256_run": record["predictions_sha256"],
        "val_full_sha256_rerun": sha256(tmp),
        "test_sha256_run1": sha256(test_path),
        "checked_at_kyiv": o7.kyiv_now(),
    }
    out["val_full_identical"] = (
        out["val_full_sha256_run"] == out["val_full_sha256_rerun"]
    )
    save_json(EVID / "c_final" / cid / "determinism.json", out)
    print(json.dumps(out, indent=2))


def cmd_test(args) -> None:
    cid = args.name
    params = phase_c(cid)
    cdir = EVID / "c_final" / cid
    record, det = load_json(cdir / "gate.json"), load_json(cdir / "determinism.json")
    if not (record["passes_G1_G7_G9"] and det["val_full_identical"]):
        raise SystemExit("not upload-eligible: no CSV (rule)")
    ready = sorted(
        p.parent.parent.name for p in (EVID / "c_final").glob("*/test/run.json")
    )
    if cid not in ready and len(ready) >= MAX_NEW_CSVS:
        raise SystemExit(f"already {len(ready)} new CSVs: {ready}")
    ckpt = s9.weights_path(params["soup"])
    ckpt_sha = sha256(ckpt)
    preds = pipeline(params, "test")
    pred_path = OUT / "c_final" / cid / "test" / "test_predictions.json"
    save_json(pred_path, preds)
    test_sha = sha256(pred_path)
    if test_sha != det["test_sha256_run1"]:
        raise SystemExit("test predictions differ across processes: stop")
    csv_path = SUB / f"opt09_{cid}_test.csv"
    images = submission.test_image_table()
    rows, stats = submission.build_rows(preds, images)
    submission.write_csv(csv_path, rows)
    problems, vsum = submission.validate_rows(submission.read_csv(csv_path), images)
    tdir = cdir / "test"
    run = {
        "candidate": cid,
        "params": params,
        "rule": "results/opt-09-e6-tuning/selection_rule.md",
        "rule_commit": o8.commit_of(RULE),
        "csv": csv_path.resolve().as_posix(),
        "csv_bytes": csv_path.stat().st_size,
        "csv_sha256": sha256(csv_path),
        "test_predictions": pred_path.resolve().as_posix(),
        "test_predictions_sha256": test_sha,
        "test_predictions_deterministic": True,
        "checkpoint": rel(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_composition": f"{opt09.SOUP_WEIGHTS[params['soup']]} x Model A epoch0.pt ({s8.A[1][:8]}) + "
        f"{round(1 - opt09.SOUP_WEIGHTS[params['soup']], 4)} x epoch1.pt ({s8.B[1][:8]})",
        "writer": stats,
        "project_validator_inline": {
            "valid": not problems,
            "problems": problems,
            **vsum,
        },
        "uploaded": False,
    }
    save_json(tdir / "run.json", run)
    py = sys.executable
    v1 = subprocess.run(
        [py, "scripts/validate_submission.py", str(csv_path)],
        check=False,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    (tdir / "project_validator.txt").write_text(v1.stdout + v1.stderr, encoding="utf-8")
    v2 = subprocess.run(
        [
            py,
            "scripts/check_submission_independent.py",
            str(csv_path),
            "--predictions",
            str(pred_path),
            "--run-json",
            str(tdir / "run.json"),
            "--expected-checkpoint-sha256",
            ckpt_sha,
            "--out",
            str(tdir / "independent_validator.json"),
        ],
        check=False,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    run["validators_exit"] = {"project": v1.returncode, "independent": v2.returncode}
    save_json(tdir / "run.json", run)
    print(v1.stdout[-600:], v2.stdout[-600:], "\nexit", run["validators_exit"])
    if v1.returncode or v2.returncode:
        (tdir / "run.json").rename(tdir / "run_FAILED_validation.json")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cmds = {
        "verify": cmd_verify,
        "baseline": cmd_baseline,
        "screen": cmd_screen,
        "confirm": cmd_confirm,
        "plan-c": cmd_plan_c,
        "full": cmd_full,
        "determinism": cmd_determinism,
        "test": cmd_test,
    }
    parser.add_argument("command", choices=list(cmds))
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    if args.command in ("screen", "confirm") and args.name not in opt09.GRIDS:
        parser.error(f"grid required: {sorted(opt09.GRIDS)}")
    if args.command in ("full", "determinism", "test") and args.name not in (
        "c1",
        "c2",
    ):
        parser.error("Phase C candidate required: c1 or c2")
    cmds[args.command](args)


if __name__ == "__main__":
    main()
