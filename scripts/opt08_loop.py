"""OPT-08: staged, predeclared candidates against the protected OPT-07 x2 baseline (CPU, cached predictions).

    python scripts/opt08_loop.py verify            # protected hashes: Model A, x2 / x4 predictions + CSVs, raw caches
    python scripts/opt08_loop.py split             # the frozen 4-clip discovery / 3-clip confirmation split
    python scripts/opt08_loop.py baseline          # x2 on every subset, once (must equal the OPT-07 x2 record)
    python scripts/opt08_loop.py screen E          # S1: every frozen candidate of E on the discovery clips only
    python scripts/opt08_loop.py confirm E         # S2: the S1 selection on the confirmation clips, once
    python scripts/opt08_loop.py full E            # S3: val_full + gate subsets, once, only after S1 and S2 pass
    python scripts/opt08_loop.py determinism E     # fresh process: val_full + test transform, sha256 compared
    python scripts/opt08_loop.py test E            # only if eligible: lossless test CSV + both validators

Frozen grids: movd/opt08.py EXPERIMENTS; rules: results/opt-08-overnight/E/rule.md (committed before `screen`);
stage rules and the upload-eligibility gate U1-U8: results/opt-08-overnight/TASK.md. Nothing is uploaded.
Large files go to git-ignored outputs/opt08/ and submissions/.
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

from movd import data, opt07, opt08, submission
from movd import evaluate as ev

HW2 = REPO.parent
EVID = REPO / "results" / "opt-08-overnight"
OUT = REPO / "outputs" / "opt08"
SUB = REPO / "submissions"
MAX_UPLOAD_READY = 4
OPT07 = HW2 / "MOVD-opt07"

# Protected inputs (read only, never written).
X2 = {
    "val_full": (
        OPT07 / "outputs/opt07/x2_persistence_boost/val_full.json",
        "6c3a929f9e0f968963d7f292b3a9e9bef763515803b608ad0b041b7d3ab22c03",
    ),
    "test": (
        OPT07 / "outputs/opt07/x2_persistence_boost/test/test_predictions.json",
        "fc4fdc8f50d077c3012578f26aa8f4e2300d4ef2e83889ec5b30b0f5107d9191",
    ),
    "csv": (
        OPT07 / "submissions/opt07_x2_persistence_boost_test.csv",
        "9dc8fb03dc708b2b78b05afeff29289317a810f0219e9be4e2fd50029433c867",
    ),
}
X4 = {
    "val_full": (
        OPT07 / "outputs/opt07/x4_border_confirmation/val_full.json",
        "a7c97b072cd091c1b78981991c848085760b26e77a928f97132889f0ac721a5f",
    ),
    "test": (
        OPT07 / "outputs/opt07/x4_border_confirmation/test/test_predictions.json",
        "bb7adfaa6111b825b478e38ee69ec555e5efdf5886869abb472c7662be176b8c",
    ),
    "csv": (
        OPT07 / "submissions/opt07_x4_border_confirmation_test.csv",
        "09a5c8d14633f4bade9b9c5b2edd6122c38ca8c3de0cf8afa6b5f79184cd2fcb",
    ),
}
X2_RECORD = REPO / "results/opt-07-overnight/x2_persistence_boost/gate.json"

STAGE = {"S1_min_delta_discovery": 0.0003, "S2_min_delta_confirmation": 0.0}
GATE = {
    "U1_min_delta_mAP": 0.0003,
    "U2_min_delta_mAP_width_le_1920": 0.0,
    "U2_preferred_delta": 0.0005,
    "U3_max_AP_small_loss": 0.003,
    "U4_max_density_ge100_loss": 0.003,
    "U5_min_clips_non_negative": 5,
    "U6_min_delta_mAP_without_4k_clip": 0.0003,
}
FROZEN_FILES = (
    "movd/evaluate.py",
    "movd/submission.py",
    "configs",
    "scripts/validate_submission.py",
)


# ---------------------------------------------------------------- helpers

sha256, save_json, load_json, rel, write_rows = (
    o7.sha256,
    o7.save_json,
    o7.load_json,
    o7.rel,
    o7.write_rows,
)


def checked(path: Path, expected: str):
    if sha256(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from the protected value")
    return load_json(path)


def split_record() -> dict:
    return load_json(EVID / "split.json")


def subsets() -> dict[str, list[int]]:
    out = o7.subsets()
    gt = data.load_coco("val")
    clip_of = {im["id"]: im["video"] for im in gt["images"]}
    sp = split_record()
    for stage in ("discovery", "confirmation"):
        out[stage] = [i for i in out["val_full"] if clip_of[i] in sp[stage]]
    return out


def _eval_job(pred_path: str, name: str, ids: list[int]) -> tuple[str, dict]:
    preds = load_json(Path(pred_path))
    return name, ev.evaluate(preds, data.load_coco("val"), ids)


def eval_paths(
    jobs: list[tuple[Path, str, list[int]]], workers: int = 3
) -> dict[str, dict]:
    jobs = sorted(jobs, key=lambda j: -len(j[2]))
    out = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_eval_job, str(p), n, ids) for p, n, ids in jobs]
        for f in cf.as_completed(futures):
            name, result = f.result()
            out[name] = result
    return out


def restrict(preds: list[dict], ids: list[int]) -> list[dict]:
    keep = set(ids)
    return [p for p in preds if p["image_id"] in keep]


def neighbours(split: str) -> dict:
    if split == "test":
        return opt07.test_neighbours(data.load_test_images())
    return opt07.val_neighbours(data.load_coco("val"))


def transform(params: dict, split: str) -> list[dict]:
    """The candidate's predictions on `split` from protected cached inputs (never from evaluation results)."""
    nbrs = neighbours(split)
    if params["kind"] == "x4_then_x2":
        sys.path.insert(0, str(REPO / "scripts"))
        import opt03_edge

        if (params["factor"], params["margin"]) != opt03_edge.CANDIDATES["e2"]:
            raise SystemExit("E1 border parameters differ from OPT-03 e2 / OPT-07 x4")
        x4, _ = opt03_edge.run_candidate(split, params["factor"], params["margin"])
        same = o7.same_predictions(x4, load_json(X4[split][0]))
        if not same["identical"]:
            raise SystemExit(
                f"raw cache does not reproduce OPT-07 x4 on {split}: {same}"
            )
        return opt08.apply(params, x4, nbrs)
    return opt08.apply(params, o7.load_a(split), nbrs)


def rule_committed(exp: str) -> Path:
    rule = EVID / exp / "rule.md"
    rel_rule = rule.relative_to(REPO).as_posix()
    if not rule.exists():
        raise SystemExit(f"{rel_rule} missing: write and commit the rule first")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_rule],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            rel_rule,
            "movd/opt08.py",
            "movd/opt07.py",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if tracked.returncode or dirty:
        raise SystemExit(f"{rel_rule} or movd/opt08.py is not committed unchanged")
    text = rule.read_text(encoding="utf-8")
    for cid, params in opt08.EXPERIMENTS[exp].items():
        if f"{cid} {json.dumps(params, sort_keys=True)}" not in text:
            raise SystemExit(
                f"{rel_rule} does not state the frozen parameters of {cid}"
            )
    return rule


def commit_of(path: Path) -> str:
    return subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", str(path.relative_to(REPO))],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def frozen_unchanged() -> bool:
    """Evaluator, submission writer/validator and protocol configs equal origin/develop's."""
    r = subprocess.run(
        ["git", "diff", "--quiet", "origin/develop", "--", *FROZEN_FILES],
        cwd=REPO,
        check=False,
    )
    return r.returncode == 0


def selected(exp: str) -> tuple[str, dict]:
    s1 = load_json(EVID / exp / "screen.json")
    if not s1["passes_S1"]:
        raise SystemExit(f"{exp} failed S1: stop")
    cid = s1["selected"]
    return cid, opt08.EXPERIMENTS[exp][cid]


# ---------------------------------------------------------------- commands


def cmd_verify(_args) -> None:
    items = {f"model_A_{k}": v for k, v in o7.MODEL_A.items()}
    items |= {f"x2_{k}": v for k, v in X2.items()}
    items |= {f"x4_{k}": v for k, v in X4.items()}
    record = {}
    for k, (path, expected) in items.items():
        actual = sha256(path)
        record[k] = {"path": rel(path), "sha256": actual, "matches": actual == expected}
    for split in ("val_full", "test"):
        p = o7.raw_cache(split)
        record[f"raw_cache_{split}"] = {
            "path": rel(p),
            "sha256": sha256(p),
            "matches": True,
        }
    record["frozen_files_unchanged"] = frozen_unchanged()
    record["ok"] = (
        all(v["matches"] for v in record.values() if isinstance(v, dict))
        and record["frozen_files_unchanged"]
    )
    record["checked_at_kyiv"] = o7.kyiv_now()
    save_json(EVID / "identity.json", record)
    print(json.dumps(record, indent=2))
    if not record["ok"]:
        raise SystemExit("IDENTITY FAILED: repository state unsafe, stop")


def cmd_split(_args) -> None:
    import hashlib

    target = EVID / "split.json"
    gt = data.load_coco("val")
    images = {im["id"]: im for im in gt["images"]}
    ids = data.split_image_ids("val_full")
    clips = sorted({images[i]["video"] for i in ids})
    sp = opt08.split_clips(clips)
    record = {
        "rule": f"sort clips by sha256('{opt08.SPLIT_SALT}' + clip name); first {opt08.N_DISCOVERY} = discovery",
        "digests": {
            c: hashlib.sha256((opt08.SPLIT_SALT + c).encode()).hexdigest()
            for c in clips
        },
        **sp,
        "frames": {
            stage: sum(images[i]["video"] in sp[stage] for i in ids) for stage in sp
        },
        "widths": {
            c: next(images[i]["width"] for i in ids if images[i]["video"] == c)
            for c in clips
        },
    }
    if target.exists() and load_json(target) != record:
        raise SystemExit("split.json exists and differs: the split is frozen")
    save_json(target, record)
    print(json.dumps(record, indent=2))


def cmd_baseline(_args) -> None:
    target = EVID / "baseline" / "subsets.json"
    if target.exists():
        raise SystemExit("baseline already evaluated")
    path, expected = X2["val_full"]
    checked(path, expected)
    start = time.perf_counter()
    jobs = subsets()
    results = eval_paths([(path, k, ids) for k, ids in jobs.items()])
    results = {k: results[k] for k in jobs}
    registered = load_json(X2_RECORD)["candidate_metrics"]["val_full"]
    same = all(
        results["val_full"]["metrics"][k] == registered[k]
        for k in results["val_full"]["metrics"]
    )
    ev.write_outputs(results["val_full"], EVID / "baseline" / "val_full")
    save_json(
        target,
        {
            "predictions": rel(path),
            "predictions_sha256": expected,
            "matches_opt07_x2_record": same,
            "subsets": o7.compact(results),
            "wall_seconds": round(time.perf_counter() - start, 1),
        },
    )
    print(
        "x2 val_full mAP",
        results["val_full"]["metrics"]["mAP"],
        "matches OPT-07:",
        same,
    )
    if not same:
        raise SystemExit("baseline does not reproduce the OPT-07 x2 record: stop")


def cmd_screen(args) -> None:
    exp = args.name
    rule = rule_committed(exp)
    out_dir = EVID / exp
    if (out_dir / "screen.json").exists():
        raise SystemExit(f"{exp} already screened once")
    base = load_json(EVID / "baseline" / "subsets.json")["subsets"]["discovery"]
    ids = subsets()["discovery"]
    start = time.perf_counter()
    jobs = []
    meta = {}
    for cid, params in opt08.EXPERIMENTS[exp].items():
        t0 = time.perf_counter()
        preds = transform(params, "val_full")
        full_path = OUT / exp / cid / "val_full.json"
        save_json(full_path, preds)
        disc_path = OUT / exp / cid / "discovery.json"
        save_json(disc_path, restrict(preds, ids))
        meta[cid] = {
            "params": params,
            "val_full_predictions_sha256": sha256(full_path),
            "num_predictions": len(preds),
            "transform_seconds": round(time.perf_counter() - t0, 1),
        }
        jobs.append((disc_path, cid, ids))
        del preds
    results = eval_paths(jobs)
    rows = []
    for cid, info in meta.items():
        m = results[cid]["metrics"]
        info["discovery"] = m
        info["delta_discovery_mAP"] = m["mAP"] - base["mAP"]
        rows.append(
            {
                "candidate": cid,
                **info["params"],
                "x2_mAP": base["mAP"],
                "mAP": m["mAP"],
                "delta": m["mAP"] - base["mAP"],
                "AP_small": m["AP_small"],
            }
        )
    best = max(
        meta, key=lambda c: (meta[c]["delta_discovery_mAP"], -list(meta).index(c))
    )
    passes = meta[best]["delta_discovery_mAP"] >= STAGE["S1_min_delta_discovery"]
    write_rows(out_dir / "screen.csv", rows)
    record = {
        "experiment": exp,
        "rule": rule.relative_to(REPO).as_posix(),
        "rule_commit": commit_of(rule),
        "stage_rule": STAGE,
        "baseline_x2_discovery": base,
        "candidates": meta,
        "selected": best,
        "passes_S1": passes,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "evaluated_at_kyiv": o7.kyiv_now(),
        "decision": "S1 pass: confirmation next"
        if passes
        else "rejected at S1 (discovery)",
    }
    save_json(out_dir / "screen.json", record)
    print(
        json.dumps(
            {c: round(meta[c]["delta_discovery_mAP"], 6) for c in meta}, indent=2
        )
    )
    print("selected", best, record["decision"])


def cmd_confirm(args) -> None:
    exp = args.name
    out_dir = EVID / exp
    if (out_dir / "confirm.json").exists():
        raise SystemExit(f"{exp} already confirmed once")
    cid, params = selected(exp)
    base = load_json(EVID / "baseline" / "subsets.json")["subsets"]["confirmation"]
    ids = subsets()["confirmation"]
    full_path = OUT / exp / cid / "val_full.json"
    s1 = load_json(out_dir / "screen.json")["candidates"][cid]
    if sha256(full_path) != s1["val_full_predictions_sha256"]:
        raise SystemExit("stored candidate predictions changed since S1: stop")
    conf_path = OUT / exp / cid / "confirmation.json"
    save_json(conf_path, restrict(load_json(full_path), ids))
    m = eval_paths([(conf_path, cid, ids)], workers=1)[cid]["metrics"]
    delta = m["mAP"] - base["mAP"]
    passes = delta >= STAGE["S2_min_delta_confirmation"]
    record = {
        "experiment": exp,
        "selected": cid,
        "params": params,
        "baseline_x2_confirmation": base,
        "confirmation": m,
        "delta_confirmation_mAP": delta,
        "passes_S2": passes,
        "evaluated_at_kyiv": o7.kyiv_now(),
        "decision": "S2 pass: val_full next"
        if passes
        else "rejected at S2 (confirmation)",
    }
    save_json(out_dir / "confirm.json", record)
    print(cid, "confirmation delta", delta, record["decision"])


def gate(base: dict, cand: dict) -> dict:
    g7 = o7.gate(base, cand)  # same deltas, OPT-08 thresholds below
    d = g7["deltas"]
    checks = {
        "U1 val_full mAP >= x2 + 0.0003": d["delta_mAP"] >= GATE["U1_min_delta_mAP"],
        "U2 width<=1920 mAP not below x2": d["delta_mAP_width_le_1920"]
        >= GATE["U2_min_delta_mAP_width_le_1920"],
        "U3 AP_small loss <= 0.003": d["delta_AP_small"]
        >= -GATE["U3_max_AP_small_loss"],
        "U4 density>=100 loss <= 0.003": d["delta_density_ge100"]
        >= -GATE["U4_max_density_ge100_loss"],
        "U5 >= 5 of 7 clips non-negative": d["clips_non_negative"]
        >= GATE["U5_min_clips_non_negative"],
        "U6 without-4K mAP >= x2 + 0.0003": d["delta_mAP_without_4k_clip"]
        >= GATE["U6_min_delta_mAP_without_4k_clip"],
        "U8 evaluator / protocol / CSV format unchanged": frozen_unchanged(),
    }
    return {
        "deltas": d,
        "clip_delta_mAP": g7["clip_delta_mAP"],
        "checks": checks,
        "width_le_1920_preferred_met": d["delta_mAP_width_le_1920"]
        >= GATE["U2_preferred_delta"],
        "passes_U1_U6_U8": all(checks.values()),
    }


def cmd_full(args) -> None:
    exp = args.name
    out_dir = EVID / exp
    if (out_dir / "gate.json").exists():
        raise SystemExit(f"{exp} already evaluated on val_full once")
    cid, params = selected(exp)
    if not load_json(out_dir / "confirm.json")["passes_S2"]:
        raise SystemExit(f"{exp} failed S2: no val_full run (rule)")
    full_path = OUT / exp / cid / "val_full.json"
    if (
        sha256(full_path)
        != load_json(out_dir / "screen.json")["candidates"][cid][
            "val_full_predictions_sha256"
        ]
    ):
        raise SystemExit("stored candidate predictions changed since S1: stop")
    base = load_json(EVID / "baseline" / "subsets.json")["subsets"]
    start = time.perf_counter()
    jobs = subsets()
    results = eval_paths([(full_path, k, ids) for k, ids in jobs.items()])
    results = {k: results[k] for k in jobs}
    ev.write_outputs(results["val_full"], out_dir / "val_full")
    cand = o7.compact(results)
    g = gate({k: v for k, v in base.items()}, cand)
    write_rows(
        out_dir / "subsets.csv",
        [
            {
                "subset": k,
                "x2_mAP": base[k]["mAP"],
                "cand_mAP": cand[k]["mAP"],
                "delta": cand[k]["mAP"] - base[k]["mAP"],
                "frames": cand[k]["num_images"],
                "x2_predictions": base[k]["num_predictions"],
                "cand_predictions": cand[k]["num_predictions"],
            }
            for k in cand
        ],
    )
    b_full = load_json(EVID / "baseline" / "val_full" / "metrics.json")
    write_rows(
        out_dir / "per_class.csv",
        [
            {
                "class": a["name"],
                "x2_AP": a["AP"],
                "cand_AP": c["AP"],
                "delta": c["AP"] - a["AP"],
            }
            for a, c in zip(
                b_full["per_class"], results["val_full"]["per_class"], strict=True
            )
        ],
    )
    write_rows(
        out_dir / "size_density.csv",
        [
            {
                "metric": k,
                "x2": b_full["metrics"][k],
                "cand": results["val_full"]["metrics"][k],
                "delta": results["val_full"]["metrics"][k] - b_full["metrics"][k],
            }
            for k in ("AP_small", "AP_medium", "AP_large")
        ]
        + [
            {
                "metric": f"density {a['bin']}",
                "x2": a["mAP"],
                "cand": c["mAP"],
                "delta": c["mAP"] - a["mAP"],
            }
            for a, c in zip(
                b_full["density"], results["val_full"]["density"], strict=True
            )
        ],
    )
    record = {
        "experiment": exp,
        "selected": cid,
        "params": params,
        "predictions": rel(full_path),
        "predictions_sha256": sha256(full_path),
        "baseline_x2": {
            k: base[k] for k in ("val_full", "width_le_1920", "without_4k_clip")
        },
        "candidate_metrics": {
            k: cand[k] for k in ("val_full", "width_le_1920", "without_4k_clip")
        },
        "gate": GATE,
        **g,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "evaluated_at_kyiv": o7.kyiv_now(),
        "decision": "U1-U6, U8 pass: determinism (U7) next"
        if g["passes_U1_U6_U8"]
        else "rejected (gate U1-U6/U8)",
    }
    save_json(out_dir / "gate.json", record)
    print(
        json.dumps(
            {k: record[k] for k in ("deltas", "clip_delta_mAP", "checks", "decision")},
            indent=2,
        )
    )


def cmd_determinism(args) -> None:
    exp = args.name
    cid, params = selected(exp)
    record = load_json(EVID / exp / "gate.json")
    val = transform(params, "val_full")
    tmp = OUT / exp / cid / "val_full_rerun.json"
    save_json(tmp, val)
    del val
    test = transform(params, "test")
    test_path = OUT / exp / cid / "test_predictions_run1.json"
    save_json(test_path, test)
    out = {
        "val_full_sha256_run": record["predictions_sha256"],
        "val_full_sha256_rerun": sha256(tmp),
        "test_sha256_run1": sha256(test_path),
    }
    out["val_full_identical"] = (
        out["val_full_sha256_run"] == out["val_full_sha256_rerun"]
    )
    save_json(EVID / exp / "determinism.json", out)
    print(json.dumps(out, indent=2))


def cmd_test(args) -> None:
    exp = args.name
    cid, params = selected(exp)
    record = load_json(EVID / exp / "gate.json")
    det = load_json(EVID / exp / "determinism.json")
    if not (record["passes_U1_U6_U8"] and det["val_full_identical"]):
        raise SystemExit("not upload-eligible: no CSV (rule)")
    ready = sorted(p.parent.parent.name for p in EVID.glob("*/test/run.json"))
    if exp not in ready and len(ready) >= MAX_UPLOAD_READY:
        raise SystemExit(f"already {len(ready)} upload-ready CSVs: {ready}")
    if sha256(o7.MODEL_A["checkpoint"][0]) != o7.MODEL_A["checkpoint"][1]:
        raise SystemExit("Model A checkpoint hash mismatch")
    preds = transform(params, "test")
    pred_path = OUT / exp / cid / "test" / "test_predictions.json"
    save_json(pred_path, preds)
    test_sha = sha256(pred_path)
    if test_sha != det["test_sha256_run1"]:
        raise SystemExit("test transform is not deterministic across processes: stop")
    csv_path = SUB / f"opt08_{exp}_{cid}_test.csv"
    images = submission.test_image_table()
    rows, stats = submission.build_rows(preds, images)
    submission.write_csv(csv_path, rows)
    problems, vsum = submission.validate_rows(submission.read_csv(csv_path), images)
    tdir = EVID / exp / "test"
    run = {
        "experiment": exp,
        "candidate": cid,
        "params": params,
        "rule": f"results/opt-08-overnight/{exp}/rule.md",
        "rule_commit": load_json(EVID / exp / "screen.json")["rule_commit"],
        "csv": csv_path.resolve().as_posix(),
        "csv_bytes": csv_path.stat().st_size,
        "csv_sha256": sha256(csv_path),
        "test_predictions": pred_path.resolve().as_posix(),
        "test_predictions_sha256": test_sha,
        "test_predictions_deterministic": True,
        "checkpoint_sha256": o7.MODEL_A["checkpoint"][1],
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
            o7.MODEL_A["checkpoint"][1],
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
        "split": cmd_split,
        "baseline": cmd_baseline,
        "screen": cmd_screen,
        "confirm": cmd_confirm,
        "full": cmd_full,
        "determinism": cmd_determinism,
        "test": cmd_test,
    }
    parser.add_argument("command", choices=list(cmds))
    parser.add_argument("name", nargs="?", choices=sorted(opt08.EXPERIMENTS))
    args = parser.parse_args()
    if args.command not in ("verify", "split", "baseline") and not args.name:
        parser.error("experiment name required")
    cmds[args.command](args)


if __name__ == "__main__":
    main()
