"""OPT-07: one predeclared post-processing candidate per loop iteration on protected Model A (CPU, cached).

    python scripts/opt07_loop.py verify             # protected Model A hashes (checkpoint, CSV, cached predictions)
    python scripts/opt07_loop.py baseline           # Model A on val_full and every gate subset, once
    python scripts/opt07_loop.py run X              # X's committed rule -> transform val_full once -> gate
    python scripts/opt07_loop.py determinism X      # fresh process: val_full + test transforms, sha256 compared
    python scripts/opt07_loop.py test X             # only if promoted: lossless test CSV + both validators

Candidates and frozen parameters: movd/opt07.py EXPERIMENTS; rules: results/opt-07-overnight/X/rule.md;
promotion gate: results/opt-07-overnight/progress.md (fixed before the first candidate). Nothing is uploaded.
Large files go to git-ignored outputs/opt07/ and submissions/.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from movd import (  # noqa: E402
    data,
    opt07,  # noqa: E402
    submission,
)
from movd import evaluate as ev  # noqa: E402

HW2 = REPO.parent
EVID = REPO / "results" / "opt-07-overnight"
OUT = REPO / "outputs" / "opt07"
SUB = REPO / "submissions"
MAX_UPLOAD_READY = 2

MODEL_A = {
    "checkpoint": (
        HW2 / "MOVD-ch14/outputs/final_rtdetr/train/weights/epoch0.pt",
        "6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b",
    ),
    "csv": (
        HW2 / "MOVD-ch14b/submissions/final_rtdetr_epoch1_ch13_fliptile_nms05_test.csv",
        "e587c68b517d6d59f90122946d8a45ef5beac4fb8ae3b24f66ced1268def0da7",
    ),
    "val_full": (
        HW2
        / "MOVD-ch14b/outputs/ch14b/stageA_epoch1/val_full/pipeline_predictions.json",
        "fccda2b3cc14f335468e872e95981592f4e02e5510d329a6a2c808916da7036c",
    ),
    "test": (
        HW2 / "MOVD-ch14b/outputs/ch14b/stageA_epoch1/test/test_predictions.json",
        "5f9c4913378dd5b7d740d70827255675a2035801c0dba851656a70806187930e",
    ),
}
REGISTERED_A = (
    REPO
    / "results/ch-14b-rtdetr-short-finetune/stageA_epoch1/val_full/pipeline/metrics.json"
)
CLIP_4K = "uav0000268_05773_v"
TEST_LIKE_MAX_WIDTH = 1920
DENSE = ">=100"
GATE = {
    "P1_min_delta_mAP": 0.0005,
    "P2_min_delta_mAP_width_le_1920": 0.0005,
    "P3_max_AP_small_loss": 0.003,
    "P4_max_density_ge100_loss": 0.005,
    "P5_min_clips_non_negative": 4,
    "P6_min_delta_mAP_without_4k_clip": 0.0005,
}


# ---------------------------------------------------------------- helpers


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(HW2.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_a(split: str) -> list[dict]:
    path, expected = MODEL_A[split]
    if sha256(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from the protected Model A value")
    return load_json(path)


def subsets() -> dict[str, list[int]]:
    """Gate subsets of val_full: all, width <= 1920, without the 4K clip, and each clip."""
    gt = data.load_coco("val")
    images = {im["id"]: im for im in gt["images"]}
    ids = data.split_image_ids("val_full")
    out = {
        "val_full": list(ids),
        "width_le_1920": [i for i in ids if images[i]["width"] <= TEST_LIKE_MAX_WIDTH],
        "without_4k_clip": [i for i in ids if images[i]["video"] != CLIP_4K],
    }
    for clip in sorted({images[i]["video"] for i in ids}):
        out[f"clip:{clip}"] = [i for i in ids if images[i]["video"] == clip]
    return out


def _eval_job(pred_path: str, name: str, ids: list[int]) -> tuple[str, dict]:
    preds = load_json(Path(pred_path))
    result = ev.evaluate(preds, data.load_coco("val"), ids)
    return name, result


def eval_subsets(pred_path: Path, workers: int = 3) -> dict[str, dict]:
    jobs = subsets()
    order = sorted(jobs, key=lambda k: -len(jobs[k]))  # largest first
    out = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_eval_job, str(pred_path), k, jobs[k]) for k in order]
        for f in cf.as_completed(futures):
            name, result = f.result()
            out[name] = result
    return {k: out[k] for k in jobs}


def dense(result: dict) -> float:
    return next(d["mAP"] for d in result["density"] if d["bin"] == DENSE)


def compact(results: dict[str, dict]) -> dict:
    return {
        k: {
            "num_images": r["num_images"],
            "num_predictions": r["num_predictions"],
            **r["metrics"],
            "density_ge100": dense(r) if k == "val_full" else None,
        }
        for k, r in results.items()
    }


def rule_committed(name: str) -> Path:
    """The candidate's rule must exist, be committed and unchanged, and state the frozen parameters."""
    rule = EVID / name / "rule.md"
    if not rule.exists():
        raise SystemExit(f"{rule} missing: write and commit the rule first")
    rel_rule = rule.relative_to(REPO).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_rule],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", rel_rule, "movd/opt07.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if tracked.returncode or dirty:
        raise SystemExit(f"{rel_rule} or movd/opt07.py is not committed unchanged")
    if params_text(name) not in rule.read_text(encoding="utf-8"):
        raise SystemExit(f"{rel_rule} does not state the frozen parameters of {name}")
    return rule


def params_text(name: str) -> str:
    return json.dumps(opt07.EXPERIMENTS[name], sort_keys=True)


def transform(name: str, split: str) -> list[dict]:
    preds = load_a(split)
    if split == "test":
        nbrs = opt07.test_neighbours(data.load_test_images())
    else:
        nbrs = opt07.val_neighbours(data.load_coco("val"))
    return opt07.apply(name, preds, nbrs)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def kyiv_now() -> str:
    import datetime
    import zoneinfo

    return datetime.datetime.now(zoneinfo.ZoneInfo("Europe/Kyiv")).strftime(
        "%Y-%m-%d %H:%M"
    )


# ---------------------------------------------------------------- commands


def cmd_verify(_args) -> None:
    record = {}
    for k, (path, expected) in MODEL_A.items():
        actual = sha256(path)
        record[k] = {"path": rel(path), "sha256": actual, "matches": actual == expected}
    record["ok"] = all(v["matches"] for v in record.values())
    record["checked_at_kyiv"] = kyiv_now()
    save_json(EVID / "identity.json", record)
    print(json.dumps(record, indent=2))
    if not record["ok"]:
        raise SystemExit("IDENTITY FAILED: repository state unsafe, stop")


def cmd_baseline(_args) -> None:
    target = EVID / "baseline" / "subsets.json"
    if target.exists():
        raise SystemExit("baseline already evaluated")
    path = MODEL_A["val_full"][0]
    load_a("val_full")  # hash check
    start = time.perf_counter()
    results = eval_subsets(path)
    registered = load_json(REGISTERED_A)["metrics"]
    same = results["val_full"]["metrics"] == registered
    ev.write_outputs(results["val_full"], EVID / "baseline" / "val_full")
    save_json(
        target,
        {
            "predictions": rel(path),
            "predictions_sha256": MODEL_A["val_full"][1],
            "matches_registered_ch14b": same,
            "subsets": compact(results),
            "wall_seconds": round(time.perf_counter() - start, 1),
        },
    )
    print(
        "Model A val_full mAP",
        results["val_full"]["metrics"]["mAP"],
        "matches registered:",
        same,
    )
    if not same:
        raise SystemExit(
            "baseline does not reproduce the registered CH-14b value: stop"
        )


def gate(base: dict, cand: dict) -> dict:
    b, c = base, cand
    clip_keys = [k for k in c if k.startswith("clip:")]
    clip_delta = {k[5:]: c[k]["mAP"] - b[k]["mAP"] for k in clip_keys}
    non_neg = sum(d >= 0 for d in clip_delta.values())
    d = {
        "delta_mAP": c["val_full"]["mAP"] - b["val_full"]["mAP"],
        "delta_mAP_width_le_1920": c["width_le_1920"]["mAP"]
        - b["width_le_1920"]["mAP"],
        "delta_AP_small": c["val_full"]["AP_small"] - b["val_full"]["AP_small"],
        "delta_density_ge100": c["val_full"]["density_ge100"]
        - b["val_full"]["density_ge100"],
        "clips_non_negative": non_neg,
        "delta_mAP_without_4k_clip": c["without_4k_clip"]["mAP"]
        - b["without_4k_clip"]["mAP"],
    }
    checks = {
        "P1 val_full mAP >= A + 0.0005": d["delta_mAP"] >= GATE["P1_min_delta_mAP"],
        "P2 width<=1920 mAP >= A + 0.0005": d["delta_mAP_width_le_1920"]
        >= GATE["P2_min_delta_mAP_width_le_1920"],
        "P3 AP_small loss <= 0.003": d["delta_AP_small"]
        >= -GATE["P3_max_AP_small_loss"],
        "P4 density>=100 loss <= 0.005": d["delta_density_ge100"]
        >= -GATE["P4_max_density_ge100_loss"],
        "P5 >= 4 of 7 clips non-negative": non_neg >= GATE["P5_min_clips_non_negative"],
        "P6 without-4K mAP >= A + 0.0005": d["delta_mAP_without_4k_clip"]
        >= GATE["P6_min_delta_mAP_without_4k_clip"],
    }
    return {
        "deltas": d,
        "clip_delta_mAP": clip_delta,
        "checks": checks,
        "passes_P1_P6": all(checks.values()),
    }


def cmd_run(args) -> None:
    name = args.name
    rule = rule_committed(name)
    out_dir = EVID / name
    if (out_dir / "gate.json").exists():
        raise SystemExit(
            f"{name} was already evaluated once; the rule allows one evaluation"
        )
    base = load_json(EVID / "baseline" / "subsets.json")["subsets"]
    start = time.perf_counter()
    preds = transform(name, "val_full")
    t_transform = time.perf_counter() - start
    pred_path = OUT / name / "val_full.json"
    save_json(pred_path, preds)
    results = eval_subsets(pred_path)
    ev.write_outputs(results["val_full"], out_dir / "val_full")
    cand = compact(results)
    g = gate(base, cand)
    a_full = load_json(EVID / "baseline" / "val_full" / "metrics.json")
    rows = [
        {
            "subset": k,
            "A_mAP": base[k]["mAP"],
            "cand_mAP": cand[k]["mAP"],
            "delta": cand[k]["mAP"] - base[k]["mAP"],
            "frames": cand[k]["num_images"],
            "A_predictions": base[k]["num_predictions"],
            "cand_predictions": cand[k]["num_predictions"],
        }
        for k in cand
    ]
    write_rows(out_dir / "subsets.csv", rows)
    per_class = [
        {
            "class": a["name"],
            "A_AP": a["AP"],
            "cand_AP": c["AP"],
            "delta": c["AP"] - a["AP"],
        }
        for a, c in zip(
            a_full["per_class"], results["val_full"]["per_class"], strict=True
        )
    ]
    write_rows(out_dir / "per_class.csv", per_class)
    record = {
        "candidate": name,
        "params": opt07.EXPERIMENTS[name],
        "rule": rule.relative_to(REPO).as_posix(),
        "rule_commit": subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", str(rule.relative_to(REPO))],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "predictions": rel(pred_path),
        "predictions_sha256": sha256(pred_path),
        "num_predictions": len(preds),
        "transform_seconds": round(t_transform, 2),
        "baseline": {
            k: base[k] for k in ("val_full", "width_le_1920", "without_4k_clip")
        },
        "candidate_metrics": {
            k: cand[k] for k in ("val_full", "width_le_1920", "without_4k_clip")
        },
        "gate": GATE,
        **g,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "evaluated_at_kyiv": kyiv_now(),
        "decision": "P1-P6 pass: determinism check next"
        if g["passes_P1_P6"]
        else "rejected (gate P1-P6)",
    }
    save_json(out_dir / "gate.json", record)
    print(
        json.dumps(
            {k: record[k] for k in ("deltas", "clip_delta_mAP", "checks", "decision")},
            indent=2,
        )
    )


def cmd_determinism(args) -> None:
    name = args.name
    record = load_json(EVID / name / "gate.json")
    val = transform(name, "val_full")
    tmp = OUT / name / "val_full_rerun.json"
    save_json(tmp, val)
    test = transform(name, "test")
    test_path = OUT / name / "test_predictions_run1.json"
    save_json(test_path, test)
    out = {
        "val_full_sha256_run": record["predictions_sha256"],
        "val_full_sha256_rerun": sha256(tmp),
        "test_sha256_run1": sha256(test_path),
    }
    out["val_full_identical"] = (
        out["val_full_sha256_run"] == out["val_full_sha256_rerun"]
    )
    save_json(EVID / name / "determinism.json", out)
    print(json.dumps(out, indent=2))


def cmd_test(args) -> None:
    name = args.name
    record = load_json(EVID / name / "gate.json")
    det = load_json(EVID / name / "determinism.json")
    if not (record["passes_P1_P6"] and det["val_full_identical"]):
        raise SystemExit("not promoted: no CSV (rule)")
    ready = sorted(p.parent.name for p in EVID.glob("*/test/run.json"))
    if name not in ready and len(ready) >= MAX_UPLOAD_READY:
        raise SystemExit(f"already {len(ready)} upload-ready CSVs: {ready}")
    if sha256(MODEL_A["checkpoint"][0]) != MODEL_A["checkpoint"][1]:
        raise SystemExit("Model A checkpoint hash mismatch")
    preds = transform(name, "test")
    pred_path = OUT / name / "test" / "test_predictions.json"
    save_json(pred_path, preds)
    test_sha = sha256(pred_path)
    if test_sha != det["test_sha256_run1"]:
        raise SystemExit("test transform is not deterministic across processes: stop")
    csv_path = SUB / f"opt07_{name}_test.csv"
    images = submission.test_image_table()
    rows, stats = submission.build_rows(preds, images)
    submission.write_csv(csv_path, rows)
    problems, vsum = submission.validate_rows(submission.read_csv(csv_path), images)
    run = {
        "candidate": name,
        "params": opt07.EXPERIMENTS[name],
        "rule": record["rule"],
        "rule_commit": record["rule_commit"],
        "csv": csv_path.resolve().as_posix(),
        "csv_bytes": csv_path.stat().st_size,
        "csv_sha256": sha256(csv_path),
        "test_predictions": pred_path.resolve().as_posix(),
        "test_predictions_sha256": test_sha,
        "test_predictions_deterministic": True,
        "input_test_predictions_sha256": MODEL_A["test"][1],
        "checkpoint_sha256": MODEL_A["checkpoint"][1],
        "writer": stats,
        "project_validator_inline": {
            "valid": not problems,
            "problems": problems,
            **vsum,
        },
        "uploaded": False,
    }
    tdir = EVID / name / "test"
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
            MODEL_A["checkpoint"][1],
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
    parser.add_argument(
        "command", choices=["verify", "baseline", "run", "determinism", "test"]
    )
    parser.add_argument("name", nargs="?", choices=sorted(opt07.EXPERIMENTS))
    args = parser.parse_args()
    if args.command in ("run", "determinism", "test") and not args.name:
        parser.error("candidate name required")
    {
        "verify": cmd_verify,
        "baseline": cmd_baseline,
        "run": cmd_run,
        "determinism": cmd_determinism,
        "test": cmd_test,
    }[args.command](args)


if __name__ == "__main__":
    main()
