"""OPT-06A: OPT-02 candidate c4 (A 1.0 : F 0.75, class-wise NMS 0.6, top 500) on val_full (cached, CPU only).

    python scripts/opt06a_c4.py verify     # the six cached inputs match the OPT-02 hashes; c4 parameters frozen
    python scripts/opt06a_c4.py reproduce  # c4 on val_fast reproduces the registered OPT-02 metrics exactly
    python scripts/opt06a_c4.py confirm    # c4 on val_full once, against Model A, competition gate
    python scripts/opt06a_c4.py test       # only if the gate passed: merge cached test predictions, CSV, both validators

Rule and gate: results/opt-06a-c4-valfull/selection_rule.md (committed before any result).
The merge, input loading and evaluation are the unchanged OPT-02 functions (scripts/opt02_ensemble.py).
Large files (merged predictions, CSV) go to git-ignored outputs/opt06a/ and submissions/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import opt02_ensemble as o2  # noqa: E402

from movd import submission  # noqa: E402

REPO = o2.REPO
EVID = REPO / "results" / "opt-06a-c4-valfull"
OUT = REPO / "outputs" / "opt06a"
CSV_OUT = REPO / "submissions" / "opt06a_c4_ensemble_test.csv"
CAND_ID = "c4_w075_nms06"
FROZEN = {"w_A": 1.0, "w_B": 0.75, "iou": 0.6, "route": None}
GATE = {"min_mAP": 0.2840, "max_AP_small_drop": 0.005, "max_density_ge100_drop": 0.010}
OPT02_IDENTITY = REPO / "results/opt-02-rtdetr-ensemble/identity.json"
OPT02_C4_METRICS = (
    REPO / "results/opt-02-rtdetr-ensemble/val_fast" / CAND_ID / "metrics.json"
)


def candidate() -> dict:
    grid = o2.candidates()
    cand = next(c for c in grid["candidates"] if c["id"] == CAND_ID)
    if {k: cand[k] for k in FROZEN} != FROZEN or grid["max_det"] != 500:
        raise SystemExit(f"OPT-02 {CAND_ID} differs from the frozen OPT-06A candidate")
    return cand


def load(split: str) -> dict[str, list[dict]]:
    """Both models' cached predictions; every file must match the hash OPT-02 recorded."""
    identity = json.loads(OPT02_IDENTITY.read_text(encoding="utf-8"))["inputs"]
    out = {}
    for n in o2.NAMES:
        path = o2.INPUTS[(n, split)][0]
        if o2.sha256(path) != identity[f"{n}/{split}"]["sha256"]:
            raise SystemExit(f"{path}: sha256 differs from OPT-02 identity.json")
        out[n] = json.loads(path.read_text(encoding="utf-8"))
    return out


def row(name: str, result: dict) -> dict:
    return o2.summary_row(name, result)


def cmd_verify(_args) -> None:
    cand = candidate()
    identity = json.loads(OPT02_IDENTITY.read_text(encoding="utf-8"))
    record = {"candidate": cand, "inputs": {}, "checkpoints": {}}
    for (n, split), (path, _committed) in o2.INPUTS.items():
        actual = o2.sha256(path)
        expected = identity["inputs"][f"{n}/{split}"]["sha256"]
        record["inputs"][f"{n}/{split}"] = {"path": o2.rel(path), "sha256": actual, "bytes": path.stat().st_size,
                                            "matches_opt02": actual == expected}  # fmt: skip
    for n, (path, expected) in o2.CHECKPOINTS.items():  # hashed only, never loaded
        actual = o2.sha256(path)
        record["checkpoints"][n] = {
            "path": o2.rel(path),
            "sha256": actual,
            "matches_evidence": actual == expected,
        }
    record["ok"] = all(v["matches_opt02"] for v in record["inputs"].values()) and all(
        v["matches_evidence"] for v in record["checkpoints"].values()
    )
    o2.save_json(EVID / "identity.json", record)
    print(json.dumps(record, indent=2))
    if not record["ok"]:
        raise SystemExit("IDENTITY FAILED")


def cmd_reproduce(_args) -> None:
    cand = candidate()
    inputs = load("val_fast")
    preds, seconds = o2.merge(cand, inputs)
    result = o2.evaluate(preds, "val_fast", EVID / "val_fast" / CAND_ID)
    stored = json.loads(OPT02_C4_METRICS.read_text(encoding="utf-8"))
    same = {
        k: result[k] == stored[k]
        for k in ("metrics", "per_class", "density", "num_predictions")
    }
    out = {"registered": o2.rel(OPT02_C4_METRICS), "mAP": result["metrics"]["mAP"],
           "registered_mAP": stored["metrics"]["mAP"], "identical": same, "reproduced": all(same.values()),
           "merge_seconds": round(seconds, 2), **o2.counts(preds, result["num_images"])}  # fmt: skip
    o2.save_json(EVID / "reproduce_val_fast.json", out)
    print(json.dumps(out, indent=2))
    if not out["reproduced"]:
        raise SystemExit("c4 val_fast does not reproduce: stop (rule)")


def cmd_confirm(_args) -> None:
    if not json.loads((EVID / "reproduce_val_fast.json").read_text(encoding="utf-8"))[
        "reproduced"
    ]:
        raise SystemExit("val_fast reproduction failed; val_full is not run")
    if (EVID / "gate_val_full.json").exists():
        raise SystemExit(
            "val_full was already evaluated once; the rule allows one evaluation"
        )
    cand = candidate()
    split = "val_full"
    start = time.perf_counter()
    inputs = load(split)
    base = o2.evaluate(inputs["A"], split, EVID / split / "A")
    preds, seconds = o2.merge(cand, inputs)
    result = o2.evaluate(preds, split, EVID / split / CAND_ID)
    m, mb = result["metrics"], base["metrics"]
    checks = {
        "K-a mAP >= 0.2840": m["mAP"] >= GATE["min_mAP"],
        "K-b AP_small >= A - 0.005": m["AP_small"] >= mb["AP_small"] - GATE["max_AP_small_drop"],
        "K-c density>=100 >= A - 0.010": o2.dense(result) >= o2.dense(base) - GATE["max_density_ge100_drop"],
    }  # fmt: skip
    ra, rc = row("A", base), row(CAND_ID, result)
    delta = {"id": "delta (c4 - A)", **{k: rc[k] - ra[k] for k in ra if k != "id"}}
    o2.write_csv_table(EVID / "comparison_val_full.csv", [
        {**ra, **o2.counts(inputs["A"], base["num_images"])},
        {**rc, **o2.counts(preds, result["num_images"]), "merge_seconds": round(seconds, 2)}, delta,
    ])  # fmt: skip
    passed = all(checks.values())
    o2.save_json(EVID / "gate_val_full.json", {
        "rule": "results/opt-06a-c4-valfull/selection_rule.md", "candidate": cand, "gate": GATE,
        "baseline_A": mb, "baseline_A_density_ge100": o2.dense(base),
        "candidate_metrics": m, "candidate_density_ge100": o2.dense(result),
        "delta_mAP": m["mAP"] - mb["mAP"], "delta_AP_small": m["AP_small"] - mb["AP_small"],
        "delta_density_ge100": o2.dense(result) - o2.dense(base), "checks": checks, "passes": passed,
        "decision": "gate passes: one test CSV for a Kaggle exploration" if passed else
                    "gate fails: no CSV; Model A and its 0.2774301362 submission are kept",
        "merge_seconds": round(seconds, 2), "wall_seconds": round(time.perf_counter() - start, 1),
        "peak_process_memory_bytes": o2.rss(),
    })  # fmt: skip
    print(json.dumps(checks, indent=2), "\npasses:", passed)


def cmd_test(_args) -> None:
    gate = json.loads((EVID / "gate_val_full.json").read_text(encoding="utf-8"))
    if not gate["passes"]:
        raise SystemExit("the val_full gate failed; no CSV is generated (rule)")
    cand = candidate()
    for n, (path, expected) in o2.CHECKPOINTS.items():
        if o2.sha256(path) != expected:
            raise SystemExit(f"checkpoint {n} hash mismatch")
    inputs = load("test")
    preds, seconds = o2.merge(cand, inputs)
    c = o2.counts(preds, o2.TEST_FRAMES)
    preds = o2.strip(preds)
    pred_path = OUT / "test" / "test_predictions.json"
    o2.save_json(pred_path, preds)
    images = submission.test_image_table()
    rows, stats = submission.build_rows(preds, images)
    submission.write_csv(CSV_OUT, rows)
    problems, vsum = submission.validate_rows(submission.read_csv(CSV_OUT), images)
    record = {
        "candidate": cand, "csv": CSV_OUT.resolve().as_posix(), "csv_bytes": CSV_OUT.stat().st_size,
        "csv_sha256": o2.sha256(CSV_OUT), "test_predictions": pred_path.resolve().as_posix(),
        "test_predictions_sha256": o2.sha256(pred_path),
        "checkpoint_sha256": o2.CHECKPOINTS["A"][1], "checkpoint_sha256_F": o2.CHECKPOINTS["B"][1],
        "inputs_sha256": {n: o2.INPUTS[(n, "test")][1] for n in o2.NAMES}, "merge_seconds": round(seconds, 2),
        **c, "writer": stats, "project_validator_inline": {"valid": not problems, "problems": problems, **vsum},
        "uploaded": False,
    }  # fmt: skip
    o2.save_json(EVID / "test" / "run.json", record)
    py = sys.executable
    v1 = subprocess.run([py, "scripts/validate_submission.py", str(CSV_OUT)],
                        check=False, cwd=REPO, capture_output=True, text=True)  # fmt: skip
    (EVID / "test" / "project_validator.txt").write_text(
        v1.stdout + v1.stderr, encoding="utf-8"
    )
    v2 = subprocess.run([py, "scripts/check_submission_independent.py", str(CSV_OUT), "--predictions", str(pred_path),
                         "--run-json", str(EVID / "test" / "run.json"), "--expected-checkpoint-sha256",
                         o2.CHECKPOINTS["A"][1], "--out", str(EVID / "test" / "independent_validator.json")],
                        check=False, cwd=REPO, capture_output=True, text=True)  # fmt: skip
    record["validators_exit"] = {"project": v1.returncode, "independent": v2.returncode}
    o2.save_json(EVID / "test" / "run.json", record)
    print(
        v1.stdout[-600:],
        "\nproject exit",
        v1.returncode,
        "| independent exit",
        v2.returncode,
    )
    if v1.returncode or v2.returncode:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["verify", "reproduce", "confirm", "test"])
    args = parser.parse_args()
    {
        "verify": cmd_verify,
        "reproduce": cmd_reproduce,
        "confirm": cmd_confirm,
        "test": cmd_test,
    }[args.command](args)


if __name__ == "__main__":
    main()
