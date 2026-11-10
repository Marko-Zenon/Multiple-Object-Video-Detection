"""OPT-02: ensemble of the two validated RT-DETR-l models on their cached CH-13-pipeline predictions (CPU only).

    python scripts/opt02_ensemble.py verify      # input/checkpoint hashes; cached A and B reproduce their registered metrics
    python scripts/opt02_ensemble.py screen      # val_fast: Model A + the 7 predeclared candidates, gate, choice
    python scripts/opt02_ensemble.py confirm     # val_full: the chosen candidate once, against Model A
    python scripts/opt02_ensemble.py test        # only if confirm passed: merge 3,811 test frames, CSV, both validators
    python scripts/opt02_ensemble.py determinism # re-merge and compare hashes

Rule, grid and gates: results/opt-02-rtdetr-ensemble/selection_rule.md and candidates.json (committed before any result).
Large files (merged predictions, CSV) go to git-ignored outputs/opt02/ and submissions/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, ensemble, submission  # noqa: E402
from movd import evaluate as ev  # noqa: E402

REPO = data.REPO_ROOT
HW2 = REPO.parent
EVID = REPO / "results" / "opt-02-rtdetr-ensemble"
OUT = REPO / "outputs" / "opt02"
CSV_OUT = REPO / "submissions" / "opt02_rtdetr_ensemble_test.csv"
NAMES = ["A", "B"]

CHECKPOINTS = {
    "A": (
        HW2 / "MOVD-ch14/outputs/final_rtdetr/train/weights/epoch0.pt",
        "6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b",
    ),
    "B": (
        HW2 / "MOVD-ch11/outputs/rtdetr_baseline/train/weights/last.pt",
        "da5d36258918e7fdddcb9cfe6e87f6a78ef8ee1a04e65deb3c0eedb42fb6e872",
    ),
}
# cached CH-13-pipeline predictions (post class-wise NMS 0.5 and the per-frame top-500 cap)
INPUTS = {
    ("A", "val_fast"): (
        HW2 / "MOVD-ch14b/outputs/ch14b/stageA_epoch1/val_fast/pipeline_predictions.json",
        "87137d7df1478253733cf0f1d5d611067c5c22252a63df0dfee5b6da67275a0e",
    ),
    ("A", "val_full"): (
        HW2 / "MOVD-ch14b/outputs/ch14b/stageA_epoch1/val_full/pipeline_predictions.json",
        "fccda2b3cc14f335468e872e95981592f4e02e5510d329a6a2c808916da7036c",
    ),
    ("A", "test"): (
        HW2 / "MOVD-ch14b/outputs/ch14b/stageA_epoch1/test/test_predictions.json",
        "5f9c4913378dd5b7d740d70827255675a2035801c0dba851656a70806187930e",
    ),
    ("B", "val_fast"): (HW2 / "MOVD-ch13/outputs/ch13/preds/val_fast_combo_flip_tile_nms05.json", None),
    ("B", "val_full"): (HW2 / "MOVD-ch13/outputs/ch13/preds/val_full_combo_flip_tile_nms05.json", None),
    ("B", "test"): (
        HW2 / "MOVD-ch13/outputs/ch13_test/test_predictions.json",
        "184ddccc963df0bfa5f31665500c671388d4e8b4bdd48c605640d6acc3647ccd",
    ),
}  # fmt: skip
# registered metrics the cached predictions must reproduce exactly (B val files have no committed hash)
REGISTERED = {
    ("A", "val_fast"): REPO / "results/ch-14b-rtdetr-short-finetune/stageA_epoch1/val_fast/pipeline/metrics.json",
    ("A", "val_full"): REPO / "results/ch-14b-rtdetr-short-finetune/stageA_epoch1/val_full/pipeline/metrics.json",
    ("B", "val_fast"): REPO / "results/ch-13-inference/runs/val_fast/combo_flip_tile_nms05/metrics.json",
    ("B", "val_full"): REPO / "results/ch-13-inference/runs/val_full/combo_flip_tile_nms05/metrics.json",
}  # fmt: skip
# measured full test inference (CH-13 pipeline, RTX 5070) of each model, for the runtime projection
TEST_SECONDS = {
    "A": (REPO / "results/ch-14b-rtdetr-short-finetune/final_submission.json", ("test_inference", "seconds")),
    "B": (HW2 / "MOVD-ch13/outputs/ch13_test/run.json", ("inference_seconds_all_shards",)),
}  # fmt: skip
TEST_FRAMES = 3811
DENSE = ">=100"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(HW2.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_input(model: str, split: str) -> list[dict]:
    path, expected = INPUTS[(model, split)]
    if expected is not None and sha256(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from the committed evidence")
    return json.loads(path.read_text(encoding="utf-8"))


def candidates() -> dict:
    return json.loads((EVID / "candidates.json").read_text(encoding="utf-8"))


def merge(cand: dict, inputs: dict[str, list[dict]]) -> tuple[list[dict], float]:
    """Ensemble one candidate; returns (predictions, seconds). Records are tagged with their source model."""
    tagged = [[{**r, "_m": n} for r in inputs[n]] for n in NAMES]
    start = time.perf_counter()
    out = ensemble.ensemble(
        tagged,
        [cand["w_A"], cand["w_B"]],
        cand["iou"],
        candidates()["max_det"],
        ensemble.routes_from_config(cand["route"], NAMES),
    )
    return out, time.perf_counter() - start


def strip(preds: list[dict]) -> list[dict]:
    return [{k: v for k, v in p.items() if k != "_m"} for p in preds]


def counts(preds: list[dict], num_images: int, max_det: int = 500) -> dict:
    per_image: dict[int, int] = {}
    from_b = 0
    for p in preds:
        per_image[p["image_id"]] = per_image.get(p["image_id"], 0) + 1
        from_b += p.get("_m") == "B"
    return {
        "detections": len(preds),
        "mean_per_frame": round(len(preds) / num_images, 2),
        "frames_at_cap": sum(n >= max_det for n in per_image.values()),
        "frames": num_images,
        "share_from_B": round(from_b / len(preds), 4) if preds and "_m" in preds[0] else None,
    }  # fmt: skip


def summary_row(name: str, result: dict) -> dict:
    m = result["metrics"]
    row = {
        "id": name,
        **{
            k: m[k]
            for k in ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
        },
    }
    row.update({f"AP_{r['name']}": r["AP"] for r in result["per_class"]})
    row.update({f"density_{d['bin']}": d["mAP"] for d in result["density"]})
    return row  # fmt: skip


def dense(result: dict) -> float:
    return next(d["mAP"] for d in result["density"] if d["bin"] == DENSE)


def evaluate(preds: list[dict], split: str, out_dir: Path) -> dict:
    result = ev.evaluate_split(strip(preds), split)
    ev.write_outputs(result, out_dir)
    return result


def test_seconds() -> dict:
    out = {}
    for n, (path, keys) in TEST_SECONDS.items():
        value = json.loads(path.read_text(encoding="utf-8"))
        for k in keys:
            value = value[k]
        out[n] = {"seconds": float(value), "source": rel(path) + ":" + ".".join(keys)}
    return out


def write_csv_table(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = list(
            dict.fromkeys(k for r in rows for k in r)
        )  # union of the rows' keys
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rss() -> int:
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


# ---------------------------------------------------------------- commands


def cmd_verify(_args) -> None:
    record = {"checkpoints": {}, "inputs": {}, "registered_metrics_reproduced": {}}
    for n, (path, expected) in CHECKPOINTS.items():
        actual = sha256(path)
        record["checkpoints"][n] = {
            "path": rel(path),
            "sha256": actual,
            "matches_evidence": actual == expected,
        }
    for (n, split), (path, expected) in INPUTS.items():
        actual = sha256(path)
        record["inputs"][f"{n}/{split}"] = {
            "path": rel(path), "sha256": actual, "bytes": path.stat().st_size,
            "committed_sha256": expected, "matches_evidence": None if expected is None else actual == expected,
        }  # fmt: skip
    for (n, split), reg_path in REGISTERED.items():
        preds = load_input(n, split)
        result = ev.evaluate_split(preds, split)
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        same = all(
            result[k] == reg[k]
            for k in ("metrics", "per_class", "density", "num_predictions")
        )
        c = counts(preds, result["num_images"])
        record["registered_metrics_reproduced"][f"{n}/{split}"] = {
            "registered": rel(reg_path), "identical": same, "mAP": result["metrics"]["mAP"], **c,
        }  # fmt: skip
        print(
            f"{n} {split}: mAP {result['metrics']['mAP']:.4f}, identical to registered: {same}, {c}"
        )
    for n in NAMES:
        preds = load_input(n, "test")
        c = counts(preds, TEST_FRAMES)
        record["inputs"][f"{n}/test"].update(c)
        print(f"{n} test: {c}")
    record["ok"] = all(v["matches_evidence"] for v in record["checkpoints"].values()) and all(
        v["matches_evidence"] is not False for v in record["inputs"].values()
    ) and all(v["identical"] for v in record["registered_metrics_reproduced"].values())  # fmt: skip
    save_json(EVID / "identity.json", record)
    print("identity ok" if record["ok"] else "IDENTITY FAILED")
    if not record["ok"]:
        raise SystemExit(1)


def cmd_screen(_args) -> None:
    grid = candidates()
    gate = grid["gate_val_fast"]
    split = "val_fast"
    inputs = {n: load_input(n, split) for n in NAMES}
    base = evaluate(inputs["A"], split, EVID / split / "A")
    rows = [
        {**summary_row("A (baseline)", base), **counts(inputs["A"], base["num_images"])}
    ]
    runtime = test_seconds()
    decisions = []
    for cand in grid["candidates"]:
        preds, seconds = merge(cand, inputs)
        result = evaluate(preds, split, EVID / split / cand["id"])
        c = counts(preds, result["num_images"])
        merge_per_frame = seconds / result["num_images"]
        projected = (
            sum(v["seconds"] for v in runtime.values()) + merge_per_frame * TEST_FRAMES
        ) / 60
        m, mb = result["metrics"], base["metrics"]
        checks = {
            "G-a mAP >= A + 0.003": m["mAP"] >= mb["mAP"] + gate["min_mAP_gain"],
            "G-b AP_small >= A - 0.003": m["AP_small"] >= mb["AP_small"] - gate["max_AP_small_drop"],
            "G-c density>=100 >= A - 0.005": dense(result) >= dense(base) - gate["max_density_ge100_drop"],
            "G-d projected test minutes <= 90": projected <= gate["max_projected_test_minutes"],
        }  # fmt: skip
        decisions.append({
            "id": cand["id"], "mAP": m["mAP"], "AP_small": m["AP_small"], "density_ge100": dense(result),
            "delta_mAP": m["mAP"] - mb["mAP"], "delta_AP_small": m["AP_small"] - mb["AP_small"],
            "delta_density_ge100": dense(result) - dense(base),
            "merge_seconds": round(seconds, 2), "projected_test_minutes": round(projected, 2),
            "checks": checks, "passes": all(checks.values()),
        })  # fmt: skip
        rows.append({**summary_row(cand["id"], result), **c, "merge_seconds": round(seconds, 2),
                     "projected_test_minutes": round(projected, 2), "passes_gate": all(checks.values())})  # fmt: skip
        print(f"{cand['id']}: mAP {m['mAP']:.4f} ({m['mAP'] - mb['mAP']:+.4f}), AP_S {m['AP_small']:.4f}, "
              f">=100 {dense(result):.4f}, merge {seconds:.1f} s, pass {all(checks.values())}", flush=True)  # fmt: skip
    passing = [d for d in decisions if d["passes"]]
    passing.sort(
        key=lambda d: (
            -d["mAP"],
            -d["AP_small"],
            -d["density_ge100"],
            d["projected_test_minutes"],
            d["id"],
        )
    )
    chosen = passing[0]["id"] if passing else None
    write_csv_table(EVID / "screening_val_fast.csv", rows)
    save_json(EVID / "selection_val_fast.json", {
        "rule": "results/opt-02-rtdetr-ensemble/selection_rule.md",
        "baseline_A": {"mAP": base["metrics"]["mAP"], "AP_small": base["metrics"]["AP_small"], "density_ge100": dense(base)},
        "test_inference_seconds": runtime, "candidates": decisions,
        "passing": [d["id"] for d in passing], "chosen": chosen,
        "decision": f"{chosen} goes to the single val_full check" if chosen else
                    "no candidate passes: model selection stops, Model A and its 0.2774301362 submission are kept",
        "peak_process_memory_bytes": rss(), "platform": platform.platform(),
    })  # fmt: skip
    print(f"passing: {[d['id'] for d in passing]}; chosen: {chosen}")


def cmd_confirm(_args) -> None:
    sel = json.loads((EVID / "selection_val_fast.json").read_text(encoding="utf-8"))
    if not sel["chosen"]:
        raise SystemExit("no candidate passed val_fast; val_full is not run (rule)")
    if (EVID / "selection_val_full.json").exists():
        raise SystemExit(
            "val_full was already evaluated once; the rule allows one evaluation"
        )
    grid = candidates()
    gate = grid["gate_val_full"]
    cand = next(c for c in grid["candidates"] if c["id"] == sel["chosen"])
    split = "val_full"
    inputs = {n: load_input(n, split) for n in NAMES}
    base = evaluate(inputs["A"], split, EVID / split / "A")
    preds, seconds = merge(cand, inputs)
    result = evaluate(preds, split, EVID / split / cand["id"])
    m, mb = result["metrics"], base["metrics"]
    checks = {
        "mAP >= A + 0.003": m["mAP"] >= mb["mAP"] + gate["min_mAP_gain"],
        "AP_small >= A - 0.003": m["AP_small"] >= mb["AP_small"] - gate["max_AP_small_drop"],
        "density>=100 >= A - 0.005": dense(result) >= dense(base) - gate["max_density_ge100_drop"],
    }  # fmt: skip
    rows = [{**summary_row("A (baseline)", base), **counts(inputs["A"], base["num_images"])},
            {**summary_row(cand["id"], result), **counts(preds, result["num_images"]), "merge_seconds": round(seconds, 2)}]  # fmt: skip
    write_csv_table(EVID / "confirm_val_full.csv", rows)
    selected = all(checks.values())
    save_json(EVID / "selection_val_full.json", {
        "candidate": cand, "baseline_A": mb, "baseline_A_density_ge100": dense(base),
        "candidate_metrics": m, "candidate_density_ge100": dense(result),
        "delta_mAP": m["mAP"] - mb["mAP"], "checks": checks, "selected_over_A": selected,
        "decision": f"{cand['id']} replaces Model A" if selected else
                    "the ensemble does not pass val_full: Model A and its 0.2774301362 submission are kept",
        "merge_seconds": round(seconds, 2), "peak_process_memory_bytes": rss(),
    })  # fmt: skip
    print(json.dumps(checks, indent=2), "\nselected over A:", selected)


def cmd_test(_args) -> None:
    sel = json.loads((EVID / "selection_val_full.json").read_text(encoding="utf-8"))
    if not sel["selected_over_A"]:
        raise SystemExit(
            "the ensemble did not pass val_full; no CSV is generated (rule)"
        )
    cand = sel["candidate"]
    for n, (path, expected) in CHECKPOINTS.items():
        if sha256(path) != expected:
            raise SystemExit(f"checkpoint {n} hash mismatch")
    inputs = {n: load_input(n, "test") for n in NAMES}
    preds, seconds = merge(cand, inputs)
    c = counts(preds, TEST_FRAMES)
    preds = strip(preds)
    pred_path = OUT / "test" / "test_predictions.json"
    save_json(pred_path, preds)
    images = submission.test_image_table()
    rows, stats = submission.build_rows(preds, images)
    submission.write_csv(CSV_OUT, rows)
    problems, vsum = submission.validate_rows(submission.read_csv(CSV_OUT), images)
    record = {
        "candidate": cand, "csv": CSV_OUT.resolve().as_posix(), "csv_bytes": CSV_OUT.stat().st_size,
        "csv_sha256": sha256(CSV_OUT), "test_predictions": pred_path.resolve().as_posix(),
        "test_predictions_sha256": sha256(pred_path),
        "checkpoint_sha256": CHECKPOINTS["A"][1], "checkpoint_sha256_B": CHECKPOINTS["B"][1],
        "inputs_sha256": {n: INPUTS[(n, "test")][1] for n in NAMES}, "merge_seconds": round(seconds, 2),
        **c, "writer": stats, "project_validator_inline": {"valid": not problems, "problems": problems, **vsum},
    }  # fmt: skip
    save_json(EVID / "test" / "run.json", record)
    py = sys.executable
    v1 = subprocess.run(
        [py, "scripts/validate_submission.py", str(CSV_OUT)],
        check=False,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    (EVID / "test" / "project_validator.txt").write_text(
        v1.stdout + v1.stderr, encoding="utf-8"
    )
    v2 = subprocess.run([py, "scripts/check_submission_independent.py", str(CSV_OUT), "--predictions", str(pred_path),
                         "--run-json", str(EVID / "test" / "run.json"), "--expected-checkpoint-sha256", CHECKPOINTS["A"][1],
                         "--out", str(EVID / "test" / "independent_validator.json")],
                        check=False, cwd=REPO, capture_output=True, text=True)  # fmt: skip
    print(
        v1.stdout[-600:],
        "\nproject exit",
        v1.returncode,
        "| independent exit",
        v2.returncode,
    )
    record["validators_exit"] = {"project": v1.returncode, "independent": v2.returncode}
    save_json(EVID / "test" / "run.json", record)


def cmd_determinism(_args) -> None:
    grid = candidates()
    sel = json.loads((EVID / "selection_val_fast.json").read_text(encoding="utf-8"))
    cand_id = sel["chosen"] or grid["candidates"][0]["id"]
    cand = next(c for c in grid["candidates"] if c["id"] == cand_id)
    inputs = {n: load_input(n, "val_fast") for n in NAMES}
    digests = []
    for _ in range(2):
        preds, _s = merge(cand, inputs)
        digests.append(hashlib.sha256(json.dumps(strip(preds)).encode()).hexdigest())
    # every tracked metrics.json of the screen must be reproduced by a fresh merge + evaluation
    regenerated = {}
    for c in grid["candidates"]:
        preds, _s = merge(c, inputs)
        res = ev.evaluate_split(strip(preds), "val_fast")
        stored = json.loads(
            (EVID / "val_fast" / c["id"] / "metrics.json").read_text(encoding="utf-8")
        )
        regenerated[c["id"]] = all(
            res[k] == stored[k] for k in ("metrics", "per_class", "density")
        )
    out = {"candidate": cand_id, "merge_sha256_run1": digests[0], "merge_sha256_run2": digests[1],
           "identical": digests[0] == digests[1], "val_fast_metrics_regenerated": regenerated,
           "all_regenerated": all(regenerated.values())}  # fmt: skip
    save_json(EVID / "determinism.json", out)
    print(json.dumps(out, indent=2))
    if not (out["identical"] and out["all_regenerated"]):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=["verify", "screen", "confirm", "test", "determinism"]
    )
    args = parser.parse_args()
    {"verify": cmd_verify, "screen": cmd_screen, "confirm": cmd_confirm, "test": cmd_test,
     "determinism": cmd_determinism}[args.command](args)  # fmt: skip


if __name__ == "__main__":
    main()
