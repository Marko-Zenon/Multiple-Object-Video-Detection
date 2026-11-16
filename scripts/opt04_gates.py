"""OPT-04 gates G0-G5 from stored artifacts (rule: results/opt-04-model-scale/selection_rule.md, frozen in 1ba4118).

    python scripts/opt04_gates.py g0 --cand P --smoke-run opt04_rtdetrx_smoke --speed g0_P --log <smoke logs...>
    python scripts/opt04_gates.py g1 --cand P --run opt04_rtdetrx_e1 --log <train log> [--restarts 0]
    python scripts/opt04_gates.py g2 --cand P --name P_rtdetrx_e1
    python scripts/opt04_gates.py g3 --cand P --name P_rtdetrx_e1          (G3 and G5 from val_full)
    python scripts/opt04_gates.py g4 --cand P --name P_rtdetrx_e1 --csv submissions/<file>.csv
    python scripts/opt04_gates.py verify                                   (recompute every stored gate, compare)

Each command recomputes one gate from files only (no GPU) and stores it in `results/opt-04-model-scale/gates.json`
under the candidate (P or FB), with its inputs, every check and `pass`. Thresholds are the literal values of the rule;
metrics are compared unrounded. Every mAP comes from the project evaluator (`movd/evaluate.py`): the stored
`metrics.json` of `scripts/opt04_infer.py`, and for the G2 test-like-subset guard a fresh `movd.evaluate.evaluate`
call restricted to the val_fast frames of width <= 1920, for the candidate and for Model A's stored predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data  # noqa: E402
from movd import evaluate as ev  # noqa: E402

HERE = data.REPO_ROOT / "results" / "opt-04-model-scale"
OUTPUTS = data.REPO_ROOT / "outputs" / "opt04"
GATES = HERE / "gates.json"
HW2 = data.REPO_ROOT.parent

MODEL_A = {
    "checkpoint": HW2
    / "MOVD-ch14"
    / "outputs"
    / "final_rtdetr"
    / "train"
    / "weights"
    / "epoch0.pt",
    "checkpoint_sha256": "6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b",
    "evidence": data.REPO_ROOT
    / "results"
    / "ch-14b-rtdetr-short-finetune"
    / "stageA_epoch1",
    "val_fast_predictions": HW2
    / "MOVD-ch14b"
    / "outputs"
    / "ch14b"
    / "stageA_epoch1"
    / "val_fast"
    / "pipeline_predictions.json",
    "val_fast_predictions_sha256": "87137d7df1478253733cf0f1d5d611067c5c22252a63df0dfee5b6da67275a0e",
}
PROTECTED_CSV_SHA256 = {
    "e587c68b517d6d59f90122946d8a45ef5beac4fb8ae3b24f66ced1268def0da7": "Model A",
    "73f73e219dfbf3b674277703f7314d14c316d0800bcd046634aaecd27f2b793b": "F",
}
CANDIDATES = {"P": {"imgsz": 1344}, "FB": {"imgsz": 1600}}

# Literal thresholds of the frozen rule.
G0_MAX_RESERVED_BYTES = (
    7.0e9  # "peak reserved <= 7.0 GB" (decimal GB, the stricter reading)
)
G0_MAX_EPOCH_MINUTES = 120.0
G0_MAX_TEST_MINUTES = 90.0
G2 = {"mAP": 0.2922, "AP_small": 0.1397, "bin_ge100": 0.3465}
G3 = {"mAP": 0.2890, "AP_small": 0.1474, "bin_ge100": 0.3447, "secondary_wins": 5}
G5_MAP = 0.2923
SPREAD = 0.0063
TEST_LIKE_MAX_WIDTH = 1920
TRAIN_FINAL_FRAMES = 6072
SECONDARY = ("AP50", "AP75", "AP_small", "AP_medium", "AP_large")
OOM_PATTERN = re.compile(
    r"out of memory|OutOfMemory|CUDA error|reduc\w* batch|batch size reduced|AutoBatch",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(data.REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def bins(metrics: dict) -> dict:
    """{bin label: mAP} of an evaluator result."""
    return {row["bin"]: row["mAP"] for row in metrics["density"]}


def check(name: str, value, threshold, op: str = ">=") -> dict:
    ok = value is not None and (
        value >= threshold if op == ">=" else value <= threshold
    )
    return {
        "check": name,
        "value": value,
        "threshold": threshold,
        "op": op,
        "pass": bool(ok),
    }


def log_problems(paths) -> list[str]:
    """Lines of the logs that look like an OOM, a CUDA error or an automatic batch reduction."""
    hits = []
    for path in paths or []:
        path = (
            data.REPO_ROOT / path
        )  # stored relative to the repo root (an absolute path stays absolute)
        for line in (
            Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            if OOM_PATTERN.search(line):
                hits.append(f"{Path(path).name}: {line.strip()[:200]}")
    return hits


# ---------------------------------------------------------------- gates


def gate_g0(cand: str, smoke_run: str, speed_name: str, logs) -> dict:
    """Smoke feasibility: finite losses, stop + resume, peak reserved, projected epoch and test time, clean logs."""
    runtime = load(data.REPO_ROOT / "results" / smoke_run / "runtime.json")
    processes = runtime.get("smoke_processes", [])
    statuses = [p["status"] for p in processes]
    peak = max((p.get("peak_cuda_reserved_bytes", 0) for p in processes), default=None)
    rates = []
    for p in processes:
        n = p.get("batches", 0)
        if n > 1 and "first_batch_elapsed_seconds" in p:
            rates.append(
                (p["last_batch_elapsed_seconds"] - p["first_batch_elapsed_seconds"])
                / (n - 1)
            )
    epoch_minutes = round(max(rates) * TRAIN_FINAL_FRAMES / 60, 1) if rates else None
    losses_finite = all(
        all(
            math.isfinite(p[k])
            for k in ("first_loss", "last_loss", "minimum_loss", "maximum_loss")
        )
        for p in processes
    ) and bool(processes)
    speed = load(HERE / "speed" / f"{speed_name}.json")
    pipeline_rows = [r for r in speed["rows"] if r["mode"] == "pipeline"]
    test_minutes = max(r["test_minutes_3811_frames"] for r in pipeline_rows)
    problems = log_problems(logs)
    checks = [
        {
            "check": "finite losses in every smoke process",
            "value": losses_finite,
            "pass": losses_finite,
        },
        {
            "check": "stop after epoch 1, then resume to completion",
            "value": statuses,
            "pass": statuses == ["checkpoint_stop", "completed"]
            and runtime.get("epoch_checkpoints") == ["epoch0.pt", "epoch1.pt"],
        },
        check("peak CUDA reserved bytes", peak, G0_MAX_RESERVED_BYTES, "<="),
        {
            "check": "no OOM / CUDA error / batch-reduce log line",
            "value": problems,
            "pass": not problems,
        },
        check(
            "projected train_final epoch minutes",
            epoch_minutes,
            G0_MAX_EPOCH_MINUTES,
            "<=",
        ),
        check(
            "projected test minutes (30-frame pipeline speed)",
            test_minutes,
            G0_MAX_TEST_MINUTES,
            "<=",
        ),
    ]
    return {
        "inputs": {
            "smoke_run": smoke_run,
            "runtime": f"results/{smoke_run}/runtime.json",
            "speed": rel(HERE / "speed" / f"{speed_name}.json"),
            "logs": [rel(p) for p in logs or []],
            "seconds_per_batch": [round(r, 4) for r in rates],
        },
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def gate_g1(cand: str, run_id: str, logs, restarts: int) -> dict:
    """Training identity of the epoch-1 checkpoint."""
    import torch

    weights = data.REPO_ROOT / "outputs" / run_id / "train" / "weights" / "epoch0.pt"
    config = load(data.REPO_ROOT / "results" / run_id / "config.json")
    runtime = load(data.REPO_ROOT / "results" / run_id / "runtime.json")
    process = (runtime.get("training_processes") or [{}])[-1]
    exists = weights.exists()
    args = {}
    if exists:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        args = ckpt.get("train_args", {})
        epoch_field = ckpt.get("epoch")
        del ckpt
    else:
        epoch_field = None
    finite = all(
        k in process and math.isfinite(process[k])
        for k in ("first_loss", "last_loss", "minimum_loss", "maximum_loss")
    )
    want_imgsz = CANDIDATES[cand]["imgsz"]
    problems = log_problems(logs)
    checks = [
        {"check": "epoch0.pt written", "value": rel(weights), "pass": exists},
        {
            "check": "checkpoint epoch field",
            "value": epoch_field,
            "pass": epoch_field == 0,
        },
        {
            "check": "run status",
            "value": process.get("status"),
            "pass": process.get("status") == "stopped_after_epoch_1",
        },
        {"check": "finite losses", "value": finite, "pass": finite},
        {
            "check": "train_args epochs",
            "value": args.get("epochs"),
            "pass": args.get("epochs") == 6,
        },
        {
            "check": "train_args imgsz",
            "value": args.get("imgsz"),
            "pass": args.get("imgsz") == want_imgsz,
        },
        {
            "check": "train_args batch",
            "value": args.get("batch"),
            "pass": args.get("batch") == 1,
        },
        {
            "check": "train_args seed",
            "value": args.get("seed"),
            "pass": args.get("seed") == 42,
        },
        {
            "check": "train_args fraction",
            "value": args.get("fraction"),
            "pass": args.get("fraction") in (None, 1.0),
        },
        {
            "check": "train split train_final (config + data.yaml)",
            "value": [config.get("train_split"), args.get("data")],
            "pass": config.get("train_split") == "train_final"
            and str(args.get("data", ""))
            .replace("\\", "/")
            .endswith("yolo_data_final/data.yaml"),
        },
        {
            "check": "samples and batches of epoch 1",
            "value": [process.get("samples"), process.get("batches")],
            "pass": process.get("samples") == TRAIN_FINAL_FRAMES
            and process.get("batches") == TRAIN_FINAL_FRAMES,
        },
        {
            "check": "no OOM-retry / batch-reduce event in the log",
            "value": problems,
            "pass": not problems,
        },
        check("restarts", restarts, 1, "<="),
    ]
    return {
        "inputs": {
            "run_id": run_id,
            "checkpoint_sha256": sha256(weights) if exists else None,
            "train_seconds": process.get("seconds"),
            "peak_cuda_reserved_bytes": process.get("peak_cuda_reserved_bytes"),
            "peak_cuda_allocated_bytes": process.get("peak_cuda_allocated_bytes"),
            "logs": [rel(p) for p in logs or []],
        },
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def test_like_ids(split: str = "val_fast") -> list[int]:
    images = {im["id"]: im for im in data.load_coco("val")["images"]}
    return [
        i
        for i in data.split_image_ids(split)
        if images[i]["width"] <= TEST_LIKE_MAX_WIDTH
    ]


def subset_guard(name: str) -> dict:
    """mAP of the candidate and of Model A on the val_fast frames of width <= 1920 (same evaluator, same frames)."""
    run = load(HERE / name / "val_fast" / "run.json")
    x_path = OUTPUTS / name / "val_fast" / "pipeline_predictions.json"
    if sha256(x_path) != run["pipeline_predictions_sha256"]:
        raise SystemExit(f"{x_path} does not match its run.json hash")
    a_path = MODEL_A["val_fast_predictions"]
    if sha256(a_path) != MODEL_A["val_fast_predictions_sha256"]:
        raise SystemExit(f"{a_path} does not match the registered Model A hash")
    ids = test_like_ids()
    gt = data.load_coco("val")
    x = ev.evaluate(ev.load_predictions(x_path), gt, ids)
    a = ev.evaluate(ev.load_predictions(a_path), gt, ids)
    return {
        "frames": len(ids),
        "max_width": TEST_LIKE_MAX_WIDTH,
        "candidate_mAP": x["metrics"]["mAP"],
        "model_A_mAP": a["metrics"]["mAP"],
        "candidate_predictions_sha256": run["pipeline_predictions_sha256"],
        "model_A_predictions_sha256": MODEL_A["val_fast_predictions_sha256"],
    }


def gate_g2(cand: str, name: str) -> dict:
    x = load(HERE / name / "val_fast" / "pipeline" / "metrics.json")
    guard = subset_guard(name)
    checks = [
        check("val_fast mAP", x["metrics"]["mAP"], G2["mAP"]),
        check("val_fast AP_small", x["metrics"]["AP_small"], G2["AP_small"]),
        check("val_fast >=100 bin mAP", bins(x)[">=100"], G2["bin_ge100"]),
        check(
            "test-like subset mAP (width <= 1920) vs Model A",
            guard["candidate_mAP"],
            guard["model_A_mAP"],
        ),
    ]
    return {
        "inputs": {
            "metrics": rel(HERE / name / "val_fast" / "pipeline" / "metrics.json"),
            "subset_guard": guard,
        },
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def secondary(x: dict, a: dict) -> list[dict]:
    """The 9 secondary checks: X leads A (X > A, unrounded)."""
    rows = [
        {"check": k, "candidate": x["metrics"][k], "model_A": a["metrics"][k]}
        for k in SECONDARY
    ]
    xb, ab = bins(x), bins(a)
    rows += [
        {"check": f"density {b}", "candidate": xb[b], "model_A": ab[b]} for b in ab
    ]
    for row in rows:
        row["lead"] = row["candidate"] > row["model_A"]
    return rows


def gate_g3_g5(cand: str, name: str) -> tuple[dict, dict]:
    x = load(HERE / name / "val_full" / "pipeline" / "metrics.json")
    a = load(MODEL_A["evidence"] / "val_full" / "pipeline" / "metrics.json")
    sec = secondary(x, a)
    wins = sum(r["lead"] for r in sec)
    guards = [
        check("val_full AP_small", x["metrics"]["AP_small"], G3["AP_small"]),
        check("val_full >=100 bin mAP", bins(x)[">=100"], G3["bin_ge100"]),
        check("secondary checks led vs Model A (of 9)", wins, G3["secondary_wins"]),
    ]
    inputs = {
        "metrics": rel(HERE / name / "val_full" / "pipeline" / "metrics.json"),
        "model_A_metrics": rel(
            MODEL_A["evidence"] / "val_full" / "pipeline" / "metrics.json"
        ),
        "secondary": sec,
        "delta_mAP_vs_A": x["metrics"]["mAP"] - a["metrics"]["mAP"],
    }
    g3_checks = [check("val_full mAP", x["metrics"]["mAP"], G3["mAP"]), *guards]
    g5_checks = [
        check("val_full mAP (A + 0.0063)", x["metrics"]["mAP"], G5_MAP),
        *guards,
    ]
    g3 = {
        "inputs": inputs,
        "checks": g3_checks,
        "pass": all(c["pass"] for c in g3_checks),
    }
    g5 = {
        "inputs": {"metrics": inputs["metrics"]},
        "checks": g5_checks,
        "pass": all(c["pass"] for c in g5_checks),
    }
    return g3, g5


def run_validator(args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, *args],
        check=False,
        cwd=data.REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "command": " ".join(args),
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-1500:],
    }


def gate_g4(cand: str, name: str, csv: Path) -> dict:
    csv = Path(csv)
    run = load(HERE / name / "test" / "run.json")
    predictions = data.REPO_ROOT / run["test_predictions"]
    out = HERE / name / "submission"
    out.mkdir(parents=True, exist_ok=True)
    project = run_validator(["scripts/validate_submission.py", str(csv)])
    independent = run_validator(
        [
            "scripts/check_submission_independent.py", str(csv),
            "--predictions", str(predictions),
            "--run-json", str(HERE / name / "test" / "run.json"),
            "--expected-checkpoint-sha256", run["checkpoint_sha256"],
            "--out", str(out / "independent_validator.json"),
        ]
    )  # fmt: skip
    (out / "project_validator.txt").write_text(project["stdout_tail"], encoding="utf-8")
    report = (
        load(out / "independent_validator.json")
        if (out / "independent_validator.json").exists()
        else {}
    )
    with open(csv, encoding="utf-8") as f:
        rows = sum(1 for _ in f) - 1
    csv_sha = sha256(csv)
    differ = report.get("rows_differing_from_predictions", report.get("rows_differ"))
    checks = [
        {
            "check": "non-smoke test run",
            "value": run["smoke"],
            "pass": run["smoke"] is False,
        },
        {"check": "rows", "value": rows, "pass": rows == 3811},
        {
            "check": "validate_submission.py exit code",
            "value": project["exit_code"],
            "pass": project["exit_code"] == 0,
        },
        {
            "check": "check_submission_independent.py exit code",
            "value": independent["exit_code"],
            "pass": independent["exit_code"] == 0,
        },
        {
            "check": "rows differing from the raw predictions",
            "value": differ,
            "pass": differ == 0,
        },
        {
            "check": "test inference time recorded",
            "value": run.get("inference_seconds_this_run"),
            "pass": bool(run.get("inference_seconds_this_run")),
        },
        {
            "check": "CSV differs from the protected Model A / F CSVs",
            "value": PROTECTED_CSV_SHA256.get(csv_sha, "new"),
            "pass": csv_sha not in PROTECTED_CSV_SHA256,
        },
    ]
    return {
        "inputs": {
            "csv_path": csv.resolve().as_posix(),
            "csv_sha256": csv_sha,
            "csv_bytes": csv.stat().st_size,
            "test_predictions_sha256": run["test_predictions_sha256"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "test_seconds": run.get("inference_seconds_this_run"),
            "independent_report": rel(out / "independent_validator.json"),
            "independent_report_keys": sorted(report),
        },
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


# ---------------------------------------------------------------- storage


def load_gates() -> dict:
    if GATES.exists():
        return load(GATES)
    return {
        "rule": "results/opt-04-model-scale/selection_rule.md (commit 1ba4118)",
        "P": {},
        "FB": {},
    }


def store(cand: str, gate: str, value: dict, request: dict) -> None:
    gates = load_gates()
    value = {"request": request, **value}
    gates.setdefault(cand, {})[gate] = value
    GATES.parent.mkdir(parents=True, exist_ok=True)
    GATES.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    print(f"{cand} {gate}: {'PASS' if value['pass'] else 'FAIL'}")
    for c in value["checks"]:
        print(
            f"  [{'x' if c['pass'] else ' '}] {c['check']}: {c.get('value')} (threshold {c.get('threshold', '-')})"
        )


def compute(gate: str, cand: str, request: dict) -> dict | tuple[dict, dict]:
    if gate == "G0":
        return gate_g0(cand, request["smoke_run"], request["speed"], request.get("log"))
    if gate == "G1":
        return gate_g1(
            cand, request["run"], request.get("log"), request.get("restarts", 0)
        )
    if gate == "G2":
        return gate_g2(cand, request["name"])
    if gate in ("G3", "G5"):
        g3, g5 = gate_g3_g5(cand, request["name"])
        return g3 if gate == "G3" else g5
    raise ValueError(gate)


def strip_volatile(value):
    return json.loads(json.dumps(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("g0", "g1", "g2", "g3", "g4"):
        p = sub.add_parser(name)
        p.add_argument("--cand", choices=tuple(CANDIDATES), required=True)
        if name == "g0":
            p.add_argument("--smoke-run", required=True)
            p.add_argument("--speed", required=True)
            p.add_argument("--log", nargs="*", default=[])
        elif name == "g1":
            p.add_argument("--run", required=True)
            p.add_argument("--log", nargs="*", default=[])
            p.add_argument("--restarts", type=int, default=0)
        else:
            p.add_argument("--name", required=True)
        if name == "g4":
            p.add_argument("--csv", type=Path, required=True)
    sub.add_parser("verify")
    args = parser.parse_args()

    if args.command == "verify":
        gates, bad = load_gates(), 0
        for cand in CANDIDATES:
            for gate, stored in gates.get(cand, {}).items():
                if gate == "G4":
                    continue  # re-running the validators is the G4 command itself
                fresh = strip_volatile(
                    {
                        "request": stored["request"],
                        **compute(gate, cand, stored["request"]),
                    }
                )
                same = fresh == stored
                bad += not same
                print(f"{cand} {gate}: {'identical' if same else 'DIFFERS'}")
        raise SystemExit(1 if bad else 0)

    cand = args.cand
    if args.command == "g0":
        request = {
            "smoke_run": args.smoke_run,
            "speed": args.speed,
            "log": [rel(p) for p in args.log],
        }
        store(cand, "G0", compute("G0", cand, request), request)
    elif args.command == "g1":
        request = {
            "run": args.run,
            "log": [rel(p) for p in args.log],
            "restarts": args.restarts,
        }
        store(cand, "G1", compute("G1", cand, request), request)
    elif args.command == "g2":
        request = {"name": args.name}
        store(cand, "G2", compute("G2", cand, request), request)
    elif args.command == "g3":
        request = {"name": args.name}
        g3, g5 = gate_g3_g5(cand, args.name)
        store(cand, "G3", g3, request)
        store(cand, "G5", g5, request)
    elif args.command == "g4":
        request = {"name": args.name, "csv": args.csv.resolve().as_posix()}
        store(cand, "G4", gate_g4(cand, args.name, args.csv), request)


if __name__ == "__main__":
    main()
