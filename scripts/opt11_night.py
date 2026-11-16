"""OPT-11: final-night candidates on the protected E6 pipeline. Resumable orchestrator; never uploads anything.

Candidates (all use E6's post-processing unchanged: CH-13 merge, inner tile-border drop 0.005, five-frame boost):
  c1  OPT-09 soup w075 (0.75 x Model A epoch0.pt + 0.25 x epoch1.pt), rebuilt on test.
  p1  0.75 x e1 + 0.15 x e2 + 0.10 x e3 of CH-14 final_rtdetr (EMA weights of epoch0/1/2.pt).
  p2  0.70 x e1 + 0.10 x e2 + 0.10 x e3 + 0.10 x e4 (epoch0/1/2/3.pt).
  c2  class mix: person and motor from E6, the other six classes from c1 (top 500 per image).
p1/p2 are screened on the frozen OPT-08 discovery clips; the better one (P*) goes to test only if
delta vs E6 >= +0.0011 and >= 2 of the 4 discovery clips are >= E6 ("strong" if delta >= +0.0025).

Commands:
  run                       the state machine (idempotent; resumes from artifacts; hard Kyiv cutoffs)
  status                    print state.json
  build                     build p1/p2 soups (+ identity check vs soup_w075.pt)
  gpu-cache TAG STAGE       raw 6-view cache of a soup (GPU)
  screen TAG                discovery screen vs E6
  post TAG                  test predictions -> CSV -> both validators
  classmix                  c2 on val_full, then on test if val_full >= c1's
  valfull TAG               advisory val_full of a soup (needs discovery + confirmation caches)
  handoff                   write results/opt-11-final-night/HANDOFF.md
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HW2 = REPO.parent
OUT = REPO / "outputs" / "opt11"
EVID = REPO / "results" / "opt-11-final-night"
SUB = REPO / "submissions"
STATE = OUT / "state.json"
LOCK = OUT / "gpu.lock"
LOG = OUT / "logs" / "events.log"
PY = sys.executable

KYIV = timezone(timedelta(hours=2))
DAY = "2026-11-14"


def at(hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hhmm}:00").replace(tzinfo=KYIV)


CUT = {
    "J1_latest_start": at("22:30"),
    "J2_latest_start": at("22:20"),
    "J3_latest_start": at("22:22"),
    "J2J3_kill": at("22:40"),
    "decision": at("22:40"),
    "J4_latest_start": at("22:42"),
    "J4_kill": at("23:13"),
    "J5_latest_start": at("23:05"),
    "J5_kill": at("23:18"),
    "final_stop": at("23:27"),
    "exit": at("23:30"),
}

E6_DISCOVERY_MAP = 0.3344880450293952  # results/opt-09-e6-tuning/baseline (= OPT-10 m2 decision.json S1.E6)
E6_DISCOVERY_CLIPS = {  # MOVD-opt10 results/opt-10-soup-bn/phase1-r1/decision.json (same evaluator)
    "uav0000086_00000_v": 0.2790392466181147,
    "uav0000137_00458_v": 0.35673890434178185,
    "uav0000182_00000_v": 0.23585130418013547,
    "uav0000339_00001_v": 0.3253063305382974,
}
C1_VAL_FULL = (
    0.297399  # results/opt-09-e6-tuning/c_final/c1 (0.2973990..., +0.000817 vs E6)
)
PROMOTE_MIN_DELTA, STRONG_DELTA, MIN_CLIPS = 0.0011, 0.0025, 2

W080 = (
    HW2 / "MOVD-opt08/outputs/opt08/soup/soup_w080.pt",
    "2d60d3fa66d9fd37032bcaaaeb297e98510217b600b5809f73873b36360cd939",
)
W075 = (
    HW2 / "MOVD-opt09/outputs/opt09/soup/soup_w075.pt",
    "cfaaf0a4097c96a972340cfde7aa7a35eba066a64be9c00a3f8f2bbb8ec1b39d",
)
SOUPS = {  # tag -> [(epoch file of CH-14 final_rtdetr, weight)]
    "p1": [("epoch0.pt", 0.75), ("epoch1.pt", 0.15), ("epoch2.pt", 0.10)],
    "p2": [
        ("epoch0.pt", 0.70),
        ("epoch1.pt", 0.10),
        ("epoch2.pt", 0.10),
        ("epoch3.pt", 0.10),
    ],
}
PROTECTED = {
    "E6_csv": (
        HW2 / "MOVD-opt08/submissions/opt08_e6_combination_k1_test.csv",
        "35a225c905ac56ed86532314046821d10085f73781666ee843fb4c6d28127c35",
    ),
    "E4_csv": (
        HW2 / "MOVD-opt08/submissions/opt08_e4_checkpoint_soup_s1_test.csv",
        "7dc9b318e354d4839b5dffae8f5973dd7da234fdbeb3c5f84057694878852014",
    ),
}
E6_TEST_PREDS = (
    HW2 / "MOVD-opt08/outputs/opt08/e6_combination/k1/test/test_predictions.json",
    "ef3f5c2ff42598696dfcd44aa1431c0d422598203bbfce8f0370c333c14cb0aa",
)
E6_VAL_FULL = (
    HW2 / "MOVD-opt08/outputs/opt08/e6_combination/k1/val_full.json",
    "bdb2806edad09fa093d5b59369780a8dbbb6fea4758522c72c86b8c8e98cad52",
)
C1_VAL_FULL_PREDS = HW2 / "MOVD-opt09/outputs/opt09/c_final/c1/val_full.json"
FROM_E6_CLASSES = (
    1,
    8,
)  # person, motor (movd.data.CLASS_NAMES order, category_id = index + 1)
CSV_NAME = {
    "w075": "opt11_c1_w075_test.csv",
    "p1": "opt11_p1_trajsoup_test.csv",
    "p2": "opt11_p2_trajsoup_test.csv",
}


# ---------------------------------------------------------------- small helpers


def now() -> datetime:
    return datetime.now(KYIV)


def stamp() -> str:
    return now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{stamp()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state() -> dict:
    return load_json(STATE) if STATE.exists() else {"events": {}}


def mark(key: str, value) -> None:
    s = state()
    s["events"][key] = value
    s["updated"] = stamp()
    save_json(STATE, s)


# ---------------------------------------------------------------- tag registry (patches OPT-09 code)


def soup_file(tag: str) -> Path:
    return OUT / "soup" / f"{tag}.pt"


def weights_of(tag: str) -> tuple[Path, str]:
    if tag == "w080":
        return W080
    if tag == "w075":
        return W075
    rec = load_json(EVID / "soup_build.json")[tag]
    return soup_file(tag), rec["sha256"]


def raw_dir(tag: str, stage: str) -> Path:
    if tag == "w080":
        return W080[0].parent / "raw" / "w080" / stage
    if tag == "w075" and stage != "test":
        return W075[0].parent / "raw" / "w075" / stage
    return OUT / "raw" / tag / stage


def checked_weights(tag: str) -> Path:
    path, expected = weights_of(tag)
    if sha256(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from the record")
    return path


def patched():
    """OPT-09's cache / pipeline code, redirected to the OPT-11 tag registry (no OPT-09 file is changed)."""
    import opt09_loop as ol
    import opt09_soup as s9

    s9.weights_path = checked_weights
    s9.raw_dir = raw_dir
    if not getattr(ol, "_opt11", False):
        ol.merged = functools.partial(ol.merged, workers=5)
        ol._opt11 = True
    return ol, s9


def params(tag: str) -> dict:
    from movd import opt09

    return {**opt09.E6, "soup": tag}


def n_shards(tag: str, stage: str) -> tuple[int, int]:
    import opt08_soup as s8

    clips = sorted(s8.stage_frames(stage))
    d = raw_dir(tag, stage)
    return sum((d / f"{c}.npz").exists() for c in clips), len(clips)


# ---------------------------------------------------------------- CPU / GPU work commands


def nway_state(states: list[dict], weights: list[float]) -> dict:
    """sum_i w_i * state_i for floating tensors (float32 math, stored in the original dtype); others from the first."""
    out = {}
    for k, t0 in states[0].items():
        if t0.is_floating_point():
            acc = weights[0] * t0.float()
            for s, w in zip(states[1:], weights[1:]):
                acc = acc + w * s[k].float()
            out[k] = acc.to(t0.dtype)
        else:
            out[k] = t0.clone()
    return out


def cmd_build(_args) -> None:
    import copy

    import opt08_soup as s8
    import torch

    wdir = s8.WEIGHTS_DIR
    needed = sorted({f for parts in SOUPS.values() for f, _ in parts})
    ckpts, record = {}, {"weights_dir": wdir.as_posix(), "inputs": {}}
    for name in needed:
        path = wdir / name
        digest = sha256(path)
        if name == "epoch0.pt" and digest != s8.A[1]:
            raise SystemExit("epoch0.pt is not Model A")
        if name == "epoch1.pt" and digest != s8.B[1]:
            raise SystemExit("epoch1.pt differs from the OPT-08 record")
        ckpts[name] = torch.load(path, map_location="cpu", weights_only=False)
        record["inputs"][name] = {"sha256": digest, "epoch": ckpts[name].get("epoch")}
    states = {k: v["ema"].state_dict() for k, v in ckpts.items()}
    # identity: the N-way code with (0.75, 0.25) on (e1, e2) equals OPT-09's soup_w075.pt tensor for tensor
    ref = torch.load(checked_weights("w075"), map_location="cpu", weights_only=False)[
        "ema"
    ].state_dict()
    rebuilt = nway_state([states["epoch0.pt"], states["epoch1.pt"]], [0.75, 0.25])
    identity = list(rebuilt) == list(ref) and all(
        torch.equal(rebuilt[k], ref[k]) for k in ref
    )
    record["identity_w075_tensor_equal"] = identity
    if not identity:
        save_json(EVID / "soup_build_FAILED.json", record)
        raise SystemExit("N-way rebuild of w075 differs: P candidates dropped")
    a = ckpts["epoch0.pt"]
    sa = states["epoch0.pt"]
    for tag, parts in SOUPS.items():
        weights = [w for _, w in parts]
        assert abs(sum(weights) - 1.0) < 1e-9
        st = nway_state([states[f] for f, _ in parts], weights)
        model = copy.deepcopy(a["ema"])
        model.load_state_dict(st)
        ok = list(model.state_dict()) == list(sa) and all(
            model.state_dict()[k].shape == sa[k].shape
            and model.state_dict()[k].dtype == sa[k].dtype
            for k in sa
        )
        if not ok:
            raise SystemExit(f"{tag}: names / shapes / dtypes differ")
        ckpt = {
            k: v
            for k, v in a.items()
            if k not in ("optimizer", "scaler", "ema", "model")
        }
        ckpt.update({"ema": model, "model": None, "optimizer": None, "scaler": None})
        ckpt["opt11_soup"] = {
            "parts": [
                {"file": f, "weight": w, "sha256": record["inputs"][f]["sha256"]}
                for f, w in parts
            ]
        }
        path = soup_file(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        record[tag] = {
            "parts": ckpt["opt11_soup"]["parts"],
            "path": path.as_posix(),
            "sha256": sha256(path),
            "names_shapes_dtypes_equal_model_A": ok,
            "num_tensors": len(st),
        }
    record["built_at_kyiv"] = stamp()
    save_json(EVID / "soup_build.json", record)
    print(json.dumps({k: record[k]["sha256"] for k in SOUPS}, indent=2))


def cmd_gpu_cache(args) -> None:
    _, s9 = patched()
    res = s9.cache(args.tag, args.stage, out_dir=raw_dir(args.tag, args.stage))
    print(json.dumps(res), flush=True)


def eval_ids() -> dict:
    import opt08_loop as o8
    import opt08_soup as s8

    subs = o8.subsets()
    disc = set(subs["discovery"])
    clips = {
        c: [i for i, _ in fr if i in disc]
        for c, fr in s8.stage_frames("discovery").items()
    }
    return {
        "discovery": subs["discovery"],
        "val_full": subs["val_full"],
        "clips": clips,
    }


def cmd_screen(args) -> None:
    import opt08_loop as o8

    ol, _ = patched()
    tag = args.tag
    t0 = time.perf_counter()
    preds = ol.pipeline(params(tag), "val_full", ["discovery"])
    path = OUT / "screen" / tag / "discovery.json"
    save_json(path, preds)
    ids = eval_ids()
    jobs = [(path, "discovery", ids["discovery"])] + [
        (path, c, v) for c, v in ids["clips"].items()
    ]
    res = o8.eval_paths(jobs)
    m = res["discovery"]["metrics"]
    clips = {
        c: {
            "mAP": res[c]["metrics"]["mAP"],
            "E6": E6_DISCOVERY_CLIPS[c],
            "delta": res[c]["metrics"]["mAP"] - E6_DISCOVERY_CLIPS[c],
        }
        for c in ids["clips"]
    }
    out = {
        "tag": tag,
        "params": params(tag),
        "discovery": m,
        "E6_discovery_mAP": E6_DISCOVERY_MAP,
        "delta": m["mAP"] - E6_DISCOVERY_MAP,
        "clips": clips,
        "clips_non_negative": sum(v["delta"] >= 0 for v in clips.values()),
        "discovery_predictions_sha256": sha256(path),
        "num_predictions": len(preds),
        "seconds": round(time.perf_counter() - t0, 1),
        "evaluated_at_kyiv": stamp(),
    }
    save_json(EVID / "screen" / f"{tag}.json", out)
    print(
        json.dumps(
            {
                "tag": tag,
                "delta": out["delta"],
                "clips_non_negative": out["clips_non_negative"],
            }
        )
    )


def write_and_validate(
    name: str, preds: list[dict], ckpt_sha: str, extra: dict
) -> dict:
    from movd import submission

    tdir = EVID / "test" / name
    pred_path = OUT / "test" / name / "test_predictions.json"
    save_json(pred_path, preds)
    csv_path = SUB / extra.pop("csv_name")
    SUB.mkdir(parents=True, exist_ok=True)
    images = submission.test_image_table()
    rows, stats = submission.build_rows(preds, images)
    submission.write_csv(csv_path, rows)
    problems, vsum = submission.validate_rows(submission.read_csv(csv_path), images)
    run = {
        "candidate": name,
        **extra,
        "csv": csv_path.resolve().as_posix(),
        "csv_bytes": csv_path.stat().st_size,
        "csv_sha256": sha256(csv_path),
        "test_predictions": pred_path.resolve().as_posix(),
        "test_predictions_sha256": sha256(pred_path),
        "checkpoint_sha256": ckpt_sha,
        "writer": stats,
        "project_validator_inline": {
            "valid": not problems,
            "problems": problems,
            **vsum,
        },
        "uploaded": False,
        "written_at_kyiv": stamp(),
    }
    save_json(tdir / "run.json", run)
    v1 = subprocess.run(
        [PY, "scripts/validate_submission.py", str(csv_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    (tdir / "project_validator.txt").write_text(v1.stdout + v1.stderr, encoding="utf-8")
    v2 = subprocess.run(
        [
            PY,
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
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    run["validators_exit"] = {"project": v1.returncode, "independent": v2.returncode}
    run["valid"] = (
        (not problems)
        and v1.returncode == 0
        and v2.returncode == 0
        and vsum.get("data_rows") == 3811
        and vsum.get("max_detections_in_a_row", 0) <= 500
    )
    save_json(tdir / "run.json", run)
    print(
        json.dumps(
            {
                "csv": run["csv"],
                "sha256": run["csv_sha256"],
                "valid": run["valid"],
                "validators_exit": run["validators_exit"],
            }
        ),
        flush=True,
    )
    if not run["valid"]:
        os.replace(tdir / "run.json", tdir / "run_FAILED_validation.json")
        raise SystemExit(1)
    return run


def cmd_post(args) -> None:
    ol, _ = patched()
    tag = args.tag
    ckpt_sha = weights_of(tag)[1]
    preds = ol.pipeline(params(tag), "test")
    name = "c1" if tag == "w075" else tag
    write_and_validate(
        name,
        preds,
        ckpt_sha,
        {
            "csv_name": CSV_NAME[tag],
            "params": params(tag),
            "checkpoint": weights_of(tag)[0].as_posix(),
        },
    )


def mix(e6: list[dict], other: list[dict]) -> list[dict]:
    from collections import defaultdict

    by = defaultdict(list)
    for p in e6:
        if p["category_id"] in FROM_E6_CLASSES:
            by[p["image_id"]].append(p)
    for p in other:
        if p["category_id"] not in FROM_E6_CLASSES:
            by[p["image_id"]].append(p)
    out = []
    for image_id in sorted(by):
        out.extend(sorted(by[image_id], key=lambda p: -p["score"])[:500])
    return out


def cmd_classmix(_args) -> None:
    import opt08_loop as o8

    if sha256(E6_VAL_FULL[0]) != E6_VAL_FULL[1]:
        raise SystemExit("E6 val_full predictions changed")
    val = mix(load_json(E6_VAL_FULL[0]), load_json(C1_VAL_FULL_PREDS))
    vpath = OUT / "classmix" / "val_full.json"
    save_json(vpath, val)
    del val
    m = o8.eval_paths([(vpath, "val_full", eval_ids()["val_full"])])["val_full"][
        "metrics"
    ]
    rec = {
        "rule": "person+motor from E6, other classes from c1, top 500",
        "val_full": m,
        "c1_val_full": C1_VAL_FULL,
        "passes": m["mAP"] >= C1_VAL_FULL,
        "evaluated_at_kyiv": stamp(),
        "note": "class choice made on val_full per-class deltas, so this val gain is optimistic",
    }
    save_json(EVID / "classmix" / "val.json", rec)
    print(json.dumps({"val_full": m["mAP"], "passes": rec["passes"]}), flush=True)
    if not rec["passes"]:
        return
    c1 = load_json(EVID / "test" / "c1" / "run.json")
    if sha256(E6_TEST_PREDS[0]) != E6_TEST_PREDS[1]:
        raise SystemExit("E6 test predictions changed")
    preds = mix(load_json(E6_TEST_PREDS[0]), load_json(Path(c1["test_predictions"])))
    write_and_validate(
        "c2",
        preds,
        W075[1],
        {
            "csv_name": "opt11_c2_classmix_test.csv",
            "composition": rec["rule"],
            "e6_checkpoint_sha256": W080[1],
        },
    )


def cmd_valfull(args) -> None:
    import opt08_loop as o8

    ol, _ = patched()
    preds = ol.pipeline(params(args.tag), "val_full", ["discovery", "confirmation"])
    path = OUT / "valfull" / args.tag / "val_full.json"
    save_json(path, preds)
    del preds
    ids = o8.subsets()
    jobs = [(path, k, ids[k]) for k in ("val_full", "confirmation") if k in ids]
    res = o8.eval_paths(jobs)
    out = {k: res[k]["metrics"] for k in res}
    out["E6_val_full"] = 0.296583
    out["evaluated_at_kyiv"] = stamp()
    save_json(EVID / "valfull" / f"{args.tag}.json", out)
    print(json.dumps({k: v["metrics"]["mAP"] for k, v in res.items()}))


# ---------------------------------------------------------------- orchestration


def pid_alive(pid: int) -> bool:
    r = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return str(pid) in r.stdout


def foreign_gpu_python(own: set[int]) -> list[str]:
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    bad = []
    for line in r.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if (
            len(parts) >= 2
            and parts[0].isdigit()
            and "python" in parts[1].lower()
            and int(parts[0]) not in own
        ):
            bad.append(line)
    return bad


def wait_pid(pid: int, kill_at: datetime) -> None:
    while pid_alive(pid):
        if now() >= kill_at:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
            log(f"killed pid {pid} at cutoff")
            return
        time.sleep(15)


def gpu_job(
    job: str, args: list[str], latest_start: datetime, kill_at: datetime
) -> bool:
    """Run one GPU child process under the lock file; True if it exited 0."""
    if LOCK.exists():
        info = load_json(LOCK)
        if pid_alive(int(info["pid"])):
            log(f"{job}: waiting for lock holder {info}")
            wait_pid(int(info["pid"]), kill_at)
        LOCK.unlink(missing_ok=True)
        if args[0] == "gpu-cache" and complete(args[1], args[2]):
            log(f"{job}: complete after waiting for the lock holder")
            mark(job, {"status": "ok_adopted", "ended": stamp()})
            return True
    if now() > latest_start:
        log(f"{job}: skipped, past latest start {latest_start:%H:%M}")
        mark(job, {"status": "skipped_late", "at": stamp()})
        return False
    if LOCK.exists():
        info = load_json(LOCK)
        if pid_alive(int(info["pid"])):
            log(f"{job}: waiting for lock holder {info}")
            wait_pid(int(info["pid"]), kill_at)
        LOCK.unlink(missing_ok=True)
    bad = foreign_gpu_python(set())
    if bad:
        log(f"{job}: foreign python on GPU {bad}; refusing")
        mark(job, {"status": "refused_gpu_busy", "at": stamp()})
        return False
    out = (OUT / "logs" / f"{job}.out").open("a", encoding="utf-8")
    p = subprocess.Popen(
        [PY, str(Path(__file__)), *args], cwd=REPO, stdout=out, stderr=subprocess.STDOUT
    )
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"pid": p.pid, "job": job, "started": stamp()}).encode())
    os.close(fd)
    log(f"{job}: started pid {p.pid} {args}")
    mark(job, {"status": "running", "pid": p.pid, "started": stamp()})
    while p.poll() is None:
        if now() >= kill_at:
            subprocess.run(
                ["taskkill", "/PID", str(p.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
            p.wait()
            log(f"{job}: killed at cutoff {kill_at:%H:%M}")
            break
        time.sleep(10)
    LOCK.unlink(missing_ok=True)
    ok = p.returncode == 0
    mark(job, {"status": "ok" if ok else f"exit {p.returncode}", "ended": stamp()})
    log(f"{job}: ended rc={p.returncode}")
    return ok


CPU: dict[str, subprocess.Popen] = {}


def cpu_start(name: str, args: list[str]) -> None:
    if name in CPU:
        return
    out = (OUT / "logs" / f"{name}.out").open("a", encoding="utf-8")
    CPU[name] = subprocess.Popen(
        [PY, str(Path(__file__)), *args], cwd=REPO, stdout=out, stderr=subprocess.STDOUT
    )
    log(f"cpu {name}: started pid {CPU[name].pid} {args}")
    mark(f"cpu_{name}", {"status": "running", "started": stamp()})


def cpu_wait(name: str, deadline: datetime) -> bool:
    p = CPU.get(name)
    if p is None:
        return False
    while p.poll() is None:
        if now() >= deadline:
            subprocess.run(
                ["taskkill", "/PID", str(p.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
            p.wait()
            log(f"cpu {name}: killed at {deadline:%H:%M}")
            break
        time.sleep(10)
    mark(
        f"cpu_{name}",
        {
            "status": "ok" if p.returncode == 0 else f"exit {p.returncode}",
            "ended": stamp(),
        },
    )
    log(f"cpu {name}: rc={p.returncode}")
    return p.returncode == 0


def complete(tag: str, stage: str) -> bool:
    have, total = n_shards(tag, stage)
    return have == total


def run_json(name: str) -> dict | None:
    p = EVID / "test" / name / "run.json"
    return load_json(p) if p.exists() else None


def decide() -> dict:
    res = {}
    for tag in SOUPS:
        p = EVID / "screen" / f"{tag}.json"
        if p.exists():
            res[tag] = load_json(p)
    dec = {
        "screens": {
            t: {"delta": r["delta"], "clips_non_negative": r["clips_non_negative"]}
            for t, r in res.items()
        },
        "rule": f"best of p1/p2; promote iff delta >= {PROMOTE_MIN_DELTA} and >= {MIN_CLIPS}/4 clips >= E6; "
        f"strong iff delta >= {STRONG_DELTA}",
        "decided_at_kyiv": stamp(),
    }
    if res:
        best = max(res, key=lambda t: res[t]["delta"])
        r = res[best]
        dec["best"] = best
        dec["promoted"] = (
            r["delta"] >= PROMOTE_MIN_DELTA and r["clips_non_negative"] >= MIN_CLIPS
        )
        dec["strong"] = dec["promoted"] and r["delta"] >= STRONG_DELTA
    else:
        dec.update({"best": None, "promoted": False, "strong": False})
    save_json(EVID / "decision.json", dec)
    log(f"decision: {dec}")
    return dec


def keep_awake() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except (AttributeError, OSError) as exc:  # not Windows / no kernel32
        log(f"keep_awake unavailable: {exc}")


def cmd_run(_args) -> None:
    keep_awake()
    log(f"run: start (pid {os.getpid()})")
    # J1 (c1 test raw) may already be running from the session; adopt it through the lock file.
    if not complete("w075", "test"):
        gpu_job(
            "J1", ["gpu-cache", "w075", "test"], CUT["J1_latest_start"], CUT["J4_kill"]
        )
    if complete("w075", "test") and not run_json("c1"):
        cpu_start("c1_post", ["post", "w075"])
    soups_ok = (EVID / "soup_build.json").exists()
    if not soups_ok:
        log("soups not built: P path dropped")
    # J2 / J3: discovery caches of p1 / p2, screens on CPU
    if soups_ok:
        if not complete("p1", "discovery"):
            gpu_job(
                "J2",
                ["gpu-cache", "p1", "discovery"],
                CUT["J2_latest_start"],
                CUT["J2J3_kill"],
            )
        if complete("p1", "discovery") and not (EVID / "screen" / "p1.json").exists():
            cpu_start("screen_p1", ["screen", "p1"])
        if complete("p1", "discovery") and not complete("p2", "discovery"):
            gpu_job(
                "J3",
                ["gpu-cache", "p2", "discovery"],
                CUT["J3_latest_start"],
                CUT["J2J3_kill"],
            )
        if complete("p2", "discovery") and not (EVID / "screen" / "p2.json").exists():
            cpu_start("screen_p2", ["screen", "p2"])
        for n in ("screen_p1", "screen_p2"):
            cpu_wait(n, CUT["decision"])
    dec = (
        load_json(EVID / "decision.json")
        if (EVID / "decision.json").exists()
        else decide()
    )
    # C2 after c1's CSV exists
    if "c1_post" in CPU:
        cpu_wait("c1_post", CUT["final_stop"])
    if run_json("c1") and not (EVID / "classmix" / "val.json").exists():
        cpu_start("classmix", ["classmix"])
    # J4: P* on test
    star = dec.get("best") if dec.get("promoted") else None
    if star:
        if not complete(star, "test"):
            gpu_job(
                "J4",
                ["gpu-cache", star, "test"],
                CUT["J4_latest_start"],
                CUT["J4_kill"],
            )
        if complete(star, "test") and not run_json(star):
            cpu_start("p_post", ["post", star])
        # J5: advisory confirmation + val_full
        if complete(star, "test") and not complete(star, "confirmation"):
            gpu_job(
                "J5",
                ["gpu-cache", star, "confirmation"],
                CUT["J5_latest_start"],
                CUT["J5_kill"],
            )
        if (
            complete(star, "confirmation")
            and not (EVID / "valfull" / f"{star}.json").exists()
        ):
            cpu_start("valfull", ["valfull", star])
    for n in list(CPU):
        cpu_wait(n, CUT["final_stop"])
    cmd_handoff(None)
    log("run: done")


def cmd_handoff(_args) -> None:
    lines = [
        f"# OPT-11 hand-off ({stamp()} Kyiv)",
        "",
        "Nothing was uploaded. Kaggle final selections stay E6 + E4 until a new public score beats them.",
        "",
        "## Protected CSVs (re-hashed now)",
        "",
    ]
    for k, (p, h) in PROTECTED.items():
        actual = sha256(p) if p.exists() else "MISSING"
        lines.append(
            f"- {k}: `{p.as_posix()}` sha256 `{actual}` {'unchanged' if actual == h else 'CHANGED!'}"
        )
    dec = load_json(EVID / "decision.json") if (EVID / "decision.json").exists() else {}
    lines += [
        "",
        "## Validated new CSVs",
        "",
        "| Candidate | CSV | sha256 | rows | detections | validators | val evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    valid = {}
    for name in ("c1", "p1", "p2", "c2"):
        r = run_json(name)
        if not r:
            continue
        valid[name] = r
        v = r["project_validator_inline"]
        if name == "c1":
            ev = "val_full 0.297399 (+0.0008 vs E6), confirmation +0.0037 (OPT-09)"
        elif name == "c2":
            cm = load_json(EVID / "classmix" / "val.json")
            ev = f"val_full {cm['val_full']['mAP']:.6f} (optimistic: classes chosen on val)"
        else:
            sc = load_json(EVID / "screen" / f"{name}.json")
            ev = f"discovery delta {sc['delta']:+.6f}, {sc['clips_non_negative']}/4 clips >= E6"
            vf = EVID / "valfull" / f"{name}.json"
            if vf.exists():
                x = load_json(vf)
                ev += f"; val_full {x['val_full']['mAP']:.6f} (E6 0.296583)"
                if "confirmation" in x:
                    ev += f", confirmation {x['confirmation']['mAP']:.6f} (E6 0.270895)"
        lines.append(
            f"| {name} | `{r['csv']}` | `{r['csv_sha256']}` | {v.get('data_rows')} | {v.get('detections')} | "
            f"project {r['validators_exit']['project']}, independent {r['validators_exit']['independent']} | {ev} |"
        )
    star = dec.get("best") if dec.get("promoted") else None
    order = []
    if star and star in valid:
        conf_neg = False
        vf = EVID / "valfull" / f"{star}.json"
        if vf.exists() and "confirmation" in load_json(vf):
            conf_neg = load_json(vf)["confirmation"]["mAP"] < 0.27089469511565134
        if dec.get("strong") and not conf_neg:
            order = [star, "c1", "c2"]
        elif conf_neg:
            order = ["c1", "c2", star]
        else:
            order = ["c1", star, "c2"]
    else:
        order = ["c1", "c2"]
    order = [o for o in order if o in valid]
    lines += [
        "",
        f"P decision: `{json.dumps(dec)}`",
        "",
        "## Recommended manual upload order",
        "",
    ]
    lines += [f"{i}. {o}: `{valid[o]['csv']}`" for i, o in enumerate(order, 1)] or [
        "(no new validated CSV)"
    ]
    lines += [
        "",
        "Do not append to results/submissions.csv until the public scores are known.",
    ]
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "command",
        choices=[
            "run",
            "status",
            "build",
            "gpu-cache",
            "screen",
            "post",
            "classmix",
            "valfull",
            "handoff",
            "decide",
        ],
    )
    ap.add_argument("tag", nargs="?")
    ap.add_argument("stage", nargs="?")
    args = ap.parse_args()
    cmds = {
        "run": cmd_run,
        "status": lambda a: print(json.dumps(state(), indent=2)),
        "build": cmd_build,
        "gpu-cache": cmd_gpu_cache,
        "screen": cmd_screen,
        "post": cmd_post,
        "classmix": cmd_classmix,
        "valfull": cmd_valfull,
        "handoff": cmd_handoff,
        "decide": lambda a: decide(),
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
