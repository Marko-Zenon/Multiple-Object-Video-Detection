"""OPT-10: checkpoints (weighted averages, BatchNorm re-estimation), raw caches and the Phase 0 diagnostics.

    python scripts/opt10_soup.py check                     # CPU: weighted_state (0.8, 0.2) == E6's soup w080 exactly
    python scripts/opt10_soup.py smoke                     # GPU: 20 frames of w080 == OPT-08's protected raw shard
    python scripts/opt10_soup.py diag d1                   # GPU: anchor, then BatchNorm per-image statistics on val_fast
    python scripts/opt10_soup.py diag d2                   # GPU: eval mode on 714 train_final frames
    python scripts/opt10_soup.py bn c0                     # GPU: soup w080 with re-estimated BatchNorm statistics
    python scripts/opt10_soup.py cache c0 --stage discovery   # GPU: raw 6-view output per clip (resumable)

Rule: results/opt-10-soup-bn/selection_rule.md (frozen before any result). The soup arithmetic, the views and the raw
format are OPT-08's; the protected E6 soup, its raw caches, Model A and epoch1.pt are read from their worktrees and
never written. Every GPU job is logged in results/opt-10-soup-bn/gpu_log.json against the phase budget.
Large files go to git-ignored outputs/opt10/.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import opt07_loop as o7
import opt08_soup as s8
import opt09_soup as s9
from ch14_infer import frames

from movd import data, infer, opt10, rtdetr, ultra
from movd import evaluate as ev

HW2 = REPO.parent
EVID = REPO / "results" / "opt-10-soup-bn"
OUT = REPO / "outputs" / "opt10"
E6_TAG = "w080"
CHECKPOINTS = json.loads(
    (REPO / "results/ch-14-final/checkpoints.json").read_text(encoding="utf-8")
)["files"]
# The masked train_final export the trainer read (CH-14 worktree; read only).
EXPORT = HW2 / "MOVD-ch14" / "outputs" / "yolo_data_final"
EXPORT_CHECK = REPO / "results/ch-14-final/export/export_check.json"
PHASE0_JOBS = (
    "smoke",
    "d1",
    "d2",
    "bn_c0",
    "cache_c0_discovery",
    "cache_c0_confirmation",
)


# ---------------------------------------------------------------- GPU budget log


def gpu_log() -> list[dict]:
    path = EVID / "gpu_log.json"
    return o7.load_json(path)["jobs"] if path.exists() else []


def gpu_minutes(phase: int) -> float:
    return sum(j["seconds"] for j in gpu_log() if j["phase"] == phase) / 60


@contextlib.contextmanager
def gpu_job(name: str, phase: int, estimate_min: float):
    """Refuse a job that would pass the phase budget; log its wall-clock seconds when it ends (also on error)."""
    cap = opt10.BUDGET[f"gpu_minutes_phase{phase}"]
    used = gpu_minutes(phase)
    if used + estimate_min > cap:
        raise SystemExit(
            f"GPU budget: phase {phase} used {used:.1f} min + {name} ~{estimate_min} min > cap {cap} min"
        )
    start, started = time.perf_counter(), o7.kyiv_now()
    status = "failed"
    try:
        yield
        status = "ok"
    finally:
        jobs = gpu_log()
        jobs.append(
            {
                "job": name,
                "phase": phase,
                "seconds": round(time.perf_counter() - start, 1),
                "started_kyiv": started,
                "ended_kyiv": o7.kyiv_now(),
                "status": status,
            }
        )
        total = {
            f"phase{p}_minutes": round(
                sum(j["seconds"] for j in jobs if j["phase"] == p) / 60, 1
            )
            for p in sorted({j["phase"] for j in jobs})
        }
        o7.save_json(
            EVID / "gpu_log.json", {"budget": opt10.BUDGET, **total, "jobs": jobs}
        )


# ---------------------------------------------------------------- checkpoints


def run_checkpoint(run: str, epoch: int) -> tuple[Path, str]:
    """(path, expected sha256) of a run's epoch checkpoint. Run A = CH-14 final_rtdetr (Model A's run)."""
    if run == "A":
        name = f"epoch{epoch - 1}.pt"
        return s8.WEIGHTS_DIR / name, CHECKPOINTS[name]["sha256"]
    record = (
        EVID / "replicates" / f"{run}.json"
    )  # written right after a replicate is trained (Phase 1-2)
    if not record.exists():
        raise SystemExit(
            f"replicate {run} has not been trained (needs a separate approval)"
        )
    rec = o7.load_json(record)["checkpoints"][f"epoch{epoch}"]
    return HW2 / rec["path"], rec[
        "sha256"
    ]  # o7.rel paths are relative to the hw2 folder


def load_checkpoint(path: Path, expected: str) -> dict:
    import torch

    if o7.sha256(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from the recorded value")
    return torch.load(path, map_location="cpu", weights_only=False)


def tag_path(tag: str) -> Path:
    return s9.E6_SOUP[0] if tag == E6_TAG else OUT / "soup" / f"{tag}.pt"


def weights_path(tag: str) -> Path:
    """The checkpoint of a tag, hash-checked: w080 against OPT-08, OPT-10 candidates against their build record."""
    if tag == E6_TAG:
        expected = s9.E6_SOUP[1]
    else:
        expected = o7.load_json(EVID / tag / "build.json")["sha256"]
    path = tag_path(tag)
    if o7.sha256(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from the build record")
    return path


def raw_dir(tag: str, stage: str) -> Path:
    """Raw 6-view caches: E6's are OPT-08's protected shards; OPT-10 candidates live in outputs/opt10/raw/."""
    if tag == E6_TAG:
        return s9.raw_dir(E6_TAG, stage)
    return OUT / "raw" / tag / stage


def averaged_state(cid: str) -> tuple[dict, dict, list[dict]]:
    """(first checkpoint, averaged state dict, the list of averaged checkpoints) of a frozen candidate."""
    parts, states, weights, first = [], [], [], None
    for run, epoch, weight in opt10.candidate_weights(cid):
        path, expected = run_checkpoint(run, epoch)
        ckpt = load_checkpoint(path, expected)
        first = first or ckpt
        states.append(ckpt["ema"].state_dict())
        weights.append(weight)
        parts.append(
            {
                "run": run,
                "epoch": epoch,
                "weight": weight,
                "path": o7.rel(path),
                "sha256": expected,
            }
        )
    return first, opt10.weighted_state(states, weights), parts


def cmd_check(_args) -> None:
    """CPU identity: the generalised average with E6's weights on E6's two checkpoints is soup w080, tensor for tensor."""
    import torch

    _, state, parts = averaged_state(
        "c0"
    )  # c0's weights before BatchNorm re-estimation = E6's formula
    e6 = load_checkpoint(*s9.E6_SOUP)["ema"].state_dict()
    identical = list(state) == list(e6) and all(
        torch.equal(state[k], e6[k]) for k in e6
    )
    record = {
        "weights": parts,
        "E6_soup": {"path": o7.rel(s9.E6_SOUP[0]), "sha256": s9.E6_SOUP[1]},
        "num_tensors": len(state),
        "weighted_state_identical_to_E6_soup": identical,
        "checked_at_kyiv": o7.kyiv_now(),
    }
    o7.save_json(EVID / "soup_identity.json", record)
    print(json.dumps(record, indent=2))
    if not identical:
        raise SystemExit("weighted_state does not reproduce the E6 soup: stop")


# ---------------------------------------------------------------- frozen frame lists


def frozen_ids(name: str) -> list[int]:
    """Ids of a frozen frame file, re-derived from the rule and checked against the stored list and sha256."""
    record = o7.load_json(EVID / f"{name}.json")
    ids = opt10.spaced_ids(ultra.train_split_ids("train_final"), record["count"])
    if (
        ids != record["image_ids"]
        or opt10.ids_sha256(ids) != record["sha256_of_comma_joined_ids"]
    ):
        raise SystemExit(f"{name}.json does not follow the frozen rule")
    if record["sha256_of_comma_joined_ids"] not in (
        EVID / "selection_rule.md"
    ).read_text(encoding="utf-8"):
        raise SystemExit(
            f"{name}.json sha256 is not the one frozen in selection_rule.md"
        )
    return ids


def bn_image_paths() -> list[Path]:
    """The 1,024 masked train_final export images of the frozen BatchNorm frame list (manifest = CH-14's)."""
    manifest = EXPORT / "train_final_manifest.json"
    expected = o7.load_json(EXPORT_CHECK)["manifests"]["train_final"]
    if o7.load_json(manifest) != expected:
        raise SystemExit(f"{manifest} differs from the CH-14 export manifest")
    images = {im["id"]: im for im in data.load_coco("train")["images"]}
    paths = []
    for image_id in frozen_ids("bn_frames"):
        path = (
            EXPORT
            / "train_final"
            / "images"
            / ultra._relative_frame_path(images[image_id])
        )
        if not path.exists():
            raise SystemExit(f"{path} missing from the masked export")
        paths.append(path)
    return paths


# ---------------------------------------------------------------- BatchNorm re-estimation


def unfused(ckpt: dict, device: str):
    """The training-form network of a checkpoint (BatchNorm not folded), FP32, eval mode, on the GPU."""
    return ckpt["ema"].float().eval().to(f"cuda:{device}")


def image_batches(paths: list[Path], device: str, digest):
    """Frames as network inputs, one per batch; `digest` collects the sha256 of the image files read."""
    import cv2

    for path in paths:
        raw = path.read_bytes()
        digest.update(raw)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read {path}")
        yield opt10.preprocess(image).to(f"cuda:{device}")


def cmd_bn(args) -> None:
    """Build a `bn: true` candidate: average the weights, re-estimate the BatchNorm statistics, save like a soup."""
    import torch

    cid = args.tag
    if not opt10.GRID[cid]["bn"]:
        raise SystemExit(f"{cid} has no BatchNorm re-estimation")
    target = EVID / cid / "build.json"
    if target.exists():
        raise SystemExit(f"{cid} already built")
    first, state, parts = averaged_state(cid)
    paths = bn_image_paths()
    with gpu_job(f"bn_{cid}", opt10.GRID[cid]["phase"], estimate_min=4):
        start = time.perf_counter()
        template = copy.deepcopy(first["ema"])
        template.load_state_dict(state)
        net = template.float().eval().to(f"cuda:{args.device}")
        modes = [m.training for m in net.modules()]
        momenta = [m.momentum for m in opt10.bn_layers(net)]
        before = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        digest = hashlib.sha256()
        record = opt10.reestimate_bn(net, image_batches(paths, args.device, digest))
        seconds = time.perf_counter() - start
        after = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    buffers = set(opt10.bn_buffer_names(net))
    others_identical = all(
        torch.equal(after[k], before[k]) for k in after if k not in buffers
    )
    stat_keys = [
        k for k in after if k in buffers and not k.endswith("num_batches_tracked")
    ]
    rel_change = {
        kind: float(
            np.median(
                [
                    ((after[k] - before[k]).norm() / (before[k].norm() + 1e-12)).item()
                    for k in stat_keys
                    if k.endswith(kind)
                ]
            )
        )
        for kind in ("running_mean", "running_var")
    }
    # Save like a soup: the averaged float16 state with only the BatchNorm buffers replaced.
    new_state = dict(state)
    for k in buffers:
        new_state[k] = after[k].to(state[k].dtype)
    model = copy.deepcopy(first["ema"])
    model.load_state_dict(new_state)
    ckpt = {
        k: v
        for k, v in first.items()
        if k not in ("optimizer", "scaler", "ema", "model")
    }
    ckpt.update({"ema": model, "model": None, "optimizer": None, "scaler": None})
    ckpt["opt10"] = {"candidate": cid, "weights": parts, "bn_frames": len(paths)}
    path = tag_path(cid)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)["ema"].state_dict()
    out = {
        "candidate": cid,
        "spec": opt10.GRID[cid],
        "weights": parts,
        "path": o7.rel(path),
        "sha256": o7.sha256(path),
        **record,
        "bn_frames": len(paths),
        "bn_frames_ids_sha256": o7.load_json(EVID / "bn_frames.json")[
            "sha256_of_comma_joined_ids"
        ],
        "bn_image_files_sha256": digest.hexdigest(),
        "export": o7.rel(EXPORT),
        "non_bn_tensors_identical_to_the_average": others_identical,
        "saved_parameters_identical_to_the_average": all(
            torch.equal(saved[k], state[k]) for k in saved if k not in buffers
        ),
        "num_tensors": len(after),
        "bn_buffers": len(buffers),
        "num_batches_tracked": sorted(
            {int(after[k]) for k in buffers if k.endswith("tracked")}
        ),
        "statistics_finite": all(
            bool(torch.isfinite(after[k]).all()) for k in stat_keys
        ),
        "variance_positive": all(
            bool((after[k] > 0).all()) for k in stat_keys if k.endswith("running_var")
        ),
        "variance_nonnegative": all(
            bool((after[k] >= 0).all()) for k in stat_keys if k.endswith("running_var")
        ),
        "median_relative_change_vs_stored_statistics": rel_change,
        "modes_restored": [m.training for m in net.modules()] == modes,
        "momenta_restored": [m.momentum for m in opt10.bn_layers(net)] == momenta,
        "momentum_values": sorted(set(momenta)),
        "precision": "FP32, torch.no_grad, batch 1, 1344 x 1344 stretched, RGB / 255",
        "seconds": round(seconds, 1),
        "built_at_kyiv": o7.kyiv_now(),
    }
    out["ok"] = (
        all(
            out[k]
            for k in (
                "non_bn_tensors_identical_to_the_average",
                "saved_parameters_identical_to_the_average",
                "statistics_finite",
                "variance_nonnegative",
                "modes_restored",
                "momenta_restored",
            )
        )
        and out["batches"] == opt10.BN_FRAMES
        and out["num_batches_tracked"] == [opt10.BN_FRAMES]
    )
    o7.save_json(target if out["ok"] else EVID / cid / "build_FAILED.json", out)
    print(json.dumps(out, indent=2))
    if not out["ok"]:
        raise SystemExit(f"{cid}: BatchNorm re-estimation checks failed: stop")


def cmd_build(args) -> None:
    """Build a `bn: false` candidate (m2, m4, m4e1): the frozen weighted average, saved like OPT-08's soups (CPU)."""
    import torch

    cid = args.tag
    if opt10.GRID[cid]["bn"]:
        raise SystemExit(f"{cid} needs BatchNorm re-estimation: use `bn`")
    target = EVID / cid / "build.json"
    if target.exists():
        raise SystemExit(f"{cid} already built")
    first, state, parts = averaged_state(cid)
    model = copy.deepcopy(first["ema"])
    model.load_state_dict(state)
    ckpt = {
        k: v
        for k, v in first.items()
        if k not in ("optimizer", "scaler", "ema", "model")
    }
    ckpt.update({"ema": model, "model": None, "optimizer": None, "scaler": None})
    ckpt["opt10"] = {"candidate": cid, "weights": parts}
    path = tag_path(cid)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)["ema"].state_dict()
    # Independent recomputation of the average from freshly loaded, hash-checked sources.
    states = [
        load_checkpoint(run_checkpoint(p["run"], p["epoch"])[0], p["sha256"])[
            "ema"
        ].state_dict()
        for p in parts
    ]
    floating = [k for k, v in states[0].items() if v.is_floating_point()]
    others = [k for k in states[0] if k not in floating]
    max_err = 0.0
    for k in floating:
        ref = sum(
            p["weight"] * s[k].float() for p, s in zip(parts, states, strict=True)
        )
        diff = saved[k].float() - ref.to(saved[k].dtype).float()
        max_err = max(max_err, float(diff.abs().max()))
    sources_after = {
        f"{p['run']}_epoch{p['epoch']}": o7.sha256(
            run_checkpoint(p["run"], p["epoch"])[0]
        )
        == p["sha256"]
        for p in parts
    }
    out = {
        "candidate": cid,
        "spec": opt10.GRID[cid],
        "weights": parts,
        "weights_sum": sum(p["weight"] for p in parts),
        "path": o7.rel(path),
        "sha256": o7.sha256(path),
        "num_tensors": len(saved),
        "floating_tensors": len(floating),
        "non_floating_tensors": len(others),
        "saved_identical_to_weighted_state": list(saved) == list(state)
        and all(torch.equal(saved[k], state[k]) for k in state),
        "max_abs_error_vs_independent_recomputation": max_err,
        "non_floating_identical_to_first_source": all(
            torch.equal(saved[k], states[0][k]) for k in others
        ),
        "floating_tensors_changed_vs_first_source": sum(
            not torch.equal(saved[k], states[0][k]) for k in floating
        ),
        "dtypes": sorted({str(v.dtype) for v in saved.values()}),
        "all_finite": all(bool(torch.isfinite(saved[k]).all()) for k in floating),
        "sources_unchanged_after_build": sources_after,
        "built_at_kyiv": o7.kyiv_now(),
    }
    out["ok"] = (
        out["saved_identical_to_weighted_state"]
        and out["non_floating_identical_to_first_source"]
        and out["all_finite"]
        and all(sources_after.values())
        and abs(out["weights_sum"] - 1) < 1e-9
    )
    o7.save_json(target if out["ok"] else EVID / cid / "build_FAILED.json", out)
    print(json.dumps(out, indent=2))
    if not out["ok"]:
        raise SystemExit(f"{cid}: build checks failed: stop")


# ---------------------------------------------------------------- replicate runs (Phase 1-2, chunk 10.4 / 10.6)

REP_EVID = EVID / "replicates"
SMOKE_CONFIG = REPO / "configs/runs/opt10_rep_smoke.json"
TRAIN_PROJECTED_MAX_MIN = 70  # rule: a run projected beyond 70 min stops
LOSS_KEYS = ("first_loss", "last_loss", "minimum_loss", "maximum_loss")


def rep_config(run: str) -> Path:
    return REPO / f"configs/runs/opt10_rep_s{opt10.REPLICATE_SEEDS[run]}.json"


def nvidia_snapshot(name: str) -> None:
    """`nvidia-smi` before every GPU job (rule), kept as evidence."""
    import subprocess

    text = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=True
    ).stdout
    REP_EVID.mkdir(parents=True, exist_ok=True)
    (REP_EVID / f"nvidia_before_{name}.txt").write_text(text, encoding="utf-8")


def train_process(
    config: Path, *extra: str, watch_epoch_seconds: float | None = None
) -> int:
    """Run scripts/train_ultra.py; stop it if epoch 1 ends after `watch_epoch_seconds` (projection rule)."""
    import subprocess

    run_id = o7.load_json(config)["run_id"]
    suffix = "".join(
        e.strip("-").replace("-", "_") for e in extra if e.startswith("--")
    )
    log = OUT / "logs" / f"train_{run_id}_{suffix or 'run'}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "scripts/train_ultra.py"),
        "--config",
        str(config),
        "--device",
        "0",
        *extra,
    ]
    obs = REPO / "results" / run_id / "observations.json"
    with log.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(20)
            if watch_epoch_seconds and obs.exists():
                o = o7.load_json(obs)
                if (
                    o.get("saved_epochs") == [0]
                    and o["elapsed_seconds"] > watch_epoch_seconds
                ):
                    proc.terminate()
                    proc.wait()
                    raise SystemExit(
                        f"{run_id}: epoch 1 took {o['elapsed_seconds'] / 60:.1f} min, projected beyond "
                        f"{TRAIN_PROJECTED_MAX_MIN} min: stopped (rule)"
                    )
    return proc.returncode


def cmd_rep_smoke(_args) -> None:
    """Replicate smoke: ~100 frames x 2 epochs at seed 43, stop after epoch 1 + resume, then an inference check."""
    import torch
    import yaml

    run_id = o7.load_json(SMOKE_CONFIG)["run_id"]
    target = REP_EVID / "smoke.json"
    if target.exists():
        raise SystemExit("replicate smoke already passed")
    nvidia_snapshot("rep_smoke")
    with gpu_job("rep_smoke", 1, estimate_min=4):
        codes = [
            train_process(SMOKE_CONFIG, "--stop-after-epoch", "1"),
            train_process(SMOKE_CONFIG, "--resume"),
        ]
        weights = REPO / "outputs" / run_id / "train" / "weights"
        net = unfused(
            torch.load(weights / "epoch1.pt", map_location="cpu", weights_only=False),
            "0",
        )
        items = [(i, p) for _, i, p in frames("val_fast", None)][:8]
        preds = single_view(net, items, "0")
    args_yaml = yaml.safe_load(
        (REPO / "outputs" / run_id / "train" / "args.yaml").read_text(encoding="utf-8")
    )
    runtime = o7.load_json(REPO / "results" / run_id / "runtime.json")
    procs = runtime["smoke_processes"]
    out = {
        "config": o7.rel(SMOKE_CONFIG),
        "return_codes": codes,
        "args_yaml_seed": args_yaml["seed"],
        "args_yaml_deterministic": args_yaml["deterministic"],
        "processes": procs,
        "stop_then_resume": [p["status"] for p in procs]
        == ["checkpoint_stop", "completed"]
        and procs[1]["resume"]
        and procs[1]["start_epoch"] == 1
        and procs[1]["optimizer_state_entries"] > 0,
        "epoch_checkpoints": runtime["epoch_checkpoints"],
        "losses_finite": all(
            bool(np.isfinite([p[k] for k in LOSS_KEYS]).all()) for p in procs
        ),
        "inference_frames": len(items),
        "inference_predictions": len(preds),
        "inference_finite": bool(
            np.isfinite([v for p in preds for v in (*p["bbox"], p["score"])]).all()
        ),
        "peak_cuda_reserved_gb": round(
            max(p["peak_cuda_reserved_bytes"] for p in procs) / 2**30, 2
        ),
        "checked_at_kyiv": o7.kyiv_now(),
    }
    out["ok"] = (
        codes == [0, 0]
        and out["args_yaml_seed"] == opt10.REPLICATE_SEEDS["R1"]
        and out["stop_then_resume"]
        and out["epoch_checkpoints"] == ["epoch0.pt", "epoch1.pt"]
        and out["losses_finite"]
        and out["inference_predictions"] > 0
        and out["inference_finite"]
    )
    o7.save_json(target if out["ok"] else REP_EVID / "smoke_FAILED.json", out)
    print(json.dumps(out, indent=2))
    if not out["ok"]:
        raise SystemExit("replicate smoke failed: stop")


def cmd_train(args) -> None:
    """Train one replicate with its frozen config (a stop_after_epoch run, never resumed); record its checkpoints."""
    import yaml

    run = args.tag
    if not (REP_EVID / "smoke.json").exists():
        raise SystemExit("the replicate smoke must pass first")
    record_path = REP_EVID / f"{run}.json"
    if record_path.exists():
        raise SystemExit(f"{run} already trained")
    config_path = rep_config(run)
    config = ultra.load_run_config(config_path)
    run_id = config["run_id"]
    nvidia_snapshot(f"train_{run}")
    with gpu_job(f"train_{run}", 1 if run == "R1" else 2, estimate_min=55):
        code = train_process(
            config_path, watch_epoch_seconds=TRAIN_PROJECTED_MAX_MIN * 60 / 2
        )
    if code != 0:
        raise SystemExit(f"{run_id}: train_ultra.py exited with {code}")
    results = REPO / "results" / run_id
    weights = REPO / "outputs" / run_id / "train" / "weights"
    cfg = o7.load_json(results / "config.json")
    proc = o7.load_json(results / "runtime.json")["training_processes"][-1]
    args_yaml = yaml.safe_load(
        (weights.parent / "args.yaml").read_text(encoding="utf-8")
    )
    model_a = o7.load_json(REPO / "results/final_rtdetr/config.json")
    kwargs_diff = {
        k: [model_a["ultralytics_train_kwargs"].get(k), v]
        for k, v in cfg["ultralytics_train_kwargs"].items()
        if k not in ("project", "data")
        and model_a["ultralytics_train_kwargs"].get(k) != v
    }
    stop = config["stop_after_epoch"]
    checkpoints = {
        f"epoch{e + 1}": {
            "file": f"epoch{e}.pt",
            "path": o7.rel(weights / f"epoch{e}.pt"),
            "sha256": o7.sha256(weights / f"epoch{e}.pt"),
        }
        for e in range(stop)
    }
    out = {
        "run": run,
        "run_id": run_id,
        "config": o7.rel(config_path),
        "config_sha256": o7.sha256(config_path),
        "seed": args_yaml["seed"],
        "deterministic": args_yaml["deterministic"],
        "schedule_epochs": args_yaml["epochs"],
        "stop_after_epoch": stop,
        "ultralytics_train_kwargs_diff_vs_model_A": kwargs_diff,
        "start_weights": cfg["weights"],
        "start_weights_sha256": cfg["weights_sha256"],
        "start_weights_equal_model_A": cfg["weights_sha256"]
        == model_a["weights_sha256"],
        "train_split": cfg["train_split"],
        "train_split_file_sha256": cfg["train_split_file_sha256"],
        "export_manifest_equal_model_A": cfg["export_manifest"]
        == model_a["export_manifest"],
        "frames": proc["samples"],
        "batches_per_epoch": proc["batches_per_epoch"],
        "images_seen": proc["batches"],
        "saved_epochs": proc["saved_epochs"],
        "status": proc["status"],
        "losses": {k: proc[k] for k in LOSS_KEYS},
        "losses_finite": bool(np.isfinite([proc[k] for k in LOSS_KEYS]).all()),
        "train_seconds": proc["seconds"],
        "peak_cuda_allocated_gb": round(proc["peak_cuda_allocated_bytes"] / 2**30, 2),
        "peak_cuda_reserved_gb": round(proc["peak_cuda_reserved_bytes"] / 2**30, 2),
        "checkpoints": checkpoints,
        "recorded_at_kyiv": o7.kyiv_now(),
    }
    out["ok"] = (
        out["seed"] == opt10.REPLICATE_SEEDS[run]
        and set(kwargs_diff) == {"seed"}
        and out["start_weights_equal_model_A"]
        and out["export_manifest_equal_model_A"]
        and out["frames"] == 6072
        and out["images_seen"] == 6072 * stop
        and out["saved_epochs"] == list(range(stop))
        and out["status"] == f"stopped_after_epoch_{stop}"
        and out["losses_finite"]
    )
    o7.save_json(record_path if out["ok"] else REP_EVID / f"{run}_FAILED.json", out)
    print(json.dumps(out, indent=2))
    if not out["ok"]:
        raise SystemExit(f"{run}: training record checks failed: stop")


def cmd_valfast(args) -> None:
    """Reported only: each replicate checkpoint alone, single full-frame view on val_fast (D1's eval-mode path)."""
    import torch

    run = args.tag
    target = REP_EVID / f"{run}_val_fast.json"
    if target.exists():
        raise SystemExit(f"{run} val_fast already run once")
    gt = data.load_coco("val")
    ids = data.split_image_ids("val_fast")
    items = [(i, p) for _, i, p in frames("val_fast", None)]
    curve = o7.load_json(REPO / "results/final_rtdetr/curve.json")
    stored = {row["epoch"]: row["val_fast"]["mAP"] for row in curve}
    record = {
        "run": run,
        "path": "unfused network, single full-frame view at 1344, top-300 decoding, conf 0.001 (D1 eval mode)",
        "epochs": {},
    }
    nvidia_snapshot(f"valfast_{run}")
    with gpu_job(f"valfast_{run}", 1 if run == "R1" else 2, estimate_min=6):
        for epoch in (1, 2):
            path, expected = run_checkpoint(run, epoch)
            net = unfused(load_checkpoint(path, expected), args.device)
            m = summary(ev.evaluate(single_view(net, items, args.device), gt, ids))
            record["epochs"][str(epoch)] = {
                "checkpoint_sha256": expected,
                "val_fast": m,
                "model_A_stored_val_fast_mAP": stored[epoch],
                "delta_vs_model_A_stored": m["mAP"] - stored[epoch],
            }
            print(epoch, json.dumps(record["epochs"][str(epoch)]), flush=True)
            del net
            torch.cuda.empty_cache()
    record["peak_mem_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    record["run_at_kyiv"] = o7.kyiv_now()
    o7.save_json(target, record)


# ---------------------------------------------------------------- raw cache (OPT-08's loop, checkpoint by path)


def cache(
    tag: str,
    stage: str,
    device: str = "0",
    max_frames: int = 0,
    clips: list[str] | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Raw 6-view RT-DETR output per clip, exactly as scripts/opt08_soup.py `cache`; never writes OPT-08 shards."""
    import cv2
    import torch

    if stage == "test":
        raise SystemExit("test inference is not part of this session (rule)")
    if tag == E6_TAG and out_dir is None:
        raise SystemExit(
            "the E6 raw caches are protected (OPT-08): use them, do not rebuild"
        )
    cfg = s8.config()
    specs = infer.pass_specs(cfg)
    grid, frac = cfg.get("tile_grid", 2), cfg.get("tile_frac", 0.6)
    by_clip = s8.stage_frames(stage)
    model = rtdetr.load_model(weights_path(tag), "rtdetr")
    torch.zeros(1, device=f"cuda:{device}")
    out_dir = out_dir or raw_dir(tag, stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds, computed = 0.0, 0
    for clip in clips or sorted(by_clip):
        shard = out_dir / f"{clip}.npz"
        if shard.exists():
            continue
        ids, boxes, scores = [], [], []
        start = time.perf_counter()
        for image_id, path in by_clip[clip][: max_frames or None]:
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"cannot read {path}")
            raws = [
                infer.raw_pass(model, image, s["imgsz"], device, s["view"], grid, frac)
                for s in specs
            ]
            ids.append(image_id)
            boxes.append(np.stack([r.boxes for r in raws]))
            scores.append(np.stack([r.scores for r in raws]))
        seconds += time.perf_counter() - start
        computed += len(ids)
        tmp = shard.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            image_ids=np.array(ids, np.int64),
            boxes=np.stack(boxes),
            scores=np.stack(scores),
        )
        tmp.replace(shard)
        print(
            f"{tag} {stage} {clip}: {len(ids)} frames, {seconds / 60:.1f} min",
            flush=True,
        )
    return {
        "frames": computed,
        "seconds": round(seconds, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }


def cmd_cache(args) -> None:
    tag, stage = args.tag, args.stage
    phase = opt10.GRID[tag]["phase"]
    estimate = {"discovery": 12, "confirmation": 13}[stage]
    with gpu_job(f"cache_{tag}_{stage}", phase, estimate):
        rec = cache(tag, stage, args.device)
    shards = {p.name: o7.sha256(p) for p in sorted(raw_dir(tag, stage).glob("*.npz"))}
    o7.save_json(
        EVID / tag / f"cache_{stage}.json",
        {
            "candidate": tag,
            "stage": stage,
            "checkpoint_sha256": o7.sha256(weights_path(tag)),
            **rec,
            "shards_sha256": shards,
            "cached_at_kyiv": o7.kyiv_now(),
        },
    )
    print(json.dumps(rec))


def cmd_smoke(args) -> None:
    """20 frames of the first discovery clip through the OPT-10 cache path: w080 must equal OPT-08's shard exactly."""
    clip = min(s8.stage_frames("discovery"))
    d = OUT / "raw_smoke" / E6_TAG
    shard = d / f"{clip}.npz"
    if shard.exists():
        shard.unlink()  # only this command's own earlier smoke output
    with gpu_job("smoke", 0, estimate_min=2):
        info = cache(
            E6_TAG, "discovery", args.device, max_frames=20, clips=[clip], out_dir=d
        )
    z, ref = np.load(shard), np.load(raw_dir(E6_TAG, "discovery") / f"{clip}.npz")
    n = len(z["image_ids"])
    out = {
        "clip": clip,
        **info,
        "identical_to_opt08_shard": bool(
            np.array_equal(ref["image_ids"][:n], z["image_ids"])
            and np.array_equal(ref["boxes"][:n], z["boxes"])
            and np.array_equal(ref["scores"][:n], z["scores"])
        ),
        "checked_at_kyiv": o7.kyiv_now(),
    }
    o7.save_json(EVID / "smoke.json", out)
    print(json.dumps(out, indent=2))
    if not out["identical_to_opt08_shard"]:
        raise SystemExit(
            "smoke failed: the cache path does not reproduce E6's raw output: stop"
        )


# ---------------------------------------------------------------- diagnostics D1 / D2


def single_view(net, items: list[tuple[int, Path]], device: str) -> list[dict]:
    """One full-frame pass per frame through the unfused network -> COCO results (top-300 decoding, conf 0.001)."""
    import cv2

    preds = []
    for image_id, path in items:
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"cannot read {path}")
        raw = opt10.direct_raw(net, image, f"cuda:{device}")
        preds.extend(infer.to_coco(image_id, infer.decode(raw, k=300, conf=0.001)))
    return ev.cap_per_image(preds, 500)


def summary(result: dict) -> dict:
    keys = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
    return {k: result["metrics"][k] for k in keys} | {
        "num_predictions": result["num_predictions"]
    }


def diag_checkpoint(epoch: int) -> dict:
    name = f"epoch{epoch - 1}.pt"
    return load_checkpoint(s8.WEIGHTS_DIR / name, CHECKPOINTS[name]["sha256"])


def cmd_diag(args) -> None:
    import torch

    which = args.tag
    target = EVID / "diagnostics" / f"{which}.json"
    if target.exists():
        raise SystemExit(f"{which} already run once")
    device = args.device
    record = {"diagnostic": which, "epochs": {}, "thresholds": opt10.DIAG}
    if which == "d1":
        gt = data.load_coco("val")
        ids = data.split_image_ids("val_fast")
        items = [(i, p) for _, i, p in frames("val_fast", None)]
        curve = o7.load_json(REPO / "results/final_rtdetr/curve.json")
        stored = {row["epoch"]: row["val_fast"]["mAP"] for row in curve}
        with gpu_job("d1", 0, estimate_min=9):
            for epoch in opt10.DIAG_EPOCHS:
                net = unfused(diag_checkpoint(epoch), device)
                before = {k: v.clone() for k, v in net.state_dict().items()}
                start = time.perf_counter()
                eval_mode = summary(
                    ev.evaluate(single_view(net, items, device), gt, ids)
                )
                row = {
                    "stored_fused_val_fast_mAP": stored[epoch],
                    "eval_mode": eval_mode,
                    "unfused_minus_stored": eval_mode["mAP"] - stored[epoch],
                }
                if epoch == opt10.DIAG_EPOCHS[0]:
                    anchor = opt10.DIAG["anchor_model_A_val_fast_mAP"]
                    row["anchor_ok"] = (
                        stored[epoch] == anchor
                        and abs(eval_mode["mAP"] - anchor)
                        <= opt10.DIAG["anchor_tolerance"]
                    )
                    if not row["anchor_ok"]:
                        record["epochs"][str(epoch)] = row
                        record["decision"] = (
                            "ANCHOR FAILED: no diagnostic, no c0 (rule)"
                        )
                        o7.save_json(
                            EVID / "diagnostics" / "d1_ANCHOR_FAILED.json", record
                        )
                        raise SystemExit(json.dumps(record, indent=2))
                with opt10.bn_batch_stats(net):
                    batch_mode = summary(
                        ev.evaluate(single_view(net, items, device), gt, ids)
                    )
                after = net.state_dict()
                row.update(
                    {
                        "per_image_statistics": batch_mode,
                        "lift_mAP": batch_mode["mAP"] - eval_mode["mAP"],
                        "state_byte_identical_after": all(
                            torch.equal(after[k], before[k]) for k in before
                        ),
                        "seconds": round(time.perf_counter() - start, 1),
                    }
                )
                record["epochs"][str(epoch)] = row
                print(epoch, json.dumps(row), flush=True)
                del net
                torch.cuda.empty_cache()
        lift6 = record["epochs"]["6"]["lift_mAP"]
        record["H1_material"] = lift6 >= opt10.DIAG["D1_H1_min_lift_epoch6"]
        record["reading"] = (
            f"epoch 6 lift {lift6:+.4f} vs threshold +{opt10.DIAG['D1_H1_min_lift_epoch6']}: "
            + ("H1 material" if record["H1_material"] else "H1 not supported by D1")
        )
    elif which == "d2":
        ids = frozen_ids("d2_frames")
        coco = data.load_coco("train")
        keep = set(ids)
        gt = {
            **{k: v for k, v in coco.items() if k not in ("images", "annotations")},
            "images": [im for im in coco["images"] if im["id"] in keep],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
        }
        images = {im["id"]: im for im in gt["images"]}
        items = [(i, data.image_path("train", images[i])) for i in ids]
        del coco
        with gpu_job("d2", 0, estimate_min=5):
            for epoch in opt10.DIAG_EPOCHS:
                net = unfused(diag_checkpoint(epoch), device)
                start = time.perf_counter()
                row = {
                    "eval_mode_train_frames": summary(
                        ev.evaluate(single_view(net, items, device), gt, ids)
                    )
                }
                row["seconds"] = round(time.perf_counter() - start, 1)
                record["epochs"][str(epoch)] = row
                print(epoch, json.dumps(row), flush=True)
                del net
                torch.cuda.empty_cache()
        rise = (
            record["epochs"]["6"]["eval_mode_train_frames"]["mAP"]
            - record["epochs"]["1"]["eval_mode_train_frames"]["mAP"]
        )
        record["rise_epoch1_to_6"] = rise
        record["frames"] = {
            "count": len(ids),
            "ids_sha256": opt10.ids_sha256(ids),
            "images": "unmasked originals",
        }
        record["H3_confirmed"] = rise >= opt10.DIAG["D2_H3_min_rise_epoch1_to_6"]
        record["reading"] = (
            f"train-frame mAP rise epoch 1 -> 6 {rise:+.4f} vs threshold +{opt10.DIAG['D2_H3_min_rise_epoch1_to_6']}: "
            + ("H3 confirmed" if record["H3_confirmed"] else "H3 not confirmed by D2")
        )
    record["run_at_kyiv"] = o7.kyiv_now()
    o7.save_json(target, record)
    print(json.dumps({k: v for k, v in record.items() if k != "thresholds"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cmds = {
        "check": cmd_check,
        "smoke": cmd_smoke,
        "diag": cmd_diag,
        "bn": cmd_bn,
        "build": cmd_build,
        "cache": cmd_cache,
        "rep-smoke": cmd_rep_smoke,
        "train": cmd_train,
        "valfast": cmd_valfast,
    }
    ap.add_argument("command", choices=list(cmds))
    ap.add_argument("tag", nargs="?")
    ap.add_argument("--stage", choices=["discovery", "confirmation"])
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    if args.command == "diag" and args.tag not in ("d1", "d2"):
        ap.error("diag needs d1 or d2")
    if args.command in ("bn", "build", "cache") and args.tag not in opt10.GRID:
        ap.error(f"candidate required: {list(opt10.GRID)}")
    if args.command in ("train", "valfast") and args.tag not in opt10.REPLICATE_SEEDS:
        ap.error(f"replicate required: {list(opt10.REPLICATE_SEEDS)}")
    if args.command == "cache" and not args.stage:
        ap.error("cache needs --stage")
    cmds[args.command](args)


if __name__ == "__main__":
    main()
