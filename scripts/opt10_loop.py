"""OPT-10: stages and promotion gate of a frozen candidate against the protected E6 pipeline (CPU).

    python scripts/opt10_loop.py verify          # protected hashes, frozen files, E6 val_full rebuilt byte-identically
    python scripts/opt10_loop.py screen C        # S1: candidate C on the discovery clips (needs its discovery raw cache)
    python scripts/opt10_loop.py confirm C       # S2: C on the confirmation clips, once, only after S1 passed
    python scripts/opt10_loop.py full C          # S3: val_full + subsets + leave-one-clip-out, gate G1-G7 and G9
    python scripts/opt10_loop.py determinism C   # G8 (val part): a fresh process rebuilds val_full, sha256 compared

Candidates: movd/opt10.py GRID; rule: results/opt-10-soup-bn/selection_rule.md (committed before `screen`).
Every candidate keeps E6's post-processing (`movd.opt09.E6`: border drop 0.005, five-frame boost IoU 0.5); only the
checkpoint behind the raw cache differs. The clip split, subsets and evaluator are OPT-08's / the project's,
unchanged. No test inference and no CSV here: those belong to chunk 10.8 and need a separate approval.
Large files go to git-ignored outputs/opt10/.
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
import opt09_loop as o9
import opt09_soup as s9
import opt10_soup as s10

from movd import data, opt09, opt10
from movd import evaluate as ev

EVID = REPO / "results" / "opt-10-soup-bn"
OUT = REPO / "outputs" / "opt10"
RULE = EVID / "selection_rule.md"
E6_VAL_FULL = o9.PROTECTED["E6_val_full_predictions"]
FROZEN_FILES = (
    *o9.FROZEN_FILES,
    "movd/opt09.py",
    "scripts/opt08_soup.py",
    "scripts/opt09_soup.py",
    "scripts/opt09_loop.py",
)
OWN_CONFIGS = ":(exclude)configs/runs/opt10_rep_*.json"  # additions (replicate runs), not protocol changes

sha256, save_json, load_json, rel, write_rows = (
    o7.sha256,
    o7.save_json,
    o7.load_json,
    o7.rel,
    o7.write_rows,
)


# ---------------------------------------------------------------- pipeline


def pipeline(tag: str, stages: list[str] | None = None, workers: int = 7) -> list[dict]:
    """E6's post-processing on a tag's raw caches: CH-13 merge with the border rule, then the five-frame boost.

    Shard order is (stage, sorted clip), as in scripts/opt09_loop.py, so tag "w080" rebuilds E6 byte for byte.
    """
    stages = stages or ["discovery", "confirmation"]
    shards = []
    for stage in stages:
        for clip in sorted(s8.stage_frames(stage)):
            shard = s10.raw_dir(tag, stage) / f"{clip}.npz"
            if not shard.exists():
                raise SystemExit(f"{shard} missing: cache {tag} {stage} first")
            shards.append(str(shard))
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        parts = pool.map(
            o9._merge_job,
            shards,
            ["val"] * len(shards),
            [opt09.E6["factor"]] * len(shards),
            [opt09.E6["margin"]] * len(shards),
        )
        pre = [p for part in parts for p in part]
    return opt09.five_frame_boost(
        pre, o8.neighbours("val_full"), opt09.E6["iou"], opt09.E6["strength"]
    )


# ---------------------------------------------------------------- helpers


def frozen_unchanged() -> bool:
    """Evaluator, writer, validators, protocol configs and the E6 pipeline code equal origin/develop's."""
    r = subprocess.run(
        ["git", "diff", "--quiet", "origin/develop", "--", *FROZEN_FILES, OWN_CONFIGS],
        cwd=REPO,
        check=False,
    )
    modified = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=MDR",
            "origin/develop",
            "--",
            *FROZEN_FILES,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return r.returncode == 0 and not modified


def rule_committed() -> str:
    """The rule and the grid module are committed unchanged, and the rule states every candidate."""
    files = [RULE.relative_to(REPO).as_posix(), "movd/opt10.py"]
    files += [f"results/opt-10-soup-bn/{n}.json" for n in ("bn_frames", "d2_frames")]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *files],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *files],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if tracked.returncode or dirty:
        raise SystemExit(
            "selection_rule.md, the frame lists or movd/opt10.py are not committed unchanged"
        )
    text = RULE.read_text(encoding="utf-8")
    for cid, spec in opt10.GRID.items():
        if f"{cid} {json.dumps(spec, sort_keys=True)}" not in text:
            raise SystemExit(f"selection_rule.md does not state {cid}")
    return o8.commit_of(RULE)


def baseline() -> dict:
    """Unchanged E6 on every subset (results/opt-09-e6-tuning/baseline, equal to the OPT-08 E6 record)."""
    record = load_json(REPO / "results/opt-09-e6-tuning/baseline/subsets.json")
    if record["predictions_sha256"] != E6_VAL_FULL[1]:
        raise SystemExit("the stored E6 baseline is not the protected E6 val_full")
    return record["subsets"]


def loco_subsets() -> dict[str, list[int]]:
    """Leave-one-clip-out: val_full without each of the 7 clips in turn."""
    subsets = o8.subsets()
    clips = {k: set(v) for k, v in subsets.items() if k.startswith("clip:")}
    return {
        f"without:{k[5:]}": [i for i in subsets["val_full"] if i not in ids]
        for k, ids in clips.items()
    }


def baseline_loco() -> dict:
    """E6's leave-one-clip-out mAP, evaluated once from the protected val_full predictions."""
    target = EVID / "baseline" / "leave_one_clip_out.json"
    if target.exists():
        return load_json(target)["subsets"]
    path, expected = E6_VAL_FULL
    o8.checked(path, expected)
    results = o8.eval_paths([(path, k, ids) for k, ids in loco_subsets().items()])
    out = {
        k: {"mAP": r["metrics"]["mAP"], "num_images": r["num_images"]}
        for k, r in results.items()
    }
    save_json(
        target,
        {"predictions": rel(path), "predictions_sha256": expected, "subsets": out},
    )
    return out


def need(cid: str, stage_file: str, key: str) -> dict:
    path = EVID / cid / stage_file
    if not path.exists():
        raise SystemExit(f"{cid}: {stage_file} missing (earlier stage not run)")
    record = load_json(path)
    if not record[key]:
        raise SystemExit(f"{cid} failed {key}: it stops here (rule)")
    return record


# ---------------------------------------------------------------- commands


def cmd_verify(_args) -> None:
    import torch

    record = {}
    for k, (path, expected) in o9.PROTECTED.items():
        actual = sha256(path)
        record[k] = {
            "path": rel(path),
            "sha256": actual,
            "matches": expected is None or actual == expected,
        }
    shards = {}
    for stage in ("discovery", "confirmation", "test"):
        d = s10.raw_dir(s10.E6_TAG, stage)
        shards[stage] = {p.name: sha256(p) for p in sorted(d.glob("*.npz"))}
    reference = load_json(REPO / "results/opt-09-e6-tuning/identity.json")[
        "E6_raw_shards"
    ]["sha256"]
    record["E6_raw_shards_equal_opt09_record"] = shards == reference
    gt_path = data.split_dir("val") / "annotations_coco.json"
    record["val_annotations"] = {"path": rel(gt_path), "sha256": sha256(gt_path)}
    record["frozen_files_unchanged"] = frozen_unchanged()
    t0 = time.perf_counter()
    rebuilt = OUT / "e6_rebuild" / "val_full.json"
    save_json(rebuilt, pipeline(s10.E6_TAG))
    record["E6_val_full_rebuilt_sha256"] = sha256(rebuilt)
    record["E6_val_full_rebuilt_identical"] = (
        record["E6_val_full_rebuilt_sha256"] == E6_VAL_FULL[1]
    )
    record["rebuild_seconds"] = round(time.perf_counter() - t0, 1)
    # the generalised average with E6's weights must be soup w080, tensor for tensor
    _, state, _ = s10.averaged_state("c0")
    e6 = s10.load_checkpoint(*s9.E6_SOUP)["ema"].state_dict()
    record["weighted_state_identical_to_E6_soup"] = list(state) == list(e6) and all(
        torch.equal(state[k], e6[k]) for k in e6
    )
    record["ok"] = (
        all(v["matches"] for k, v in record.items() if k in o9.PROTECTED)
        and record["E6_raw_shards_equal_opt09_record"]
        and record["frozen_files_unchanged"]
        and record["E6_val_full_rebuilt_identical"]
        and record["weighted_state_identical_to_E6_soup"]
    )
    record["checked_at_kyiv"] = o7.kyiv_now()
    history = EVID / "identity.json"
    runs = load_json(history)["runs"] if history.exists() else []
    save_json(history, {"runs": [*runs, record]})
    print(json.dumps(record, indent=2))
    if not record["ok"]:
        raise SystemExit("IDENTITY FAILED: stop")


def cmd_screen(args) -> None:
    cid = args.name
    commit = rule_committed()
    out_dir = EVID / cid
    if (out_dir / "screen.json").exists():
        raise SystemExit(f"{cid} already screened once")
    base = baseline()["discovery"]
    ids = o8.subsets()["discovery"]
    start = time.perf_counter()
    path = OUT / cid / "discovery.json"
    preds = pipeline(cid, ["discovery"])
    save_json(path, preds)
    m = o8.eval_paths([(path, cid, ids)], workers=1)[cid]["metrics"]
    delta = m["mAP"] - base["mAP"]
    passes = delta >= opt10.STAGE["S1_min_delta_discovery"]
    record = {
        "candidate": cid,
        "spec": opt10.GRID[cid],
        "checkpoint_sha256": sha256(s10.weights_path(cid)),
        "rule_commit": commit,
        "stage_rule": opt10.STAGE,
        "baseline_E6_discovery": base,
        "discovery": m,
        "delta_discovery_mAP": delta,
        "delta_discovery_AP_small": m["AP_small"] - base["AP_small"],
        "discovery_predictions_sha256": sha256(path),
        "num_predictions": len(preds),
        "passes_S1": passes,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "evaluated_at_kyiv": o7.kyiv_now(),
        "decision": "S1 pass: confirmation next"
        if passes
        else "rejected at S1 (discovery)",
    }
    if cid == "c0":
        record["m4bn_eligible"] = delta >= opt10.M4BN_NEEDS_C0_DISCOVERY_DELTA
        record["H2_material"] = delta >= opt10.DIAG["c0_H2_min_delta_discovery"]
    save_json(out_dir / "screen.json", record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("delta_discovery_mAP", "delta_discovery_AP_small", "decision")
            },
            indent=2,
        )
    )


def cmd_confirm(args) -> None:
    cid = args.name
    out_dir = EVID / cid
    if (out_dir / "confirm.json").exists():
        raise SystemExit(f"{cid} already confirmed once")
    need(cid, "screen.json", "passes_S1")
    base = baseline()["confirmation"]
    ids = o8.subsets()["confirmation"]
    path = OUT / cid / "confirmation.json"
    save_json(path, pipeline(cid, ["confirmation"]))
    m = o8.eval_paths([(path, cid, ids)], workers=1)[cid]["metrics"]
    delta = m["mAP"] - base["mAP"]
    passes = delta >= opt10.STAGE["S2_min_delta_confirmation"]
    save_json(
        out_dir / "confirm.json",
        {
            "candidate": cid,
            "baseline_E6_confirmation": base,
            "confirmation": m,
            "delta_confirmation_mAP": delta,
            "confirmation_predictions_sha256": sha256(path),
            "passes_S2": passes,
            "evaluated_at_kyiv": o7.kyiv_now(),
            "decision": "S2 pass: val_full gate next"
            if passes
            else "rejected at S2 (confirmation)",
        },
    )
    print(cid, "confirmation delta", delta, "S2 pass" if passes else "S2 FAIL")


def gate(
    base: dict, cand: dict, base_loco: dict, cand_loco: dict, per_class_rows: int
) -> dict:
    g = opt10.GATE
    clip_delta = {
        k[5:]: cand[k]["mAP"] - base[k]["mAP"] for k in cand if k.startswith("clip:")
    }
    loco_delta = {k[8:]: cand_loco[k]["mAP"] - base_loco[k]["mAP"] for k in cand_loco}
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
        "min_leave_one_clip_out_delta": min(loco_delta.values()),
        "delta_discovery": cand["discovery"]["mAP"] - base["discovery"]["mAP"],
        "delta_confirmation": cand["confirmation"]["mAP"] - base["confirmation"]["mAP"],
    }
    checks = {
        "G1 val_full mAP >= E6 + 0.0010": d["delta_mAP"] >= g["G1_min_delta_mAP"],
        "G2 width<=1920 mAP >= E6 + 0.0010": d["delta_mAP_width_le_1920"]
        >= g["G2_min_delta_mAP_width_le_1920"],
        "G3 >= 5 of 7 clips non-negative": d["clips_non_negative"]
        >= g["G3_min_clips_non_negative"],
        "G4 without-4K mAP >= E6 + 0.0010": d["delta_mAP_without_4k_clip"]
        >= g["G4_min_delta_mAP_without_4k_clip"],
        "G5 AP_small loss <= 0.002": d["delta_AP_small"] >= -g["G5_max_AP_small_loss"],
        "G6 density>=100 loss <= 0.002": d["delta_density_ge100"]
        >= -g["G6_max_density_ge100_loss"],
        "G7 leave-one-clip-out delta > 0 for every clip": d[
            "min_leave_one_clip_out_delta"
        ]
        > g["G7_min_leave_one_clip_out_delta_exclusive"],
        "all 8 class results reported": per_class_rows == 8,
        "G9 evaluator / protocol / validators unchanged": frozen_unchanged(),
    }
    return {
        "deltas": d,
        "clip_delta_mAP": clip_delta,
        "leave_one_clip_out_delta_mAP": loco_delta,
        "checks": checks,
        "passes_G1_G7_G9": all(checks.values()),
        "selection_tier_val_part": d["delta_mAP"]
        >= opt10.SELECTION_TIER["min_delta_mAP"],
        "label_T": cand["val_full"]["mAP"] >= opt10.TARGET_LABEL_VAL_FULL,
    }


def cmd_full(args) -> None:
    cid = args.name
    out_dir = EVID / cid
    if (out_dir / "gate.json").exists():
        raise SystemExit(f"{cid} already evaluated on val_full once")
    need(cid, "screen.json", "passes_S1")
    need(cid, "confirm.json", "passes_S2")
    base, base_loco = baseline(), baseline_loco()
    start = time.perf_counter()
    path = OUT / cid / "val_full.json"
    save_json(path, pipeline(cid))
    jobs = {**o8.subsets(), **loco_subsets()}
    results = o8.eval_paths([(path, k, ids) for k, ids in jobs.items()])
    ev.write_outputs(results["val_full"], out_dir / "val_full")
    cand = o7.compact({k: results[k] for k in o8.subsets()})
    cand_loco = {k: {"mAP": results[k]["metrics"]["mAP"]} for k in loco_subsets()}
    b_full = load_json(REPO / "results/opt-09-e6-tuning/baseline/val_full/metrics.json")
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
            }
            for k in cand
        ]
        + [
            {
                "subset": k,
                "E6_mAP": base_loco[k]["mAP"],
                "cand_mAP": cand_loco[k]["mAP"],
                "delta": cand_loco[k]["mAP"] - base_loco[k]["mAP"],
                "frames": base_loco[k]["num_images"],
            }
            for k in cand_loco
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
    g = gate(base, cand, base_loco, cand_loco, len(per_class))
    keys = ("val_full", "width_le_1920", "without_4k_clip", "discovery", "confirmation")
    record = {
        "candidate": cid,
        "spec": opt10.GRID[cid],
        "predictions": rel(path),
        "predictions_sha256": sha256(path),
        "E6": {k: base[k] for k in keys},
        "candidate_metrics": {k: cand[k] for k in keys},
        "gate": opt10.GATE,
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
    record = need(cid, "gate.json", "passes_G1_G7_G9")
    tmp = OUT / cid / "val_full_rerun.json"
    save_json(tmp, pipeline(cid))
    out = {
        "val_full_sha256_run": record["predictions_sha256"],
        "val_full_sha256_rerun": sha256(tmp),
        "test_part": "not run: test inference belongs to chunk 10.8 (separate approval)",
        "checked_at_kyiv": o7.kyiv_now(),
    }
    out["val_full_identical"] = (
        out["val_full_sha256_run"] == out["val_full_sha256_rerun"]
    )
    save_json(EVID / cid / "determinism.json", out)
    print(json.dumps(out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cmds = {
        "verify": cmd_verify,
        "screen": cmd_screen,
        "confirm": cmd_confirm,
        "full": cmd_full,
        "determinism": cmd_determinism,
    }
    parser.add_argument("command", choices=list(cmds))
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    if args.command != "verify" and args.name not in opt10.GRID:
        parser.error(f"candidate required: {list(opt10.GRID)}")
    cmds[args.command](args)


if __name__ == "__main__":
    main()
