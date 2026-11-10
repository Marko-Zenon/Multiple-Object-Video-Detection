"""OPT-05: FB (OPT-04 RT-DETR-l@1600) as a small-object specialist added to protected Model A (CPU only).

    python scripts/opt05_specialist.py verify       # input/checkpoint hashes; Model A reproduces its registered metrics;
                                                    # A merged with no eligible FB box == A
    python scripts/opt05_specialist.py screen       # val_fast: the 8 frozen candidates, metrics + width<=1920 subset
    python scripts/opt05_specialist.py determinism  # independent re-merge + re-evaluation of every candidate
    python scripts/opt05_specialist.py select       # Gate V from the stored files -> selection_val_fast.json

Rule, grid and gates: results/opt-05-small-specialist/selection_rule.md and candidates.json (frozen before any result).
Large files (merged predictions) go to the git-ignored outputs/opt05/.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, specialist  # noqa: E402
from movd import evaluate as ev  # noqa: E402

REPO = data.REPO_ROOT
HW2 = REPO.parent
EVID = REPO / "results" / "opt-05-small-specialist"
OUT = REPO / "outputs" / "opt05"
TEST_LIKE_MAX_WIDTH = 1920
DENSE = ">=100"

CHECKPOINTS = {
    "A": (
        HW2 / "MOVD-ch14/outputs/final_rtdetr/train/weights/epoch0.pt",
        "6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b",
    ),
    "FB": (
        HW2 / "MOVD-opt04/outputs/opt04_rtdetrl1600_e1/train/weights/epoch0.pt",
        "e06a1a4b3a4213147fe98083b5a8d3cd50aaa74dc7c38c72c0047e900acca33f",
    ),
}
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
    ("FB", "val_fast"): (
        HW2 / "MOVD-opt04/outputs/opt04/FB_rtdetrl1600_e1/val_fast/pipeline_predictions.json",
        "f0c86d7077d9d05d3f4fd50da508d844819479572d90b2c6120d6a30b43107e0",
    ),
}  # fmt: skip
REGISTERED = {
    ("A", "val_fast"): REPO / "results/ch-14b-rtdetr-short-finetune/stageA_epoch1/val_fast/pipeline/metrics.json",
    ("FB", "val_fast"): REPO / "results/opt-04-model-scale/FB_rtdetrl1600_e1/val_fast/pipeline/metrics.json",
}  # fmt: skip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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


def load_input(model: str, split: str) -> list[dict]:
    path, expected = INPUTS[(model, split)]
    if sha256(path) != expected:
        raise SystemExit(
            f"{path}: sha256 differs from the committed evidence; stop (rule)"
        )
    return load_json(path)


def grid() -> dict:
    return load_json(EVID / "candidates.json")


@functools.cache
def val_gt() -> dict:
    return data.load_coco("val")


@functools.cache
def val_sizes() -> dict[int, tuple[int, int]]:
    return {im["id"]: (im["width"], im["height"]) for im in val_gt()["images"]}


def test_like_ids(split: str = "val_fast") -> list[int]:
    sizes = val_sizes()
    return [
        i for i in data.split_image_ids(split) if sizes[i][0] <= TEST_LIKE_MAX_WIDTH
    ]


def run_merge(
    cand: dict, a: list[dict], fb: list[dict], split: str
) -> tuple[list[dict], dict]:
    """Merge one candidate; returns (clean predictions, counts). Sources are tagged only for counting."""
    tagged_a = [{**r, "_src": "A"} for r in a]
    tagged_fb = [{**r, "_src": "FB"} for r in fb]
    ids = data.split_image_ids(split)
    start = time.perf_counter()
    out = specialist.merge(
        tagged_a, tagged_fb, val_sizes(), cand["size_rule"], cand["fb_multiplier"], cand["nms_iou"],
        grid()["max_det"], image_ids=ids,
    )  # fmt: skip
    seconds = time.perf_counter() - start
    sizes = val_sizes()
    eligible = sum(
        specialist.is_small(r["bbox"], *sizes[r["image_id"]], cand["size_rule"])
        for r in fb
    )
    per_frame: dict[int, int] = {}
    for p in out:
        per_frame[p["image_id"]] = per_frame.get(p["image_id"], 0) + 1
    from_fb = sum(p["_src"] == "FB" for p in out)
    clean = [{k: v for k, v in p.items() if k != "_src"} for p in out]
    counts = {
        "frames": len(ids),
        "detections": len(clean),
        "A_input": len(a),
        "A_kept": len(clean) - from_fb,
        "A_removed": len(a) - (len(clean) - from_fb),
        "FB_input": len(fb),
        "FB_eligible": eligible,
        "FB_added": from_fb,
        "frames_at_cap": sum(n >= grid()["max_det"] for n in per_frame.values()),
        "merge_seconds": round(seconds, 2),
        "predictions_sha256": sha256_json(clean),
    }
    return clean, counts


def evaluate(preds: list[dict], split: str) -> tuple[dict, float]:
    result = ev.evaluate_split(preds, split)
    subset = ev.evaluate(preds, val_gt(), test_like_ids(split))
    return result, subset["metrics"]["mAP"]


def dense(result: dict) -> float:
    return next(d["mAP"] for d in result["density"] if d["bin"] == DENSE)


def summary_row(
    cid: str,
    result: dict,
    subset: float,
    counts: dict | None = None,
    cand: dict | None = None,
) -> dict:
    m = result["metrics"]
    row = {"id": cid}
    if cand:
        row.update(
            size_rule=cand["size_rule"],
            fb_multiplier=cand["fb_multiplier"],
            nms_iou=cand["nms_iou"],
        )
    row.update(
        {k: m[k] for k in ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")}
    )
    row.update({f"density_{d['bin']}": d["mAP"] for d in result["density"]})
    row["subset_w_le_1920"] = subset
    row.update({f"AP_{r['name']}": r["AP"] for r in result["per_class"]})
    if counts:
        row.update(
            {
                k: counts[k]
                for k in ("FB_eligible", "FB_added", "A_removed", "frames_at_cap")
            }
        )
    return row  # fmt: skip


def write_table(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- commands


def cmd_verify(_args) -> None:
    record: dict = {
        "checkpoints": {},
        "inputs": {},
        "registered_metrics_reproduced": {},
    }
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
            "path": rel(path), "sha256": actual, "bytes": path.stat().st_size, "matches_evidence": actual == expected,
        }  # fmt: skip
    for (n, split), reg_path in REGISTERED.items():
        result = ev.evaluate_split(load_input(n, split), split)
        reg = load_json(reg_path)
        same = all(
            result[k] == reg[k]
            for k in ("metrics", "per_class", "density", "num_predictions")
        )
        record["registered_metrics_reproduced"][f"{n}/{split}"] = {
            "registered": rel(reg_path), "identical": same, "mAP": result["metrics"]["mAP"],
        }  # fmt: skip
        print(
            f"{n} {split}: mAP {result['metrics']['mAP']:.6f}, identical to registered: {same}"
        )
    # identity: Model A with no eligible FB box must come out unchanged (NMS >= 0.5 never re-suppresses A)
    a = load_input("A", "val_fast")
    ids = data.split_image_ids("val_fast")
    identity = {}
    for iou in sorted({c["nms_iou"] for c in grid()["candidates"]}):
        out = specialist.merge(
            a, [], val_sizes(), "coco32", 1.0, iou, grid()["max_det"], image_ids=ids
        )
        expected = sorted(
            a, key=lambda r: (r["image_id"], -r["score"])
        )  # stable: ties keep stored order
        same = sha256_json(out) == sha256_json(expected)
        kept = {json.dumps(r, sort_keys=True) for r in out}
        removed = [r for r in a if json.dumps(r, sort_keys=True) not in kept]
        identity[str(iou)] = {
            "detections": len(out), "A_detections": len(a), "unchanged": same and len(out) == len(a),
            "A_removed": len(removed), "removed_max_score": max((r["score"] for r in removed), default=None),
            "mAP_A_only_merge": ev.evaluate_split(out, "val_fast")["metrics"]["mAP"],
            "subset_ids_count": len(test_like_ids()),
        }  # fmt: skip
        # an A box removed by A-only NMS is a float32 boundary effect (stored pair IoU ~0.49998, recomputed > 0.5
        # after the xywh -> xyxy rebuild); it is recorded, and tolerated only if it moves mAP by < 1e-6
        identity[str(iou)]["ok"] = abs(identity[str(iou)]["mAP_A_only_merge"] - grid()["reference_model_A"][
            "val_fast"]["mAP"]) < 1e-6  # fmt: skip
    record["A_only_merge_unchanged"] = identity
    record["A_only_merge_note"] = (
        "At NMS 0.5, A-only re-NMS removes a few low-score A boxes whose stored pair IoU is ~0.49998 (CH-13 NMS ran on "
        "the original xyxy; the xywh -> xyxy rebuild in float32 lands just above 0.5). Frozen rule unchanged; deviation recorded."
    )
    subset_a = ev.evaluate(a, val_gt(), test_like_ids())["metrics"]["mAP"]
    record["model_A_subset_w_le_1920"] = {
        "frames": len(test_like_ids()),
        "mAP": subset_a,
    }
    ref = grid()["reference_model_A"]["val_fast"]["subset_w_le_1920"]
    record["ok"] = (
        all(v["matches_evidence"] for v in record["checkpoints"].values())
        and all(v["matches_evidence"] for v in record["inputs"].values())
        and all(
            v["identical"] for v in record["registered_metrics_reproduced"].values()
        )
        and all(v["ok"] for v in identity.values())
        and subset_a == ref
    )
    save_json(EVID / "identity.json", record)
    print(json.dumps(identity), "subset A", subset_a, "==", ref)
    print("identity ok" if record["ok"] else "IDENTITY FAILED")
    if not record["ok"]:
        raise SystemExit(1)


def cmd_screen(args) -> None:
    a, fb = load_input("A", "val_fast"), load_input("FB", "val_fast")
    base, base_subset = evaluate(a, "val_fast")
    rows = [summary_row("model_A", base, base_subset)]
    fb_result = load_json(REGISTERED[("FB", "val_fast")])
    rows.append(
        summary_row(
            "FB_alone",
            fb_result,
            load_json(REPO / "results/opt-04-model-scale/gates.json")["FB"]["G2"][
                "inputs"
            ]["subset_guard"]["candidate_mAP"],
        )
    )
    tag = "rerun" if args.rerun else "screen"
    for cand in grid()["candidates"]:
        preds, counts = run_merge(cand, a, fb, "val_fast")
        result, subset = evaluate(preds, "val_fast")
        out_dir = (
            EVID / "val_fast" / cand["id"]
            if not args.rerun
            else OUT / "rerun" / cand["id"]
        )
        ev.write_outputs(result, out_dir)
        save_json(out_dir / "run.json", {"candidate": cand, **counts, "subset_w_le_1920": {
            "frames": len(test_like_ids()), "mAP": subset}})  # fmt: skip
        if not args.rerun:
            save_json(OUT / "val_fast" / f"{cand['id']}.json", preds)
        rows.append(summary_row(cand["id"], result, subset, counts, cand))
        m = result["metrics"]
        print(f"{tag} {cand['id']}: mAP {m['mAP']:.4f} AP_S {m['AP_small']:.4f} >=100 {dense(result):.4f} "
              f"subset {subset:.4f} FB added {counts['FB_added']} A removed {counts['A_removed']}", flush=True)  # fmt: skip
    if not args.rerun:
        write_table(EVID / "screening_val_fast.csv", rows)


def cmd_determinism(_args) -> None:
    """Compare the independent rerun (outputs/opt05/rerun) with the stored screen, file by file."""
    rows = {}
    for cand in grid()["candidates"]:
        first = EVID / "val_fast" / cand["id"]
        second = OUT / "rerun" / cand["id"]
        r1, r2 = load_json(first / "run.json"), load_json(second / "run.json")
        m1, m2 = load_json(first / "metrics.json"), load_json(second / "metrics.json")
        rows[cand["id"]] = {
            "predictions_sha256": r1["predictions_sha256"],
            "same_predictions": r1["predictions_sha256"] == r2["predictions_sha256"],
            "same_metrics": all(
                m1[k] == m2[k]
                for k in ("metrics", "per_class", "density", "num_predictions")
            ),
            "same_subset": r1["subset_w_le_1920"] == r2["subset_w_le_1920"],
        }
        rows[cand["id"]]["pass"] = all(
            v for k, v in rows[cand["id"]].items() if k.startswith("same")
        )
    save_json(EVID / "determinism.json", rows)
    print(json.dumps(rows, indent=1))


def check(name: str, value, threshold, op: str = ">=") -> dict:
    return {
        "check": name,
        "value": value,
        "threshold": threshold,
        "op": op,
        "pass": value >= threshold,
    }


def cmd_select(_args) -> None:
    g = grid()
    gate = g["gate_V"]
    ref = g["reference_model_A"]["val_fast"]
    det = load_json(EVID / "determinism.json")
    out = {
        "rule": "results/opt-05-small-specialist/selection_rule.md (Gate V)",
        "candidates": {},
    }
    for cand in g["candidates"]:
        d = EVID / "val_fast" / cand["id"]
        m, run = load_json(d / "metrics.json"), load_json(d / "run.json")
        checks = [
            check("val_fast mAP (A + 0.003)", m["metrics"]["mAP"], gate["mAP"]),
            check(
                "val_fast AP_small (>= A)", m["metrics"]["AP_small"], gate["AP_small"]
            ),
            check("val_fast >=100 bin (A - 0.005)", dense(m), gate["bin_ge100"]),
            check(
                "width <= 1920 subset mAP (>= A)",
                run["subset_w_le_1920"]["mAP"],
                ref["subset_w_le_1920"],
            ),
            {
                "check": "deterministic rerun",
                "value": det[cand["id"]]["pass"],
                "pass": det[cand["id"]]["pass"],
            },
        ]
        out["candidates"][cand["id"]] = {
            **cand, "delta_mAP_vs_A": m["metrics"]["mAP"] - ref["mAP"], "checks": checks,
            "pass": all(c["pass"] for c in checks),
        }  # fmt: skip
    passing = [c for c in g["candidates"] if out["candidates"][c["id"]]["pass"]]
    best = None
    if passing:  # highest mAP, exact tie -> lower id (grid order)
        best = max(
            passing,
            key=lambda c: (
                out["candidates"][c["id"]]["checks"][0]["value"],
                -g["candidates"].index(c),
            ),
        )
    out["passing"] = [c["id"] for c in passing]
    out["selected"] = best["id"] if best else None
    out["outcome"] = (
        f"{best['id']} passes Gate V -> val_full (Gate F)" if best
        else "V0: no candidate passes Gate V -> no val_full, no GPU, no test CSV; Model A kept"
    )  # fmt: skip
    save_json(EVID / "selection_val_fast.json", out)
    for cid, c in out["candidates"].items():
        print(cid, f"{c['delta_mAP_vs_A']:+.4f}", "PASS" if c["pass"] else
              "fail: " + ", ".join(x["check"] for x in c["checks"] if not x["pass"]))  # fmt: skip
    print(out["outcome"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    s = sub.add_parser("screen")
    s.add_argument(
        "--rerun",
        action="store_true",
        help="independent second pass into outputs/opt05/rerun",
    )
    s.set_defaults(fn=cmd_screen)
    sub.add_parser("determinism").set_defaults(fn=cmd_determinism)
    sub.add_parser("select").set_defaults(fn=cmd_select)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
