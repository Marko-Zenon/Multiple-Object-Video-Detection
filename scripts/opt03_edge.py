"""OPT-03 phase 2/3: the frozen tile-border experiment (`results/opt-03-modelA-diagnosis/selection_rule.md`).

    python scripts/opt03_edge.py screen              # 6 candidates on val_fast, gate G-a..G-d, selection
    python scripts/opt03_edge.py confirm             # the selected candidate on val_full, exactly once, F-a..F-c
    python scripts/opt03_edge.py test                # the selected candidate on the test raw cache -> predictions
    python scripts/opt03_edge.py live --weights <epoch0.pt>   # live pipeline == cache path on a few test frames

All post-processing runs on the CPU from the Model A raw caches written by `scripts/opt03_cache.py`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, infer, opt03  # noqa: E402
from movd import evaluate as ev  # noqa: E402

ROOT = data.REPO_ROOT
RES = ROOT / "results" / "opt-03-modelA-diagnosis"
RAW = ROOT / "outputs" / "opt03" / "raw"
PREDS = ROOT / "outputs" / "opt03" / "preds"
PIPELINE = ROOT / "configs" / "runs" / "final_inference.json"
CANDIDATES = {
    "e1": (0.0, 0.002),
    "e2": (0.0, 0.005),
    "e3": (0.0, 0.01),
    "e4": (0.0, 0.02),
    "e5": (0.5, 0.005),
    "e6": (0.25, 0.005),
}
BASE_FAST = {
    "mAP": 0.2892166441381564,
    "AP_small": 0.14273391809388225,
    "dens100": 0.35150175230579234,
}
BASE_FULL_MAP = 0.28596193196293734
GATE = {
    "G-a": BASE_FAST["mAP"] + 0.003,
    "G-b": BASE_FAST["AP_small"] - 0.003,
    "G-c": BASE_FAST["dens100"] - 0.005,
}
TEST_MINUTES_MODEL_A = (
    1729.5 / 60
)  # CH-14b full test inference, same 6 passes (results/ch-14b-…/README.md)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def config() -> dict:
    return json.loads(PIPELINE.read_text(encoding="utf-8"))["config"]


def sizes(split: str) -> dict[int, tuple[int, int]]:
    if split == "test":
        return {r["id"]: (r["width"], r["height"]) for r in data.load_test_images()}
    return {
        im["id"]: (im["width"], im["height"]) for im in data.load_coco("val")["images"]
    }


def run_candidate(split: str, factor: float, margin: float) -> tuple[list[dict], dict]:
    z = np.load(RAW / f"{split}.npz")
    ids, boxes, scores = z["image_ids"], z["boxes"], z["scores"]
    wh, cfg = sizes(split), config()
    preds, counts = [], []
    t0 = time.perf_counter()
    for r, image_id in enumerate(ids.tolist()):
        raws = [infer.Raw(boxes[r, v], scores[r, v]) for v in range(boxes.shape[1])]
        d = opt03.edge_filter_combine(raws, *wh[image_id], factor, margin, cfg)
        counts.append(len(d))
        preds.extend(infer.to_coco(image_id, d))
    sec = time.perf_counter() - t0
    stats = {
        "frames": len(ids),
        "detections": len(preds),
        "frames_at_500": int(sum(c >= 500 for c in counts)),
        "cpu_ms_per_frame": round(1000 * sec / len(ids), 2),
        "raw_cache_sha256": sha256(RAW / f"{split}.npz"),
    }
    return preds, stats


def row_of(name, factor, margin, result, stats) -> dict:
    m = result["metrics"]
    dens = {d["bin"]: d["mAP"] for d in result["density"]}
    return {
        "candidate": name,
        "factor": factor,
        "margin": margin,
        "mAP": m["mAP"],
        "AP50": m["AP50"],
        "AP75": m["AP75"],
        "AP_small": m["AP_small"],
        "AP_medium": m["AP_medium"],
        "AP_large": m["AP_large"],
        **{f"dens_{k}": v for k, v in dens.items()},
        **{f"AP_{r['name']}": r["AP"] for r in result["per_class"]},
        **stats,
    }


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def screen(_args) -> None:
    rows = []
    for name, (f, m) in CANDIDATES.items():
        preds, stats = run_candidate("val_fast", f, m)
        result = ev.evaluate_split(preds, "val_fast")
        ev.write_outputs(result, RES / "val_fast" / name)
        cpu_extra_min = 3811 * stats["cpu_ms_per_frame"] / 1000 / 60
        row = row_of(name, f, m, result, stats)
        row["projected_test_minutes"] = round(TEST_MINUTES_MODEL_A + cpu_extra_min, 1)
        row["G-a"] = row["mAP"] >= GATE["G-a"]
        row["G-b"] = row["AP_small"] >= GATE["G-b"]
        row["G-c"] = row["dens_>=100"] >= GATE["G-c"]
        row["G-d"] = row["projected_test_minutes"] < 90
        row["passes"] = all(row[g] for g in ("G-a", "G-b", "G-c", "G-d"))
        rows.append(row)
        print(
            name,
            f,
            m,
            round(row["mAP"], 4),
            round(row["AP_small"], 4),
            round(row["dens_>=100"], 4),
            row["passes"],
            flush=True,
        )
    write_csv(RES / "screening_val_fast.csv", rows)
    passing = [r for r in rows if r["passes"]]
    selected = None
    if passing:
        best = max(r["mAP"] for r in passing)
        near = [r for r in passing if best - r["mAP"] <= 0.0005]
        near.sort(
            key=lambda r: (r["factor"] != 0, r["margin"])
        )  # drop before down-weight, then smaller margin
        selected = near[0]["candidate"]
    save_json(
        RES / "selection_val_fast.json",
        {
            "rule": "results/opt-03-modelA-diagnosis/selection_rule.md (commit 528167a, before any result)",
            "baseline_val_fast": BASE_FAST,
            "thresholds": GATE,
            "passing": [r["candidate"] for r in passing],
            "selected": selected,
            "selected_metrics": next(
                (r for r in rows if r["candidate"] == selected), None
            ),
        },
    )
    print("selected:", selected)


def confirm(_args) -> None:
    sel = json.loads((RES / "selection_val_fast.json").read_text(encoding="utf-8"))
    name = sel["selected"]
    if name is None:
        raise SystemExit(
            "no candidate passed the val_fast gate: val_full is not run (rule)"
        )
    out = RES / "val_full" / name
    if (out / "metrics.json").exists():
        raise SystemExit(f"{out} exists: val_full is evaluated exactly once")
    f, m = CANDIDATES[name]
    preds, stats = run_candidate("val_full", f, m)
    path = PREDS / f"val_full_{name}.json"
    save_json(path, preds)
    result = ev.evaluate_split(preds, "val_full")
    ev.write_outputs(result, out)
    base_dir = ROOT / "results" / "ch-14b-rtdetr-short-finetune"
    base_cls = {
        r["name"]: float(r["AP_stageA_epoch1"])
        for r in read_csv(base_dir / "per_class_stageA_epoch1.csv")
    }
    base_den = {
        r["bin"]: float(r["mAP_stageA_epoch1"])
        for r in read_csv(base_dir / "density_stageA_epoch1.csv")
    }
    cls_d = {
        r["name"]: round(r["AP"] - base_cls[r["name"]], 4) for r in result["per_class"]
    }
    den_d = {
        d["bin"]: round(d["mAP"] - base_den[d["bin"]], 4) for d in result["density"]
    }
    checks = {
        "F-a": result["metrics"]["mAP"] >= BASE_FULL_MAP + 0.003,
        "F-b": min(cls_d.values()) >= -0.010 and min(den_d.values()) >= -0.010,
        "F-c": True,
    }
    save_json(
        RES / "selection_val_full.json",
        {
            "candidate": name,
            "factor": f,
            "margin": m,
            "model_a_val_full_mAP": BASE_FULL_MAP,
            "threshold_F-a": BASE_FULL_MAP + 0.003,
            "metrics": result["metrics"],
            "delta_mAP": result["metrics"]["mAP"] - BASE_FULL_MAP,
            "per_class_delta": cls_d,
            "density_delta": den_d,
            "checks": checks,
            "passes": all(checks.values()),
            "predictions": path.relative_to(ROOT).as_posix(),
            "predictions_sha256": sha256(path),
            **stats,
        },
    )
    print(
        json.dumps(
            {
                "mAP": result["metrics"]["mAP"],
                "checks": checks,
                "cls": cls_d,
                "dens": den_d,
            },
            indent=1,
        )
    )


def test(_args) -> None:
    sel = json.loads((RES / "selection_val_full.json").read_text(encoding="utf-8"))
    if not sel["passes"]:
        raise SystemExit("final gate failed: no test predictions (rule)")
    f, m = CANDIDATES[sel["candidate"]]
    preds, stats = run_candidate("test", f, m)
    if stats["frames"] != 3811:
        raise SystemExit(f"test cache has {stats['frames']} frames")
    path = PREDS / f"test_{sel['candidate']}.json"
    save_json(path, preds)
    save_json(
        RES / "test_predictions.json",
        {
            "candidate": sel["candidate"],
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            **stats,
        },
    )
    print(path, stats)


def live(args) -> None:
    """Live pipeline (6 GPU passes + edge step) == cache path, on the first frames of two test clips."""
    import cv2

    from movd import rtdetr

    sel = json.loads((RES / "selection_val_full.json").read_text(encoding="utf-8"))
    f, m = CANDIDATES[sel["candidate"]]
    cfg = config()
    z = np.load(RAW / "test.npz")
    row_of_id = {int(i): r for r, i in enumerate(z["image_ids"].tolist())}
    rows = data.load_test_images()
    pick = [
        r
        for r in rows
        if r["frame"] <= 5 and r["clip"] in (rows[0]["clip"], rows[-1]["clip"])
    ]
    model = rtdetr.load_model(args.weights, "rtdetr")
    same = 0
    for r in pick:
        image = cv2.imread(str(data.test_image_path(r)))
        raws = [
            infer.raw_pass(model, image, s["imgsz"], "0", s["view"])
            for s in infer.pass_specs(cfg)
        ]
        live_d = opt03.edge_filter_combine(raws, r["width"], r["height"], f, m, cfg)
        k = row_of_id[r["id"]]
        cached = [infer.Raw(z["boxes"][k, v], z["scores"][k, v]) for v in range(6)]
        cache_d = opt03.edge_filter_combine(cached, r["width"], r["height"], f, m, cfg)
        same += bool(
            np.array_equal(live_d.boxes, cache_d.boxes)
            and np.array_equal(live_d.scores, cache_d.scores)
        )
    save_json(
        RES / "live_vs_cache.json",
        {"frames": len(pick), "identical": same, "candidate": sel["candidate"]},
    )
    print(same, "/", len(pick))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("screen", "confirm", "test"):
        sub.add_parser(n)
    lv = sub.add_parser("live")
    lv.add_argument("--weights", type=Path, required=True)
    args = ap.parse_args()
    {"screen": screen, "confirm": confirm, "test": test, "live": live}[args.cmd](args)


if __name__ == "__main__":
    main()
