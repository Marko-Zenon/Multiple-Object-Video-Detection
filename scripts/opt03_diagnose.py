"""OPT-03 phase 1: diagnose Model A's CH-13 pipeline on val_fast from the raw per-view cache (CPU only).

    python scripts/opt03_diagnose.py identity      # cache -> CH-13 merge == stored Model A predictions, mAP 0.2892
    python scripts/opt03_diagnose.py ablations     # project evaluator: views on their own, leave-one-view-out, no cap
    python scripts/opt03_diagnose.py tables        # diagnostic matcher at IoU 0.5: FP / FN breakdowns

Everything is written to results/opt-03-modelA-diagnosis/diagnosis/. Only val_fast is read (val_full is kept for
the single final check). mAP numbers come from movd.evaluate; "oracle" rows (no 500 cap) are diagnostics only
and are labelled as such, because Kaggle scores at most 500 boxes per image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, infer, opt03  # noqa: E402
from movd import evaluate as ev  # noqa: E402

RAW = data.REPO_ROOT / "outputs" / "opt03" / "raw"
OUT = data.REPO_ROOT / "results" / "opt-03-modelA-diagnosis" / "diagnosis"
PIPELINE = data.REPO_ROOT / "configs" / "runs" / "final_inference.json"
MODEL_A_PREDS = Path(
    "C:/UCU/AI/hw2/MOVD-ch14b/outputs/ch14b/stageA_epoch1/val_fast/pipeline_predictions.json"
)
SPLIT = "val_fast"


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_cache(split: str = SPLIT):
    z = np.load(RAW / f"{split}.npz")
    return z["image_ids"], z["boxes"], z["scores"]


def raws_of(boxes, scores, row) -> list[infer.Raw]:
    return [infer.Raw(boxes[row, v], scores[row, v]) for v in range(boxes.shape[1])]


def to_preds(ids, dets_list) -> list[dict]:
    out = []
    for image_id, d in zip(ids.tolist(), dets_list):
        out.extend(infer.to_coco(image_id, d))
    return out


def summary(result: dict) -> dict:
    m = result["metrics"]
    dens = {d["bin"]: d["mAP"] for d in result["density"]}
    return {
        "mAP": round(m["mAP"], 4),
        "AP50": round(m["AP50"], 4),
        "AP_small": round(m["AP_small"], 4),
        "AP_medium": round(m["AP_medium"], 4),
        "AP_large": round(m["AP_large"], 4),
        "density_ge100": round(dens.get(">=100"), 4)
        if dens.get(">=100") is not None
        else None,
        **{f"AP_{r['name']}": round(r["AP"], 4) for r in result["per_class"]},
    }


# ---------------------------------------------------------------- identity


def identity(_args) -> None:
    config = json.loads(PIPELINE.read_text(encoding="utf-8"))["config"]
    ids, boxes, scores = load_cache()
    dets = [infer.combine(raws_of(boxes, scores, r), config) for r in range(len(ids))]
    preds = to_preds(ids, dets)
    stored = json.loads(MODEL_A_PREDS.read_text(encoding="utf-8"))
    key = lambda p: (
        p["image_id"],
        p["category_id"],
        round(p["score"], 6),
        tuple(round(x, 3) for x in p["bbox"]),
    )  # noqa: E731
    same = sorted(map(key, preds)) == sorted(map(key, stored))
    max_diff = 0.0
    if len(preds) == len(stored):
        a = sorted(preds, key=key)
        b = sorted(stored, key=key)
        max_diff = max(
            max(
                abs(x - y)
                for x, y in zip(p["bbox"] + [p["score"]], q["bbox"] + [q["score"]])
            )
            for p, q in zip(a, b)
        )
    result = ev.evaluate_split(preds, SPLIT)
    record = {
        "split": SPLIT,
        "frames": len(ids),
        "detections_from_cache": len(preds),
        "detections_stored_model_a": len(stored),
        "stored_predictions": MODEL_A_PREDS.as_posix(),
        "stored_predictions_sha256": hashlib.sha256(
            MODEL_A_PREDS.read_bytes()
        ).hexdigest(),
        "identical_after_rounding_1e-3px_1e-6score": same,
        "max_abs_difference": max_diff,
        "mAP_from_cache": result["metrics"]["mAP"],
        "registered_mAP": 0.2892,
    }
    save_json(OUT / "identity.json", record)
    print(json.dumps(record, indent=2))


# ---------------------------------------------------------------- ablations (project evaluator)


def ablations(_args) -> None:
    ids, boxes, scores = load_cache()
    gt = data.load_coco("val")
    split_ids = data.split_image_ids(SPLIT)
    coco_gt = ev._coco_gt(gt)
    rows = []
    sets = {
        "all6 (Model A pipeline)": None,
        "full only (single pass)": [0],
        "flip only": [1],
        "full+flip": [0, 1],
        "tiles only": [2, 3, 4, 5],
        "full+tiles": [0, 2, 3, 4, 5],
        **{
            f"all but {opt03.VIEWS[v]}": [u for u in range(6) if u != v]
            for v in range(6)
        },
    }
    for name, views in sets.items():
        traced = [
            opt03.combine_traced(raws_of(boxes, scores, r), keep_views=views)
            for r in range(len(ids))
        ]
        capped = to_preds(ids, [t.dets(500) for t in traced])
        res = ev.evaluate_split(capped, SPLIT)
        row = {"set": name, "max_dets": 500, **summary(res)}
        row["frames_at_cap"] = int(sum(len(t.scores) >= 500 for t in traced))
        row["mean_boxes_after_nms_precap"] = round(
            float(np.mean([len(t.scores) for t in traced])), 1
        )
        rows.append(row)
        print(row["set"], row["mAP"], flush=True)
        if views is None:
            # Diagnostic only: the same merge without the 500 cap (pycocotools maxDets raised to 3000).
            full = to_preds(ids, [t.dets(None) for t in traced])
            E = ev._run_cocoeval(coco_gt, full, split_ids, 3000)
            s = ev._summary(E)
            rows.append(
                {
                    "set": "all6, NO cap (oracle, not scoreable on Kaggle)",
                    "max_dets": 3000,
                    "mAP": round(s["mAP"], 4),
                    "AP50": round(s["AP50"], 4),
                    "AP_small": round(s["AP_small"], 4),
                    "AP_medium": round(s["AP_medium"], 4),
                    "AP_large": round(s["AP_large"], 4),
                }
            )
            print("oracle no cap", s["mAP"], flush=True)
    write_csv(OUT / "ablations_val_fast.csv", rows)


# ---------------------------------------------------------------- matcher tables


def size_label(area: float) -> str:
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def density_label(n: int) -> str:
    for low, high in ev.DENSITY_BINS:
        if low <= n < high:
            return ev._bin_label(low, high)
    raise ValueError(n)


def score_bin(s: float) -> str:
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
    for lo, hi in pairwise(edges):
        if lo <= s < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return "?"


def tables(_args) -> None:
    ids, boxes, scores = load_cache()
    gt = data.load_coco("val")
    images = {im["id"]: im for im in gt["images"]}
    anns = defaultdict(list)
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    names = (
        data.CLASS_NAMES if hasattr(data, "CLASS_NAMES") else [str(i) for i in range(8)]
    )

    box_rows = []  # one per kept box (pre-cap), with attributes
    gt_rows = []  # one per target GT
    hits_by_frame = {}  # image id -> set of track ids hit (capped)
    frame_rows = []
    for r, image_id in enumerate(ids.tolist()):
        im = images[image_id]
        W, H = im["width"], im["height"]
        t = opt03.combine_traced(raws_of(boxes, scores, r))
        tg = [a for a in anns[image_id] if not a.get("iscrowd", 0)]
        cr = [a for a in anns[image_id] if a.get("iscrowd", 0)]
        to_xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]  # noqa: E731
        gb = np.array([to_xyxy(a["bbox"]) for a in tg], np.float64).reshape(-1, 4)
        gc = np.array([a["category_id"] - 1 for a in tg], np.int64)
        crowd = np.unique(
            np.array([to_xyxy(a["bbox"]) for a in cr], np.float64).reshape(-1, 4),
            axis=0,
        )
        n_cap = min(500, len(t.scores))
        out_cap, hit_cap = opt03.match_frame(
            t.boxes[:n_cap], t.classes[:n_cap], gb, gc, crowd
        )
        out_all, hit_all = opt03.match_frame(t.boxes, t.classes, gb, gc, crowd)
        windows = infer.tile_windows(W, H, 2, 0.6)
        edge = np.zeros(len(t.scores), bool)
        for v in range(2, 6):
            sel = t.src == v
            edge[sel] = opt03.touches_inner_tile_edge(
                t.boxes[sel], windows[v - 2], W, H
            )
        dens = density_label(len(tg))
        for i in range(len(t.scores)):
            box_rows.append(
                (
                    image_id, int(t.classes[i]), float(t.scores[i]), int(t.src[i]), i, i < 500,
                    int(out_cap[i]) if i < n_cap else -1, int(out_all[i]), bool(edge[i]), dens,
                )
            )  # fmt: skip
        for j, a in enumerate(tg):
            gt_rows.append(
                {
                    "image_id": image_id,
                    "clip": data.clip_name(im),
                    "frame": im["frame_id"],
                    "track": a.get("track_id"),
                    "cls": int(gc[j]),
                    "size": size_label(a["area"]),
                    "occlusion": a.get("occlusion"),
                    "truncation": a.get("truncation"),
                    "density": dens,
                    "width": W,
                    "hit_cap": bool(hit_cap[j] >= 0),
                    "hit_precap": bool(hit_all[j] >= 0),
                    "hit_src": int(t.src[hit_cap[j]]) if hit_cap[j] >= 0 else -1,
                    "hit_score": float(t.scores[hit_cap[j]])
                    if hit_cap[j] >= 0
                    else None,
                }
            )
        hits_by_frame[image_id] = (
            {gt_rows[-len(tg) + j]["track"] for j in range(len(tg)) if hit_cap[j] >= 0}
            if tg
            else set()
        )
        frame_rows.append(
            {
                "image_id": image_id,
                "targets": len(tg),
                "density": dens,
                "width": W,
                "candidates_into_nms": t.n_candidates,
                "after_nms": len(t.scores),
                "cut_by_cap": max(0, len(t.scores) - 500),
                "score_at_rank500": float(t.scores[499])
                if len(t.scores) >= 500
                else None,
            }
        )

    B = np.array(
        [(b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8]) for b in box_rows],
        dtype=float,
    )
    cls_, score_, src_, _rank, incap_, oc_, oa_, edge_ = B.T
    dens_ = np.array([b[9] for b in box_rows])
    incap_ = incap_.astype(bool)
    edge_ = edge_.astype(bool)
    G = gt_rows
    res = {}

    def recall_table(key):
        out = []
        for val in sorted({g[key] for g in G}, key=lambda x: (str(type(x)), x)):
            sub = [g for g in G if g[key] == val]
            out.append(
                {
                    key: names[val] if key == "cls" else val,
                    "gt": len(sub),
                    "recall50_capped": round(np.mean([g["hit_cap"] for g in sub]), 4),
                    "recall50_precap": round(
                        np.mean([g["hit_precap"] for g in sub]), 4
                    ),
                    "lost_to_cap": sum(
                        g["hit_precap"] and not g["hit_cap"] for g in sub
                    ),
                }
            )
        return out

    for key in ("cls", "size", "density", "occlusion", "truncation", "width"):
        res[f"recall_by_{key}"] = recall_table(key)

    # False positives of the scored (capped) set by class and type.
    fp_rows = []
    for c in range(8):
        sel = incap_ & (cls_ == c)
        cnt = Counter(oc_[sel].astype(int))
        n = int(sel.sum())
        fp_rows.append(
            {
                "cls": names[c],
                "boxes": n,
                **{opt03.OUTCOMES[k]: cnt.get(k, 0) for k in range(5)},
            }
        )
    res["capped_outcomes_by_class"] = fp_rows
    dens_rows = []
    for d in sorted(set(dens_)):
        sel = incap_ & (dens_ == d)
        cnt = Counter(oc_[sel].astype(int))
        dens_rows.append(
            {
                "density": d,
                "boxes": int(sel.sum()),
                **{opt03.OUTCOMES[k]: cnt.get(k, 0) for k in range(5)},
            }
        )
    res["capped_outcomes_by_density"] = dens_rows

    # Source views: share of kept boxes, of TPs, and precision, capped set.
    src_rows = []
    for v, name in enumerate(opt03.VIEWS):
        sel = incap_ & (src_ == v)
        cnt = Counter(oc_[sel].astype(int))
        scored = int(sel.sum()) - cnt.get(opt03.IGNORED, 0)
        src_rows.append(
            {
                "view": name,
                "kept_boxes": int(sel.sum()),
                "tp": cnt.get(opt03.TP, 0),
                "precision50": round(cnt.get(opt03.TP, 0) / max(scored, 1), 4),
                "fp_duplicate": cnt.get(opt03.FP_DUP, 0),
                "mean_score": round(float(score_[sel].mean()), 4)
                if sel.any()
                else None,
            }
        )
    res["capped_by_source_view"] = src_rows

    # Tile boxes touching an inner tile edge (truncated objects) vs other tile boxes.
    tile = incap_ & (src_ >= 2)
    edge_rows = []
    for flag in (True, False):
        sel = tile & (edge_ == flag)
        cnt = Counter(oc_[sel].astype(int))
        scored = int(sel.sum()) - cnt.get(opt03.IGNORED, 0)
        edge_rows.append(
            {
                "tile_box_touches_inner_edge": flag,
                "boxes": int(sel.sum()),
                "tp": cnt.get(opt03.TP, 0),
                "precision50": round(cnt.get(opt03.TP, 0) / max(scored, 1), 4),
                "fp_duplicate": cnt.get(opt03.FP_DUP, 0),
                "fp_background": cnt.get(opt03.FP_BG, 0),
                "mean_score": round(float(score_[sel].mean()), 4)
                if sel.any()
                else None,
            }
        )
    res["tile_inner_edge"] = edge_rows

    # Calibration: precision at IoU 0.5 by class and score bin (capped set, ignored boxes excluded).
    cal = []
    for c in range(8):
        for sb in sorted({score_bin(s) for s in score_}):
            sel = (
                incap_
                & (cls_ == c)
                & np.array([score_bin(s) == sb for s in score_])
                & (oc_ != opt03.IGNORED)
            )
            if sel.sum() >= 20:
                cal.append(
                    {
                        "cls": names[c],
                        "score_bin": sb,
                        "boxes": int(sel.sum()),
                        "precision50": round(float((oc_[sel] == opt03.TP).mean()), 4),
                    }
                )
    res["calibration_precision_by_class_score"] = cal

    # The cap: what lies below rank 500.
    below = ~incap_
    cnt = Counter(oa_[below].astype(int))
    res["below_cap"] = {
        "frames": len(frame_rows),
        "frames_at_cap": sum(f["after_nms"] >= 500 for f in frame_rows),
        "boxes_after_nms_mean": round(
            float(np.mean([f["after_nms"] for f in frame_rows])), 1
        ),
        "boxes_cut_total": int(below.sum()),
        "cut_outcomes_when_matched_without_cap": {
            opt03.OUTCOMES[k]: cnt.get(k, 0) for k in range(5)
        },
        "gt_lost_to_cap": sum(g["hit_precap"] and not g["hit_cap"] for g in G),
        "gt_total": len(G),
        "score_at_rank500_median": float(
            np.median(
                [
                    f["score_at_rank500"]
                    for f in frame_rows
                    if f["score_at_rank500"] is not None
                ]
            )
        ),
        "cut_by_class": {names[c]: int((below & (cls_ == c)).sum()) for c in range(8)},
        "cut_tp_by_class": {
            names[c]: int((below & (cls_ == c) & (oa_ == opt03.TP)).sum())
            for c in range(8)
        },
    }
    res["cap_by_density"] = [
        {
            "density": d,
            "frames": sum(f["density"] == d for f in frame_rows),
            "mean_after_nms": round(
                float(
                    np.mean([f["after_nms"] for f in frame_rows if f["density"] == d])
                ),
                1,
            ),
            "mean_cut": round(
                float(
                    np.mean([f["cut_by_cap"] for f in frame_rows if f["density"] == d])
                ),
                1,
            ),
            "gt_lost_to_cap": sum(
                g["hit_precap"] and not g["hit_cap"] for g in G if g["density"] == d
            ),
        }
        for d in sorted({f["density"] for f in frame_rows})
    ]

    # Temporal: a missed GT whose track is detected in the previous or next val_fast frame (4 frames apart).
    by_clip = defaultdict(list)
    for f in frame_rows:
        by_clip[data.clip_name(images[f["image_id"]])].append(f["image_id"])
    neighbours = {}
    for fids in by_clip.values():
        fids = sorted(fids, key=lambda i: images[i]["frame_id"])
        for k, i in enumerate(fids):
            neighbours[i] = fids[max(k - 1, 0) : k] + fids[k + 1 : k + 2]
    missed = [g for g in G if not g["hit_cap"]]
    recoverable = [
        g
        for g in missed
        if any(g["track"] in hits_by_frame[n] for n in neighbours[g["image_id"]])
    ]
    res["temporal"] = {
        "frame_spacing_in_val_fast": 4,
        "gt_missed_capped": len(missed),
        "missed_but_track_hit_in_a_neighbour_frame": len(recoverable),
        "share_of_all_gt": round(len(recoverable) / len(G), 4),
        "by_class": dict(Counter(names[g["cls"]] for g in recoverable)),
        "by_size": dict(Counter(g["size"] for g in recoverable)),
    }
    # Scores of TPs vs FPs (capped set).
    sel_tp = incap_ & (oc_ == opt03.TP)
    sel_fp = incap_ & np.isin(oc_, [opt03.FP_DUP, opt03.FP_CLASS, opt03.FP_BG])
    res["score_quantiles"] = {
        "tp": [
            round(float(q), 4)
            for q in np.quantile(score_[sel_tp], [0.1, 0.25, 0.5, 0.75, 0.9])
        ],
        "fp": [
            round(float(q), 4)
            for q in np.quantile(score_[sel_fp], [0.1, 0.25, 0.5, 0.75, 0.9])
        ],
        "share_of_capped_boxes_below_0.05": round(
            float((score_[incap_] < 0.05).mean()), 4
        ),
        "tp_share_below_0.05": round(float((score_[sel_tp] < 0.05).mean()), 4),
    }
    save_json(OUT / "tables.json", res)
    write_csv(OUT / "frames_val_fast.csv", frame_rows)
    np.savez_compressed(
        data.REPO_ROOT / "outputs" / "opt03" / "diag_boxes_val_fast.npz",
        B=B,
        dens=dens_,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in res.items()
                if k != "calibration_precision_by_class_score"
            },
            indent=1,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("identity", "ablations", "tables"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"identity": identity, "ablations": ablations, "tables": tables}[args.cmd](args)


if __name__ == "__main__":
    main()
