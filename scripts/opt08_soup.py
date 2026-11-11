"""OPT-08 E4: weight soup of Model A (epoch0.pt) with the adjacent CH-14 checkpoint epoch1.pt.

    python scripts/opt08_soup.py check                       # hashes, class, YAML, parameter names / shapes / dtypes
    python scripts/opt08_soup.py build                       # one soup checkpoint per frozen weight (outputs/opt08/soup/)
    python scripts/opt08_soup.py cache W --stage discovery   # raw 6-view RT-DETR output per clip (GPU, resumable)
    python scripts/opt08_soup.py smoke                       # Model A through the same cache + merge path on 20 frames

`W` is a frozen weight tag (w080, w065, w050) or `modelA` (smoke only). Stages: discovery / confirmation clips of
the frozen OPT-08 split, or test. The raw output is what `movd.infer.raw_pass` returns for each CH-13 view
(as scripts/opt03_cache.py); `merged(W, split)` turns it into CH-13 pipeline predictions with the unchanged
`movd.infer.combine`. Rule: results/opt-08-overnight/e4_checkpoint_soup/rule.md (committed before any soup inference).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import opt07_loop as o7
from ch14_infer import frames

from movd import data, infer, opt03, rtdetr

WEIGHTS_DIR = REPO.parent / "MOVD-ch14/outputs/final_rtdetr/train/weights"
A = (
    WEIGHTS_DIR / "epoch0.pt",
    "6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b",
)
B = (
    WEIGHTS_DIR / "epoch1.pt",
    "5594fb0e9a1ae97bd0ca42fa6c6bc42cfbed9c77697ddc3b3e2336f6bb015799",
)
SOUP_WEIGHTS = {
    "w080": 0.8,
    "w065": 0.65,
    "w050": 0.5,
}  # weight on Model A; frozen before any soup inference
OUT = REPO / "outputs" / "opt08" / "soup"
EVID = REPO / "results" / "opt-08-overnight" / "e4_checkpoint_soup"
PIPELINE = REPO / "configs" / "runs" / "final_inference.json"


def config() -> dict:
    return json.loads(PIPELINE.read_text(encoding="utf-8"))["config"]


def load_ckpt(path: Path, expected: str):
    import torch

    if o7.sha256(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from checkpoints.json")
    return torch.load(path, map_location="cpu", weights_only=False)


def compatibility() -> dict:
    a, b = load_ckpt(*A), load_ckpt(*B)
    ea, eb = a.get("ema"), b.get("ema")
    sa, sb = ea.state_dict(), eb.state_dict()
    rec = {
        "model_A": {
            "path": o7.rel(A[0]),
            "sha256": A[1],
            "epoch": a["epoch"],
            "updates": a["updates"],
        },
        "adjacent": {
            "path": o7.rel(B[0]),
            "sha256": B[1],
            "epoch": b["epoch"],
            "updates": b["updates"],
        },
        "weights_entry": "ema (model is None in both)",
        "same_class": type(ea).__name__ == type(eb).__name__,
        "class": type(ea).__name__,
        "same_yaml": str(ea.yaml) == str(eb.yaml),
        "num_tensors": len(sa),
        "same_names": list(sa) == list(sb),
        "same_shapes": all(sa[k].shape == sb[k].shape for k in sa),
        "same_dtypes": all(sa[k].dtype == sb[k].dtype for k in sa),
        "dtypes": sorted({str(v.dtype) for v in sa.values()}),
        "same_names_list_in_model_attr": a["model"] is None and b["model"] is None,
        "adjacent_epochs": b["epoch"] == a["epoch"] + 1,
    }
    rec["compatible"] = all(
        rec[k]
        for k in (
            "same_class",
            "same_yaml",
            "same_names",
            "same_shapes",
            "same_dtypes",
            "adjacent_epochs",
        )
    )
    return rec


def soup_state(sa: dict, sb: dict, w: float) -> dict:
    """w * A + (1 - w) * B for floating tensors (computed in float32, stored in the original dtype); others from A."""
    out = {}
    for k, ta in sa.items():
        if ta.is_floating_point():
            mixed = w * ta.float() + (1.0 - w) * sb[k].float()
            out[k] = mixed.to(ta.dtype)
        else:
            out[k] = ta.clone()
    return out


def soup_path(tag: str) -> Path:
    return OUT / f"soup_{tag}.pt"


def weights_path(tag: str) -> Path:
    if tag == "modelA":
        return A[0]
    built = json.loads((EVID / "soup_build.json").read_text(encoding="utf-8"))
    path = soup_path(tag)
    if o7.sha256(path) != built[tag]["sha256"]:
        raise SystemExit(f"{path}: sha256 differs from soup_build.json")
    return path


def cmd_check(_args) -> None:
    rec = compatibility()
    o7.save_json(EVID / "compatibility.json", rec)
    print(json.dumps(rec, indent=2))
    if not rec["compatible"]:
        raise SystemExit("compatibility imperfect: E4 rejected (rule)")


def cmd_build(_args) -> None:
    import copy

    import torch

    if not compatibility()["compatible"]:
        raise SystemExit("compatibility imperfect: E4 rejected (rule)")
    a, b = load_ckpt(*A), load_ckpt(*B)
    sa, sb = a["ema"].state_dict(), b["ema"].state_dict()
    record = {}
    for tag, w in SOUP_WEIGHTS.items():
        model = copy.deepcopy(a["ema"])
        model.load_state_dict(soup_state(sa, sb, w))
        ckpt = {
            k: v
            for k, v in a.items()
            if k not in ("optimizer", "scaler", "ema", "model")
        }
        ckpt.update({"ema": model, "model": None, "optimizer": None, "scaler": None})
        ckpt["opt08_soup"] = {
            "weight_model_A": w,
            "model_A_sha256": A[1],
            "adjacent_sha256": B[1],
        }
        path = soup_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        record[tag] = {
            "weight_model_A": w,
            "path": o7.rel(path),
            "sha256": o7.sha256(path),
        }
    o7.save_json(EVID / "soup_build.json", record)
    print(json.dumps(record, indent=2))


def stage_frames(stage: str) -> dict[str, list[tuple[int, Path]]]:
    if stage == "test":
        items = frames("test", None)
    else:
        sp = json.loads(
            (REPO / "results/opt-08-overnight/split.json").read_text(encoding="utf-8")
        )
        items = [f for f in frames("val_full", None) if f[0] in sp[stage]]
    by_clip = defaultdict(list)
    for clip, image_id, path in items:
        by_clip[clip].append((image_id, path))
    return dict(by_clip)


def shard_dir(tag: str, stage: str) -> Path:
    return OUT / "raw" / tag / stage


def cache(
    tag: str,
    stage: str,
    device: str = "0",
    max_frames: int = 0,
    clips: list[str] | None = None,
    out_dir: Path | None = None,
) -> dict:
    import cv2
    import torch

    cfg = config()
    specs = infer.pass_specs(cfg)
    grid, frac = cfg.get("tile_grid", 2), cfg.get("tile_frac", 0.6)
    by_clip = stage_frames(stage)
    model = rtdetr.load_model(weights_path(tag), "rtdetr")
    torch.zeros(1, device=f"cuda:{device}")
    if out_dir is None:
        out_dir = shard_dir(tag, stage) if not max_frames else OUT / "raw_smoke" / tag
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
        if max_frames:
            break
    return {
        "frames": computed,
        "seconds": round(seconds, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }


def sizes(stage: str) -> dict[int, tuple[int, int]]:
    if stage == "test":
        return {r["id"]: (r["width"], r["height"]) for r in data.load_test_images()}
    return {
        im["id"]: (im["width"], im["height"]) for im in data.load_coco("val")["images"]
    }


def merge_shard(
    path: Path, wh: dict, factor: float = 1.0, margin: float = 0.0
) -> list[dict]:
    """CH-13 pipeline predictions from one raw shard: `opt03.edge_filter_combine` with factor 1 is the unchanged
    CH-13 merge (OPT-07 identity: reproduces Model A's stored predictions exactly); factor 0, margin 0.005 is x4."""
    cfg = config()
    z = np.load(path)
    preds = []
    for r, image_id in enumerate(z["image_ids"].tolist()):
        raws = [
            infer.Raw(z["boxes"][r, v], z["scores"][r, v])
            for v in range(z["boxes"].shape[1])
        ]
        d = opt03.edge_filter_combine(raws, *wh[image_id], factor, margin, cfg)
        preds.extend(infer.to_coco(image_id, d))
    return preds


def merged(
    tag: str, stages: list[str], factor: float = 1.0, margin: float = 0.0
) -> list[dict]:
    """Merged predictions of every cached clip of `stages` (each stage must be complete)."""
    wh = sizes("test" if stages == ["test"] else "val")
    preds = []
    for stage in stages:
        clips = sorted(stage_frames(stage))
        for clip in clips:
            shard = shard_dir(tag, stage) / f"{clip}.npz"
            if not shard.exists():
                raise SystemExit(f"{shard} missing: cache {tag} {stage} first")
            preds.extend(merge_shard(shard, wh, factor, margin))
    return preds


def cmd_smoke(_args) -> None:
    """Model A through this cache + merge path on the first 20 frames of one clip == Model A's stored predictions."""
    clip = min(stage_frames("discovery"))
    rec = cache("modelA", "discovery", max_frames=20, clips=[clip])
    shard = OUT / "raw_smoke" / "modelA" / f"{clip}.npz"
    preds = merge_shard(shard, sizes("val"))
    ids = set(np.load(shard)["image_ids"].tolist())
    ref = [p for p in o7.load_a("val_full") if p["image_id"] in ids]
    same = o7.same_predictions(preds, ref)
    out = {
        "clip": clip,
        "frames": len(ids),
        **rec,
        **same,
        "seconds_per_frame": round(rec["seconds"] / max(1, rec["frames"]), 3),
    }
    o7.save_json(EVID / "smoke.json", out)
    print(json.dumps(out, indent=2))
    if not same["identical"]:
        raise SystemExit("smoke: cache + merge path does not reproduce Model A")


def cmd_recheck(args) -> None:
    """U7 (GPU part): re-infer confirmation clip 305 in a new process; must equal the cached raw shard exactly."""
    clip = "uav0000305_00000_v"
    tag = args.tag
    ref = shard_dir(tag, "confirmation") / f"{clip}.npz"
    rec_dir = OUT / "raw_recheck" / tag
    shard = rec_dir / f"{clip}.npz"
    if shard.exists():
        shard.unlink()  # only this command's own earlier output
    info = cache(tag, "confirmation", clips=[clip], out_dir=rec_dir)
    a, b = np.load(ref), np.load(shard)
    same_ids = bool(np.array_equal(a["image_ids"], b["image_ids"]))
    diff = (
        max(float(np.abs(a[k] - b[k]).max()) for k in ("boxes", "scores"))
        if same_ids
        else None
    )
    out = {
        "tag": tag,
        "clip": clip,
        **info,
        "same_image_ids": same_ids,
        "max_abs_diff": diff,
        "identical": same_ids and diff == 0.0,
    }
    o7.save_json(EVID / f"gpu_recheck_{tag}.json", out)
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["check", "build", "cache", "smoke", "recheck"])
    ap.add_argument("tag", nargs="?")
    ap.add_argument("--stage", choices=["discovery", "confirmation", "test"])
    ap.add_argument("--clips", nargs="*")
    args = ap.parse_args()
    if args.command == "check":
        cmd_check(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "recheck":
        cmd_recheck(args)
    elif args.command == "smoke":
        cmd_smoke(args)
    elif args.command == "cache":
        rec = cache(args.tag, args.stage, clips=args.clips)
        print(json.dumps(rec))


if __name__ == "__main__":
    main()
