"""Create (or verify) the frozen frame splits in configs/splits/.

    python scripts/make_splits.py           # write train_rubric.json, val_fast.json, val_full.json
    python scripts/make_splits.py --check   # regenerate in memory, compare SHA-256 with the committed files

The rules come from configs/protocol.json (CH-02). The splits are deterministic: they depend only on the
COCO files and the protocol, not on randomness. Seed 42 is recorded for completeness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data  # noqa: E402


def build_split(name: str, source: str, rule: str, image_ids: list[int], coco: dict) -> dict:
    images = {im["id"]: im for im in coco["images"]}
    chosen = [images[i] for i in image_ids]
    return {
        "name": name,
        "source_split": source,
        "rule": rule,
        "seed": data.SEED,
        "num_images": len(chosen),
        "num_clips": len({data.clip_name(im) for im in chosen}),
        "frames_per_clip": dict(sorted(Counter(data.clip_name(im) for im in chosen).items())),
        "image_ids": image_ids,
        "file_names": [im["file_name"] for im in chosen],
    }


def build_all(root: Path | None) -> dict[str, dict]:
    protocol = data.load_protocol()["splits"]
    train = data.load_coco("train", root)
    val = data.load_coco("val", root)
    overlap = data.clip_overlap(train, val)
    if overlap:
        raise SystemExit(f"train/val clip overlap: {sorted(overlap)}")

    per_clip = protocol["train_rubric"]["frames_per_clip"]
    k = protocol["val_fast"]["every_kth_frame"]
    return {
        "train_rubric": build_split(
            "train_rubric", "train", f"np.linspace: {per_clip} evenly spaced frames per clip",
            data.evenly_spaced_per_clip(train["images"], per_clip), train),
        "val_fast": build_split(
            "val_fast", "val", f"every {k}th frame of every clip, from the first frame",
            data.every_kth_per_clip(val["images"], k), val),
        "val_full": build_split(
            "val_full", "val", "all val frames",
            data.every_kth_per_clip(val["images"], 1), val),
    }


def serialise(split: dict) -> bytes:
    return (json.dumps(split, indent=1) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None, help="dataset root (default: repo dataset folder)")
    parser.add_argument("--check", action="store_true", help="verify the committed files instead of writing")
    args = parser.parse_args()

    ok = True
    for name, split in build_all(args.root).items():
        blob = serialise(split)
        digest = hashlib.sha256(blob).hexdigest()
        path = data.SPLITS_DIR / f"{name}.json"
        if args.check:
            # Hash the canonical form of the committed file, so Git line-ending conversion does not matter.
            committed = "missing"
            if path.exists():
                committed = hashlib.sha256(serialise(json.loads(path.read_text(encoding="utf-8")))).hexdigest()
            same = committed == digest
            ok &= same
            print(f"[{'OK' if same else 'FAIL'}] {name}: {split['num_images']} images, "
                  f"{split['num_clips']} clips, sha256 {digest[:16]} (committed {committed[:16]})")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            print(f"wrote {path.relative_to(data.REPO_ROOT)}: {split['num_images']} images, "
                  f"{split['num_clips']} clips, sha256 {digest}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
