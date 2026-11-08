"""CH-02 dataset inspection: ignore semantics, label map, image sizes, target/ignore overlap.

    python results/ch-02-protocol/inspect_dataset.py   # writes results/ch-02-protocol/data_facts.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from movd import data  # noqa: E402

facts = {}
for split in data.DATA_SPLITS:
    coco = data.load_coco(split)
    images = {im["id"]: im for im in coco["images"]}
    crowd = [a for a in coco["annotations"] if a["iscrowd"]]
    targets = [a for a in coco["annotations"] if not a["iscrowd"]]
    regions = defaultdict(set)
    for a in crowd:
        regions[(a["image_id"], tuple(a["bbox"]))].add(a["category_id"])
    index = data.split_annotations(coco)

    # ignore txt files vs COCO crowd regions
    txt_lines = matched = 0
    for image in coco["images"]:
        txt = data.read_ignore_file(data.ignore_path(split, image), image["width"], image["height"])
        txt_lines += len(txt)
        for b in index[image["id"]]["ignore"]:
            matched += any(np.abs(np.subtract(b, g)).sum() < 4 for g in txt)

    # share of each target's area inside an ignore region (largest single region)
    ioa = []
    for entry in index.values():
        if not entry["targets"]:
            continue
        t = np.array([a["bbox"] for a in entry["targets"]], float)
        if not entry["ignore"]:
            ioa += [0.0] * len(t)
            continue
        g = np.array(entry["ignore"], float)
        iw = np.minimum(t[:, None, 0] + t[:, None, 2], g[None, :, 0] + g[None, :, 2]) - np.maximum(t[:, None, 0], g[None, :, 0])
        ih = np.minimum(t[:, None, 1] + t[:, None, 3], g[None, :, 1] + g[None, :, 3]) - np.maximum(t[:, None, 1], g[None, :, 1])
        ioa += (np.clip(iw, 0, None) * np.clip(ih, 0, None)).max(1).__truediv__(t[:, 2] * t[:, 3]).tolist()
    ioa = np.array(ioa)
    frac = [sum(b[2] * b[3] for b in e["ignore"]) / (images[i]["width"] * images[i]["height"])
            for i, e in index.items() if e["ignore"]]
    sizes_per_clip = defaultdict(set)
    for im in coco["images"]:
        sizes_per_clip[im["video"]].add((im["width"], im["height"]))
    facts[split] = {
        "images": len(coco["images"]),
        "clips": len(coco["videos"]),
        "target_annotations": len(targets),
        "crowd_annotations": len(crowd),
        "crowd_flags": sorted({(a["iscrowd"], a.get("ignore")) for a in crowd}),
        "unique_ignore_regions_in_coco": len(regions),
        "categories_per_ignore_region": sorted(Counter(len(v) for v in regions.values()).items()),
        "ignore_region_orig_category": dict(Counter(a["orig_category_id"] for a in crowd if a["category_id"] == 1)),
        "ignore_txt_files": sum(1 for _ in (data.split_dir(split) / "ignore").rglob("*.txt")),
        "ignore_txt_lines": txt_lines,
        "coco_regions_matched_in_txt": matched,
        "images_with_ignore_regions": len(frac),
        "ignore_area_fraction_mean_median_max": [float(np.mean(frac)), float(np.median(frac)), float(np.max(frac))],
        "targets_touching_ignore": int((ioa > 0).sum()),
        "targets_ioa_ge_0.5": int((ioa >= 0.5).sum()),
        "targets_inside_ignore_ioa_ge_0.99": int((ioa >= 0.99).sum()),
        "orig_to_category": {f"{o}->{c}": n for (o, c), n in sorted(Counter((a["orig_category_id"], a["category_id"]) for a in targets).items())},
        "image_sizes": {f"{w}x{h}": n for (w, h), n in Counter((im["width"], im["height"]) for im in coco["images"]).most_common()},
        "clips_with_several_sizes": [c for c, s in sizes_per_clip.items() if len(s) > 1],
        "zero_size_or_out_of_bounds_boxes": sum(1 for a in coco["annotations"] if a["bbox"][2] <= 0 or a["bbox"][3] <= 0
                                                or a["bbox"][0] + a["bbox"][2] > images[a["image_id"]]["width"]
                                                or a["bbox"][1] + a["bbox"][3] > images[a["image_id"]]["height"]),
    }
test = data.load_test_images()
facts["test"] = {
    "images": len(test),
    "clips": len({r["clip"] for r in test}),
    "csv_columns": ["image_id", "file_name", "clip", "frame", "width", "height"],
    "image_id_example": test[0]["image_key"],
    "image_sizes": {f"{w}x{h}": n for (w, h), n in Counter((r["width"], r["height"]) for r in test).most_common()},
}
facts["train_val_clip_overlap"] = sorted(data.clip_overlap(data.load_coco("train"), data.load_coco("val")))
out = Path(__file__).resolve().parent / "data_facts.json"
out.write_text(json.dumps(facts, indent=1) + "\n")
print(json.dumps(facts, indent=1))
