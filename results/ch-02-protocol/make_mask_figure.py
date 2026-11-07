"""CH-02 visual check of the training-side ignore policy on 2 frames (one val, one train).

    python results/ch-02-protocol/make_mask_figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from movd import data  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures" / "ignore_masking_check.png"


def pick(split):
    """Frame with the largest ignore area that also has a target overlapping an ignore region."""
    coco = data.load_coco(split)
    index = data.split_annotations(coco)
    best, best_area = None, -1.0
    for image in coco["images"][::7]:
        entry = index[image["id"]]
        area = sum(b[2] * b[3] for b in entry["ignore"]) / (image["width"] * image["height"])
        overlaps = any(
            t["bbox"][0] < g[0] + g[2] and g[0] < t["bbox"][0] + t["bbox"][2]
            and t["bbox"][1] < g[1] + g[3] and g[1] < t["bbox"][1] + t["bbox"][3]
            for t in entry["targets"] for g in entry["ignore"])
        if overlaps and area > best_area:
            best, best_area = image, area
    return best, index[best["id"]], best_area


fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
for row, split in enumerate(["val", "train"]):
    image, entry, area = pick(split)
    masked, _, _ = data.training_sample(split, image, entry)
    import cv2
    original = cv2.cvtColor(cv2.imread(str(data.image_path(split, image))), cv2.COLOR_BGR2RGB)
    for col, (img, title) in enumerate([(original, "original + boxes"), (masked, "training input (masked)")]):
        ax = axes[row, col]
        ax.imshow(img)
        for x, y, w, h in entry["ignore"]:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="red", lw=1.5))
        if col == 0:
            for t in entry["targets"]:
                x, y, w, h = t["bbox"]
                ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="lime", lw=0.6))
        ax.set_title(f"{split}: {image['file_name']} — {title}\n"
                     f"{len(entry['ignore'])} ignore regions ({area:.1%} of frame), {len(entry['targets'])} targets",
                     fontsize=9)
        ax.axis("off")
fig.suptitle("CH-02 ignore policy: red = ignore regions (filled with ImageNet mean), green = targets (pixels kept)")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=110)
print("wrote", OUT)
