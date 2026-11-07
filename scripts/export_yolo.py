"""Export train_rubric (masked, CH-02 policy) and val_fast (unmodified) to YOLO format, then check the export.

CPU only. Writes the data to the git-ignored `outputs/yolo_data/` and the small evidence to
`results/ch-09-yolo/` (export_check.json, figures/export_check.png).

    python scripts/export_yolo.py --dataset-root <dataset>
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, ultra  # noqa: E402

RESULTS = data.REPO_ROOT / "results" / "ch-09-yolo"


def figure(out_root: Path, root, path: Path, count: int = 2) -> list[int]:
    """Original (ignore regions red, targets green) next to the exported frame, for the frames with the most fill."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    coco = data.load_coco("train", root)
    images = {img["id"]: img for img in coco["images"]}
    entries = data.split_annotations(coco)
    ids = data.split_image_ids("train_rubric")

    def filled(image_id):
        image, entry = images[image_id], entries[image_id]
        return data.ignore_mask(
            image["height"],
            image["width"],
            entry["ignore"],
            [a["bbox"] for a in entry["targets"]],
        ).sum()

    # Two frames from different clips with the largest filled area.
    chosen, clips = [], set()
    for image_id in sorted(
        (i for i in ids if entries[i]["ignore"]), key=filled, reverse=True
    ):
        if images[image_id]["video"] not in clips:
            chosen.append(image_id)
            clips.add(images[image_id]["video"])
        if len(chosen) == count:
            break

    fig, axes = plt.subplots(len(chosen), 2, figsize=(16, 4.8 * len(chosen)))
    for row, image_id in zip(axes, chosen):
        image, entry = images[image_id], entries[image_id]
        original = cv2.cvtColor(
            cv2.imread(str(data.image_path("train", image, root))), cv2.COLOR_BGR2RGB
        )
        relative = (
            Path(Path(image["file_name"]).parent.name) / Path(image["file_name"]).name
        )
        exported = cv2.cvtColor(
            cv2.imread(str(out_root / "train_rubric" / "images" / relative)),
            cv2.COLOR_BGR2RGB,
        )
        row[0].imshow(original)
        for x, y, w, h in entry["ignore"]:
            row[0].add_patch(Rectangle((x, y), w, h, fill=False, ec="red", lw=1.5))
        for ann in entry["targets"]:
            x, y, w, h = ann["bbox"]
            row[0].add_patch(Rectangle((x, y), w, h, fill=False, ec="lime", lw=0.6))
        row[0].set_title(f"original {image['file_name']} — red: ignore, green: targets")
        row[1].imshow(exported)
        row[1].set_title("exported training image (ignore filled, targets kept)")
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ultra.DEFAULT_EXPORT_ROOT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip writing, only check an existing export",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    splits = [
        ("train_rubric", "train", data.split_image_ids("train_rubric"), True),
        ("val_fast", "val", data.split_image_ids("val_fast"), False),
    ]
    manifests = {}
    if not args.check_only:
        for name, source, ids, mask in splits:
            manifests[name] = ultra.export_split(
                name, source, ids, args.out, args.dataset_root, mask=mask
            )
        ultra.write_data_yaml(args.out, "train_rubric", "val_fast")
    else:  # keep the manifests written by the export run
        manifests = {
            name: json.loads(
                (args.out / f"{name}_manifest.json").read_text(encoding="utf-8")
            )
            for name, *_ in splits
        }
    export_seconds = time.perf_counter() - start

    checks = {
        name: ultra.check_export(
            name, source, ids, args.out, args.dataset_root, masked=mask
        )
        for name, source, ids, mask in splits
    }
    chosen = figure(
        args.out, args.dataset_root, RESULTS / "figures" / "export_check.png"
    )

    report = {
        "export_root": "outputs/yolo_data (git-ignored)",
        "data_yaml": "outputs/yolo_data/data.yaml",
        "policy": "configs/protocol.json ignore_policy.training (CH-02): crowd never a target; ignore regions filled with "
        f"{list(data.IGNORE_FILL_RGB)} except inside labelled target boxes; masked frames re-encoded JPEG q{ultra.JPEG_QUALITY}; "
        "frames without ignore regions byte-copied",
        "val_note": "val_fast is an unmodified byte copy, only for the Ultralytics trainer's internal loader; reported numbers use the original val images and movd.evaluate",
        "manifests": manifests or None,
        "checks": checks,
        "disk_bytes": {
            name: ultra.directory_bytes(args.out / name) for name, *_ in splits
        },
        "disk_bytes_total": ultra.directory_bytes(args.out),
        "figure": "results/ch-09-yolo/figures/export_check.png",
        "figure_image_ids": chosen,
        "seconds": round(time.perf_counter() - start, 1),
        "export_seconds": round(export_seconds, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "export_check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for name, check in checks.items():
        print(
            f"[OK] {name}: {check['num_images']} images, {check['num_labels']} labels, classes {check['class_ids_found']}, "
            f"max box error {check['max_label_box_error_px']:.4f} px, masked frames {check['masked_frames']}, "
            f"max fill MAE {check['pixel_check_max_fill_mae']}"
        )
    print(
        f"disk: {report['disk_bytes_total'] / 1e9:.2f} GB total {report['disk_bytes']}"
    )


if __name__ == "__main__":
    main()
