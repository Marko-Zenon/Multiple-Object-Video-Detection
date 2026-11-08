"""Build the CH-09 transfer inventory after full training/evaluation; never uploads files."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = "yolo_baseline_marko_3050"
EXCLUDE = {"artifact_manifest.json", "artifact_manifest.sha256", "transfer_files.txt"}


def main():
    results = ROOT / "results" / RUN
    output = ROOT / "outputs" / RUN
    curve = json.loads((results / "curve.json").read_text())
    runtime = json.loads((results / "runtime.json").read_text())
    assert [row["epoch"] for row in curve] == list(range(1, 7))
    assert sum(p["batches"] for p in runtime["training_processes"]) == 16800
    assert runtime["training_processes"][-1]["status"] == "completed"
    for i in range(6):
        assert (output / "train/weights" / f"epoch{i}.pt").is_file()
        assert (output / "val_fast_predictions" / f"epoch{i}.json").is_file()
    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["split"] == "val_full" and metrics["num_images"] == 2846
    for name in [
        "speed.json",
        "per_class.csv",
        "failure_cases.csv",
        "frcnn_comparison.csv",
    ]:
        assert (results / name).is_file()
    files = set()
    for folder in [
        f"outputs/{RUN}",
        f"results/{RUN}",
        "outputs/yolo_smoke_marko_3050",
        "results/yolo_smoke_marko_3050",
        "results/ch-09-yolo",
    ]:
        files.update(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and p.name not in EXCLUDE
        )
    for folder in [
        "outputs/ch09_marko_baseline_20261108",
        "outputs/ch09_marko_setup_20261108",
    ]:
        files.update(
            p
            for p in (ROOT / folder).iterdir()
            if p.is_file() and p.name not in EXCLUDE
        )
    for name in [
        "outputs/weights/yolo11s.pt",
        "configs/runs/yolo_baseline_marko_3050.json",
        "configs/runs/yolo_smoke_marko_3050.json",
        "configs/protocol.json",
        "movd/ultra.py",
        "scripts/train_ultra.py",
        "scripts/export_yolo.py",
        "scripts/predict_ultra.py",
        "scripts/failures_ultra.py",
    ]:
        files.add(ROOT / name)
    rows = []
    for path in sorted(files):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    inventory = {
        "run_id": RUN,
        "count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "transfer_status": "NOT TRANSFERRED; destination/permission and recipient verification pending",
        "note": "Relative to repository root. Manifest/list files excluded. Historical smoke manifests retain their original code hashes; use this inventory for current transfer integrity.",
        "files": rows,
    }
    (results / "artifact_manifest.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    (results / "artifact_manifest.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    )
    (results / "transfer_files.txt").write_text(
        "".join(row["path"] + "\n" for row in rows)
    )
    print(
        f"Inventoried {len(rows)} files, {inventory['total_bytes']} bytes; no upload performed."
    )


if __name__ == "__main__":
    main()
