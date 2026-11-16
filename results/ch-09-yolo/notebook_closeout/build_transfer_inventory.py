"""Create a supplemental inventory without modifying the original 145-file manifest."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent
EXCLUDED = {"transfer_manifest.json", "transfer_manifest.sha256", "transfer_files.txt"}


def main():
    assert not any((FOLDER / name).exists() for name in EXCLUDED), (
        "Do not overwrite an existing transfer inventory"
    )
    base_path = ROOT / "results/yolo_baseline_marko_3050/artifact_manifest.json"
    original = json.loads(base_path.read_text())
    paths = {ROOT / row["path"] for row in original["files"]}
    for row in original["files"]:
        path = ROOT / row["path"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
        assert path.stat().st_size == row["bytes"]
    paths.update(
        path
        for path in FOLDER.iterdir()
        if path.is_file() and path.name not in EXCLUDED
    )
    for name in [
        "artifact_manifest.json",
        "artifact_manifest.sha256",
        "transfer_files.txt",
    ]:
        paths.add(base_path.parent / name)
    paths.update((ROOT / "configs/runs").glob("yolo_*.json"))
    paths.update(
        ROOT / name
        for name in [
            "homework.ipynb",
            "README.md",
            "requirements.txt",
            "scripts/check_ultra_adapter.py",
            "tests/test_ultra.py",
        ]
    )
    required = {
        ROOT / "outputs/yolo_baseline_marko_3050/train/weights" / f"epoch{i}.pt"
        for i in range(6)
    }
    required.update(
        ROOT / "outputs/yolo_baseline_marko_3050/train/weights" / name
        for name in ["last.pt", "best.pt"]
    )
    required.update(
        ROOT
        / "outputs/yolo_baseline_marko_3050/val_fast_predictions"
        / f"epoch{i}.json"
        for i in range(6)
    )
    required.add(ROOT / "outputs/yolo_baseline_marko_3050/val_full_predictions.json")
    assert required <= paths
    rows = []
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix()
        assert not relative.startswith(
            (
                ".git/",
                ".venv/",
                "multiple-object-video-detection-hw-2-ucu-ai-2026/",
                "outputs/yolo_data/",
            )
        )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "status": "PREPARED; NOT TRANSFERRED",
        "recipient_verification": "PENDING",
        "original_manifest": str(base_path.relative_to(ROOT)),
        "original_entries_reverified": original["count"],
        "notebook_commit": "7150f8228156bdab1a3672da3dc9080e7c347dcb",
        "count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "note": "Paths relative to repository root. These three inventory files are excluded from their own payload; copy them separately.",
    }
    (FOLDER / "transfer_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (FOLDER / "transfer_manifest.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    )
    (FOLDER / "transfer_files.txt").write_text(
        "".join(row["path"] + "\n" for row in rows)
    )
    print(
        f"Prepared {len(rows)} files, {manifest['total_bytes']} bytes; no transfer initiated"
    )


if __name__ == "__main__":
    main()
