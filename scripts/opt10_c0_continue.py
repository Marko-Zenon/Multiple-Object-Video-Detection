"""OPT-10 chunk 10.3c: separate accounting for the existing c0 confirmation.

Only verification and confirmation caching live here. S2, S3 and determinism use
opt10_loop.py unchanged. Historical Phase 0 evidence and accounting are read only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import opt10_loop as loop
import opt10_soup as soup

EVID = REPO / "results/opt-10-soup-bn/continuation-c0-eval"
OUT = REPO / "outputs/opt10/continuation-c0-eval"
HISTORICAL = (
    "selection_rule.md",
    "bn_frames.json",
    "d2_frames.json",
    "identity.json",
    "gpu_log.json",
    "session_decision.json",
    "smoke.json",
    "soup_identity.json",
    "diagnostics/d1.json",
    "diagnostics/d2.json",
    "c0/build.json",
    "c0/build_FAILED.json",
    "c0/variance_audit.json",
    "c0/cache_discovery.json",
    "c0/screen.json",
)


def artifact_identity() -> dict:
    contract = loop.load_json(EVID / "contract.json")
    files = [soup.EVID / name for name in HISTORICAL]
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            contract["base_commit"],
            "--",
            *[str(p.relative_to(REPO)) for p in files],
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    paths = {str(p.relative_to(REPO)): loop.sha256(p) for p in files}
    checkpoint = soup.weights_path("c0")
    if loop.sha256(checkpoint) != contract["checkpoint_sha256"]:
        raise SystemExit("c0 differs from the authorized checkpoint")
    cache = loop.load_json(soup.EVID / "c0/cache_discovery.json")
    for name, expected in cache["shards_sha256"].items():
        shard = soup.raw_dir("c0", "discovery") / name
        if loop.sha256(shard) != expected:
            raise SystemExit(f"historical discovery shard differs: {shard}")
        paths[loop.rel(shard)] = expected
    screen = loop.load_json(soup.EVID / "c0/screen.json")
    predictions = soup.OUT / "c0/discovery.json"
    if loop.sha256(predictions) != screen["discovery_predictions_sha256"]:
        raise SystemExit("historical c0 discovery predictions differ")
    paths[loop.rel(predictions)] = screen["discovery_predictions_sha256"]
    paths[loop.rel(checkpoint)] = contract["checkpoint_sha256"]
    return paths


def verify() -> None:
    paths = artifact_identity()
    target = EVID / "artifact_identity.json"
    runs = loop.load_json(target)["runs"] if target.exists() else []
    if runs and paths != runs[0]["sha256"]:
        raise SystemExit("historical artifact bytes changed during continuation")
    loop.cmd_verify(SimpleNamespace(evidence_dir=EVID, output_dir=OUT))
    loop.save_json(
        target,
        {
            "runs": [
                *runs,
                {
                    "checked_at_kyiv": soup.o7.kyiv_now(),
                    "sha256": paths,
                    "ok": True,
                },
            ]
        },
    )
    print("c0, historical evidence, discovery caches and protected identity PASS")


@contextlib.contextmanager
def gpu_job():
    contract = loop.load_json(EVID / "contract.json")
    path = EVID / "gpu_log.json"
    jobs = loop.load_json(path)["jobs"] if path.exists() else []
    used = sum(j["seconds"] for j in jobs) / 60
    cap = contract["gpu_budget_minutes"]
    estimate = contract["gpu_job_estimate_minutes"]
    if used + estimate > cap:
        raise SystemExit(f"continuation budget: {used:.2f} + {estimate} > {cap} min")
    start, started = time.perf_counter(), soup.o7.kyiv_now()
    status = "failed"
    try:
        yield
        status = "ok"
    finally:
        jobs.append(
            {
                "job": contract["gpu_job"],
                "seconds": round(time.perf_counter() - start, 1),
                "started_kyiv": started,
                "ended_kyiv": soup.o7.kyiv_now(),
                "status": status,
            }
        )
        loop.save_json(
            path,
            {
                "chunk": contract["chunk"],
                "budget_minutes": cap,
                "used_minutes": sum(j["seconds"] for j in jobs) / 60,
                "contract_sha256": loop.sha256(EVID / "contract.json"),
                "jobs": jobs,
            },
        )


def cache() -> None:
    contract = EVID / "contract.json"
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(contract.relative_to(REPO))],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", str(contract.relative_to(REPO))],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    if not loop.load_json(EVID / "artifact_identity.json")["runs"][-1]["ok"]:
        raise SystemExit("continuation identity verification required")
    artifact_identity()
    loop.need("c0", "screen.json", "passes_S1")
    if (soup.EVID / "c0/confirm.json").exists():
        raise SystemExit("S2 already evaluated: do not rerun confirmation")
    if (soup.EVID / "c0/cache_confirmation.json").exists():
        raise SystemExit("confirmation cache already complete")
    before = subprocess.check_output(["nvidia-smi"], text=True)
    (EVID / "nvidia_before_confirmation.txt").write_text(
        "\n".join(line.rstrip() for line in before.splitlines()) + "\n",
        encoding="utf-8",
    )
    with gpu_job():
        record = soup.cache("c0", "confirmation")
    shards = {
        p.name: loop.sha256(p)
        for p in sorted(soup.raw_dir("c0", "confirmation").glob("*.npz"))
    }
    loop.save_json(
        soup.EVID / "c0/cache_confirmation.json",
        {
            "candidate": "c0",
            "stage": "confirmation",
            "continuation_chunk": "10.3c",
            "checkpoint_sha256": loop.sha256(soup.weights_path("c0")),
            **record,
            "shards_sha256": shards,
            "cached_at_kyiv": soup.o7.kyiv_now(),
        },
    )
    print(json.dumps(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "cache"))
    args = parser.parse_args()
    {"verify": verify, "cache": cache}[args.command]()


if __name__ == "__main__":
    main()
