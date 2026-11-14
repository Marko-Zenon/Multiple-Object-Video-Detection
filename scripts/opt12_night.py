"""OPT-12 detached runner: supervises the training child, then hands over to scripts/opt12_morning.py.

Deterministic, resumable, no agent loop and no network code. Started outside the Claude session's process tree
(Win32_Process.Create), so it survives the session. State: outputs/opt12/state.json (atomic). Events:
outputs/opt12/logs/events.jsonl (append-only). GPU mutex: outputs/opt12/gpu.lock (one GPU child at a time).

    python scripts/opt12_night.py run

Stages, decided from artifacts on every (re)start:
  train    - adopt a live trainer by pid, else start it (latest 03:45) or resume it once from a checkpoint that still
             holds an optimizer (never after 05:30); kill at 07:00. The trainer stops itself after a saved epoch when
             the next epoch could not finish by 06:55 (scripts/opt12_train.py --stop-at).
  morning  - run scripts/opt12_morning.py run as a child (selection, caches, gates, CSVs, validators, hand-off). It
             takes the GPU lock per job. If the file does not exist yet, wait for it until 07:30.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import opt12  # noqa: E402
from movd.opt12 import CLOCK, event, get_state, now, update_state  # noqa: E402

PY = sys.executable
RUN_ID = "opt12_reg_ft"
CONFIG = "configs/runs/opt12_reg_ft.json"
TRAIN_OUT = opt12.REPO / "outputs" / RUN_ID
TRAIN_RES = opt12.REPO / "results" / RUN_ID
MAX_TRAIN_LAUNCHES = 2  # first start + one resume


def train_status() -> str | None:
    """Status of the latest training process from observations.json, or None if it never finished cleanly."""
    p = TRAIN_RES / "observations.json"
    if not p.exists():
        return None
    procs = opt12.load_json(p).get("processes", [])
    return procs[-1].get("status") if procs else None


def finished(status: str | None) -> bool:
    return bool(status) and (status == "completed" or status.startswith(("clock_stop", "catastrophe")))


def run_child(job: str, args: list[str], kill_at, log_name: str) -> int:
    """One GPU child under the lock; kill the process tree at kill_at. Returns the exit code (-9 if killed)."""
    while not opt12.acquire_lock(job, os.getpid()):
        info = opt12.load_json(opt12.LOCK)
        event("lock_wait", job=job, holder=info)
        time.sleep(30)
        if now() >= kill_at:
            return -8
    bad = opt12.foreign_gpu_python({os.getpid()})
    if bad:
        event("refused_gpu_busy", job=job, foreign=bad)
        opt12.release_lock()
        return -7
    out = (opt12.OUT / "logs" / log_name).open("a", encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen([PY, *args], cwd=opt12.REPO, stdout=out, stderr=subprocess.STDOUT, env=env)
    opt12.set_lock_pid(p.pid, job)
    update_state("jobs", job, {"status": "running", "pid": p.pid, "started": opt12.stamp(), "args": args})
    event("child_start", job=job, pid=p.pid, args=args)
    rc = wait_or_kill(p.pid, kill_at, popen=p)
    opt12.release_lock()
    update_state("jobs", job, {"status": "ended", "pid": p.pid, "rc": rc, "ended": opt12.stamp(), "args": args})
    event("child_end", job=job, pid=p.pid, rc=rc)
    return rc


def wait_or_kill(pid: int, kill_at, popen=None) -> int:
    while (popen.poll() is None) if popen else opt12.pid_alive(pid):
        if now() >= kill_at:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
            event("killed_at_cutoff", pid=pid, cutoff=opt12.stamp(kill_at))
            if popen:
                popen.wait()
            return -9
        update_state("runner", "heartbeat", opt12.stamp())
        time.sleep(20)
    return popen.returncode if popen else 0


def stage_train() -> str:
    s = get_state()
    job = s.get("jobs", {}).get("train", {})
    # Adopt a live trainer (runner restart).
    if job.get("status") == "running" and job.get("pid") and opt12.pid_alive(int(job["pid"])):
        event("adopt_trainer", pid=job["pid"])
        if not opt12.LOCK.exists():
            opt12.set_lock_pid(int(job["pid"]), "train")
        wait_or_kill(int(job["pid"]), CLOCK["train_kill"])
        opt12.release_lock()
    launches = int(s.get("decisions", {}).get("train_launches", 0))
    reference = s.get("decisions", {}).get("reference_R")
    stop_at = opt12.stamp(CLOCK["train_stop"])[:16]
    while not finished(train_status()):
        if launches >= MAX_TRAIN_LAUNCHES:
            event("train_give_up", launches=launches, status=train_status())
            return "failed"
        resume = TRAIN_OUT.exists()
        if not resume and now() > CLOCK["train_latest_start"]:
            event("train_skipped_late")
            return "skipped_late"
        if resume and now() > CLOCK["no_new_training"]:
            event("train_no_resume_after_0530", status=train_status())
            return "partial"
        args = ["scripts/opt12_train.py", "--config", CONFIG, "--device", "0", "--stop-at", stop_at]
        if reference is not None:
            args += ["--reference", str(reference)]
        if resume:
            args.append("--resume")
        launches += 1
        update_state("decisions", "train_launches", launches)
        rc = run_child("train", args, CLOCK["train_kill"], "train.out")
        if rc in (-9, -8):
            return "killed"
        if rc == 3:
            return "catastrophe"
        if rc == -7:
            time.sleep(60)
    st = train_status()
    return "catastrophe" if st and st.startswith("catastrophe") else "done"


def cmd_run() -> None:
    opt12.keep_awake()
    update_state("runner", "pid", os.getpid())
    update_state("runner", "started", opt12.stamp())
    event("runner_start", pid=os.getpid(), ppid=os.getppid())
    s = get_state()
    if "train_result" not in s.get("decisions", {}):
        result = stage_train()
        update_state("decisions", "train_result", result)
        event("train_stage_done", result=result, status=train_status())
    morning = opt12.REPO / "scripts" / "opt12_morning.py"
    deadline = opt12.at("07:30")
    while not morning.exists() and now() < deadline:
        update_state("runner", "heartbeat", opt12.stamp())
        time.sleep(60)
    if not morning.exists():
        event("morning_script_missing")
        return
    out = (opt12.OUT / "logs" / "morning.out").open("a", encoding="utf-8")
    p = subprocess.Popen([PY, str(morning), "run"], cwd=opt12.REPO, stdout=out, stderr=subprocess.STDOUT,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    update_state("jobs", "morning", {"status": "running", "pid": p.pid, "started": opt12.stamp()})
    event("morning_start", pid=p.pid)
    rc = wait_or_kill(p.pid, CLOCK["hard_handoff"], popen=p)
    update_state("jobs", "morning", {"status": "ended", "pid": p.pid, "rc": rc, "ended": opt12.stamp()})
    event("runner_done", morning_rc=rc)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "run":
        raise SystemExit("usage: opt12_night.py run")
    cmd_run()
