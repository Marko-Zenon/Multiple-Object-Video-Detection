"""OPT-12: frozen constants, clocks, run-state helpers and pure decision rules (no GPU, no network).

Plan: C:/Users/nazar/.claude/plans/pasted-content-id-58d0-act-as-delegated-cosmos.md (student-approved 2026-11-15).
Everything here is deterministic and unit-tested (tests/test_opt12.py). The OPT-12 scripts import it:
scripts/opt12_train.py (training launcher), scripts/opt12_night.py (detached runner), scripts/opt12_morning.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HW2 = REPO.parent
OUT = REPO / "outputs" / "opt12"
EVID = REPO / "results" / "opt-12-regularized-finetune"
SUB = REPO / "submissions"
STATE = OUT / "state.json"
EVENTS = OUT / "logs" / "events.jsonl"
LOCK = OUT / "gpu.lock"

KYIV = timezone(timedelta(hours=2))  # EET, fixed (no DST in November)
DAY = "2026-11-15"


def at(hhmm: str, day: str = DAY) -> datetime:
    return datetime.fromisoformat(f"{day}T{hhmm}:00").replace(tzinfo=KYIV)


# Clocks (Kyiv). The 10:00 hand-off target and the 12:00 hard limit follow the student's 2026-11-15 instruction.
CLOCK = {
    "train_latest_start": at("03:45"),
    "no_new_training": at("05:30"),  # no training start or resume after this
    "train_end_target": at("06:15"),  # E is sized so the last epoch ends about here
    "train_stop": at("06:55"),  # in-trainer stop after a saved epoch that could not finish by this time
    "train_kill": at("07:00"),
    "last_gpu_start": at("11:20"),  # no GPU job after this; a job must also be able to end by gpu_end
    "gpu_end": at("11:50"),
    "no_new_experiment": at("11:30"),
    "interim_handoff": at("11:00"),
    "hard_handoff": at("12:00"),
}

# Protected artifacts (read-only for OPT-12; re-hashed at start and end).
PROTECTED = {
    "c1_csv": (
        HW2 / "MOVD-opt11/submissions/opt11_c1_w075_test.csv",
        "04fcd221137f033af8557c561d6046d2fa3a08fcad8eb566380dfb6f91361e8c",
    ),
    "c2_csv": (
        HW2 / "MOVD-opt11/submissions/opt11_c2_classmix_test.csv",
        "3e25acda0794df5da3b097f43f9a567806109e82f6639cf34075c49c21fc14ab",
    ),
    "p2_csv": (
        HW2 / "MOVD-opt11/submissions/opt11_p2_trajsoup_test.csv",
        "69fd9896ee645de5c7debae4f134bd0f092f35e02888a43479f83bdc8a5a7e2b",
    ),
    "E6_csv": (
        HW2 / "MOVD-opt08/submissions/opt08_e6_combination_k1_test.csv",
        "35a225c905ac56ed86532314046821d10085f73781666ee843fb4c6d28127c35",
    ),
    "E4_csv": (
        HW2 / "MOVD-opt08/submissions/opt08_e4_checkpoint_soup_s1_test.csv",
        "7dc9b318e354d4839b5dffae8f5973dd7da234fdbeb3c5f84057694878852014",
    ),
    "soup_w075": (
        HW2 / "MOVD-opt09/outputs/opt09/soup/soup_w075.pt",
        "cfaaf0a4097c96a972340cfde7aa7a35eba066a64be9c00a3f8f2bbb8ec1b39d",
    ),
    "soup_w080": (
        HW2 / "MOVD-opt08/outputs/opt08/soup/soup_w080.pt",
        "2d60d3fa66d9fd37032bcaaaeb297e98510217b600b5809f73873b36360cd939",
    ),
    "modelA_epoch0": (
        HW2 / "MOVD-ch14/outputs/final_rtdetr/train/weights/epoch0.pt",
        "6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b",
    ),
}
W075 = PROTECTED["soup_w075"]
MODEL_A = PROTECTED["modelA_epoch0"]
MODEL_A_E2 = HW2 / "MOVD-ch14/outputs/final_rtdetr/train/weights/epoch1.pt"

# ---------------------------------------------------------------- frozen decision constants (plan section D / E)

CATASTROPHE_EPOCH = 2  # 1-based epoch after which the catastrophe check runs
CATASTROPHE_DROP = 0.060  # abort if the Ultralytics monitor < R - 0.060
NSTAR_MARGIN = 0.003  # last / best epoch replaces AVG3 only if better by this much
NSTAR_SCREEN = 0.010  # N-star is cached only if within this of w075 (single view, val_fast)
ALPHAS = (0.15, 0.25, 0.5)
ALPHA_TIE = 0.001  # within this, the smaller alpha wins
WSTAR_SCREEN = 0.002  # W-star is cached only if it beats w075 by this much
W070 = 0.70  # fallback soup weight on Model A epoch0.pt (rest on epoch1.pt)

GATES = {
    "C1": {"val_full": 0.0010, "width_le_1920": 0.0, "discovery": 0.0, "confirmation": 0.0, "clips": 5},
    "C2": {"val_full": 0.0005, "width_le_1920": 0.0, "discovery": 0.0, "confirmation": 0.0, "clips": 5},
    "C3": {"val_full": 0.0030, "width_le_1920": 0.0010, "confirmation": 0.0, "clips": 5},
    "C4": {"val_full": 0.0005, "width_le_1920": 0.0, "confirmation": 0.0, "clips": 5},
}
C3_NSTAR_MIN = -0.010  # C3 needs N-star val_full >= c1 - 0.010
C3_GRID = [((1.0, 1.0), 0.5), ((1.0, 1.0), 0.6), ((1.0, 0.75), 0.5), ((1.0, 0.75), 0.6),
           ((0.75, 1.0), 0.5), ((0.75, 1.0), 0.6)]  # (c1 weight, N-star weight), class-wise NMS IoU
TLR_DEMOTE = -0.005  # advisory low-resolution check: demote a candidate below this delta
PUBLIC_C1 = 0.2960489045
PUBLIC_SEQ_THRESHOLD = 0.2930  # c1 public - 0.003


# ---------------------------------------------------------------- small helpers


def now() -> datetime:
    return datetime.now(KYIV)


def stamp(t: datetime | None = None) -> str:
    return (t or now()).strftime("%Y-%m-%d %H:%M:%S")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    """Atomic JSON write (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def event(kind: str, **fields) -> dict:
    """Append one JSON line to the append-only event log (flushed + fsynced)."""
    rec = {"t": stamp(), "kind": kind, **fields}
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(rec, default=str), flush=True)
    return rec


def get_state(path: Path = STATE) -> dict:
    return load_json(path) if Path(path).exists() else {"jobs": {}, "decisions": {}}


def update_state(section: str, key: str, value, path: Path = STATE) -> dict:
    s = get_state(path)
    s.setdefault(section, {})[key] = value
    s["updated"] = stamp()
    save_json(path, s)
    return s


def pid_alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"],
                       capture_output=True, text=True, check=False)
    return f'"{int(pid)}"' in r.stdout


def foreign_gpu_python(own: set[int]) -> list[str]:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
                       capture_output=True, text=True, check=False)
    bad = []
    for line in r.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and "python" in parts[1].lower() and int(parts[0]) not in own:
            bad.append(line)
    return bad


def acquire_lock(job: str, pid: int, lock: Path = LOCK, alive=pid_alive) -> bool:
    """Create the GPU lock atomically. A lock whose holder pid is dead is reclaimed. False if a live holder exists."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                info = load_json(lock)
            except (OSError, ValueError):
                info = {"pid": -1}
            if int(info.get("pid", -1)) > 0 and alive(int(info["pid"])):
                return False
            lock.unlink(missing_ok=True)
            continue
        os.write(fd, json.dumps({"pid": pid, "job": job, "started": stamp()}).encode())
        os.close(fd)
        return True
    return False


def set_lock_pid(pid: int, job: str, lock: Path = LOCK) -> None:
    save_json(lock, {"pid": pid, "job": job, "started": stamp()})


def release_lock(lock: Path = LOCK) -> None:
    lock.unlink(missing_ok=True)


def keep_awake() -> None:
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except (AttributeError, OSError):
        pass


# ---------------------------------------------------------------- pure decision rules


def choose_epochs(start: datetime, epoch_seconds: float, end_target: datetime | None = None,
                  lo: int = 4, hi: int = 10) -> int:
    """E = clamp(floor((end_target - start) / epoch_seconds), lo, hi); 0 if even lo epochs cannot fit before train_stop."""
    end_target = end_target or CLOCK["train_end_target"]
    n = int((end_target - start).total_seconds() // epoch_seconds)
    if start + timedelta(seconds=lo * epoch_seconds) > CLOCK["train_stop"]:
        return 0
    return max(lo, min(hi, n))


def close_mosaic_for(epochs: int) -> int:
    return 2 if epochs >= 6 else 1


def catastrophe(epoch_1based: int, monitor: float, reference: float) -> bool:
    return epoch_1based == CATASTROPHE_EPOCH and monitor < reference - CATASTROPHE_DROP


def clock_stop(t_now: datetime, last_epoch_seconds: float, stop: datetime | None = None) -> bool:
    """True if another epoch started now could not finish by the in-trainer stop time."""
    stop = stop or CLOCK["train_stop"]
    return t_now + timedelta(seconds=1.05 * last_epoch_seconds) > stop


def choose_nstar(scores: dict[str, float]) -> str:
    """scores: {'avg3', 'last', 'best'} -> single-view val_fast mAP. AVG3 unless last/best is better by NSTAR_MARGIN."""
    best_alt = max((k for k in ("last", "best") if k in scores), key=lambda k: scores[k], default=None)
    if "avg3" not in scores:
        return best_alt
    if best_alt is not None and scores[best_alt] >= scores["avg3"] + NSTAR_MARGIN:
        return best_alt
    return "avg3"


def choose_alpha(scores: dict[float, float]) -> float:
    """Best alpha; within ALPHA_TIE of the best, the smallest alpha wins (closest to c1)."""
    top = max(scores.values())
    return min(a for a, v in scores.items() if v >= top - ALPHA_TIE)


def nstar_screen(nstar: float, w075: float) -> bool:
    return nstar >= w075 - NSTAR_SCREEN


def wstar_screen(wstar: float, w075: float) -> bool:
    return wstar >= w075 + WSTAR_SCREEN


def gate(name: str, deltas: dict[str, float], clips_non_negative: int) -> dict:
    """Apply a frozen candidate gate. deltas: candidate minus c1 on val_full / width_le_1920 / discovery / confirmation."""
    g = GATES[name]
    checks = {k: deltas.get(k, float("-inf")) >= v for k, v in g.items() if k != "clips"}
    checks["clips"] = clips_non_negative >= g["clips"]
    return {"gate": name, "thresholds": g, "deltas": deltas, "clips_non_negative": clips_non_negative,
            "checks": checks, "passed": all(checks.values())}


def upload_order(passed: dict[str, bool], alpha_star: float | None, tlr: dict[str, float] | None = None) -> list[str]:
    """Recommended manual upload order (plan section D, sequential test). The C1 public result decides C3 vs C2 later;
    this is the order before any public score is known, with the conditional rule written next to it in the hand-off."""
    order = []
    if passed.get("C1"):
        order.append("C1")
        order += [c for c in ("C3", "C2") if passed.get(c)]
    else:
        order += [c for c in ("C2", "C3") if passed.get(c)]
    if passed.get("C4"):
        order.append("C4")
    if tlr:
        demoted = [c for c in order if tlr.get(c, 0.0) < TLR_DEMOTE]
        order = [c for c in order if c not in demoted] + demoted
    return order
