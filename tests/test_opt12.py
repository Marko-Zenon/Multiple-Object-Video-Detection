"""OPT-12 unit tests: frozen decision rules, clocks, state/lock helpers, BN-eval callback and the no-network rule."""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

import pytest

from movd import opt12
from movd.opt12 import at

REPO = Path(__file__).resolve().parents[1]


def test_choose_epochs_clamps_and_refuses_late_start():
    start = at("01:40")
    assert opt12.choose_epochs(start, 1500.0, end_target=at("06:15")) == 10  # 16,500 s / 1,500 s = 11 -> 10
    assert opt12.choose_epochs(start, 2400.0, end_target=at("06:15")) == 6
    assert opt12.choose_epochs(at("04:40"), 1800.0, end_target=at("06:15")) == 4  # clamped up to 4, fits 06:55
    assert opt12.choose_epochs(at("05:30"), 1800.0, end_target=at("06:15")) == 0  # 4 epochs end after 06:55


def test_close_mosaic_and_catastrophe():
    assert opt12.close_mosaic_for(8) == 2 and opt12.close_mosaic_for(5) == 1
    assert opt12.catastrophe(2, 0.20, 0.27) and not opt12.catastrophe(2, 0.22, 0.27)
    assert not opt12.catastrophe(1, 0.0, 0.27) and not opt12.catastrophe(3, 0.0, 0.27)


def test_clock_stop():
    assert opt12.clock_stop(at("06:30"), 1800.0)
    assert not opt12.clock_stop(at("05:00"), 1800.0)


def test_choose_nstar():
    assert opt12.choose_nstar({"avg3": 0.280, "last": 0.282, "best": 0.2825}) == "avg3"
    assert opt12.choose_nstar({"avg3": 0.280, "last": 0.284, "best": 0.284}) in ("last", "best")
    assert opt12.choose_nstar({"avg3": 0.280, "last": 0.270, "best": 0.2835}) == "best"


def test_choose_alpha_prefers_small_within_tie():
    assert opt12.choose_alpha({0.15: 0.2800, 0.25: 0.2805, 0.5: 0.2809}) == 0.15
    assert opt12.choose_alpha({0.15: 0.2800, 0.25: 0.2815, 0.5: 0.2830}) == 0.5
    assert opt12.choose_alpha({0.15: 0.2800, 0.25: 0.2830, 0.5: 0.2835}) == 0.25


def test_screens():
    assert opt12.nstar_screen(0.271, 0.280) and not opt12.nstar_screen(0.2699, 0.280)
    assert opt12.wstar_screen(0.2821, 0.280) and not opt12.wstar_screen(0.2819, 0.280)


def test_gate_all_checks_required():
    ok = opt12.gate("C1", {"val_full": 0.002, "width_le_1920": 0.0, "discovery": 0.001, "confirmation": 0.0}, 5)
    assert ok["passed"]
    bad_conf = opt12.gate("C1", {"val_full": 0.004, "width_le_1920": 0.002, "discovery": 0.01, "confirmation": -0.0001}, 7)
    assert not bad_conf["passed"] and not bad_conf["checks"]["confirmation"]
    bad_clips = opt12.gate("C2", {"val_full": 0.001, "width_le_1920": 0.0, "discovery": 0.0, "confirmation": 0.0}, 4)
    assert not bad_clips["passed"]
    c3 = opt12.gate("C3", {"val_full": 0.0031, "width_le_1920": 0.0009, "confirmation": 0.0}, 6)
    assert not c3["passed"] and not c3["checks"]["width_le_1920"]


def test_upload_order():
    assert opt12.upload_order({"C1": True, "C2": True, "C3": True}, 0.25) == ["C1", "C3", "C2"]
    assert opt12.upload_order({"C1": False, "C2": True, "C3": True, "C4": True}, 0.15) == ["C2", "C3", "C4"]
    assert opt12.upload_order({"C1": True, "C3": True}, None, tlr={"C1": -0.01}) == ["C3", "C1"]


def test_state_and_events_are_atomic_and_appendable(tmp_path, monkeypatch):
    monkeypatch.setattr(opt12, "EVENTS", tmp_path / "logs" / "events.jsonl")
    p = tmp_path / "state.json"
    opt12.update_state("jobs", "train", {"status": "running"}, path=p)
    opt12.update_state("decisions", "E", 8, path=p)
    s = opt12.get_state(p)
    assert s["jobs"]["train"]["status"] == "running" and s["decisions"]["E"] == 8
    assert not (tmp_path / "state.json.tmp").exists()
    opt12.event("a", x=1)
    opt12.event("b", y=2)
    lines = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["a", "b"]


def test_lock_reclaims_dead_holder(tmp_path):
    lock = tmp_path / "gpu.lock"
    assert opt12.acquire_lock("j1", 111, lock=lock, alive=lambda pid: True)
    assert not opt12.acquire_lock("j2", 222, lock=lock, alive=lambda pid: True)  # live holder
    assert opt12.acquire_lock("j2", 222, lock=lock, alive=lambda pid: False)  # dead holder reclaimed
    assert json.loads(lock.read_text())["job"] == "j2"
    opt12.release_lock(lock)
    assert not lock.exists()


def test_bn_eval_callback_keeps_running_stats():
    torch = pytest.importorskip("torch")
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import opt12_train as t

    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU())
    before = t.bn_hash(net)
    net.train()
    for m in t.bn_modules(net):
        m.eval()
    net(torch.randn(2, 3, 8, 8)).sum().backward()
    assert t.bn_hash(net) == before
    assert net[1].weight.grad is not None  # affine parameters still trainable
    net.train()
    net(torch.randn(2, 3, 8, 8))
    assert t.bn_hash(net) != before  # without the callback the statistics would move


def test_wiseft_identity_at_alpha_0_and_1():
    torch = pytest.importorskip("torch")
    from movd.opt12_soup import weighted_state

    a = {"w": torch.randn(3, 3).half(), "n": torch.tensor(5)}
    b = {"w": torch.randn(3, 3).half(), "n": torch.tensor(7)}
    assert torch.equal(weighted_state([a, b], [1.0, 0.0])["w"], a["w"])
    assert torch.equal(weighted_state([a, b], [0.0, 1.0])["w"], b["w"])
    assert torch.equal(weighted_state([a, b], [0.5, 0.5])["n"], a["n"])


def test_no_network_or_kaggle_code():
    files = [REPO / "movd" / "opt12.py", REPO / "movd" / "opt12_soup.py"] + sorted((REPO / "scripts").glob("opt12_*"))
    pattern = re.compile(r"\b(kaggle|requests|urllib|http\.client|socket)\b")
    for f in files:
        if f.suffix not in (".py", ".ps1"):
            continue
        code = "\n".join(line for line in f.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#"))
        code = re.sub(r'""".*?"""', "", code, flags=re.S)
        assert not pattern.search(code), f"{f.name} mentions network/Kaggle code"


def test_clock_constants_ordered():
    c = opt12.CLOCK
    assert c["train_latest_start"] < c["no_new_training"] < c["train_end_target"] < c["train_stop"] < c["train_kill"]
    assert c["interim_handoff"] < c["no_new_experiment"] < c["hard_handoff"]
    assert c["train_stop"] + timedelta(minutes=5) == c["train_kill"]
