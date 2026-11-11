"""OPT-10: frozen frame lists and grid, weight averaging, BatchNorm re-estimation and the per-image-statistics
diagnostic, on tiny synthetic networks (CPU only)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from movd import opt10

REPO = Path(__file__).resolve().parents[1]
EVID = REPO / "results" / "opt-10-soup-bn"


# ---------------------------------------------------------------- frozen frame lists


def test_bn_frames_are_exactly_1024_unique_and_span_the_whole_list():
    idx = opt10.spaced_indices(6072, opt10.BN_FRAMES)
    assert len(idx) == 1024 and len(set(idx)) == 1024
    assert idx[0] == 0 and idx[-1] == 6071
    assert idx == sorted(idx)
    gaps = {b - a for a, b in itertools.pairwise(idx)}
    assert gaps == {5, 6}  # evenly spread, never the first 1,024 frames
    assert len(range(6072)[::6][:1024]) == 1012  # the rejected first draft of the rule


def test_spaced_indices_use_round_half_up_and_reject_bad_sizes():
    assert opt10.spaced_indices(5, 2) == [0, 4]
    assert opt10.spaced_indices(5, 3) == [0, 2, 4]
    assert opt10.spaced_indices(4, 3) == [0, 2, 3]  # 1.5 rounds up
    assert opt10.spaced_indices(7, 7) == list(range(7))
    for total, count in ((5, 6), (5, 1), (0, 0)):
        with pytest.raises(ValueError):
            opt10.spaced_indices(total, count)


def test_spaced_ids_sort_first_and_reject_duplicates():
    assert opt10.spaced_ids([9, 1, 5, 3, 7], 3) == [1, 5, 9]
    with pytest.raises(ValueError):
        opt10.spaced_ids([1, 1, 2], 2)
    assert opt10.ids_sha256([1, 5, 9]) == opt10.ids_sha256((1, 5, 9))
    assert opt10.ids_sha256([1, 5, 9]) != opt10.ids_sha256([1, 5, 10])


@pytest.mark.parametrize(
    ("name", "count"), [("bn_frames", opt10.BN_FRAMES), ("d2_frames", opt10.D2_FRAMES)]
)
def test_frozen_frame_files_follow_the_rule(name, count):
    from movd import ultra

    record = json.loads((EVID / f"{name}.json").read_text(encoding="utf-8"))
    ids = opt10.spaced_ids(ultra.train_split_ids("train_final"), count)
    assert record["image_ids"] == ids and len(set(ids)) == count
    assert record["sha256_of_comma_joined_ids"] == opt10.ids_sha256(ids)
    assert record["first_index"] == 0 and record["last_index"] == 6071
    rule = (EVID / "selection_rule.md").read_text(encoding="utf-8")
    assert record["sha256_of_comma_joined_ids"] in rule


# ---------------------------------------------------------------- frozen grid


def test_grid_is_the_five_approved_candidates():
    assert list(opt10.GRID) == ["c0", "m2", "m4", "m4e1", "m4bn"]
    assert opt10.REPLICATE_SEEDS == {"R1": 43, "R2": 44, "R3": 45}
    assert opt10.GRID["c0"] == {
        "bn": True,
        "epoch_weights": [0.8, 0.2],
        "phase": 0,
        "runs": ["A"],
    }
    assert [c for c, s in opt10.GRID.items() if s["bn"]] == ["c0", "m4bn"]
    assert opt10.STAGE == {
        "S1_min_delta_discovery": 0.0010,
        "S2_min_delta_confirmation": 0.0,
    }
    assert opt10.GATE["G1_min_delta_mAP"] == 0.0010
    assert opt10.GATE["G2_min_delta_mAP_width_le_1920"] == 0.0010
    assert opt10.SELECTION_TIER["min_delta_mAP"] == 0.0015
    assert (
        opt10.BUDGET["gpu_minutes_phase0"] == 45 and opt10.BUDGET["max_new_csvs"] == 3
    )


def test_rule_file_states_every_candidate():
    rule = (EVID / "selection_rule.md").read_text(encoding="utf-8")
    for cid, spec in opt10.GRID.items():
        assert f"{cid} {json.dumps(spec, sort_keys=True)}" in rule


def test_candidate_weights_average_runs_uniformly_and_sum_to_one():
    assert opt10.candidate_weights("c0") == [("A", 1, 0.8), ("A", 2, 0.2)]
    assert opt10.candidate_weights("m2") == [
        ("A", 1, 0.4),
        ("R1", 1, 0.4),
        ("A", 2, 0.1),
        ("R1", 2, 0.1),
    ]
    assert opt10.candidate_weights("m4e1") == [
        (run, 1, 0.25) for run in ("A", "R1", "R2", "R3")
    ]
    for cid in opt10.GRID:
        assert sum(w for _, _, w in opt10.candidate_weights(cid)) == pytest.approx(1.0)


# ---------------------------------------------------------------- weight averaging


def state(seed, dtype=torch.float16):
    g = torch.Generator().manual_seed(seed)
    return {
        "w": torch.randn(4, 3, generator=g).to(dtype),
        "b": torch.randn(4, generator=g).to(dtype),
        "n": torch.tensor(seed, dtype=torch.int64),
    }


def test_average_of_identical_states_is_the_state():
    a = state(1)
    out = opt10.weighted_state([a, a, a, a], [0.25] * 4)
    assert all(torch.equal(out[k], a[k]) and out[k].dtype == a[k].dtype for k in a)
    out = opt10.weighted_state([a], [1.0])
    assert all(torch.equal(out[k], a[k]) for k in a)


@pytest.mark.parametrize("w", [0.8, 0.75, 0.5])
def test_two_states_equal_the_opt08_soup_formula_exactly(w):
    a, b = state(1), state(2)
    out = opt10.weighted_state([a, b], [w, 1.0 - w])
    for k in ("w", "b"):  # scripts/opt08_soup.py soup_state
        ref = (w * a[k].float() + (1.0 - w) * b[k].float()).to(a[k].dtype)
        assert torch.equal(out[k], ref)
    assert torch.equal(out["n"], a["n"])  # non-float tensors come from the first state
    assert out["n"] is not a["n"]


def test_four_states_are_the_weighted_mean():
    states = [state(s, torch.float32) for s in range(4)]
    weights = [0.4, 0.4, 0.1, 0.1]
    out = opt10.weighted_state(states, weights)
    ref = sum(w * s["w"] for w, s in zip(weights, states, strict=True))
    assert torch.allclose(out["w"], ref, atol=1e-6)


def test_weighted_state_rejects_bad_input():
    a, b = state(1), state(2)
    with pytest.raises(ValueError):
        opt10.weighted_state([a, b], [0.5])
    with pytest.raises(ValueError):
        opt10.weighted_state([a, b], [0.5, 0.4])
    with pytest.raises(ValueError):
        opt10.weighted_state([a, {"w": a["w"]}], [0.5, 0.5])
    with pytest.raises(ValueError):
        opt10.weighted_state([a, {**b, "b": b["b"].float()}], [0.5, 0.5])
    with pytest.raises(ValueError):
        opt10.weighted_state([], [])


# ---------------------------------------------------------------- BatchNorm


class Probe(nn.Module):
    """Records, at every forward, its own training flag and whether gradients are enabled."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append((self.training, torch.is_grad_enabled(), x.dtype))
        return x


def toy(momentum=0.03):
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.BatchNorm2d(3, momentum=momentum),
        Probe(),
        nn.Conv2d(3, 5, 3, padding=1),
        nn.Dropout(0.5),
        nn.Sequential(nn.BatchNorm2d(5, momentum=momentum), nn.ReLU()),
    )
    with torch.no_grad():  # statistics that are clearly not the reset values
        for m in opt10.bn_layers(net):
            m.running_mean.uniform_(-1, 1)
            m.running_var.uniform_(0.5, 2)
            m.num_batches_tracked.fill_(77)
    return net


def batches(n=6, seed=3):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 3, 8, 8, generator=g) * (i + 1) + i for i in range(n)]


def buffers(net):
    return {
        k: v.clone()
        for k, v in net.state_dict().items()
        if k in opt10.bn_buffer_names(net)
    }


def flags(net):
    return [m.training for m in net.modules()]


def test_bn_layers_and_buffer_names():
    net = toy()
    assert len(opt10.bn_layers(net)) == 2
    assert opt10.bn_buffer_names(net) == [
        "0.running_mean",
        "0.running_var",
        "0.num_batches_tracked",
        "4.0.running_mean",
        "4.0.running_var",
        "4.0.num_batches_tracked",
    ]


def test_reestimate_bn_gives_the_plain_average_of_the_per_batch_statistics():
    net = toy()
    xs = batches()
    record = opt10.reestimate_bn(net, xs)
    assert record == {"bn_layers": 2, "batches": len(xs)}
    first = net[0]
    mean = torch.stack([x.mean(dim=(0, 2, 3)) for x in xs]).mean(0)
    var = torch.stack(
        [x.transpose(0, 1).reshape(3, -1).var(dim=1, unbiased=True) for x in xs]
    ).mean(0)
    assert torch.allclose(first.running_mean, mean, atol=1e-5)
    assert torch.allclose(first.running_var, var, rtol=1e-5)
    for m in opt10.bn_layers(net):
        assert int(m.num_batches_tracked) == len(xs)  # reset to 0, then one per batch
        assert torch.isfinite(m.running_mean).all() and (m.running_var > 0).all()


def test_reestimate_bn_resets_all_three_buffers_first():
    """With one batch the result is that batch's statistics alone: nothing of the old values is left."""
    net = toy()
    x = batches(1)[0]
    opt10.reestimate_bn(net, [x])
    assert torch.allclose(net[0].running_mean, x.mean(dim=(0, 2, 3)), atol=1e-6)
    assert int(net[0].num_batches_tracked) == 1


@pytest.mark.parametrize("start", ["eval", "train", "mixed"])
def test_reestimate_bn_restores_every_mode_and_momentum(start):
    net = toy(momentum=0.03)
    net.train() if start == "train" else net.eval()
    if start == "mixed":
        net[2].train()
        net[4][0].train()
    before = flags(net)
    opt10.reestimate_bn(net, batches())
    assert flags(net) == before
    assert [m.momentum for m in opt10.bn_layers(net)] == [0.03, 0.03]


def test_reestimate_bn_runs_only_batchnorm_in_training_mode_fp32_without_gradients():
    net = toy().train()  # worst case: everything starts in training mode
    seen_bn = []
    hooks = [
        m.register_forward_hook(
            lambda mod, _i, _o: seen_bn.append((mod.training, mod.momentum))
        )
        for m in opt10.bn_layers(net)
    ]
    xs = [x.double() for x in batches(4)]  # the function casts every batch to FP32
    net.double()
    net.float()
    opt10.reestimate_bn(net, xs)
    for h in hooks:
        h.remove()
    assert seen_bn == [(True, None)] * 8  # BatchNorm: training mode, cumulative average
    probe = net[1]
    assert (
        probe.seen == [(False, False, torch.float32)] * 4
    )  # others: eval mode, no gradients, FP32
    assert net[3].training  # Dropout's own flag is back to what it was (train)


def test_reestimate_bn_leaves_parameters_untouched():
    net = toy()
    params = {k: v.clone() for k, v in net.named_parameters()}
    opt10.reestimate_bn(net, batches())
    assert all(torch.equal(v, params[k]) for k, v in net.named_parameters())
    assert not any(p.grad is not None for p in net.parameters())


def test_reestimate_bn_is_deterministic_and_independent_of_the_old_statistics():
    a, b = toy(), toy()
    with torch.no_grad():
        for m in opt10.bn_layers(b):
            m.running_mean.fill_(123.0)
            m.running_var.fill_(456.0)
    opt10.reestimate_bn(a, batches())
    opt10.reestimate_bn(b, batches())
    assert all(torch.equal(v, buffers(b)[k]) for k, v in buffers(a).items())


def test_failed_reestimation_restores_statistics_modes_and_momentum():
    net = toy().eval()
    before, modes = buffers(net), flags(net)

    def broken(_x):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        opt10.reestimate_bn(net, batches(), forward=broken)
    with pytest.raises(ValueError):
        opt10.reestimate_bn(net, [])
    assert all(torch.equal(v, before[k]) for k, v in buffers(net).items())
    assert flags(net) == modes
    assert [m.momentum for m in opt10.bn_layers(net)] == [0.03, 0.03]
    with pytest.raises(ValueError):
        opt10.reestimate_bn(nn.Sequential(nn.Conv2d(3, 3, 1)), batches())


def test_reestimated_statistics_are_used_by_normal_inference():
    net = toy().eval()
    x = batches(1)[0]
    old = net(x)
    opt10.reestimate_bn(net, batches())
    assert not torch.allclose(net(x), old)
    assert torch.equal(net(x), net(x))  # eval mode: deterministic (Dropout inactive)


# ---------------------------------------------------------------- D1 context


@pytest.mark.parametrize("start", ["eval", "train", "mixed"])
def test_bn_batch_stats_restores_buffers_byte_identically_and_modes(start):
    net = toy()
    net.train() if start == "train" else net.eval()
    if start == "mixed":
        net[4][0].train()
    before, modes = buffers(net), flags(net)
    with opt10.bn_batch_stats(net) as inside, torch.no_grad():
        assert inside is net
        for x in batches():
            net(x)
        assert [m.training for m in opt10.bn_layers(net)] == [True, True]
        assert not net[2].training and not net[3].training and not net[1].training
        assert [m.momentum for m in opt10.bn_layers(net)] == [0.0, 0.0]
    after = buffers(net)
    assert list(after) == list(before) and len(after) == 6
    for k in before:  # running_mean, running_var and num_batches_tracked
        assert torch.equal(after[k], before[k]) and after[k].dtype == before[k].dtype
    assert flags(net) == modes
    assert [m.momentum for m in opt10.bn_layers(net)] == [0.03, 0.03]


def test_bn_batch_stats_restores_after_an_error():
    net = toy().eval()
    before, modes = buffers(net), flags(net)
    with pytest.raises(RuntimeError), opt10.bn_batch_stats(net):
        net(batches(1)[0])
        raise RuntimeError("boom")
    assert all(torch.equal(v, before[k]) for k, v in buffers(net).items())
    assert flags(net) == modes


def test_bn_batch_stats_normalises_with_the_current_image():
    net = nn.Sequential(nn.BatchNorm2d(3, affine=False)).eval()
    with torch.no_grad():
        net[0].running_mean.fill_(5.0)
        net[0].running_var.fill_(9.0)
    x = batches(1)[0]
    with torch.no_grad():
        stored = net(x)
        with opt10.bn_batch_stats(net):
            own = net(x)
        again = net(x)
    assert torch.allclose(own.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(
        own.var(dim=(0, 2, 3), unbiased=False), torch.ones(3), atol=1e-3
    )
    assert not torch.allclose(own, stored)
    assert torch.equal(
        again, stored
    )  # outside the block the stored statistics are used again


# ---------------------------------------------------------------- input


def test_preprocess_stretches_to_the_square_input_in_rgb_0_1():
    image = np.zeros((6, 12, 3), np.uint8)
    image[..., 0] = 255  # blue channel of a BGR frame
    out = opt10.preprocess(image, imgsz=8)
    assert out.shape == (1, 3, 8, 8) and out.dtype == torch.float32
    assert (
        float(out[0, 2].min()) == 1.0 and float(out[0, :2].max()) == 0.0
    )  # blue is last in RGB
    square = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    out = opt10.preprocess(square, imgsz=8)  # already square: no resampling
    assert torch.equal(
        out[0], torch.from_numpy(square[..., ::-1].copy()).permute(2, 0, 1) / 255.0
    )


# ---------------------------------------------------------------- replicate configs and the gate


@pytest.mark.parametrize(("run", "seed"), sorted(opt10.REPLICATE_SEEDS.items()))
def test_replicate_configs_change_only_the_seed_and_stop_after_epoch_2(run, seed):
    from movd import ultra

    base = ultra.load_run_config(REPO / "configs/runs/final_rtdetr.json")
    rep = ultra.load_run_config(REPO / f"configs/runs/opt10_rep_s{seed}.json")
    assert rep["overrides"] == {**base["overrides"], "seed": seed}
    assert rep["stop_after_epoch"] == 2 and "stop_after_epoch" not in base
    same = set(base) - {"run_id", "note", "overrides"}
    assert all(rep[k] == base[k] for k in same)
    assert ultra.train_kwargs(rep, "d.yaml", "p", "0")["seed"] == seed
    assert ultra.train_kwargs(base, "d.yaml", "p", "0")["seed"] == 42


def gate_inputs(
    delta=0.002, clip_deltas=(0.002,) * 7, loco=(0.002,) * 7, small=0.0, dense=0.0
):
    clips = [f"clip:c{i}" for i in range(7)]
    base = {
        k: {"mAP": 0.29, "AP_small": 0.17, "density_ge100": 0.35}
        for k in (
            "val_full",
            "width_le_1920",
            "without_4k_clip",
            "discovery",
            "confirmation",
            *clips,
        )
    }
    cand = {
        k: {
            "mAP": 0.29 + delta,
            "AP_small": 0.17 + small,
            "density_ge100": 0.35 + dense,
        }
        for k in base
    }
    for k, d in zip(clips, clip_deltas, strict=True):
        cand[k] = {"mAP": 0.29 + d}
    base_loco = {f"without:c{i}": {"mAP": 0.29} for i in range(7)}
    cand_loco = {f"without:c{i}": {"mAP": 0.29 + d} for i, d in enumerate(loco)}
    return base, cand, base_loco, cand_loco


def test_gate_thresholds_and_leave_one_clip_out(monkeypatch):
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import opt10_loop

    monkeypatch.setattr(opt10_loop, "frozen_unchanged", lambda: True)
    ok = opt10_loop.gate(*gate_inputs(), 8)
    assert ok["passes_G1_G7_G9"] and ok["selection_tier_val_part"] and not ok["label_T"]
    # the OPT-09 w075 case: +0.0008 is below the +0.0010 floor
    low = opt10_loop.gate(*gate_inputs(delta=0.0008), 8)
    assert (
        not low["passes_G1_G7_G9"]
        and not low["checks"]["G1 val_full mAP >= E6 + 0.0010"]
    )
    # +0.0012 is upload-ready but below the +0.0015 selection tier
    mid = opt10_loop.gate(*gate_inputs(delta=0.0012), 8)
    assert mid["passes_G1_G7_G9"] and not mid["selection_tier_val_part"]
    # one leave-one-clip-out removal that erases the gain fails G7
    loco = opt10_loop.gate(*gate_inputs(loco=(0.002,) * 6 + (0.0,)), 8)
    assert not loco["checks"]["G7 leave-one-clip-out delta > 0 for every clip"]
    # three negative clips fail G3; AP_small and dense-bin losses beyond 0.002 fail G5 / G6
    assert not opt10_loop.gate(
        *gate_inputs(clip_deltas=(-0.001,) * 3 + (0.002,) * 4), 8
    )["passes_G1_G7_G9"]
    assert not opt10_loop.gate(*gate_inputs(small=-0.0021), 8)["passes_G1_G7_G9"]
    assert not opt10_loop.gate(*gate_inputs(dense=-0.0021), 8)["passes_G1_G7_G9"]
    # label T: val_full >= 0.2999 (here 0.29 + 0.0100)
    assert opt10_loop.gate(*gate_inputs(delta=0.0100), 8)["label_T"]


def test_bn_image_paths_uses_the_manifest_embedded_in_ch14_export_check(
    tmp_path, monkeypatch
):
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import opt10_soup

    export = tmp_path / "export"
    image = export / "train_final/images/clip/0000001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"frame")
    manifest = {"name": "train_final", "num_images": 6072, "masked": True}
    manifest_path = export / "train_final_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    reference = tmp_path / "export_check.json"
    reference.write_text(json.dumps({"manifests": {"train_final": manifest}}))
    monkeypatch.setattr(opt10_soup, "EXPORT", export)
    monkeypatch.setattr(opt10_soup, "EXPORT_CHECK", reference)
    monkeypatch.setattr(opt10_soup, "frozen_ids", lambda _name: [1])
    monkeypatch.setattr(
        opt10_soup.data,
        "load_coco",
        lambda _split: {"images": [{"id": 1, "file_name": "images/clip/0000001.jpg"}]},
    )
    assert opt10_soup.bn_image_paths() == [image]
    manifest_path.write_text(json.dumps({**manifest, "masked": False}))
    with pytest.raises(SystemExit, match="differs from the CH-14 export manifest"):
        opt10_soup.bn_image_paths()
