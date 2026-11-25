"""Tests for scripts/p3b_breaching_cifar10.py (v0.16.10 P3b) and scripts/p3b_power_table.py.

They check the pieces that decide the gates and verdicts: the nested victim records, the views, the floor, the
metrics, the resume logic, the JSON schema, the gate logic (a row behind a failed gate gets no PASS or FAIL) and the
cross file consistency with the power table and the pre-registration. They do not run the real experiment. The end
to end test needs `breaching` and skips cleanly when it is missing. No test needs the network or real CIFAR-10.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "p3b_breaching_cifar10.py"
POWER = ROOT / "scripts" / "p3b_power_table.py"
PREREG = ROOT / "docs" / "preregistration" / "v0.16-p3b.md"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p3b():
    return _load("p3b_breaching_cifar10", SCRIPT)


@pytest.fixture(scope="module")
def power():
    return _load("p3b_power_table", POWER)


def _init(p3b, seed=0):
    torch.manual_seed(seed)
    return {k: v.clone() for k, v in p3b.SmallCNN().state_dict().items()}


def _client_data(b, seed=0, n=4):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n, b, 3, 32, 32)).astype(np.float32), rng.integers(0, 10, size=(n, b))


class _Args:
    quick = True


# ---- constants, seeds and records -------------------------------------------------------------------------
def test_seed_ranges_do_not_overlap_p3(p3b):
    ev = set(range(p3b.EVAL_SEED_BASE, p3b.EVAL_SEED_BASE + p3b.EVAL_SEEDS_REQUIRED))
    assert not ev & set(p3b.P3_EVAL_SEEDS) and not set(p3b.TUNING_SEEDS) & set(p3b.P3_TUNING_SEEDS)
    assert not ev & set(p3b.TUNING_SEEDS)
    assert (p3b.EVAL_SEED_BASE, p3b.EVAL_SEEDS_REQUIRED, p3b.TUNING_SEEDS) == (100, 20, (9100, 9101, 9102))


def test_preregistered_constants(p3b):
    assert p3b.ARMS == (4, 2) and p3b.N_CLIENTS == 4 and p3b.BLOCK == 8 and p3b.CONTROL_RECORDS == 1
    assert p3b.TV_GRID == (0.05, 0.2, 0.8, 3.2) and p3b.BREACHING_ITERATIONS == 4000 and p3b.LOCAL_LR == 0.15
    assert p3b.GATE == {"P": 1.0, "S": 0.20} and p3b.EQUIV == {"P": 1.0, "S": 0.10}
    assert p3b.AT_FLOOR == {"P": 1.0, "S": 0.10} and p3b.MIX_COST == {"P": 1.0, "S": 0.10} and p3b.MARGIN_CONTROL == 2.0
    assert p3b.P3_MODEL_SHA256.startswith("66ef08e1")


def test_model_has_documented_size_and_no_buffers(p3b):
    m = p3b.SmallCNN()
    assert sum(p.numel() for p in m.parameters()) == 25578
    assert len(list(m.buffers())) == 0


def test_split_keeps_clients_out_of_the_training_subset(p3b):
    sp = p3b.Split(2000, 400, 200)
    a, b = sp.client_records(100), sp.client_records(101)
    assert len(a) == 32 and len(set(a.tolist())) == 32
    assert not set(a.tolist()) & set(sp.train_idx.tolist())
    assert not np.array_equal(a, b) and np.array_equal(a, sp.client_records(100))


def test_arms_are_nested_and_share_floor_images(p3b):
    x, y = p3b.make_standin(1200, seed=2)
    split = p3b.Split(len(y), 200, 100)
    ctx = p3b.Context(_Args, x, y, "SYNTHETIC_STANDIN", split, _init(p3b), {"state_sha256": "m"}, "cpu")
    sd = ctx.seed_data(100)
    assert sd["cx"].shape == (4, 8, 3, 32, 32) and sd["cy"].shape == (4, 8)
    a4, a2 = sd["cx"][:, :4], sd["cx"][:, :2]
    assert np.array_equal(a4[0][:2], a2[0])                 # victim images of the 2 record arm are the first two of the 4 record arm
    assert np.array_equal(sd["cx"][0][:1], a2[0][:1])       # and the control record is the first
    assert len(sd["floor"]["per_image_mse"]) == 8            # one floor for the 8 victim records, sliced per arm
    assert sd["pool_size"] == len(y) - 32


def test_standin_is_refused_outside_quick(p3b, tmp_path):
    with pytest.raises(SystemExit):
        p3b.load_data(tmp_path, "auto", standin=True, quick=False)
    x, y, src = p3b.load_data(tmp_path, "auto", standin=True, quick=True)
    assert src == "SYNTHETIC_STANDIN" and x.shape == (2000, 32, 32, 3)


def test_context_flags_the_p3_checkpoint(p3b):
    x, y = p3b.make_standin(300, seed=1)
    split = p3b.Split(len(y), 50, 20)
    same = p3b.Context(_Args, x, y, "s", split, _init(p3b), {"state_sha256": p3b.P3_MODEL_SHA256}, "cpu")
    other = p3b.Context(_Args, x, y, "s", split, _init(p3b), {"state_sha256": "x"}, "cpu")
    assert same.model_meta["is_p3_checkpoint"] is True and other.model_meta["is_p3_checkpoint"] is False


# ---- metrics and floor -----------------------------------------------------------------------------------
def test_psnr_values(p3b):
    assert p3b.psnr_from_mse(0.01) == pytest.approx(20.0)
    assert p3b.psnr_from_mse(0.0) == pytest.approx(100.0)


def test_ssim_is_one_for_identical_and_falls_with_noise(p3b):
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, size=(3, 3, 32, 32))
    assert p3b.ssim_per_image(a, a) == pytest.approx(np.ones(3))
    noisy = np.clip(a + rng.normal(0, 0.2, a.shape), 0, 1)
    assert (p3b.ssim_per_image(a, noisy) < 0.9).all()


def test_matching_undoes_a_permutation(p3b):
    rng = np.random.default_rng(1)
    orig = rng.uniform(0, 1, size=(6, 3, 32, 32)).astype(np.float32)
    perm = rng.permutation(6)
    assert np.array_equal(p3b.match_to_originals(orig, orig[perm]), orig)


def test_class_stats_subtract_exactly_the_client_records(p3b):
    x, y = p3b.make_standin(300, seed=3)
    stats = p3b.ClassStats(x, y)
    xf = p3b.to_float(x)
    clients = np.arange(8)
    avg, mean_img = stats.pool_means(xf[clients], y[clients])
    mask = np.ones(len(y), bool)
    mask[clients] = False
    for c in range(10):
        assert avg[c] == pytest.approx(xf[mask & (y == c)].astype(np.float64).mean(axis=0), abs=1e-6)
    assert mean_img == pytest.approx(xf[mask].astype(np.float64).mean(axis=0), abs=1e-6)


def test_label_floor_is_the_class_average_of_the_pool(p3b):
    cls = np.zeros((10, 3, 32, 32))
    for c in range(10):
        cls[c] = c / 10.0
    victim_y = np.array([2, 5, 5, 9])
    victim_x = np.stack([np.full((3, 32, 32), v) for v in (0.2, 0.5, 0.6, 0.9)]).astype(np.float32)
    fl = p3b.label_floor(victim_x, victim_y, cls, np.full((3, 32, 32), 0.5))
    assert fl["per_image_mse"] == pytest.approx([0.0, 0.0, 0.01, 0.0], abs=1e-6)
    assert fl["matched_per_image_mse"][2] <= fl["per_image_mse"][2] + 1e-9


# ---- views -------------------------------------------------------------------------------------------------
def test_quantiser_matches_installed_flower(p3b):
    flq = pytest.importorskip("flwr.common.secure_aggregation.quantization")
    arr = np.random.default_rng(0).normal(0, 0.1, 500)
    np.random.seed(7)
    ours = p3b.quantize(arr, p3b.CLIP_RANGE, p3b.QUANT_RANGE)
    np.random.seed(7)
    theirs = flq.quantize([arr], p3b.CLIP_RANGE, p3b.QUANT_RANGE)[0]
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("b", [2, 4])
def test_views_follow_the_v014_definitions_at_each_arm(p3b, b):
    init = _init(p3b)
    cx, cy = _client_data(b)
    v = p3b.build_views(init, cx, cy, seed=100)
    flat = lambda g: torch.cat([t.flatten() for t in g])   # noqa: E731
    model = p3b.SmallCNN()
    model.load_state_dict(init)
    ce = torch.nn.functional.cross_entropy
    own = torch.autograd.grad(ce(model(torch.from_numpy(cx[0])), torch.from_numpy(cy[0]).long()), model.parameters())
    assert torch.allclose(flat(v["grads"]["plaintext"]), flat(own), atol=1e-5, rtol=1e-4)
    joint = torch.autograd.grad(ce(model(torch.from_numpy(cx.reshape(-1, 3, 32, 32))), torch.from_numpy(cy.reshape(-1)).long()), model.parameters())
    assert torch.allclose(flat(v["grads"]["aggregate_exact"]), flat(joint), atol=1e-5, rtol=1e-4)    # N times B images
    pl = flat(v["grads"]["plaintext"])
    assert float((flat(v["grads"]["collusion"]) - pl).norm() / pl.norm()) < 0.25
    assert float((flat(v["grads"]["aggregate"]) - pl).norm() / pl.norm()) > 0.2
    assert len(v["labels"]["aggregate"]) == 4 * b and len(v["labels"]["collusion"]) == b
    assert v["truth"]["aggregate"].shape == (4 * b, 3, 32, 32) and v["truth"]["plaintext"].shape == (b, 3, 32, 32)


def test_secagg_error_is_inside_the_quantisation_bound(p3b):
    rng = np.random.default_rng(2)
    vecs = [rng.normal(0, 0.2, 4000) for _ in range(4)]
    sizes = [4] * 4
    agg = p3b.secagg_aggregate(vecs, sizes, max_weight=p3b.MAX_WEIGHT, rng_seed=3)
    step = 2 * p3b.CLIP_RANGE / p3b.QUANT_RANGE
    bound = step * p3b.QUANT_RANGE / round(4 / p3b.MAX_WEIGHT * p3b.QUANT_RANGE)
    assert np.abs(agg - p3b.exact_mean(vecs, sizes)).max() <= 1.01 * bound


# ---- tuning --------------------------------------------------------------------------------------------------
def test_tuning_selects_highest_mean_psnr_ties_earlier_and_flags_the_edge(p3b):
    def seed_result(best):
        r = {"seed": 9100, "breaching": {}}
        for view in p3b.VIEWS:
            r["breaching"][view] = {repr(tv): [10.0 + (3.0 if tv == best[view] else 0.0)] * 4 for tv in p3b.TV_GRID}
        return r

    sel = p3b.select_tuning([seed_result({"plaintext": 0.8, "aggregate": 3.2, "collusion": 0.2})])
    assert sel["selected"] == {"plaintext": 0.8, "aggregate": 3.2, "collusion": 0.2}
    assert sel["winner_at_grid_edge"] == {"plaintext": False, "aggregate": True, "collusion": False}
    tied = {"seed": 9100, "breaching": {v: {repr(tv): [10.0] * 4 for tv in p3b.TV_GRID} for v in p3b.VIEWS}}
    assert p3b.select_tuning([tied])["selected"]["plaintext"] == p3b.TV_GRID[0]


# ---- verdict rules and the gates --------------------------------------------------------------------------------
def test_basic_verdict_rules(p3b):
    ci = lambda m, lo, hi: {"mean": m, "lo": lo, "hi": hi}   # noqa: E731
    assert p3b.verdict_lower(ci(3, 2, 4), 1.0)["verdict"] == "PASS"
    assert p3b.verdict_lower(ci(1.5, 0.5, 2.5), 1.0)["verdict"] == "FAIL, underpowered"
    assert p3b.verdict_lower(ci(0.2, -0.5, 0.9), 1.0)["verdict"] == "FAIL"
    assert p3b.verdict_lower(ci(3, 2.0, 4), 2.0, inclusive=True)["verdict"] == "PASS"
    assert p3b.verdict_upper(ci(-3, -4, -2), 1.0)["verdict"] == "PASS"
    assert p3b.verdict_upper(ci(0.5, -0.2, 1.4), 1.0)["verdict"] == "FAIL, underpowered"
    assert p3b.verdict_upper(ci(3, 2, 4), 1.0)["verdict"] == "FAIL"
    assert p3b.verdict_equiv(ci(0.1, -0.6, 0.8), 1.0)["verdict"] == "PASS"
    assert p3b.verdict_equiv(ci(0.1, -0.6, 1.1), 1.0)["verdict"] == "FAIL"
    assert p3b.verdict_equiv(ci(0.0, -1.0, 1.0), 1.0)["verdict"] == "PASS"       # boundary is inside


def _fake_run(p3b, seed, gains, ssim_gains, quick=False, standin=False, rng=None):
    rng = rng or np.random.default_rng(seed)
    rows = {}
    keys = ["b1/plaintext"] + [f"b{a}/{v}" for a in (4, 2) for v in ("plaintext", "collusion", "aggregate")]
    for k in keys:
        n = 1 if k.startswith("b1") else int(k[1])
        floor_psnr = 13.0 + rng.normal(0, 1, n)
        att_psnr = floor_psnr + gains.get(k, 0.0) + rng.normal(0, 0.5, n)
        floor_ssim = np.full(n, 0.3)
        rows[k] = {"attack": "breaching", "view": k.split("/")[1], "per_image_psnr": att_psnr.tolist(),
                   "per_image_mse": (10 ** (-att_psnr / 10)).tolist(),
                   "per_image_ssim": (floor_ssim + ssim_gains.get(k, 0.0) + rng.normal(0, 0.02, n)).tolist(),
                   "floor": {"per_image_psnr": floor_psnr.tolist(), "per_image_mse": (10 ** (-floor_psnr / 10)).tolist(),
                             "per_image_ssim": floor_ssim.tolist(), "matched_per_image_mse": (10 ** (-floor_psnr / 10)).tolist(),
                             "mean_image_per_image_mse": (10 ** (-floor_psnr / 10)).tolist()},
                   "n_images": n, "seconds": 1.0, "invalid_reconstructions": 0}
    return {"schema_version": 1, "seed": seed, "quick": quick, "synthetic_standin": standin, "data_source": "test", "device": "cpu",
            "model": {"state_sha256": "x"}, "rows": rows}


GOOD = {"b1/plaintext": 5.0, "b4/plaintext": 4.0, "b4/collusion": 3.8, "b4/aggregate": -1.0,
        "b2/plaintext": 5.0, "b2/collusion": 4.8, "b2/aggregate": -0.5}
GOOD_S = {"b1/plaintext": 0.4, "b4/plaintext": 0.35, "b4/collusion": 0.34, "b4/aggregate": 0.0,
          "b2/plaintext": 0.4, "b2/collusion": 0.39, "b2/aggregate": 0.0}


def _summary(p3b, gains, ssim, n=20, base=None, **kw):
    base = p3b.EVAL_SEED_BASE if base is None else base
    return p3b.summarise([_fake_run(p3b, base + i, gains, ssim, **kw) for i in range(n)])


def test_everything_passes_when_the_plaintext_gates_pass(p3b):
    s = _summary(p3b, GOOD, GOOD_S)
    assert s["status"] == "verdicts computed"
    v = s["verdicts"]
    for key in ("K0", "G4P", "G4S", "G2P", "G2S"):
        assert v[key]["verdict"] == "PASS", key
    for arm in (4, 2):
        for m in "PS":
            for rule in "CQAD":
                assert v[f"{rule}{arm}{m}"]["verdict"] == "PASS", f"{rule}{arm}{m}"
    assert s["rows"]["b4/plaintext"]["n_images"] == 80 and s["rows"]["b2/plaintext"]["n_images"] == 40 and s["rows"]["b1/plaintext"]["n_images"] == 20
    assert v["T1"]["verdict"] in ("PASS", "FAIL", "FAIL, underpowered")


def test_failed_gate_makes_its_rows_not_interpretable_and_gives_no_pass(p3b):
    """The vacuous pass of P3: plaintext at the floor, aggregate below it. A would pass if it were read."""
    gains = dict(GOOD)
    gains["b4/plaintext"] = 0.1
    gains["b4/collusion"] = 0.1
    gains["b4/aggregate"] = -3.0
    s = _summary(p3b, gains, GOOD_S)
    v = s["verdicts"]
    assert v["G4P"]["verdict"].startswith("FAIL")
    for rule in "CQAD":
        assert v[f"{rule}4P"]["verdict"] == "NOT INTERPRETABLE", rule
        assert "G4P" in v[f"{rule}4P"]["reason"]
        assert v[f"{rule}4P"]["mean"] is not None          # the numbers stay
    assert v["A4P"]["mean"] < -2.0                            # it would have passed as a vacuous pass
    # the other metric and the other arm are unaffected because their gates passed
    assert v["G4S"]["verdict"] == "PASS" and v["A4S"]["verdict"] == "PASS"
    assert v["G2P"]["verdict"] == "PASS" and v["A2P"]["verdict"] == "PASS"


def test_ssim_gate_failure_withholds_only_the_ssim_rows_of_that_arm(p3b):
    ssim = dict(GOOD_S)
    ssim["b2/plaintext"] = 0.05
    s = _summary(p3b, GOOD, ssim)
    v = s["verdicts"]
    assert v["G2S"]["verdict"].startswith("FAIL")
    assert all(v[f"{r}2S"]["verdict"] == "NOT INTERPRETABLE" for r in "CQAD")
    assert all(v[f"{r}2P"]["verdict"] == "PASS" for r in "CQAD")


def test_a_failed_control_withholds_every_dependent_row(p3b):
    gains = dict(GOOD)
    gains["b1/plaintext"] = 0.0
    v = _summary(p3b, gains, GOOD_S)["verdicts"]
    assert v["K0"]["verdict"].startswith("FAIL")
    for arm in (4, 2):
        for m in "PS":
            assert all(v[f"{r}{arm}{m}"]["verdict"] == "NOT INTERPRETABLE" and "K0" in v[f"{r}{arm}{m}"]["reason"] for r in "CQAD")
    assert v["G4P"]["verdict"] == "PASS"                      # the gates themselves are still reported


def test_a_read_row_can_still_fail(p3b):
    gains = dict(GOOD)
    gains["b4/aggregate"] = 3.0              # the attack still works on the aggregate: A fails, D fails
    gains["b4/collusion"] = 0.0              # collusion falls to the floor: C fails, Q fails
    v = _summary(p3b, gains, GOOD_S)["verdicts"]
    assert v["A4P"]["verdict"] == "FAIL" and v["D4P"]["verdict"] in ("FAIL", "FAIL, underpowered")
    assert v["C4P"]["verdict"] == "FAIL" and v["Q4P"]["verdict"] == "FAIL"


def test_summary_withholds_verdicts_for_wrong_size_quick_standin_or_wrong_seeds(p3b):
    assert "no verdict" in _summary(p3b, GOOD, GOOD_S, n=5)["status"]
    assert "quick run" in _summary(p3b, GOOD, GOOD_S, quick=True)["status"]
    st = _summary(p3b, GOOD, GOOD_S, standin=True)
    assert "stand-in" in st["status"] and "withheld" in st["verdicts"]
    assert "seeds are not" in _summary(p3b, GOOD, GOOD_S, base=0)["status"]          # P3 seeds are refused


def test_power_void_when_observed_sd_is_far_above_the_assumed_one(p3b):
    entry = p3b.power_entry("G", (4, "P"), 1.8 * 1.6)
    assert entry["power_void"] is True and entry["ratio"] == pytest.approx(1.6)
    assert p3b.power_entry("G", (4, "P"), 1.8 * 1.4)["power_void"] is False
    s = _summary(p3b, GOOD, GOOD_S)
    assert set(s["power_check_observed_vs_assumed_sd"]) >= {"K0", "G4P", "C2S", "Q4P", "A2P", "D4S", "T1"}


def test_t1_uses_the_first_two_victim_images_only(p3b):
    run = _fake_run(p3b, 100, GOOD, GOOD_S)
    r4 = run["rows"]["b4/plaintext"]
    r4["per_image_psnr"] = [20.0, 20.0, -50.0, -50.0]          # the last two must not matter
    r4["floor"]["per_image_psnr"] = [10.0, 10.0, 10.0, 10.0]
    assert p3b.first_two_gain([run], "b4/plaintext") == pytest.approx([10.0])


def test_bootstrap_is_deterministic(p3b):
    assert _summary(p3b, GOOD, GOOD_S)["rows"] == _summary(p3b, GOOD, GOOD_S)["rows"]


def test_a_p3_style_checkpoint_is_reused_without_training(p3b, tmp_path):
    """The runbook copies the P3 model.pt into the output folder. Its config must be accepted as it is."""
    split = p3b.Split(50000, p3b.TRAIN_SUBSET_SIZE, p3b.HELD_OUT_SIZE)
    config = {"train_subset": 20000, "epochs": 10, "lr": 0.001, "batch": 128, "data_source": "hf-parquet", "quick": False, "subset_seed": 12345}
    state = _init(p3b)
    torch.save({"state": state, "meta": {"state_sha256": "abc"}, "config": config}, tmp_path / "model.pt")
    got, meta = p3b.train_or_load_model(tmp_path, None, None, split, epochs=p3b.TRAIN_EPOCHS, source="hf-parquet+cache", device="cpu", quick=False)
    assert meta["state_sha256"] == "abc" and all(torch.equal(got[k], state[k]) for k in state)
    with pytest.raises(SystemExit):       # a checkpoint trained on another source is refused
        p3b.train_or_load_model(tmp_path, None, None, split, epochs=p3b.TRAIN_EPOCHS, source="torchvision", device="cpu", quick=False)


# ---- resume and atomic writes -------------------------------------------------------------------------------------
def test_atomic_write_leaves_no_temporary_file(p3b, tmp_path):
    p3b.atomic_write_json(tmp_path / "a.json", {"x": 1})
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["a.json"]


def test_run_resumes_and_skips_finished_seeds(p3b, tmp_path, monkeypatch):
    calls = []

    def fake_run_seed(ctx, seed, tuning):
        calls.append(seed)
        return {"schema_version": 1, "seed": seed, "settings_hash": p3b.settings_hash(ctx, tuning), "rows": {}, "quick": True,
                "synthetic_standin": True}, {}

    monkeypatch.setattr(p3b, "run_seed", fake_run_seed)
    monkeypatch.setattr(p3b, "train_or_load_model", lambda *a, **k: (_init(p3b), {"state_sha256": "m"}))
    sel = {"selected": {v: 0.2 for v in p3b.VIEWS}}
    monkeypatch.setattr(p3b, "run_tuning", lambda ctx, out: sel)
    monkeypatch.setattr(p3b, "write_summary", lambda out, n: 0)
    argv = ["--out", str(tmp_path), "--seeds", "3", "--quick", "--synthetic-standin", "--device", "cpu", "--data-dir", str(tmp_path / "d")]
    p3b.main(argv)
    assert calls == [100, 101, 102]
    p3b.main(argv)
    assert calls == [100, 101, 102]
    (tmp_path / "seed101.json").unlink()
    (tmp_path / "seed102.json").write_text('{"schema_version": 1, "seed": 102, "rows"')    # truncated by a killed run
    p3b.main(argv)
    assert calls == [100, 101, 102, 101, 102]
    other = json.loads((tmp_path / "seed100.json").read_text())
    other["settings_hash"] = "different"
    (tmp_path / "seed100.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit):
        p3b.main(argv)
    assert json.loads((tmp_path / "seed100.json").read_text())["settings_hash"] == "different"


# ---- cross file consistency ---------------------------------------------------------------------------------------------
def test_assumed_sds_match_the_power_table_script(p3b, power):
    for name, (kind, margin, sd) in power.RULES.items():
        tag = name.split()[0]
        if tag == "K0":
            assert sd == p3b.ASSUMED_SD["K0"]["P"] and margin == p3b.MARGIN_CONTROL
        elif tag == "T1":
            assert sd == p3b.ASSUMED_SD["T1"]["P"]
        else:
            rule, arm, m = tag[0], int(tag[1]), tag[2]
            assert sd == p3b.ASSUMED_SD[rule][(arm, m)], name
            table = {"G": p3b.GATE, "A": p3b.AT_FLOOR, "D": p3b.MIX_COST, "Q": p3b.EQUIV}[rule]
            assert margin == table[m], name
            assert kind == {"G": "lower", "A": "upper", "D": "lower", "Q": "equiv"}[rule], name


def test_preregistration_names_the_hash_of_the_power_script():
    text = PREREG.read_text()
    digest = hashlib.sha256(POWER.read_bytes()).hexdigest()
    assert digest in text.split("## Deviation log")[0]


def test_power_table_reproduces_a_few_preregistered_cells(power):
    # cells quoted in the pre-registration, with 4,000 simulations. Here 1,500 are used, so the tolerance is wider
    assert power.power("lower", 1.0, 1.8, 1.9, sims=1500) == pytest.approx(0.63, abs=0.06)
    assert power.power("upper", 1.0, 1.6, 0.0, sims=1500) == pytest.approx(0.81, abs=0.06)
    assert power.power("equiv", 1.0, 0.9, 0.5, sims=1500) == pytest.approx(0.72, abs=0.06)
    assert power.power("lower", 2.0, 3.3, 5.3, sims=1500) == pytest.approx(0.99, abs=0.03)
    assert power.analytic_lower(1.0, 1.8, 1.9) == pytest.approx(0.63, abs=0.08)       # the t approximation agrees roughly


# ---- end to end schema (needs breaching) ----------------------------------------------------------------------------------
@pytest.mark.slow
def test_one_seed_end_to_end_schema(p3b, tmp_path):
    pytest.importorskip("breaching")
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    x, y = p3b.make_standin(600, seed=1)
    split = p3b.Split(len(y), 100, 50)
    ctx = p3b.Context(_Args, x, y, "SYNTHETIC_STANDIN", split, _init(p3b), {"state_sha256": "m", "held_out_accuracy": 0.1}, "cpu")
    ctx.iters = 2
    sel = {"selected": {v: 0.2 for v in p3b.VIEWS}}
    out, images = p3b.run_seed(ctx, 100, sel)
    json.dumps(out)
    assert out["schema_version"] == 1 and out["synthetic_standin"] is True and out["seed"] == 100
    assert out["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest() and out["hostname"] and out["run_started"]
    expected = {"b1/plaintext"} | {f"b{a}/{v}" for a in (4, 2) for v in ("plaintext", "collusion", "aggregate")}
    assert set(out["rows"]) == expected
    for key, row in out["rows"].items():
        n = 1 if key.startswith("b1") else int(key[1])
        assert row["n_images"] == n
        for field in ("per_image_mse", "per_image_psnr", "per_image_ssim"):
            assert len(row[field]) == n and len(row["floor"][field]) == n
    assert images["b4/aggregate"].shape == (16, 3, 32, 32) and images["b2/aggregate"].shape == (8, 3, 32, 32)
    assert out["rows"]["b4/plaintext"]["floor"]["per_image_mse"] == out["floor"]["per_image_mse"][:4]
    assert out["rows"]["b2/plaintext"]["floor"]["per_image_mse"] == out["floor"]["per_image_mse"][:2]
    assert out["rows"]["b1/plaintext"]["floor"]["per_image_mse"] == out["floor"]["per_image_mse"][:1]
