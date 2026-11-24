"""Tests for scripts/p3_breaching_cifar10.py (v0.16 P3).

They check the pieces that decide the verdicts: the floor, the metrics, the view construction, the
resume logic, the JSON schema and the verdict rules. They do not run the real experiment. The end to end
test needs `breaching` and skips cleanly when it is missing. No test needs the network or real CIFAR-10.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "p3_breaching_cifar10.py"


@pytest.fixture(scope="module")
def p3():
    spec = importlib.util.spec_from_file_location("p3_breaching_cifar10", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p3_breaching_cifar10"] = mod
    spec.loader.exec_module(mod)
    return mod


def _init(p3, seed=0):
    torch.manual_seed(seed)
    return {k: v.clone() for k, v in p3.SmallCNN().state_dict().items()}


def _client_data(seed=0, n=4, b=8):
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0, 1, size=(n, b, 3, 32, 32)).astype(np.float32)
    cy = rng.integers(0, 10, size=(n, b))
    return cx, cy


# ---- model and split ------------------------------------------------------------------------------
def test_model_has_documented_size_and_no_buffers(p3):
    m = p3.SmallCNN()
    assert sum(p.numel() for p in m.parameters()) == 25578
    assert len(list(m.buffers())) == 0


def test_split_keeps_clients_out_of_the_training_subset(p3):
    sp = p3.Split(2000, 400, 200)
    a, b = sp.client_records(0), sp.client_records(1)
    assert len(a) == 32 and len(set(a.tolist())) == 32
    assert not set(a.tolist()) & set(sp.train_idx.tolist())
    assert not np.array_equal(a, b)
    assert np.array_equal(a, sp.client_records(0))     # deterministic per seed


def test_standin_is_refused_outside_quick(p3, tmp_path):
    with pytest.raises(SystemExit):
        p3.load_data(tmp_path, "auto", standin=True, quick=False)
    x, y, src = p3.load_data(tmp_path, "auto", standin=True, quick=True)
    assert src == "SYNTHETIC_STANDIN" and x.shape == (2000, 32, 32, 3)


# ---- metrics and floor -----------------------------------------------------------------------------
def test_psnr_values(p3):
    assert p3.psnr_from_mse(0.01) == pytest.approx(20.0)
    assert p3.psnr_from_mse(0.0) == pytest.approx(100.0)     # clamped at 1e-10, no infinity


def test_ssim_is_one_for_identical_and_falls_with_noise(p3):
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, size=(3, 3, 32, 32))
    assert p3.ssim_per_image(a, a) == pytest.approx(np.ones(3))
    noisy = np.clip(a + rng.normal(0, 0.2, a.shape), 0, 1)
    assert (p3.ssim_per_image(a, noisy) < 0.9).all()


def test_matching_undoes_a_permutation(p3):
    rng = np.random.default_rng(1)
    orig = rng.uniform(0, 1, size=(6, 3, 32, 32)).astype(np.float32)
    perm = rng.permutation(6)
    assert np.array_equal(p3.match_to_originals(orig, orig[perm]), orig)


def test_class_stats_subtract_exactly_the_client_records(p3):
    x, y = p3.make_standin(300, seed=3)
    stats = p3.ClassStats(x, y)
    xf = p3.to_float(x)
    clients = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    avg, mean_img = stats.pool_means(xf[clients], y[clients])
    mask = np.ones(len(y), bool)
    mask[clients] = False
    for c in range(10):
        assert avg[c] == pytest.approx(xf[mask & (y == c)].astype(np.float64).mean(axis=0), abs=1e-6)
    assert mean_img == pytest.approx(xf[mask].astype(np.float64).mean(axis=0), abs=1e-6)


def test_label_floor_is_the_class_average_of_the_pool(p3):
    cls = np.zeros((10, 3, 32, 32))
    for c in range(10):
        cls[c] = c / 10.0
    victim_y = np.array([2, 5, 5, 9])
    victim_x = np.stack([np.full((3, 32, 32), v) for v in (0.2, 0.5, 0.6, 0.9)]).astype(np.float32)
    fl = p3.label_floor(victim_x, victim_y, cls, np.full((3, 32, 32), 0.5))
    assert fl["per_image_mse"] == pytest.approx([0.0, 0.0, 0.01, 0.0], abs=1e-6)
    assert len(fl["per_image_psnr"]) == len(fl["per_image_ssim"]) == len(fl["matched_per_image_mse"]) == 4
    assert fl["matched_per_image_mse"][2] <= fl["per_image_mse"][2] + 1e-9    # matching can only help


# ---- views -------------------------------------------------------------------------------------------
def test_quantiser_matches_installed_flower(p3):
    flq = pytest.importorskip("flwr.common.secure_aggregation.quantization")
    arr = np.random.default_rng(0).normal(0, 0.1, 500)
    np.random.seed(7)
    ours = p3.quantize(arr, p3.CLIP_RANGE, p3.QUANT_RANGE)
    np.random.seed(7)
    theirs = flq.quantize([arr], p3.CLIP_RANGE, p3.QUANT_RANGE)[0]
    assert np.array_equal(ours, theirs)
    assert p3.dequantize(ours, p3.CLIP_RANGE, p3.QUANT_RANGE) == pytest.approx(flq.dequantize([theirs], p3.CLIP_RANGE, p3.QUANT_RANGE)[0])


def test_secagg_aggregate_error_is_inside_the_quantisation_bound(p3):
    rng = np.random.default_rng(2)
    vecs = [rng.normal(0, 0.2, 4000) for _ in range(4)]
    sizes = [8] * 4
    for mw in (p3.MAX_WEIGHT, p3.MAX_WEIGHT_SECONDARY):
        agg = p3.secagg_aggregate(vecs, sizes, max_weight=mw, rng_seed=3)
        step = 2 * p3.CLIP_RANGE / p3.QUANT_RANGE
        bound = step * p3.QUANT_RANGE / round(8 / mw * p3.QUANT_RANGE)    # n units over the total weight
        err = np.abs(agg - p3.exact_mean(vecs, sizes)).max()
        assert err <= 1.01 * bound
    # a coarser weight ratio gives a larger error, which is the reason both settings are reported
    e1000 = np.abs(p3.secagg_aggregate(vecs, sizes, max_weight=1000.0, rng_seed=3) - p3.exact_mean(vecs, sizes)).max()
    e16 = np.abs(p3.secagg_aggregate(vecs, sizes, max_weight=16.0, rng_seed=3) - p3.exact_mean(vecs, sizes)).max()
    assert e1000 > 5 * e16


def test_views_follow_the_v014_definitions(p3):
    init = _init(p3)
    cx, cy = _client_data()
    v = p3.build_views(init, cx, cy, seed=0)
    flat = lambda g: torch.cat([t.flatten() for t in g])   # noqa: E731
    # plaintext is the victim's own full batch gradient
    model = p3.SmallCNN()
    model.load_state_dict(init)
    own = torch.autograd.grad(torch.nn.functional.cross_entropy(model(torch.from_numpy(cx[0])), torch.from_numpy(cy[0]).long()), model.parameters())
    assert torch.allclose(flat(v["grads"]["plaintext"]), flat(own), atol=1e-5, rtol=1e-4)
    # the exact aggregate is the gradient of the mean loss over all 32 records (the joint oracle target)
    joint = torch.autograd.grad(torch.nn.functional.cross_entropy(model(torch.from_numpy(cx.reshape(-1, 3, 32, 32))),
                                                                  torch.from_numpy(cy.reshape(-1)).long()), model.parameters())
    assert torch.allclose(flat(v["grads"]["aggregate_exact"]), flat(joint), atol=1e-5, rtol=1e-4)
    # collusion returns the victim update up to quantisation error, and the finer weight ratio is closer
    pl = flat(v["grads"]["plaintext"])
    e_c = float((flat(v["grads"]["collusion"]) - pl).norm() / pl.norm())
    e_c16 = float((flat(v["grads"]["collusion_maxw16"]) - pl).norm() / pl.norm())
    assert e_c < 0.25 and e_c16 < e_c
    # the aggregate is far from the victim update
    assert float((flat(v["grads"]["aggregate"]) - pl).norm() / pl.norm()) > 0.2
    # label and truth bookkeeping
    assert len(v["labels"]["aggregate"]) == 32 and len(v["labels"]["collusion"]) == 8 and len(v["truth"]["aggregate_exact"]) == 32
    assert v["truth"]["plaintext"].shape == (8, 3, 32, 32)


# ---- tuning and verdict rules --------------------------------------------------------------------------
def test_tuning_selects_highest_mean_psnr_and_ties_go_to_the_earlier_entry(p3):
    def seed_result(best_b, best_r):
        r = {"seed": 9000, "breaching": {}, "repo": {}}
        for view in p3.VIEWS:
            r["breaching"][view] = {repr(tv): [10.0 + (3.0 if tv == best_b else 0.0)] * 8 for tv in p3.TV_GRID_BREACHING}
            r["repo"][view] = {repr(tv): [10.0] * 8 for tv in p3.TV_GRID_REPO}      # all tied
        r["repo"]["plaintext"][repr(best_r)] = [12.0] * 8
        return r

    sel = p3.select_tuning([seed_result(0.01, 0.1)])["selected"]
    assert sel["breaching"]["aggregate"] == 0.01 and sel["breaching"]["plaintext"] == 0.01
    assert sel["repo"]["plaintext"] == 0.1
    assert sel["repo"]["aggregate"] == p3.TV_GRID_REPO[0]          # tie, earlier entry


def test_verdict_rules(p3):
    ci = lambda m, lo, hi: {"mean": m, "lo": lo, "hi": hi}   # noqa: E731
    assert p3.verdict_lower(ci(3, 2, 4), 1.0)["verdict"] == "PASS"
    assert p3.verdict_lower(ci(1.5, 0.5, 2.5), 1.0)["verdict"] == "FAIL, underpowered"
    assert p3.verdict_lower(ci(0.2, -0.5, 0.9), 1.0)["verdict"] == "FAIL"
    assert p3.verdict_lower(ci(3, 2.0, 4), 2.0, inclusive=True)["verdict"] == "PASS"
    assert p3.verdict_upper(ci(-3, -4, -2), 1.0, -1.0) == {**p3.verdict_upper(ci(-3, -4, -2), 1.0, -1.0), "verdict": "PASS", "vacuous": True}
    assert p3.verdict_upper(ci(0.0, -0.5, 0.9), 1.0, -1.0)["vacuous"] is False
    assert p3.verdict_upper(ci(0.5, -0.2, 1.4), 1.0, -1.0)["verdict"] == "FAIL, underpowered"
    assert p3.verdict_upper(ci(3, 2, 4), 1.0, -1.0)["verdict"] == "FAIL"


def _fake_run(p3, seed, gains, quick=False, standin=False, rng=None):
    rng = rng or np.random.default_rng(seed)
    rows = {}
    keys = ["breaching/ctrl_plaintext_b1", "breaching/plaintext", "breaching/aggregate", "breaching/collusion", "breaching/collusion_maxw16",
            "breaching/aggregate_exact", "breaching/plaintext_ig_default", "repo/plaintext", "repo/aggregate", "repo/collusion", "repo/collusion_maxw16"]
    for k in keys:
        n = 4 if "ctrl" in k else 8
        floor_psnr = 13.0 + rng.normal(0, 1, n)
        att_psnr = floor_psnr + gains.get(k, 0.0) + rng.normal(0, 0.5, n)
        fl_mse = 10 ** (-floor_psnr / 10)
        rows[k] = {"attack": k.split("/")[0], "view": k.split("/")[1], "per_image_psnr": att_psnr.tolist(),
                   "per_image_mse": (10 ** (-att_psnr / 10)).tolist(), "per_image_ssim": (0.3 + 0.01 * gains.get(k, 0.0) + rng.normal(0, 0.02, n)).tolist(),
                   "floor": {"per_image_psnr": floor_psnr.tolist(), "per_image_mse": fl_mse.tolist(), "per_image_ssim": [0.3] * n,
                             "matched_per_image_mse": fl_mse.tolist(), "mean_image_per_image_mse": fl_mse.tolist()},
                   "n_images": n, "seconds": 1.0, "invalid_reconstructions": 0}
    return {"schema_version": 1, "seed": seed, "quick": quick, "synthetic_standin": standin, "data_source": "test", "device": "cpu",
            "model": {"state_sha256": "x"}, "rows": rows}


GAINS = {"breaching/ctrl_plaintext_b1": 5.0, "breaching/plaintext": 4.0, "breaching/collusion": 2.5, "breaching/aggregate": -3.0,
         "repo/plaintext": 3.0, "repo/collusion": 1.5, "repo/aggregate": -3.0}


def test_summary_gives_verdicts_for_20_real_size_seeds(p3):
    runs = [_fake_run(p3, s, GAINS) for s in range(20)]
    s = p3.summarise(runs)
    assert s["status"] == "verdicts computed"
    v = s["verdicts"]
    assert v["K0"]["verdict"] == "PASS" and v["R1"]["verdict"] == "PASS" and v["R3"]["verdict"] == "PASS"
    assert v["R2"]["verdict"] == "PASS" and v["R2"]["vacuous"] is True
    assert v["O1"]["verdict"] == "PASS" and v["O2"]["verdict"] == "PASS"
    assert v["B1c"]["ratio"] > 0 and len(v["B1c"]["ratio_ci95"]) == 2
    row = s["rows"]["breaching/plaintext"]
    assert row["n_images"] == 160 and row["gain_psnr_db"]["lo"] < row["gain_psnr_db"]["mean"] < row["gain_psnr_db"]["hi"]
    assert row["recognisable_mse_below_0_05"]["floor_rate"] >= 0.0
    assert s["rows"]["breaching/ctrl_plaintext_b1"]["n_images"] == 80


def test_summary_withholds_verdicts_for_wrong_size_quick_or_standin(p3):
    assert "no verdict" in p3.summarise([_fake_run(p3, s, GAINS) for s in range(5)])["status"]
    assert "quick run" in p3.summarise([_fake_run(p3, s, GAINS, quick=True) for s in range(20)])["status"]
    st = p3.summarise([_fake_run(p3, s, GAINS, standin=True) for s in range(20)])
    assert "stand-in" in st["status"] and "withheld" in st["verdicts"]


def test_a_failing_attack_is_reported_as_fail(p3):
    weak = dict(GAINS)
    weak["breaching/plaintext"] = 0.2          # attack barely above the floor
    s = p3.summarise([_fake_run(p3, k, weak) for k in range(20)])
    assert s["verdicts"]["R1"]["verdict"].startswith("FAIL")
    leak = dict(GAINS)
    leak["breaching/aggregate"] = 3.0          # aggregate attack clearly above the floor
    s = p3.summarise([_fake_run(p3, k, leak) for k in range(20)])
    assert s["verdicts"]["R2"]["verdict"] == "FAIL"


def test_bootstrap_is_deterministic(p3):
    runs = [_fake_run(p3, s, GAINS) for s in range(20)]
    assert p3.summarise(runs)["rows"] == p3.summarise(runs)["rows"]


# ---- resume and atomic writes ----------------------------------------------------------------------------
def test_atomic_write_leaves_no_temporary_file(p3, tmp_path):
    p3.atomic_write_json(tmp_path / "a.json", {"x": 1})
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["a.json"]


def test_run_resumes_and_skips_finished_seeds(p3, tmp_path, monkeypatch):
    calls = []

    def fake_run_seed(ctx, seed, tuning):
        calls.append(seed)
        return {"schema_version": 1, "seed": seed, "settings_hash": p3.settings_hash(ctx, tuning), "rows": {}, "quick": True,
                "synthetic_standin": True}, {}

    monkeypatch.setattr(p3, "run_seed", fake_run_seed)
    monkeypatch.setattr(p3, "train_or_load_model", lambda *a, **k: (_init(p3), {"state_sha256": "m"}))
    sel = {"selected": {"breaching": {v: 0.2 for v in p3.VIEWS}, "repo": {v: 0.5 for v in p3.VIEWS}}}
    monkeypatch.setattr(p3, "run_tuning", lambda ctx, out: sel)
    monkeypatch.setattr(p3, "write_summary", lambda out, n: 0)
    argv = ["--out", str(tmp_path), "--seeds", "3", "--quick", "--synthetic-standin", "--device", "cpu", "--data-dir", str(tmp_path / "d")]
    p3.main(argv)
    assert calls == [0, 1, 2]
    p3.main(argv)
    assert calls == [0, 1, 2]                                   # nothing redone
    (tmp_path / "seed1.json").unlink()
    (tmp_path / "seed2.json").write_text('{"schema_version": 1, "seed": 2, "rows"')    # truncated file from a killed run
    p3.main(argv)
    assert calls == [0, 1, 2, 1, 2]                             # only the missing and the corrupt seed
    # a seed written under other settings is refused and not overwritten
    other = json.loads((tmp_path / "seed0.json").read_text())
    other["settings_hash"] = "different"
    (tmp_path / "seed0.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit):
        p3.main(argv)
    assert json.loads((tmp_path / "seed0.json").read_text())["settings_hash"] == "different"


# ---- end to end schema (needs breaching) ------------------------------------------------------------------
@pytest.mark.slow
def test_one_seed_end_to_end_schema(p3, tmp_path):
    pytest.importorskip("breaching")
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    x, y = p3.make_standin(600, seed=1)
    split = p3.Split(len(y), 100, 50)

    class Args:
        quick = True
        skip_library_default = False

    ctx = p3.Context(Args, x, y, "SYNTHETIC_STANDIN", split, _init(p3), {"state_sha256": "m", "held_out_accuracy": 0.1}, "cpu")
    ctx.breaching_iters = ctx.repo_iters = ctx.library_iters = 2
    sel = {"selected": {"breaching": {v: 0.2 for v in p3.VIEWS}, "repo": {v: 0.5 for v in p3.VIEWS}}}
    out, images = p3.run_seed(ctx, 0, sel)
    json.dumps(out)                                              # must be serialisable
    assert out["schema_version"] == 1 and out["synthetic_standin"] is True and out["seed"] == 0
    expected = {"breaching/ctrl_plaintext_b1", "breaching/plaintext", "breaching/aggregate", "breaching/collusion", "breaching/collusion_maxw16",
                "breaching/aggregate_exact", "breaching/plaintext_ig_default", "repo/plaintext", "repo/aggregate", "repo/collusion",
                "repo/collusion_maxw16"}
    assert set(out["rows"]) == expected
    for key, row in out["rows"].items():
        n = 4 if "ctrl" in key else 8
        assert row["n_images"] == n
        for field in ("per_image_mse", "per_image_psnr", "per_image_ssim"):
            assert len(row[field]) == n and len(row["floor"][field]) == n          # the floor is attached to every row
        assert row["invalid_reconstructions"] in (0, 1) or row["invalid_reconstructions"] <= 4
    assert images["breaching/aggregate"].shape == (32, 3, 32, 32)
    # the floor of the control row is the first four floor images
    assert out["rows"]["breaching/ctrl_plaintext_b1"]["floor"]["per_image_mse"] == out["floor"]["per_image_mse"][:4]
