"""Fashion-MNIST loader and CNN ordering study helpers (v0.15, row X1). No network is used.

The loader tests use files built in the test. Real data is used only by the one test that skips
when the verified files are absent (the hosted workflow downloads them).
"""
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from feddisclosure.attacks.view_study import flatten, secagg_aggregate, unflatten
from feddisclosure.evaluation import fashion_mnist_study as fs
from feddisclosure.evaluation import scaling_study as ss
from feddisclosure.training import fashion_mnist as fm


def _idx(images: np.ndarray, labels: np.ndarray, *, side=28, img_magic=2051, lab_magic=2049):
    n = len(labels)
    head = b"".join(v.to_bytes(4, "big") for v in (img_magic, n, side, side))
    imgs = gzip.compress(head + images.astype(np.uint8).tobytes())
    labs = gzip.compress(lab_magic.to_bytes(4, "big") + n.to_bytes(4, "big") + labels.astype(np.uint8).tobytes())
    return imgs, labs


def test_parse_idx_decodes_pixels_to_unit_range_and_labels():
    rng = np.random.default_rng(0)
    pix, lab = rng.integers(0, 256, size=(5, 784)), np.array([0, 9, 3, 3, 1])
    x, y = fm.parse_idx(*_idx(pix, lab))
    assert x.shape == (5, 784) and x.dtype == np.float32 and y.tolist() == lab.tolist()
    assert np.allclose(x, pix / 255.0) and x.min() >= 0 and x.max() <= 1


@pytest.mark.parametrize("kwargs,labels", [({"img_magic": 2049}, [1, 2]), ({"lab_magic": 2051}, [1, 2]),
                                            ({"side": 8}, [1, 2]), ({}, [1, 10])])
def test_parse_idx_refuses_malformed_files(kwargs, labels):
    side = kwargs.get("side", 28)
    pix = np.zeros((2, side * side), dtype=np.uint8)
    with pytest.raises(ValueError):
        fm.parse_idx(*_idx(pix, np.array(labels), **kwargs))


def test_missing_or_altered_files_are_refused_and_never_replaced_by_other_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        fm.load_fashion_mnist(tmp_path)
    for name in fm.SHA256:
        (tmp_path / name).write_bytes(b"not the real file")
    assert not fm.files_verified(tmp_path)
    with pytest.raises(FileNotFoundError):
        fm.load_fashion_mnist(tmp_path)


def test_download_keeps_a_file_only_if_its_sha256_matches(tmp_path, monkeypatch):
    good = {name: name.encode() * 10 for name in fm.SHA256}

    class Reply:
        def __init__(self, data): self.data = data
        def read(self): return self.data
        def __enter__(self): return self
        def __exit__(self, *a): return False

    served = {"data": dict(good)}
    monkeypatch.setattr(fm.urllib.request, "urlopen", lambda url, timeout=0: Reply(served["data"][url.rsplit("/", 1)[1]]))
    monkeypatch.setattr(fm, "SHA256", {n: hashlib.sha256(d).hexdigest() for n, d in good.items()})
    fm.fetch_fashion_mnist(tmp_path / "ok")
    assert fm.files_verified(tmp_path / "ok") and (tmp_path / "ok" / "SHA256SUMS").is_file()
    served["data"] = {n: b"tampered" for n in good}
    with pytest.raises(ValueError, match="pinned SHA-256"):
        fm.fetch_fashion_mnist(tmp_path / "bad")
    assert not list((tmp_path / "bad").glob("*.gz")) and not list((tmp_path / "bad").glob("*.part"))


def test_real_data_if_present_is_the_60000_image_train_split():
    if not fm.files_verified(fm.LOCAL_DIR):
        pytest.skip("verified Fashion-MNIST files are not present (the hosted workflow downloads them)")
    x, y, provenance = fm.load_fashion_mnist()
    assert x.shape == (60000, 784) and np.bincount(y).tolist() == [6000] * 10 and "MIT" in provenance


def test_standin_is_deterministic_labelled_and_shaped_like_the_real_data():
    x, y, label = fm.synthetic_standin(per_class=20, seed=1)
    x2, y2, _ = fm.synthetic_standin(per_class=20, seed=1)
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    assert not np.array_equal(x, fm.synthetic_standin(per_class=20, seed=2)[0])
    assert x.shape == (200, 784) and np.bincount(y).tolist() == [20] * 10 and 0 <= x.min() and x.max() <= 1
    assert "SYNTHETIC" in label and "unit tests" in label


def test_cnn_is_small_and_takes_flat_images():
    m = fs.SmallCNN()
    assert m(torch.zeros(5, 784)).shape == (5, 10)
    assert 5_000 < sum(p.numel() for p in m.parameters()) < 15_000


def test_one_step_delta_over_lr_is_the_batch_gradient_the_attack_assumes():
    x, y, _ = fm.synthetic_standin(per_class=4, seed=0)
    torch.manual_seed(0)
    init = {k: v.clone() for k, v in fs.SmallCNN().state_dict().items()}
    after = fs.client_step(init, x[:8], y[:8])
    model = fs.SmallCNN()
    model.load_state_dict(init)
    grads = torch.autograd.grad(torch.nn.functional.cross_entropy(model(torch.from_numpy(x[:8])), torch.from_numpy(y[:8])),
                                tuple(model.parameters()))
    for g, name in zip(grads, dict(model.named_parameters())):
        assert torch.allclose((init[name] - after[name]) / fs.LR, g, atol=1e-6)


def test_secagg_collusion_recovers_the_cnn_victim_update_up_to_the_quantiser_error():
    """The quantiser error scales with max_weight. At 1000 (the repo default) it is a visible share of a
    small CNN update, at 16 it is negligible. The study reports both."""
    pytest.importorskip("flwr")      # the SecAgg+ quantiser comes from Flower
    x, y, _ = fm.synthetic_standin(per_class=4, seed=0)
    torch.manual_seed(1)
    init = {k: v.clone() for k, v in fs.SmallCNN().state_dict().items()}
    states = [fs.client_step(init, x[8 * i:8 * i + 8], y[8 * i:8 * i + 8]) for i in range(4)]
    vecs = [flatten(s) for s in states]
    delta_rms = float(np.sqrt(((flatten(init) - vecs[0]) ** 2).mean()))
    rms = {}
    for mw in (1000.0, 16.0):
        agg = flatten(unflatten(secagg_aggregate(vecs, [8] * 4, max_weight=mw, rng_seed=1), init))
        rms[mw] = float(np.sqrt(((ss.collusion_view(agg, vecs, [8] * 4) - vecs[0]) ** 2).mean()))
    assert rms[1000.0] < delta_rms / 2          # the update is still recoverable, with visible noise
    assert rms[16.0] < delta_rms / 100          # a finer weight ratio makes the error negligible
    assert rms[16.0] < rms[1000.0] / 10


def test_victim_scoring_uses_optimal_matching_and_the_victim_rows():
    rng = np.random.default_rng(0)
    originals = rng.random((32, 784)).astype(np.float32)
    assert np.allclose(fs.victim_per_image_mse(originals, originals[rng.permutation(32)], 8), 0.0, atol=1e-7)
    assert np.allclose(fs.victim_per_image_mse(originals, originals[:8][::-1].copy(), 8), 0.0, atol=1e-7)
    assert fs.victim_per_image_mse(originals, rng.random((32, 784)), 8).shape == (8,)


def test_run_seed_is_deterministic_and_reports_all_views_on_the_standin():
    pytest.importorskip("flwr")      # the SecAgg+ quantiser comes from Flower
    x, y, _ = fm.synthetic_standin(per_class=30, seed=0)
    cfg = {"iterations": 2, "attack_lr": 0.1, "tv_weight": 0.0005}
    settings = {"individual": cfg, "collusion": cfg, "aggregate": cfg}
    a = fs.run_seed(x, y, 0, settings=settings)
    assert a == fs.run_seed(x, y, 0, settings=settings)
    assert set(a["views"]) == set(fs.view_names()) == {"plaintext_individual", "secagg_collusion_maxw1000",
                                                       "secagg_aggregate_maxw1000"}
    assert all(len(v) == 8 for v in a["views"].values()) and len(a["label_floor_per_image_mse"]) == 8
    summary = ss.summarise([a, fs.run_seed(x, y, 1, settings=settings)])
    assert set(summary["views"]) == set(a["views"])


def test_tuning_refuses_evaluation_seeds():
    x, y, _ = fm.synthetic_standin(per_class=30, seed=0)
    with pytest.raises(ValueError, match="tuning seeds"):
        fs.tune_one((x, y, "individual", {"iterations": 1}, [0, 1]))


def test_secondary_max_weight_views_are_added_on_request_and_only_restricts_views():
    pytest.importorskip("flwr")      # the SecAgg+ quantiser comes from Flower
    x, y, _ = fm.synthetic_standin(per_class=30, seed=0)
    cfg = {"iterations": 1, "attack_lr": 0.1, "tv_weight": 0.0005}
    settings = {"individual": cfg, "collusion": cfg, "aggregate": cfg}
    r = fs.run_seed(x, y, 0, settings=settings, max_weights=(1000.0, 16.0))
    assert set(r["views"]) == set(fs.view_names((1000.0, 16.0))) and len(r["views"]) == 5
    only = fs.run_seed(x, y, 0, settings=settings, only=("plaintext_individual",))
    assert set(only["views"]) == {"plaintext_individual"}
    assert only["views"]["plaintext_individual"] == r["views"]["plaintext_individual"]


def test_best_setting_is_lowest_mse():
    rows = [{"mean_mse": 0.3, "setting": "a"}, {"mean_mse": 0.1, "setting": "b"}, {"mean_mse": 0.1, "setting": "c"}]
    assert fs.best_setting(rows)["setting"] == "b"


def _summary(floor, plain, col, agg, n=240):
    def row(rate):
        return {"rate": rate, "wilson": [max(0, rate - 0.05), min(1, rate + 0.05)]}
    return {"label_floor": row(floor), "views": {"plaintext_individual": row(plain), "secagg_collusion_maxw1000": row(col),
                                                 "secagg_aggregate_maxw1000": row(agg)}}


def test_x1_needs_all_three_parts_and_the_thresholds_are_the_registered_ones():
    assert (fs.FLOOR_PLUS_PLAINTEXT, fs.FLOOR_PLUS_COLLUSION, fs.FLOOR_PLUS_AGGREGATE) == (0.10, 0.10, 0.05)
    assert fs.check_x1(_summary(0.5, 0.8, 0.7, 0.4))["pass"] is True
    assert fs.check_x1(_summary(0.5, 0.6, 0.6, 0.55))["pass"] is True              # exactly on all three limits
    assert fs.check_x1(_summary(0.5, 0.59, 0.8, 0.4))["parts"]["plaintext_at_least_floor_plus_10"] is False
    assert fs.check_x1(_summary(0.5, 0.8, 0.59, 0.4))["parts"]["collusion_at_least_floor_plus_10"] is False
    r = fs.check_x1(_summary(0.5, 0.8, 0.8, 0.56))
    assert r["pass"] is False and r["parts"]["aggregate_within_floor_plus_5"] is False


def test_saved_fashion_verdict_matches_a_recomputation_from_the_saved_summary():
    import json
    root = Path(__file__).resolve().parents[2] / "reports/results/privacy/v0150"
    if not (root / "fashion_mnist/summary.json").is_file():
        pytest.skip("Fashion-MNIST results are not in this checkout")
    saved = json.loads((root / "fashion_mnist/summary.json").read_text())
    check = json.loads((root / "prereg_check_fashion.json").read_text())
    for max_weight, key in ((1000.0, "X1"), (16.0, "X1_secondary_max_weight_16")):
        fresh = fs.check_x1(saved["summary"], max_weight)
        assert fresh["pass"] is saved[key]["pass"] is check[key]["pass"]
        assert fresh["parts"] == check[key]["parts"]
    # The point estimates behind every part respect the registered thresholds.
    obs = check["X1"]["observed"]
    assert check["X1"]["pass"] == (obs["plaintext_individual"]["rate_minus_floor"] >= 0.10 - 1e-9
                                   and obs["secagg_collusion_maxw1000"]["rate_minus_floor"] >= 0.10 - 1e-9
                                   and obs["secagg_aggregate_maxw1000"]["rate_minus_floor"] <= 0.05 + 1e-9)
