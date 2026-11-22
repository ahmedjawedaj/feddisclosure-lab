"""Label-conditional floor and identification metric, checked on toy data with exact answers.

Images are constant vectors of 64 equal pixels, so every squared error is easy to compute by hand.
All offsets are powers of two, so the arithmetic is exact in floating point.
"""
import numpy as np
import pytest

pytest.importorskip("flwr")
from feddisclosure.attacks.view_study import class_average_images, identification_rate, label_floor  # noqa: E402

D = 64


def img(v):
    return np.full(D, v, dtype=np.float64)


def toy_pool():
    """Class c has two public records, base_c plus and minus 1/16, so the class average is base_c."""
    base = np.arange(10) / 16
    px = np.stack([img(b + s) for b in base for s in (0.0625, -0.0625)])
    py = np.repeat(np.arange(10), 2)
    return base, px, py


def test_class_average_is_the_mean_of_each_class():
    base, px, py = toy_pool()
    cm = class_average_images(px, py)
    assert np.allclose(cm, np.stack([img(b) for b in base]), atol=0, rtol=0)


def test_label_floor_exact_mse_and_rate():
    base, px, py = toy_pool()
    # Both victims are class 0. Offsets 1/8 and 1/4 give per-image MSE 1/64 (below 0.05) and 1/16 (above).
    victims = np.stack([img(0.125), img(0.25)])
    out = label_floor(victims, np.array([0, 0]), px, py, u=2, repeats=5)
    assert out["per_image_mse"] == [1 / 64, 1 / 16]
    assert out["class_average_mse"] == pytest.approx((1 / 64 + 1 / 16) / 2)
    assert out["class_average_rate_mse_below_0_05"] == 0.5
    assert out["matched_mse"] == pytest.approx(out["class_average_mse"])      # same class image, matching cannot help


def test_matched_scoring_is_never_worse_than_direct_scoring():
    base, px, py = toy_pool()
    victims = np.stack([img(3 / 16 + 0.125), img(4 / 16 + 0.125)])           # near each other's class image
    out = label_floor(victims, np.array([3, 4]), px, py, u=2, repeats=5)
    assert out["matched_mse"] <= out["class_average_mse"] + 1e-15


def test_identification_is_certain_when_the_record_is_closest_and_zero_when_it_is_farthest():
    base, px, py = toy_pool()
    near = img(3 / 16 + 1 / 32)           # closer to the class average than both candidates at 1/16
    far = img(3 / 16 + 1 / 4)             # farther than both candidates
    recon = img(3 / 16)
    hit = identification_rate(near[None], recon[None], np.array([3]), px, py, u=2, repeats=20)
    miss = identification_rate(far[None], recon[None], np.array([3]), px, py, u=2, repeats=20)
    assert (hit, miss) == (1.0, 0.0)


def test_identification_splits_credit_on_exact_ties():
    base, px, py = toy_pool()
    victim = img(3 / 16 - 1 / 16)         # identical to one public record and equally far as the other
    out = identification_rate(victim[None], img(3 / 16)[None], np.array([3]), px, py, u=2, repeats=20)
    assert out == 0.5


def test_identification_chance_level_is_one_over_u_for_exchangeable_records():
    rng = np.random.default_rng(0)
    pool = rng.normal(0.5, 0.2, size=(400, D))
    victims = rng.normal(0.5, 0.2, size=(3000, D))
    labels = np.zeros(3000, dtype=int)
    independent = rng.normal(0.5, 0.2, size=(400, D))                          # reconstruction made without the pool
    recon = np.repeat(independent.mean(axis=0, keepdims=True), 3000, axis=0)
    rate = identification_rate(victims, recon, labels, pool, np.zeros(400, dtype=int), u=5, repeats=5,
                               rng=np.random.default_rng(1))
    assert abs(rate - 0.2) < 0.03


def test_missing_class_and_small_pool_fail_closed():
    base, px, py = toy_pool()
    with pytest.raises(ValueError, match="no record of class 9"):
        class_average_images(px[py != 9], py[py != 9])
    with pytest.raises(ValueError, match="too few"):
        identification_rate(img(0.1)[None], img(0.0)[None], np.array([0]), px, py, u=5, repeats=1)


# The next tests tie compute_label_floor to the data selection of the saved v0.14.2 runs.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import compute_label_floor as clf  # noqa: E402
from feddisclosure.attacks.view_seed import select_view_records  # noqa: E402
from feddisclosure.training.datasets import _source  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def digits():
    x, y, _ = _source("sklearn_digits", ROOT / "data/training/_lira_cache")
    return x, y


def test_floor_for_seed_uses_the_pool_of_select_view_records_and_never_a_client_record(digits, monkeypatch):
    x, y = digits
    seen = {}
    real = clf.label_floor

    def spy(victim_x, victim_y, pool_x, pool_y, **kw):
        seen.update(victim_x=victim_x, pool_x=pool_x, pool_y=pool_y)
        return real(victim_x, victim_y, pool_x, pool_y, **kw)

    monkeypatch.setattr(clf, "label_floor", spy)
    out = clf.floor_for_seed(x, y, 3, repeats=2)
    sel = select_view_records(len(y), clf.N, clf.B, 3)
    assert np.array_equal(seen["pool_x"], x[sel.pool_idx]) and np.array_equal(seen["pool_y"], y[sel.pool_idx])
    assert np.array_equal(seen["victim_x"], x[sel.idx[:clf.B]])
    assert not set(sel.pool_idx.tolist()) & set(sel.idx.tolist())              # disjoint from all 32 client records
    assert len(sel.idx) == 32 and len(sel.pool_idx) == len(y) - 32 == out["public_pool_size"]


def test_floor_for_seed_matches_an_independent_recomputation(digits):
    x, y = digits
    seed = 4
    perm = np.random.default_rng(1000 + seed).permutation(len(y))          # the v0.14.2 selection, written out again
    victim, pool = perm[:8], perm[32:]
    means = {c: x[pool][y[pool] == c].mean(axis=0) for c in range(10)}
    per_image = [float(((x[i] - means[y[i]]) ** 2).mean()) for i in victim]
    out = clf.floor_for_seed(x, y, seed, repeats=2)
    assert out["class_average_mse"] == pytest.approx(np.mean(per_image), rel=1e-6)       # images are float32
    assert out["class_average_rate_mse_below_0_05"] == np.mean(np.array(per_image) < 0.05)


def test_pool_size_sensitivity_full_pool_row_equals_the_headline_floor(digits):
    x, y = digits
    sens = clf.pool_size_sensitivity(x, y, seeds=2, sizes=(100,), subsets=3)
    per_seed = [clf.floor_for_seed(x, y, s, repeats=1) for s in range(2)]
    full = sens["rows"][-1]
    assert full["pool_size"] == "full"
    assert full["mse"] == pytest.approx(np.mean([r["class_average_mse"] for r in per_seed]), rel=1e-6)
    assert sens["rows"][0]["mse"] > full["mse"]                                # fewer public images give a worse floor


def test_subset_floor_falls_back_to_the_subset_mean_for_a_missing_class():
    sub_x = np.stack([img(0.25), img(0.75)])
    sub_y = np.array([0, 1])
    r = clf.subset_floor(np.stack([img(0.5)]), np.array([7]), sub_x, sub_y)    # class 7 is absent, mean image is 0.5
    assert r == {"mse": 0.0, "rate": 1.0, "fallback": True}
    full_x = np.stack([img(c / 16) for c in range(10)])
    r = clf.subset_floor(np.stack([img(0.25)]), np.array([3]), full_x, np.arange(10))   # class 3 image is 3/16
    assert r == {"mse": (1 / 16) ** 2, "rate": 1.0, "fallback": False}
