"""run_view_comparison: pooled Wilson rates and the guard that protects the saved v0.14.2 summary."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("flwr")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_view_comparison as rv  # noqa: E402


def test_wilson_pooled_counts_every_victim_image_of_every_seed():
    mean, lo, hi = rv._wilson_pooled([7 / 8, 7 / 8, 5 / 8, 6 / 8, 5 / 8, 5 / 8, 6 / 8, 5 / 8, 5 / 8, 3 / 8])
    assert mean == 0.675                                     # 54 of 80
    assert (lo, hi) == (pytest.approx(0.566, abs=1e-3), pytest.approx(0.768, abs=1e-3))


def test_wilson_pooled_stays_inside_zero_and_one_at_the_extremes():
    _, lo0, hi0 = rv._wilson_pooled(np.zeros(10))
    _, lo1, hi1 = rv._wilson_pooled(np.ones(10))
    assert lo0 == 0.0 and 0.0 < hi0 < 0.06                    # 0 of 80
    assert hi1 == 1.0 and 0.94 < lo1 < 1.0                    # 80 of 80


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    saved = tmp_path / "saved_v0142"
    saved.mkdir()
    (saved / "summary.json").write_text('{"keep": "me"}')
    monkeypatch.setattr(rv, "V0142_DIR", saved)
    monkeypatch.setattr(rv, "NEW_SEEDS_DIR", tmp_path / "new")
    monkeypatch.setattr(rv, "single_step_epsilon", lambda sigma, delta=1e-5: 1.0)
    calls = []

    def fake_one_seed(seed, sigmas, max_weights, iterations, legacy_seeds=True):
        calls.append((seed, legacy_seeds))
        return {"seed": seed, "floor": {"mean_image_mse": 0.07, "random_digits_mse": 0.09},
                "label_floor": {"class_average_mse": 0.046, "class_average_rate_mse_below_0_05": 0.675,
                                "matched_mse": 0.045, "matched_rate_mse_below_0_05": 0.675, "identification_rate": 0.1},
                "views": {"plaintext_individual": {"victim_mse": 0.01 + 0.001 * seed, "victim_rate_mse_below_0_05": 1.0}}}

    monkeypatch.setattr(rv, "one_seed", fake_one_seed)
    return saved, tmp_path / "new", calls


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_view_comparison.py", "--seeds", "2", *args])
    rv.main()


def test_default_run_refuses_to_replace_the_saved_summary(sandbox, monkeypatch):
    saved, _, calls = sandbox
    with pytest.raises(SystemExit) as e:
        run(monkeypatch)
    assert e.value.code == 2
    assert json.loads((saved / "summary.json").read_text()) == {"keep": "me"}
    assert calls == []


def test_overwrite_flag_allows_replacing_the_saved_summary(sandbox, monkeypatch):
    saved, _, calls = sandbox
    run(monkeypatch, "--overwrite-summary")
    assert calls == [(0, True), (1, True)]
    assert json.loads((saved / "summary.json").read_text())["legacy_seeds"] is True


def test_new_seed_scheme_never_writes_into_the_saved_directory(sandbox, monkeypatch):
    saved, new, calls = sandbox
    with pytest.raises(SystemExit):
        run(monkeypatch, "--no-legacy-seeds", "--output", str(saved), "--overwrite-summary")
    assert calls == []
    run(monkeypatch, "--no-legacy-seeds")                        # default output is its own directory
    assert calls == [(0, False), (1, False)]
    assert json.loads((new / "summary.json").read_text())["legacy_seeds"] is False
    assert json.loads((saved / "summary.json").read_text()) == {"keep": "me"}


def test_explicit_output_directory_is_not_guarded(sandbox, monkeypatch, tmp_path):
    out = tmp_path / "elsewhere"
    run(monkeypatch, "--output", str(out))
    assert (out / "summary.json").is_file()
