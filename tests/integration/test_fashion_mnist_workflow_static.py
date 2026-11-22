"""Static checks of the Fashion-MNIST workflow and runner. NOT a hosted run and no data download."""
from pathlib import Path
import re
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_fashion_mnist_study as runner  # noqa: E402
from feddisclosure.training import fashion_mnist as fm  # noqa: E402

WORKFLOW = ROOT / ".github/workflows/fashion-mnist-study.yml"
DOC = yaml.safe_load(WORKFLOW.read_text())


def test_workflow_is_dispatch_only_with_minimal_permissions_and_pinned_versions():
    triggers = DOC.get(True, DOC.get("on"))
    assert list(triggers) == ["workflow_dispatch"] and DOC["permissions"] == {"contents": "read"}
    assert set(triggers["workflow_dispatch"]["inputs"]) == {"seeds"}
    job = DOC["jobs"]["fashion-mnist"]
    assert job["runs-on"] == "ubuntu-latest"
    text = WORKFLOW.read_text()
    assert "'torch==2.5.1'" in text and "'flwr==1.27.0'" in text
    assert all(re.fullmatch(r"[\w./-]+@v\d+", u) for u in re.findall(r"uses: (\S+)", text))


def test_workflow_downloads_then_tunes_then_evaluates_and_keeps_results_on_failure():
    steps = DOC["jobs"]["fashion-mnist"]["steps"]
    order = [i for key in ("fetch_fashion_mnist", "--tune", "--seeds") for i, s in enumerate(steps)
             if key in s.get("run", "")]
    assert len(order) == 3 and order == sorted(order)
    assert steps[-1]["if"] == "always()" and steps[-1]["uses"].startswith("actions/upload-artifact")
    assert not any("standin" in s.get("run", "") for s in steps)


def test_runner_never_touches_the_stand_in_data():
    assert "standin" not in (ROOT / "scripts/run_fashion_mnist_study.py").read_text()


def test_evaluation_needs_tuning_results_and_enough_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_fashion_mnist", lambda download=False: fm.synthetic_standin(per_class=5)[:2] + ("test",))
    with pytest.raises(SystemExit):
        _run(["--seeds", "10", "--output", str(tmp_path)], monkeypatch)
    with pytest.raises(SystemExit):
        _run(["--seeds", "5", "--output", str(tmp_path)], monkeypatch)


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_fashion_mnist_study.py", *argv])
    runner.main()


def test_runner_stops_when_the_verified_files_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "LOCAL_DIR", tmp_path / "nothing")
    monkeypatch.setattr(runner, "load_fashion_mnist", lambda download=False: fm.load_fashion_mnist(tmp_path / "nothing", download=download))
    with pytest.raises(FileNotFoundError):
        _run(["--seeds", "10", "--output", str(tmp_path / "out")], monkeypatch)


def test_tuning_runs_three_stages_per_view_and_reuses_cached_rows(monkeypatch):
    x, y, _ = fm.synthetic_standin(per_class=30, seed=0)
    monkeypatch.setattr(runner, "STAGE1", {"iterations": 1})
    monkeypatch.setattr(runner, "LR_GRID", (0.1, 0.2))
    monkeypatch.setattr(runner, "TV_GRID", (0.001,))
    monkeypatch.setattr(runner, "AGG_LR_GRID", (0.1,))
    monkeypatch.setattr(runner, "AGG_TV_GRID", (0.001,))
    monkeypatch.setattr(runner, "ITER_GRID", (1, 2))
    monkeypatch.setattr(runner, "TUNING_SEEDS", [9000])
    calls = []
    real = runner.fs.tune_one

    def fake(args):
        calls.append(args[2])
        view, setting = args[2], args[3]
        return {"view": view, "setting": setting, "seeds": args[4], "mean_mse": 1.0 / (1 + setting["iterations"] + 10 * setting["tv_weight"]),
                "rate": 0.0}
    monkeypatch.setattr(runner.fs, "tune_one", fake)
    cache = {}
    first = runner.tune(x, y, 1, cache)
    n_first = len(calls)
    assert n_first == len(cache) and set(first["chosen"]) == {"individual", "collusion", "aggregate"}
    # The best setting wins and the stage 3 values were tried at the best iterations and step size.
    assert all(first["chosen"][v]["iterations"] == 2 for v in first["chosen"])
    tried = {r["setting"]["tv_weight"] for r in first["rows"]["individual"]}
    assert set(runner.TV_EXTENSIONS[0]) <= tried
    runner.tune(x, y, 1, cache)
    assert len(calls) == n_first          # nothing is recomputed on the second call
    assert runner.fs.tune_one is fake and real is not fake


def test_tv_search_stops_when_the_best_setting_is_not_at_the_edge(monkeypatch):
    x, y, _ = fm.synthetic_standin(per_class=30, seed=0)
    monkeypatch.setattr(runner, "STAGE1", {"iterations": 1})
    monkeypatch.setattr(runner, "LR_GRID", (0.1,))
    monkeypatch.setattr(runner, "TV_GRID", (0.001, 0.002))
    monkeypatch.setattr(runner, "AGG_LR_GRID", (0.1,))
    monkeypatch.setattr(runner, "AGG_TV_GRID", (0.001, 0.002))
    monkeypatch.setattr(runner, "ITER_GRID", (1,))
    monkeypatch.setattr(runner, "TUNING_SEEDS", [9000])
    monkeypatch.setattr(runner.fs, "tune_one", lambda a: {"view": a[2], "setting": a[3], "seeds": a[4],
                                                          "mean_mse": a[3]["tv_weight"], "rate": 0.0})   # smaller tv is better
    out = runner.tune(x, y, 1, {})
    tried = {r["setting"]["tv_weight"] for rows in out["rows"].values() for r in rows}
    assert tried == {0.001, 0.002} and all(c["tv_weight"] == 0.001 for c in out["chosen"].values())
