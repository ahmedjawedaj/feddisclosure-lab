"""Helper units of scripts/study_reply_order.py on tiny synthetic input."""
import importlib.util
import json
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("flwr")

SPEC = importlib.util.spec_from_file_location("study_reply_order", Path(__file__).parents[2] / "scripts/study_reply_order.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_all_orders_counts_and_rejects_single_client():
    assert len(study.all_orders(4)) == 24 and len(set(study.all_orders(4))) == 24
    with pytest.raises(ValueError):
        study.all_orders(1)


def test_gate_uses_the_unchanged_1e_5_rule():
    ref = {"w": torch.zeros(4)}
    ok, worst = study.gate({"w": torch.full((4,), 9e-6)}, ref)
    assert ok and worst == pytest.approx(9e-6)
    ok, _ = study.gate({"w": torch.full((4,), 2e-5)}, ref)
    assert not ok
    with pytest.raises(ValueError):
        study.gate({"x": torch.zeros(1)}, ref)


def test_pass_rate_and_summarize_orders():
    recs = [{"passes": True, "max_abs_diff": 1e-8}, {"passes": False, "max_abs_diff": 3e-5},
            {"passes": True, "max_abs_diff": 2e-8}, {"passes": True, "max_abs_diff": 0.0}]
    s = study.summarize_orders(recs)
    assert s["pass_rate"] == 0.75 and s["orders_passing"] == 3 and s["max_abs_diff_worst"] == 3e-5
    with pytest.raises(ValueError):
        study.pass_rate([])


def test_p3_verdict_pass_fail_and_minimum_seeds():
    base = [0.10, 0.12, 0.08, 0.15, 0.09]
    same = {f"o{i}": [v + 1e-9 * i for v in base] for i in range(4)}
    assert study.p3_verdict(same)["p3_pass"] is True
    shifted = {f"o{i}": [v + 0.2 * i for v in base] for i in range(4)}
    assert study.p3_verdict(shifted)["p3_pass"] is False
    with pytest.raises(ValueError):
        study.p3_verdict({"a": base[:4], "b": base[:4]})
    with pytest.raises(ValueError):
        study.p3_verdict({"a": base, "b": base[:5] + [0.1]})


def _tiny_run(tmp_path, clients=3, per_client=20):
    rng = np.random.default_rng(0)
    parts = []
    for c in range(clients):
        d = tmp_path / f"institution-{c:02d}"
        d.mkdir()
        x = rng.random((per_client + 4 * c, 64)).astype("float32")
        y = rng.integers(0, 10, len(x))
        np.savez(d / "train.npz", x=x, y=y)
        parts.append({"file": f"institution-{c:02d}/train.npz"})
    manifest = {"num_features": 64, "num_classes": 10, "client_partitions": parts}
    cfg = {"seed": 5, "rounds": 2, "local_epochs": 1, "batch_size": 8, "learning_rate": 0.2}
    return study.Replay(cfg, manifest, tmp_path)


def test_replay_deterministic_trajectory_equals_reference_for_every_order(tmp_path):
    rp = _tiny_run(tmp_path)
    ref, g = [], rp.init
    for _ in range(rp.rounds):
        g = study.aggregate_reference(rp.train_all(g), rp.sizes)
        ref.append(g)
    for order in study.all_orders(rp.n):
        traj = rp.run(study.aggregate_deterministic, [order] * rp.rounds)
        assert study.state_digest(traj[-1]) == study.state_digest(ref[-1])


def test_replay_cache_does_not_change_results_and_counts_trainings(tmp_path):
    rp = _tiny_run(tmp_path)
    first = rp.run(study.aggregate_upstream, [(0, 1, 2)] * 2)
    count = rp.trainings
    second = rp.run(study.aggregate_upstream, [(0, 1, 2)] * 2)
    assert rp.trainings == count
    assert study.state_digest(first[-1]) == study.state_digest(second[-1])


def test_upstream_aggregate_depends_on_order_but_deterministic_does_not():
    rng = np.random.default_rng(1)
    states = [{"w": torch.from_numpy(rng.normal(size=60).astype("float32"))} for _ in range(4)]
    sizes = [96, 100, 90, 70]
    up = {study.state_digest(study.aggregate_upstream(states, sizes, o)) for o in itertools.permutations(range(4))}
    de = {study.state_digest(study.aggregate_deterministic(states, sizes, o)) for o in itertools.permutations(range(4))}
    assert len(up) > 1 and len(de) == 1


# ---------------------------------------------------------------- annotations live in code, rerun reproduces the committed file
def _saved_inputs_present():
    out = study.OUT
    names = list(study.NATIVE_RUNS) + list(study.REPLAY_ONLY)
    needed = [out / f"replay_{n}.json" for n in names] + [out / "p3_victim_mse.json"]
    needed += [study.LIVE_DIR / r / "flower_parity.json" for r in study.LIVE_RUNS] + [study.PARITY_JSON]
    return [str(p) for p in needed if not p.is_file()]


def test_summarize_regenerates_the_committed_parity_json_byte_for_byte():
    missing = _saved_inputs_present()
    if missing:
        pytest.skip("saved replay, P3 or live parity inputs absent: " + ", ".join(missing[:3]))
    regenerated = json.dumps(study.prereg_check(study.summarize()), indent=2) + "\n"
    assert regenerated == study.PARITY_JSON.read_text(), \
        "prereg_check_parity.json differs from what `summarize` writes. Wording belongs in the constants of study_reply_order.py"


def test_hand_written_annotations_are_in_the_regenerated_file():
    missing = _saved_inputs_present()
    if missing:
        pytest.skip("saved inputs absent")
    d = study.prereg_check(study.summarize())
    assert d["P1"]["label"] == study.P1_LABEL and d["P1"]["scope"] == study.P1_SCOPE
    assert d["P3"]["label"] == study.P3_LABEL and d["P3"]["observed"]["sensitivity_note"] == study.P3_SENSITIVITY_NOTE
    assert d["P2"]["observed"]["note"] == study.P2_NOTE
    assert list(d["live_hosted_run"]["runs"]) == list(study.LIVE_RUNS)
    assert all(r["verified_numerical_parity"] for r in d["live_hosted_run"]["runs"].values())


def test_live_block_refuses_to_run_without_the_live_evidence(tmp_path):
    with pytest.raises(FileNotFoundError):
        study.live_hosted_run(tmp_path)


def test_replay_without_checkpoints_says_so_instead_of_a_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "NATIVE", tmp_path / "native")
    with pytest.raises(SystemExit) as e:
        study.require_replay_inputs("run-iid-d", "data/training/digits-docker-iid-v012", True)
    msg = str(e.value)
    assert "checkpoints not in checkout" in msg and "final_model.pt" in msg and "manifest.json" in msg
    assert "summarize" in msg
    # the replay-only runs need only the prepared partitions
    with pytest.raises(SystemExit) as e2:
        study.require_replay_inputs("c4-digits-iid-2round", "data/training/digits-docker-iid-4c-v014", False)
    assert "final_model.pt" not in str(e2.value) and "manifest.json" in str(e2.value)
