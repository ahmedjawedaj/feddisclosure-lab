import pytest
np = pytest.importorskip("numpy")

from feddisclosure.attacks.membership import confidence_scores, attack_outputs


def test_output_only_heuristics_have_expected_directions():
    p = np.array([[0.92, 0.08], [0.5, 0.5], [0.6, 0.4]])
    for method in ("max_confidence", "negative_entropy", "top2_margin"):
        scores = confidence_scores(p, method=method)
        assert scores.shape == (3,) and np.isfinite(scores).all()
        assert scores[0] > scores[2] > scores[1]
    with pytest.raises(ValueError, match="Unsupported"):
        confidence_scores(p, method="unknown")


def test_each_method_refuses_oracle_data(tmp_path):
    p = np.array([[.2,.8], [.6,.4]])
    file = tmp_path / "obs.npz"
    np.savez_compressed(file, probabilities=p, membership=[0, 1])
    for method in ("max_confidence", "negative_entropy", "top2_margin"):
        with pytest.raises(ValueError, match="probabilities only"):
            attack_outputs(file, tmp_path / method, method=method)
