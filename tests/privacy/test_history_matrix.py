import pytest
from scripts.run_history_matrix import make_matrix
from scripts.verify_history_matrix import verify


def test_matrix_integrity_and_repeatability(tmp_path):
    a = make_matrix(seeds=[241, 242], thresholds=[2, 5], output=tmp_path / 'a.json')
    b = make_matrix(seeds=[241, 242], thresholds=[2, 5], output=tmp_path / 'b.json')
    assert a['cases_sha256'] == b['cases_sha256']
    assert verify(tmp_path / 'a.json')['verified_cases'] == 8
    assert a['summary']['history_reset_id']['exact_inference_rate'] == 1
    assert a['summary']['principal_history_reset_id']['harmless_repeat_released_cases'] == 8


def test_matrix_input_validation(tmp_path):
    with pytest.raises(ValueError):
        make_matrix(seeds=[1, 1], thresholds=[5], output=tmp_path/'x')
    with pytest.raises(ValueError):
        make_matrix(seeds=[1], thresholds=[0], output=tmp_path/'x')
