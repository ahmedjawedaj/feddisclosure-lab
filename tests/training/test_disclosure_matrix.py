import pytest
from scripts.run_disclosure_matrix import summarize


def test_disclosure_matrix_pairs_identical_seed():
    rows = [{"seed": 1, "noise_std": 0.0, "mean_mse": 0.001, "mean_psnr_db": 30, "mean_ssim": 0.9},
            {"seed": 1, "noise_std": 0.01, "mean_mse": 0.03, "mean_psnr_db": 17, "mean_ssim": 0.5}]
    report = summarize(rows)
    assert report["paired_differences"][0]["noisy_minus_exact_mse"] == pytest.approx(0.029)
    assert "NONE" in report["privacy_claim"]
