"""Regenerate figures strictly from saved public-data research evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve, roc_auc_score


def plot_gradient(root: Path, output: Path):
    experiments = [("gradient_b1", "Single image\nExact gradient"),
                   ("gradient_b1_noisy", "Single image\nGaussian noise 0.01"),
                   ("gradient_b8_known_labels", "Batch of eight\nLabels revealed")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for col, (folder, title) in enumerate(experiments):
        case = root / folder
        with np.load(case / "evaluator_only" / "original.npz") as data:
            original = data["x"]
        with np.load(case / "attacker" / "recovered.npz") as data:
            recovered = data["x"]
        distances = ((original[:, None, :] - recovered[None, :, :]) ** 2).mean(axis=2)
        rows, cols = linear_sum_assignment(distances)
        a, b = int(rows[0]), int(cols[0])
        evaluation = json.loads((case / "evaluation.json").read_text())
        axes[0, col].imshow(original[a].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(recovered[b].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(title)
        axes[1, col].set_xlabel(f"Batch mean MSE: {evaluation['mean_mse']:.4g}")
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("Original (evaluator)")
    axes[1, 0].set_ylabel("Recovered (attacker)")
    fig.suptitle("Controlled gradient reconstruction on public 8×8 digit images")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_sweep(sweep: Path, output: Path):
    rows = json.loads(sweep.read_text())["paired_differences"]
    seeds = [p["seed"] for p in rows]
    iid_minus_skew = [p["iid_minus_skew_accuracy"] * 100 for p in rows]
    iid_minus_skew_worst = [p["iid_minus_skew_worst_client_accuracy"] * 100 for p in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(seeds))
    ax.bar(x - 0.2, iid_minus_skew, width=0.38, label="Global test accuracy")
    ax.bar(x + 0.2, iid_minus_skew_worst, width=0.38, label="Worst-client accuracy")
    ax.axhline(0, lw=0.8, color="black")
    ax.set_xticks(x, seeds)
    ax.set_ylabel("IID minus severe non-IID (percentage points)")
    ax.set_xlabel("Paired public-data seed")
    ax.set_title("Two-round federated training: paired-seed deltas")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_membership(root: Path, output: Path):
    with np.load(root / "attacker" / "scores.npz") as f:
        scores = f["scores"]
    with np.load(root / "evaluator_only" / "membership_labels.npz") as f:
        truth = f["membership"]
    fpr, tpr, _ = roc_curve(truth, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"Confidence heuristic, AUC={roc_auc_score(truth,scores):.3f}")
    ax.plot([0, 1], [0, 1], "--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Output-only membership inference on public digits")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    root = Path("reports/results/sprint3")
    plot_gradient(root, Path("reports/figures/sprint3_gradient_comparison.png"))
    plot_sweep(Path("reports/results/sprint2/sweep/sweep_summary.json"), Path("reports/figures/sprint2_five_seed_accuracy.png"))
    plot_membership(root / "membership_smoke", Path("reports/figures/sprint3_membership_roc.png"))
    print("Generated three figures from recorded experiment artifacts")
