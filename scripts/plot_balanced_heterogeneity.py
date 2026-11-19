"""Regenerate publication-ready matched-seed figure from saved scalar results."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"iid": "IID", "dirichlet_balanced_0.1": "Dir. α=0.1",
          "dirichlet_balanced_0.5": "Dir. α=0.5", "dirichlet_balanced_10": "Dir. α=10"}


def plot(summary_path: Path, output: Path) -> None:
    s = json.loads(summary_path.read_text())
    seeds = s["aggregate"]["completed_seeds"]
    if not seeds or len(seeds) < 2:
        raise ValueError("Need at least two matched seeds")
    records = {(r["seed"], r["partition"]): r["metrics"] for r in s["results"]}
    parts = list(s["partitions_requested"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x = np.arange(len(parts))
    for ax, field, subtitle in ((ax1, "test_accuracy", "Global held-out accuracy"),
                                (ax2, "worst_client_accuracy", "Worst-client validation accuracy")):
        measurements = np.array([[records[(seed, p)][field] for seed in seeds] for p in parts]) * 100
        means = measurements.mean(axis=1)
        stds = measurements.std(axis=1, ddof=1)
        ax.bar(x, means, yerr=stds, capsize=4, alpha=.75)
        for j in range(len(parts)):
            ax.scatter(np.full(len(seeds), x[j]) + np.linspace(-.12, .12, len(seeds)),
                       measurements[j], s=18, facecolors="none", edgecolors="black", zorder=4)
        ax.set_xticks(x, [LABELS[p] for p in parts], rotation=15)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title(subtitle)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"Matched three-client FedAvg, {len(seeds)} seeds, {s['rounds']} rounds, equal client sizes")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="reports/results/sprint5/balanced_heterogeneity/summary.json")
    parser.add_argument("--output", default="reports/figures/sprint5_balanced_heterogeneity.png")
    args = parser.parse_args()
    plot(ROOT / args.summary, ROOT / args.output)


if __name__ == "__main__":
    main()
