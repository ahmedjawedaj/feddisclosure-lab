"""Plot a matched longer-round FedAvg/FedProx pilot from actual model runs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def plot(output: Path):
    folder = ROOT / "reports/results/sprint2/extended"
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name in ("fedavg", "fedprox"):
        record = json.loads((folder / f"{name}_labelskew_8round_seed241.json").read_text())
        pairs = [(r["round"], 100*r["public_test"]["accuracy"]) for r in record["rounds"]]
        ax.plot(*zip(*pairs), marker="o", label=name.upper() if name == "fedavg" else "FedProx (mu=0.3)")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Held-out global accuracy (%)")
    ax.set_title("Severe non-IID public digits, seed 241; 3 clients")
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_partition_sensitivity(output: Path):
    folder = ROOT / "reports/results/sprint2/extended"
    configs = ("fedavg_labelskew_8round_seed241", "fedavg_dirichlet_8round_seed241")
    data = [json.loads((folder / f"{config}.json").read_text()) for config in configs]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for label, record in zip(("severe label skew", "Dirichlet alpha=0.5"), data):
        pairs = [(r["round"], 100*r["public_test"]["accuracy"]) for r in record["rounds"]]
        ax.plot(*zip(*pairs), marker="o", label=label)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Held-out global accuracy (%)")
    ax.set_title("FedAvg: distribution sensitivity, same data seed 241")
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0, 100)
    ax.legend()
    fig.text(0.11, 0.01, "Site sizes and SGD step counts differ across partitions; not optimizer-compute matched.", fontsize=8)
    fig.subplots_adjust(bottom=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)

if __name__ == "__main__":
    plot(ROOT / "reports/figures/sprint2_extended_8round_paired.png")
    plot_partition_sensitivity(ROOT / "reports/figures/sprint2_partition_sensitivity.png")
