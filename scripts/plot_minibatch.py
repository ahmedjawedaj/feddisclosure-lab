"""Plot only executed, evaluator-scored changing-minibatch inversion results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(path: Path, output: Path) -> None:
    summary = json.loads(path.read_text())
    rows = summary["cases"]
    if not rows or any("reconstruction_mse" not in row for row in rows):
        raise ValueError("Completed, evaluator-scored cases are required")
    label = [f"{x['total_distinct_records']} records\n{x['local_steps']} steps\norder {x['schedule_seed']}" for x in rows]
    values = [x["reconstruction_mse"] for x in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(label, values)
    ax.set_ylabel("Reconstruction MSE (lower means stronger attack)")
    ax.set_title("Unprotected model-update inversion: disjoint SGD minibatches")
    ax.set_ylim(0, max(values) * 1.2)
    fig.text(0.12, 0.01, "All contributing labels disclosed to attacker. Public 8x8 digits; no DP/SecAgg+.", fontsize=8)
    fig.subplots_adjust(bottom=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="reports/results/sprint3/changing_minibatches/seed241/minibatch_update_summary.json")
    parser.add_argument("--output", default="reports/figures/sprint3_changing_minibatches.png")
    args = parser.parse_args()
    plot(Path(args.summary), Path(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
