"""Render a truthful report figure from the recorded experiment artifact.

Missing releases are shown as missing, never as zero counts.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/results/sprint1_analytics_verified.json")
    parser.add_argument("--output", default="reports/figures/sprint1_analytics.png")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    report = json.loads(Path(args.input).read_text())
    baseline = report["results"]["naive"]["releases"]
    defended = report["results"]["history"]["releases"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, label, releases in zip(axes, ("Naive threshold", "History-aware heuristic"), (baseline, defended)):
        x = [0, 1]
        for pos, release in zip(x, releases):
            if release["released"]:
                ax.bar(pos, release["count"], width=0.58)
                ax.text(pos, release["count"] + 2, str(release["count"]), ha="center", fontsize=12)
            else:
                ax.text(pos, 55, "BLOCKED", ha="center", fontsize=11, weight="bold")
        ax.set_xticks(x, ["All records", "Negative records"])
        ax.set_title(label)
        ax.set_ylim(0, 140)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Released gateway-wide count")
    fig.suptitle("FedDisclosure Lab: two-query differencing experiment")
    fig.text(0.5, 0.01, "Naive released 110 and 107, allowing inference of a restricted count of 3. Blocked ≠ 0.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
