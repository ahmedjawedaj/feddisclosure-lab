"""Research figures generated only from recorded (not invented) measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot_multirecord(summary_path, output):
    summary = json.loads(Path(summary_path).read_text())
    cases = summary['runs']
    labels = [f"B={v['batch_size']}\nK={v['local_steps']}" for v in cases]
    mse = [v['image_mean_mse'] for v in cases]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, mse)
    ax.set_ylim(0, max(mse) * 1.2)
    ax.set_ylabel('Evaluator-matched image MSE (lower is better)')
    ax.set_xlabel('Distinct records B and local SGD steps K')
    ax.set_title('Unprotected per-client update inversion: 160 attack iterations')
    ax.text(0, -0.26, 'Batch >1: known-label attacker. Fixed local batch reused across steps.',
            transform=ax.transAxes, fontsize=8)
    fig.subplots_adjust(bottom=0.3)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_fedprox(summary_path, output):
    summary = json.loads(Path(summary_path).read_text())
    rows = summary['results']
    seeds = [r['seed'] for r in rows]
    positions = np.arange(len(rows))
    avg = [100*r['fedavg']['last_test_acc'] for r in rows]
    prox = [100*r['fedprox']['last_test_acc'] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(positions-0.19, avg, width=0.38, label='FedAvg')
    ax.bar(positions+0.19, prox, width=0.38, label='FedProx (mu=0.3)')
    ax.set_xticks(positions, seeds)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Held-out public-digit test accuracy (%)')
    ax.set_xlabel('Matched dataset/initialization seed')
    ax.set_title('Severe non-IID; 3 clients, 4 rounds, 2 local epochs')
    ax.legend()
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    plot_multirecord(ROOT/'reports/results/sprint3/multirecord_update/multirecord_update_summary.json',
                     ROOT/'reports/figures/sprint3_multirecord_inversion.png')
    plot_fedprox(ROOT/'reports/results/sprint2/ablations/paired/paired_summary.json',
                 ROOT/'reports/figures/sprint2_fedprox_paired.png')
    print('Regenerated two figures from measured v0.5 evidence')
