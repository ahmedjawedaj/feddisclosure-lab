"""Plot five-seed mean +/- empirical standard deviation; no significance test."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'reports/results/sprint4/aggregate_collusion'
names = ('individual', 'aggregate_only', 'all_but_one_collusion')
labels = ('Individual update', 'Aggregate only', 'Aggregate + 2 colluders')
data = [np.array([json.loads((folder/f'seed{s}'/'summary.json').read_text())['results'][name]['mse'] for s in range(241,246)]) for name in names]
fig, ax = plt.subplots(figsize=(9,4.5))
x = np.arange(3)
ax.bar(x, [d.mean() for d in data], yerr=[d.std(ddof=1) for d in data], capsize=6)
for i, d in enumerate(data):
    ax.scatter(np.full(len(d), i) + np.linspace(-.09,.09,len(d)), d, color='black', s=14, zorder=3)
ax.set_xticks(x, labels)
ax.set_ylabel('Target-image reconstruction MSE (lower better)')
ax.set_title('Individual, aggregate-only and all-but-one collusion: 5-seed pilot')
ax.text(.5, -.24, 'Idealized unencrypted sum; attacker knows target labels. NOT Flower SecAgg+.',
        transform=ax.transAxes, ha='center', fontsize=9)
fig.subplots_adjust(bottom=.28)
out = ROOT / 'reports/figures/sprint4_aggregate_collusion.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=155)
print(out, out.stat().st_size)
