"""Plot only observed reference results, without implying statistical significance."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / "reports/results/sprint2"
a = json.loads((results / "torch_iid_reference.json").read_text())
b = json.loads((results / "torch_non_iid_reference.json").read_text())
fig, ax = plt.subplots(figsize=(7.3, 4.2))
for name, data in [("IID", a), ("Label-skew non-IID", b)]:
    ax.plot([x["round"] for x in data["rounds"]],
            [x["public_test"]["accuracy"] for x in data["rounds"]], marker="o", label=name)
ax.set(title="3-client PyTorch FedAvg reference (not Flower)", xlabel="FedAvg round", ylabel="Held-out public digit accuracy")
ax.set_ylim(0, 1)
ax.set_xticks([1, 2])
ax.legend()
ax.grid(True, alpha=0.2)
fig.text(0.02, 0.015, "Single seed and two rounds. Exploratory smoke evidence, not a significance test.", fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
out = ROOT / "reports/figures/sprint2_training_reference.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(out)
