"""Write prereg_check_scaling.json (S1, S2, S3) and prereg_check_fashion.json (X1).

Reads the saved study summaries and, when present, the memory record of the hosted 8-client Docker
run. Thresholds are copied from docs/preregistration/v0.15.md and are never changed here.
S3 is PENDING until `reports/results/privacy/v0150/scaling/docker_n8/scaling_n8_memory.json` exists.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V = ROOT / "reports/results/privacy/v0150"
PREREG = "docs/preregistration/v0.15.md"


def scaling() -> dict:
    study = json.loads((V / "scaling/runs/summary.json").read_text())
    out = {"schema_version": 1, "preregistration": PREREG,
           "note": "Thresholds are copied verbatim from the pre-registration and were not changed. Pass rules are applied to "
                   "point estimates of pooled recognisable rates. Intervals are reported next to them. The primary SecAgg+ "
                   "setting is max_weight 1000 (the upstream default, fixed before the 8-client run). 30 evaluation seeds "
                   "(0 to 29), 8 victim images per seed.",
           "S1": {**study["S1"], "operationalisation":
                  "Floor is the class-average label floor on the same victim images. The aggregate view is the SecAgg+ "
                  "aggregate with joint-oracle labels. Pass uses point estimates. The aggregate is far BELOW the floor, "
                  "so S1 passes because the attack is weak on 64 joint images, not because of any guarantee."},
           "S2": {**study["S2"], "operationalisation":
                  "n-1 collusion on the primary max_weight 1000 view at 4 and 8 clients. Both counts must reach 85 percent."},
           "S1_at_4_clients_for_reference": study["S1_at_4_clients_for_reference"]}
    mem = V / "scaling/docker_n8/scaling_n8_memory.json"
    if mem.is_file():
        m = json.loads(mem.read_text())
        out["S3"] = {"id": "S3", "prediction": "Memory at 8 clients fits",
                     "pass_rule": "Peak container memory below 6 GB. If not, the 8-client run is dropped and recorded as such",
                     "observed": {k: m.get(k) for k in ("peak_total_gb", "peak_single_container_gb", "containers_seen",
                                                        "samples", "outcome", "num_shares", "reconstruction_threshold", "host")},
                     "pass": bool(m["pass"])}
    else:
        out["S3"] = {"id": "S3", "prediction": "Memory at 8 clients fits",
                     "pass_rule": "Peak container memory below 6 GB. If not, the 8-client run is dropped and recorded as such",
                     "status": "PENDING. Needs a Docker daemon, not available where this was built. Dispatch "
                               ".github/workflows/scaling-n8.yml and rerun this script on the result.",
                     "pass": None}
    return out


def fashion() -> dict:
    study = json.loads((V / "fashion_mnist/summary.json").read_text())
    return {"schema_version": 1, "preregistration": PREREG,
            "note": "Thresholds are copied verbatim from the pre-registration and were not changed. Pass rules are applied to "
                    "point estimates. Primary SecAgg+ setting is max_weight 1000. Attack settings come from tuning seeds "
                    "9000 to 9002. Evaluation seeds are " + study["evaluation_seeds"] + ".",
            "X1": {**study["X1"], "operationalisation":
                   "Plaintext and n-1 collusion must each be at least floor plus 10 points and the aggregate view at most "
                   "floor plus 5 points. All three parts must hold. The same rule at max_weight 16 is reported as "
                   "X1_secondary_max_weight_16 and does not change the verdict."},
            "X1_secondary_max_weight_16": study["X1_secondary_max_weight_16"]}


def main():
    (V / "prereg_check_scaling.json").write_text(json.dumps(scaling(), indent=2) + "\n")
    print("wrote prereg_check_scaling.json")
    if (V / "fashion_mnist/summary.json").is_file():
        (V / "prereg_check_fashion.json").write_text(json.dumps(fashion(), indent=2) + "\n")
        print("wrote prereg_check_fashion.json")


if __name__ == "__main__":
    main()
