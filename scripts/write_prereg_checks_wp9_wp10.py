"""Write prereg_check_scaling.json (S1, S2, S3) and prereg_check_fashion.json (X1).

Reads the saved study summaries and, when present, the memory record of the hosted 8-client Docker
run. Thresholds are copied from docs/preregistration/v0.15.md and are never changed here.
S3 is PENDING until `reports/results/privacy/v0150/scaling/docker_n8/scaling_n8_memory.json` exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V = ROOT / "reports/results/privacy/v0150"
PREREG = "docs/preregistration/v0.15.md"


OLD_KEY = "gap_upper_limit_is_conservative_check"
NEW_KEY = "aggregate_wilson_upper_limit_within_floor_plus_5"


def _plain_names(check: dict) -> dict:
    """The saved study summary calls this flag `gap_upper_limit_is_conservative_check`. Here it gets a plain name.
    It is true when the upper Wilson limit of the aggregate rate is at most the floor rate plus 5 points."""
    out = json.loads(json.dumps(check))
    obs = out["observed"]
    if OLD_KEY in obs:
        obs[NEW_KEY] = obs.pop(OLD_KEY)
    return out


def scaling() -> dict:
    study = json.loads((V / "scaling/runs/summary.json").read_text())
    out = {"schema_version": 1, "preregistration": PREREG,
           "note": "Thresholds are copied verbatim from the pre-registration and were not changed. Pass rules are applied to "
                   "point estimates of pooled recognisable rates. Intervals are reported next to them. The primary SecAgg+ "
                   "setting is max_weight 1000 (the upstream default, fixed before the 8-client run). 30 evaluation seeds "
                   "(0 to 29), 8 victim images per seed.",
           "S1": {**_plain_names(study["S1"]), "verdict_label": "PASS (vacuous, by the one-sided rule)",
                  "operationalisation":
                  "Floor is the class-average label floor on the same victim images. The aggregate view is the SecAgg+ "
                  "aggregate with joint-oracle labels. Pass uses point estimates. The rule is one-sided (at most floor plus "
                  "5 points) and the aggregate is far BELOW the floor, so S1 passes because this attack is weak on 64 joint "
                  "images. A two-sided reading (within 5 points of the floor) would FAIL. The pass is not privacy evidence "
                  "and any aggregate-at-floor claim is conditional on this weaker attack."},
           "S2": {**study["S2"], "operationalisation":
                  "n-1 collusion on the primary max_weight 1000 view at 4 and 8 clients. Both counts must reach 85 percent."},
           "S1_at_4_clients_for_reference": _plain_names(study["S1_at_4_clients_for_reference"])}
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
                   "X1_secondary_max_weight_16 and does not change the verdict.",
                   "verdict_effect": "decides the X1 verdict"},
            "X1_secondary_max_weight_16": {**study["X1_secondary_max_weight_16"], "verdict_effect": "none",
                   "note": "Reported for context only. The X1 verdict is the primary row at max_weight 1000. "
                           "Passing here does not change it."}}


def main(argv=None):
    argparse.ArgumentParser(
        description="Rewrite prereg_check_scaling.json and prereg_check_fashion.json from the saved study summaries "
                    "under reports/results/privacy/v0150/. Takes no options. Running it with --help writes nothing.").parse_args(argv)
    (V / "prereg_check_scaling.json").write_text(json.dumps(scaling(), indent=2) + "\n")
    print("wrote prereg_check_scaling.json")
    if (V / "fashion_mnist/summary.json").is_file():
        (V / "prereg_check_fashion.json").write_text(json.dumps(fashion(), indent=2) + "\n")
        print("wrote prereg_check_fashion.json")


if __name__ == "__main__":
    main()
