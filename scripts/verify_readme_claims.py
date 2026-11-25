"""README hygiene check: every headline number in README.md must match saved evidence.

This is a consistency check between prose and saved files. It is not a scientific validation. It
cannot tell whether an experiment was run correctly, only that the number printed in the README is
the number stored in the evidence file it points to (within the rounding the README prints).
It checks table cells, numbers quoted in the prose under the table, and the verdict words
(for example "26 points above" or "does not beat the floor"), which are derived from the saved
intervals so a stale verdict is caught when the evidence changes. The test-suite count is read from
the newest fast suite records (one per environment), which `scripts/record_fast_suite.py` writes from a real pytest
run. A record is flagged when the code changed after the commit it names, or when its package line does not fit
the environment the README says the count is for.
It fails closed: a missing README row, a missing evidence file, a missing JSON path, a value that is
not a finite number or a mismatch gives exit code 1.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VIEWS = "reports/results/privacy/v014/view_comparison/summary.json"
FLOOR = "reports/results/privacy/v0143/label_floor.json"
INTERVALS = "reports/results/privacy/v0143/view_intervals.json"
LIRA = "reports/results/privacy/v014/lira/{}/summary.json"
V15 = "reports/results/privacy/v0150/"
PROOFS, DP, WIRE, FAULTS = (V15 + f"prereg_check_{n}.json" for n in ("proofs", "dp", "wire", "faults"))
EXTERNAL, PARITY, SCALING, FASHION = (V15 + f"prereg_check_{n}.json" for n in ("external", "parity", "scaling", "fashion"))
FASHION_BOOT = V15 + "fashion_mnist/seed_bootstrap.json"
# Fast suite records live in reports/results/privacy/v*/. The newest version directory that holds the file wins.
RECORD_DIR = "reports/results/privacy/v0150"
FLOWER_DP = "reports/results/privacy/v0160/flower_dp_audit/flwr-1.27.0/flower_dp_audit.json"
P3 = "reports/results/privacy/v0160/p3_breaching/summary.json"
FEDML = "reports/results/privacy/v0160/fedml_dp_audit/fedml-0.9.6/fedml_dp_audit.json"
ADAPT = "reports/results/privacy/v0160/flower_adaptive_clipping/flwr-1.27.0/flower_adaptive_clipping.json"
NVF = "reports/results/privacy/v0160/nvflare_dp_audit/nvflare-2.9.1/nvflare_dp_audit.json"
GUARANTEE_DIR = "reports/results/privacy/v0160/flower_adaptive_guarantee/"
GUARANTEE = {v: f"{GUARANTEE_DIR}flwr-{v}/flower_adaptive_guarantee.json" for v in ("1.27.0", "1.44.0")}
ADAPT_G = GUARANTEE["1.27.0"]
SEL = "reports/results/privacy/v0160/nvflare_selection/nvflare_selection.json"
EVT = "reports/results/privacy/v0160/nvflare_events/nvflare_events.json"
TFF = "reports/results/privacy/v0160/tff_dp/tff-0.87.0/tff_dp_audit.json"
SUITE_FILES = {"full": "fast_suite.txt", "ci": "fast_suite_ci_env.txt"}
# A record is stale when any of these changed between the commit it names and the working tree.
CODE_PATHS = ("src", "services", "scripts", "tests", "pyproject.toml")


@dataclass(frozen=True)
class Claim:
    id: str
    row: str            # start of the first cell of the README table row
    col: int            # 0-based cell index in that row
    file: str           # evidence file relative to the repo root
    path: str           # JSON path with / separators, integers index lists
    scale: float = 1.0  # evidence value times scale is compared with the README number
    minus_file: str = ""   # optional second evidence value, subtracted before scaling
    minus_path: str = ""


@dataclass(frozen=True)
class Ref:
    group: int          # regex group of the prose claim holding the printed number
    file: str
    path: str
    scale: float = 1.0
    minus_file: str = ""
    minus_path: str = ""


@dataclass(frozen=True)
class ProseClaim:
    id: str
    pattern: str        # regex that must match the README text, numbers captured in groups
    refs: tuple


@dataclass(frozen=True)
class Verdict:
    """The words in a table cell must agree with the sign of a paired interval in the evidence.
    If the lower end of the interval is above zero the cell must say 'above'. Otherwise it must say
    'does not beat' and must not say 'above'."""
    id: str
    row: str
    col: int
    file: str
    low_path: str


@dataclass(frozen=True)
class VerdictWord:
    """The README must say `<token> PASS` or `<token> FAIL` as the saved pre-registration check says, and never the
    opposite. The saved value is a boolean `pass` or a string "PASS" or "FAIL"."""
    token: str
    file: str
    path: str


@dataclass(frozen=True)
class Phrase:
    id: str
    row: str
    col: int
    phrase: str


def _view_claims() -> list[Claim]:
    rows = [("Plaintext update", "plaintext_individual"), ("SecAgg+ aggregate only", "secagg_aggregate_maxw1000"),
            ("SecAgg+ with n-1 colluders", "secagg_collusion_n-1_maxw1000"), ("Opacus DP, sigma 0.2", "dp0.2_individual")]
    out = []
    for label, view in rows:
        out.append(Claim(f"inversion.{view}.mse", label, 1, VIEWS, f"views/{view}/victim_mse/0"))
        out.append(Claim(f"inversion.{view}.rate", label, 2, VIEWS, f"views/{view}/rate_mse_below_0_05/0", 100))
    for label, view in (("Plaintext update", "plaintext_individual"), ("SecAgg+ with n-1 colluders", "secagg_collusion_n-1_maxw1000")):
        out.append(Claim(f"inversion.{view}.points_above_floor", label, 3, VIEWS, f"views/{view}/rate_mse_below_0_05/0", 100,
                         FLOOR, "pooled/class_average_rate_mse_below_0_05"))
    out += [Claim("floor.label_only.mse", "Label-only floor", 1, FLOOR, "pooled/class_average_mse"),
            Claim("floor.label_only.rate", "Label-only floor", 2, FLOOR, "pooled/class_average_rate_mse_below_0_05", 100),
            Claim("floor.mean_image.mse", "Mean-image floor", 1, FLOOR, "pooled/mean_image_mse"),
            Claim("floor.mean_image.rate", "Mean-image floor", 2, FLOOR, "pooled/mean_image_rate_mse_below_0_05", 100)]
    return out


def _prose_claims() -> list[ProseClaim]:
    agg = "secagg_aggregate_maxw1000"
    sens = "pool_size_sensitivity/rows/{}/rate_mse_below_0_05"
    return [
        ProseClaim("prose.floor_counts_and_wilson",
                   r"\((\d+) of (\d+) images, Wilson 95% interval (\d+)% to (\d+)%\)",
                   (Ref(1, INTERVALS, "label_floor/recognisable/successes"), Ref(2, INTERVALS, "label_floor/recognisable/trials"),
                    Ref(3, INTERVALS, "label_floor/recognisable/wilson/0", 100), Ref(4, INTERVALS, "label_floor/recognisable/wilson/1", 100))),
        ProseClaim("prose.aggregate_against_floor",
                   r"on both error \(([\d.]+) against ([\d.]+)\) and recognisable share \((\d+)% against ([\d.]+)%\)",
                   (Ref(1, VIEWS, f"views/{agg}/victim_mse/0"), Ref(2, FLOOR, "pooled/class_average_mse"),
                    Ref(3, VIEWS, f"views/{agg}/rate_mse_below_0_05/0", 100), Ref(4, FLOOR, "pooled/class_average_rate_mse_below_0_05", 100))),
        ProseClaim("prose.dp_against_mean_image",
                   r"\(([\d.]+) is worse than the mean image at ([\d.]+)\)",
                   (Ref(1, VIEWS, "views/dp0.2_individual/victim_mse/0"), Ref(2, FLOOR, "pooled/mean_image_mse"))),
        ProseClaim("prose.maxw16_not_distinguishable",
                   r"the share is ([\d.]+)% and the paired difference to the floor is (-?[\d.]+) points with an interval of (-?[\d.]+) to \+?([\d.]+)",
                   (Ref(1, VIEWS, "views/secagg_aggregate_maxw16/rate_mse_below_0_05/0", 100),
                    Ref(2, INTERVALS, "views/secagg_aggregate_maxw16/paired_rate_minus_label_floor_bootstrap95/0", 100),
                    Ref(3, INTERVALS, "views/secagg_aggregate_maxw16/paired_rate_minus_label_floor_bootstrap95/1", 100),
                    Ref(4, INTERVALS, "views/secagg_aggregate_maxw16/paired_rate_minus_label_floor_bootstrap95/2", 100))),
        ProseClaim("prose.public_pool_size", r"(\d+)-image labelled public pool",
                   (Ref(1, FLOOR, "per_seed/0/public_pool_size"),)),
        ProseClaim("prose.pool_size_sensitivity",
                   r"pool of 20 images the floor is (\d+)% recognisable, with 50 images (\d+)%, with 100 images (\d+)% and with 300 images (\d+)%",
                   tuple(Ref(i + 1, FLOOR, "pool_size_sensitivity/rows/%d/rate_mse_below_0_05" % i, 100) for i in range(4))),
    ]


def _v15_prose() -> list[ProseClaim]:
    """Numbers quoted in the v0.15 section of the README. Each one is read from a prereg_check file (or the seed
    bootstrap file) that the work package scripts write. A sentence that is reworded or removed fails closed."""
    R = Ref
    b1, b2 = "checks/B1/headline/", "checks/B2/headline/"
    x1, x16 = "X1/observed/secagg_collusion_maxw1000/rate_minus_floor", "X1_secondary_max_weight_16/observed/secagg_collusion_maxw16/rate_minus_floor"
    gap = "rows/secagg_collusion_maxw1000/gap_seed_bootstrap/"
    return [
        ProseClaim("v15.E1", r"E1 PASS: zero counterexamples in ([\d,]+) float32 cases, largest error ([\d.]+) of the bound",
                   (R(1, PROOFS, "E1/observed/cases_float32"), R(2, PROOFS, "E1/observed/max_error_over_stated_bound"))),
        ProseClaim("v15.E2", r"bias of up to 3/16 of a quantisation unit \(([\d.]+) measured\)",
                   (R(1, PROOFS, "E2/observed/float32_bias_max_abs_units_measured"),)),
        ProseClaim("v15.E3", r"99th percentile error is ([\d.]+) to ([\d.]+) of the Hoeffding bound, and with one shared seed it breaks the bound at 256 clients \(([\d.]+) to ([\d.]+) times the bound\)",
                   (R(1, PROOFS, "E3/observed/4/independent_p99_over_bound"), R(2, PROOFS, "E3/observed/256/independent_p99_over_bound"),
                    R(3, PROOFS, "E3/observed/256/shared_different_data_p99_over_bound"), R(4, PROOFS, "E3/observed/256/shared_identical_p99_over_bound"))),
        ProseClaim("v15.M1", r"the convex L2 bound has (\d+) violations in ([\d,]+) trials",
                   (R(1, PROOFS, "M1/observed/l2_bound_violations"), R(2, PROOFS, "M1/observed/convex_trials"))),
        ProseClaim("v15.D2", r"Wilson lower limit ([\d.]+) same-label and ([\d.]+) any-label at sigma 0.2",
                   (R(1, DP, "D2/observed/wilson_lower_by_candidate_mode/same_u10"), R(2, DP, "D2/observed/wilson_lower_by_candidate_mode/any_u10"))),
        ProseClaim("v15.D3_same_label", r"at sigma 4 the same-label attack is ([\d.]+) below the bound",
                   (R(1, DP, "D3/observed/same_u10/gap"),)),
        ProseClaim("v15.D3_any_label", r"Any-label candidates pass \(([\d.]+)\)", (R(1, DP, "D3/observed/any_u10/gap"),)),
        ProseClaim("v15.W1", r"\((\d+) of \d+ victim images recognisable from one masked vector, label floor ([\d.]+)%",
                   (R(1, WIRE, "W1/observed/attack_rate", 40), R(2, WIRE, "W1/observed/floor_rate", 100))),
        ProseClaim("v15.W2", r"worst coordinate error is (\d+)% of the proven bound",
                   (R(1, WIRE, "W2/observed/worst_error_over_bound", 100),)),
        ProseClaim("v15.W3", r"captured ([\d.]+)% against simulated ([\d.]+)% recognisable",
                   (R(1, WIRE, "W3/observed/captured_rate", 100), R(2, WIRE, "W3/observed/simulated_rate", 100))),
        ProseClaim("v15.W4", r"W4: plaintext mode ([\d.]+)% recognisable", (R(1, WIRE, "W4/observed/recognisable_rate", 100),)),
        ProseClaim("v15.W5", r"visible in (\d+) of (\d+) replies",
                   (R(1, WIRE, "W5/observed/counts_visible"), R(2, WIRE, "W5/observed/replies"))),
        ProseClaim("v15.W6", r"W6: (\d+) of (\d+) objects found",
                   (R(1, WIRE, "W6/observed/found_in_relay"), R(2, WIRE, "W6/observed/objects"))),
        ProseClaim("v15.F1", r"F1 PASS: (\d+) of 5 faults caught by the detectors as specified",
                   (R(1, FAULTS, "F1/observed/strict_detectors_A_B_C_D1/n_detected"),)),
        ProseClaim("v15.F2", r"has power ([\d.]+) at noise factors 0.25, 0.5 and 2 with a false positive rate of ([\d.]+)",
                   (R(1, FAULTS, "F2/observed/power_by_noise_factor/noise_x0.5/power"), R(2, FAULTS, "F2/observed/false_positive_rate/rate"))),
        ProseClaim("v15.B1", r"ratio ([\d.]+) \(bootstrap 95% interval ([\d.]+) to ([\d.]+), limit 1.5\)",
                   (R(1, EXTERNAL, b1 + "ratio"), R(2, EXTERNAL, b1 + "ratio_bootstrap_ci95/0"), R(3, EXTERNAL, b1 + "ratio_bootstrap_ci95/1"))),
        ProseClaim("v15.B1_shipped", r"\(ratios ([\d.]+) and ([\d.]+)\)",
                   (R(1, EXTERNAL, "checks/B1/all_configurations/ig_library_default/ratio"),
                    R(2, EXTERNAL, "checks/B1/all_configurations/dlg_known_labels/ratio"))),
        ProseClaim("v15.B2", r"AUC ([\d.]+) against ([\d.]+) for the strongest reference",
                   (R(1, EXTERNAL, b2 + "repo_auc"), R(2, EXTERNAL, b2 + "reference_auc"))),
        ProseClaim("v15.P1", r"passes the unchanged 1e-5 float64 gate on (\d+) of (\d+) native runs",
                   (R(1, PARITY, "P1/observed/passing_with_deterministic"), R(2, PARITY, "P1/observed/native_runs_replayed"))),
        ProseClaim("v15.P2", r"\(([\d.]+)% of orders pass\)",
                   (R(1, PARITY, "P2/observed/exploratory_4_client_configs/c4-mnist-iid-10round/upstream_pass_rate", 100),)),
        ProseClaim("v15.S1", r"aggregate-only is ([\d.]+)% against a ([\d.]+)% floor at 8 clients",
                   (R(1, SCALING, "S1/observed/aggregate_rate", 100), R(2, SCALING, "S1/observed/floor_rate", 100))),
        ProseClaim("v15.S2", r"n-1 collusion is ([\d.]+)% at 4 clients and ([\d.]+)% at 8",
                   (R(1, SCALING, "S2/observed/by_clients/4/rate", 100), R(2, SCALING, "S2/observed/by_clients/8/rate", 100))),
        ProseClaim("v15.S3", r"peaked at ([\d.]+) GB and ([\d.]+) GB against the 6 GB rule", (R(1, SCALING, "S3/observed/peak_total_gb"), R(2, V15 + "scaling/docker_n8/scaling_n8_memory_7_4.json", "peak_total_gb"))),
        ProseClaim("v15.X1", r"n-1 collusion is \+([\d.]+) points over the floor against a \+10 rule \(20 seeds, seed bootstrap interval \+([\d.]+) to \+([\d.]+)\)",
                   (R(1, FASHION, x1, 100), R(2, FASHION_BOOT, gap + "0", 100), R(3, FASHION_BOOT, gap + "1", 100))),
        ProseClaim("v15.X1_hosted", r"gives \+([\d.]+) points \(seed bootstrap \+([\d.]+) to \+([\d.]+)\) and would pass",
                   (R(1, V15 + "fashion_mnist/hosted/seed_bootstrap.json", "rows/secagg_collusion_maxw1000/gap_to_floor", 100),
                    R(2, V15 + "fashion_mnist/hosted/seed_bootstrap.json", gap + "0", 100),
                    R(3, V15 + "fashion_mnist/hosted/seed_bootstrap.json", gap + "1", 100))),
        ProseClaim("v15.X1_secondary", r"the same rule would pass \(\+([\d.]+)\)", (R(1, FASHION, x16, 100),)),
        ProseClaim("v16.C1", r"multiplier 1 is epsilon ([\d.]+) at delta 1e-5", (R(1, FLOWER_DP, "rows/0/measured/eps_at_delta_zero_out"),)),
        ProseClaim("v16.C3_C4", r"exact value is epsilon ([\d.]+), and with `num_sampled_clients` set to 10 while 2 results are aggregated it is epsilon ([\d.]+)\. A reported[^.]*\. Empirical 95% lower bounds are ([\d.]+) and ([\d.]+)",
                   (R(1, FLOWER_DP, "rows/1/measured/eps_at_delta_zero_out"), R(2, FLOWER_DP, "rows/2/measured/eps_at_delta_zero_out"),
                    R(3, FLOWER_DP, "rows/1/empirical/eps_lower_bound"), R(4, FLOWER_DP, "rows/2/empirical/eps_lower_bound"))),
        ProseClaim("v16.A", r"update noise scale is ([\d.]+) of the reference algorithm at the default learning rate and ([\d.]+) at learning rate 1\.0",
                   (R(1, FLOWER_DP, "rows/9/measured/sigma_ratio_to_reference"), R(2, FLOWER_DP, "rows/10/measured/sigma_ratio_to_reference"))),
        ProseClaim("v16.L1", r"exact delta is ([\d.]+) times the configured delta", (R(1, FLOWER_DP, "rows/16/exact_from_formula_sigma/delta_ratio_to_configured"),)),
        ProseClaim("v161.K0", r"batch of 1 the attack gains \+([\d.]+) dB over the floor \(interval \+([\d.]+) to \+([\d.]+)\)",
                   (R(1, P3, "verdicts/K0/mean"), R(2, P3, "verdicts/K0/lo"), R(3, P3, "verdicts/K0/hi"))),
        ProseClaim("v161.R1_R3", r"gain is (-?[\d.]+) dB in the plaintext view \(interval (-?[\d.]+) to \+([\d.]+)\) and (-?[\d.]+) dB with n-1 collusion \(interval (-?[\d.]+) to \+([\d.]+)\)",
                   (R(1, P3, "verdicts/R1/mean"), R(2, P3, "verdicts/R1/lo"), R(3, P3, "verdicts/R1/hi"),
                    R(4, P3, "verdicts/R3/mean"), R(5, P3, "verdicts/R3/lo"), R(6, P3, "verdicts/R3/hi"))),
        ProseClaim("v161.R2", r"aggregate view is ([\d.]+) dB below the floor \(interval (-?[\d.]+) to (-?[\d.]+)\)",
                   (R(1, P3, "verdicts/R2/mean", -1), R(2, P3, "verdicts/R2/lo"), R(3, P3, "verdicts/R2/hi"))),
        ProseClaim("v162.checks", r"(\d+) checks were built from them", (R(1, FEDML, "summary/checks_total"),)),
        ProseClaim("v162.LP2", r"executed Laplace mechanism is epsilon ([\d.]+) where the configuration implies ([\d.]+)",
                   (R(1, FEDML, "rows/3/cases/spread/eps_pure_measured"), R(2, FEDML, "rows/3/cases/spread/eps_pure_claimed"))),
        ProseClaim("v162.LD1", r"the exact epsilon is ([\d.]+), and the empirical 95% lower bound is ([\d.]+)",
                   (R(1, FEDML, "rows/14/exact_from_measured_sigma/eps_at_configured_delta"), R(2, FEDML, "rows/14/empirical/eps_lower_bound"))),
        ProseClaim("v162.RA2", r"gives epsilon ([\d.]+) where the exact Gaussian value is ([\d.]+)",
                   (R(1, FEDML, "rows/17/cells/1/rdp_epsilon"), R(2, FEDML, "rows/17/cells/1/exact_epsilon"))),
        ProseClaim("v161.B1c", r"mean squared error is ([\d.]+) times larger in the plaintext view", (R(1, P3, "verdicts/B1c/ratio"),)),
        ProseClaim("v163.A3_A10", r"the clip after one round is ([\d.]+) from a start of 0\.1, where the paper reading gives ([\d.]+)\. After 23 such rounds it is ([\d.]+), where",
                   (R(1, ADAPT, "rows/9/measured/scenarios/0/next_clip"), R(2, ADAPT, "rows/9/measured/scenarios/0/paper"),
                    R(3, ADAPT, "rows/17/measured/clip_after_23"))),
        ProseClaim("v163.A11", r"a clip that starts at 0\.1 to ([\d.]+e-\d+) and a clip that starts at 10 to ([\d.]+e\d+),",
                   (R(1, ADAPT, "rows/18/measured/clip_after_200"), R(2, ADAPT, "rows/19/measured/clip_after_200"))),
        ProseClaim("v164.N3", r"one call records epsilon ([\d.]+), of which ([\d.]+) is the release phase",
                   (R(1, NVF, "rows/2/measured/ledger_one_call/epsilon_total"), R(2, NVF, "rows/2/measured/ledger_one_call/epsilon_release"))),
        ProseClaim("v164.N4", r"(\d+) of 1,000 entries were accepted in one pass, ([\d.]+) times the number wanted",
                   (R(1, NVF, "rows/3/measured/accepted_before_selection"), R(2, NVF, "rows/3/measured/ratio_to_n_upload"))),
        ProseClaim("v164.N5", r"returned after (\d+) passes at epsilon 1e4", (R(1, NVF, "rows/4/measured/cells/1/acceptance_passes"),)),
        ProseClaim("v164.checks", r"All 18 rows and (\d+) checks held", (R(1, NVF, "summary/checks_held"),)),
        ProseClaim("v166.ratio_default", r"that ratio averages ([\d.]+) and the model sum multiplier is ([\d.]+) of the configured one",
                   (R(1, ADAPT_G, "cells/14/rho_delta/mean"), R(2, ADAPT_G, "cells/14/rho_z/mean"))),
        ProseClaim("v166.ratio_lr1", r"At learning rate 1\.0 the ratio is ([\d.]+)\.", (R(1, ADAPT_G, "cells/16/rho_delta/mean"),)),
        ProseClaim("v166.composed_default", r"composed multiplier in that configuration is ([\d.]+) of the configured one\. With the paper's count sensitivity of 1/2 applied to the same rounds it is ([\d.]+)\.",
                   (R(1, ADAPT_G, "cells/14/z_flower_over_z/mean"), R(2, ADAPT_G, "cells/14/z_paper_count_over_z/mean"))),
        ProseClaim("v166.composed_ratio_one", r"composed multiplier with Flower's count is ([\d.]+), ([\d.]+), ([\d.]+) and ([\d.]+) of the configured one",
                   tuple(R(i + 1, ADAPT_G, f"rows/5/measured/cases/{4 + i}/z_flower_over_z") for i in range(4))),
        ProseClaim("v165.ceiling", r"largest 95% lower bound that any outcome can give is ([\d.]+) and its claim was 20",
                   (R(1, SEL, "rows/0/measured/ceiling_v0164_tails"),)),
        ProseClaim("v165.LB", r"Lower bounds on the epsilon of the event are ([\d.]+), ([\d.]+) and ([\d.]+) against claims of 1, 1 and 0\.1",
                   (R(1, SEL, "rows/1/measured/interval_and_lb/lb"), R(2, SEL, "rows/2/measured/interval_and_lb/lb"),
                    R(3, SEL, "rows/3/measured/interval_and_lb/lb"))),
        ProseClaim("v165.MDE", r"would find with 80% power is ([\d.]+), ([\d.]+) and ([\d.]+) against those claims",
                   (R(1, SEL, "rows/1/measured/power/min_detectable/epsilon_80"), R(2, SEL, "rows/2/measured/power/min_detectable/epsilon_80"),
                    R(3, SEL, "rows/3/measured/power/min_detectable/epsilon_80"))),
        ProseClaim("v165.checks", r"All (\d+) rows and (\d+) checks held\. Tables, the one prediction",
                   (R(1, SEL, "summary/rows_total"), R(2, SEL, "summary/checks_held"))),
        ProseClaim("v168.ceiling", r"largest lower bound that any outcome can give at this run count is ([\d.]+), far above every claim",
                   (R(1, EVT, "rows/0/measured/ceiling"),)),
        ProseClaim("v168.LB", r"Lower bounds on the epsilon of the event are ([\d.]+), ([\d.]+), ([\d.]+) and ([\d.]+) against a claim of 1 for the four selection configurations, ([\d.]+) against a claim of 0\.3 for the release values and ([\d.]+) against a claim of 1\.2",
                   tuple(R(i + 1, EVT, f"rows/{i + 1}/measured/interval_and_lb/lb") for i in range(6))),
        ProseClaim("v168.MDE", r"would find with 80% power is ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+) and ([\d.]+) against those claims",
                   tuple(R(i + 1, EVT, f"rows/{i + 1}/measured/power/min_detectable/epsilon_80") for i in range(6))),
        ProseClaim("v168.release_ratio", r"event epsilon is predicted at ([\d.]+) of `epsilon_release` and measured at ([\d.]+) of it",
                   (R(1, EVT, "rows/5/measured/numeric_prediction/log_ratio_over_claim"),
                    R(2, EVT, "rows/5/measured/log_ratio_hat", 1 / 0.30000000000000004))),
        ProseClaim("v168.checks", r"All (\d+) rows and (\d+) checks held, and no replication was needed",
                   (R(1, EVT, "summary/rows_total"), R(2, EVT, "summary/checks_held"))),
        ProseClaim("v163.A6", r"the noise standard deviation is ([\d.]+) by the new clip and ([\d.]+) by the old clip",
                   (R(1, ADAPT, "rows/14/measured/std_by_new_clip"), R(2, ADAPT, "rows/14/measured/std_by_old_clip"))),
    ]


def _tff_prose() -> list[ProseClaim]:
    """Numbers quoted in the v0.16.9 section of the README, read from the TFF evidence file."""
    R = Ref
    m = "rows/{}/measured/"
    return [
        ProseClaim("v169.counts", r"Result: (\d+) of (\d+) rows and (\d+) of (\d+) checks held",
                   (R(1, TFF, "summary/rows_held"), R(2, TFF, "summary/rows_total"), R(3, TFF, "summary/checks_held"), R(4, TFF, "summary/checks_total"))),
        ProseClaim("v169.F1", r"Measured over configured is ([\d.]+), ([\d.]+) and ([\d.]+) for three configurations",
                   tuple(R(i + 1, TFF, m.format(1) + f"{k}/ratio") for i, k in enumerate(("F1a", "F1b", "F1c")))),
        ProseClaim("v169.F3", r"the noise ratios are ([\d.]+) and ([\d.]+)\. A division by the reporting count would give 0\.5 and 2\.0, and the same check measured ([\d.]+) and ([\d.]+)\.",
                   (R(1, TFF, m.format(3) + "m5/ratio_to_z_C_over_n"), R(2, TFF, m.format(3) + "m20/ratio_to_z_C_over_n"),
                    R(3, TFF, m.format(3) + "m5/ratio_to_z_C_over_m"), R(4, TFF, m.format(3) + "m20/ratio_to_z_C_over_m"))),
        ProseClaim("v169.F5", r"\(ratios ([\d.]+) and ([\d.]+) for equal and unequal weights\)",
                   (R(1, TFF, m.format(5) + "equal/ratio"), R(2, TFF, m.format(5) + "unequal/ratio"))),
        ProseClaim("v169.A4", r"The measured noise is ([\d.]+) and ([\d.]+) of the value for that clip, and would be ([\d.]+) and ([\d.]+) times too high",
                   (R(1, TFF, m.format(10) + "A4a/ratio_to_old_clip"), R(2, TFF, m.format(10) + "A4b/ratio_to_old_clip"),
                    R(3, TFF, m.format(10) + "A4a/mean_ratio_to_new_clip"), R(4, TFF, m.format(10) + "A4b/mean_ratio_to_new_clip"))),
        ProseClaim("v169.A7", r"is ([\d.]+) and ([\d.]+) of the configured one in the two configurations tried\. The delivered count noise standard deviation is ([\d.]+) against 5\.0 and ([\d.]+) against 0\.6",
                   (R(1, TFF, m.format(13) + "n100/composed"), R(2, TFF, m.format(13) + "n12/composed"),
                    R(3, TFF, m.format(11) + "A5a/S_std"), R(4, TFF, m.format(11) + "A5c/S_std"))),
        ProseClaim("v169.A6", r"With 50 of 100 clients all below the clip the fraction is ([\d.]+) and not 1",
                   (R(1, TFF, m.format(12) + "noise_free/m50_u1.0/recovered_fraction"),)),
        ProseClaim("v169.A8", r"\(zeroed count (\d+) in the test\)", (R(1, TFF, m.format(14) + "zeroing/on/zeroed_count"),)),
        ProseClaim("v169.T", r"taking ([\d.]+) seconds", (R(1, TFF, m.format(15) + "total_seconds"),)),
    ]


def _verdict_words() -> list[VerdictWord]:
    out = [VerdictWord(t, PROOFS, f"{t}/pass") for t in ("E1", "E2", "E3", "M1")]
    out += [VerdictWord(t, DP, f"{t}/pass") for t in ("D1", "D2", "D3", "D4", "D5")]
    out += [VerdictWord(t, FAULTS, f"{t}/pass") for t in ("F1", "F2")]
    out += [VerdictWord(t, EXTERNAL, f"checks/{t}/verdict") for t in ("B1", "B2")]
    out += [VerdictWord(t, PARITY, f"{t}/pass") for t in ("P1", "P2")]
    out += [VerdictWord(t, SCALING, f"{t}/pass") for t in ("S1", "S2", "S3")]
    out += [VerdictWord("X1", FASHION, "X1/pass")]
    return out


def _verdicts() -> list[Verdict]:
    key = "paired_rate_minus_label_floor_bootstrap95/1"
    return [Verdict("verdict.plaintext", "Plaintext update", 3, INTERVALS, f"views/plaintext_individual/{key}"),
            Verdict("verdict.aggregate", "SecAgg+ aggregate only", 3, INTERVALS, f"views/secagg_aggregate_maxw1000/{key}"),
            Verdict("verdict.collusion", "SecAgg+ with n-1 colluders", 3, INTERVALS, f"views/secagg_collusion_n-1_maxw1000/{key}")]


def _phrases() -> list[Phrase]:
    return [Phrase("verdict.dp_is_naive_only", "Opacus DP, sigma 0.2", 0, "naive attacker"),
            Phrase("verdict.dp_no_protection_claim", "Opacus DP, sigma 0.2", 3, "not a protection claim")]


def _lira_claims() -> list[Claim]:
    rows = [("Digits, overfit regime", "large_overfit"), ("Digits, moderate regime", "large_moderate"),
            ("MNIST, overfit regime", "large_mnist_overfit")]
    return [Claim(f"lira.{d}.mode{m}", label, m + 1, LIRA.format(d), f"modes/{m}/lira_auc/0")
            for label, d in rows for m in range(3)]


CLAIMS: list[Claim] = _view_claims() + _lira_claims()
PROSE: list[ProseClaim] = _prose_claims() + _v15_prose() + _tff_prose()
VERDICTS: list[Verdict] = _verdicts()
PHRASES: list[Phrase] = _phrases()
WORDS: list[VerdictWord] = _verdict_words()
# The README states one count per environment, for example "with flwr and opacus: 506 pass and 8 skip".
SUITE_README = {"full": re.compile(r"with flwr and opacus[^:\n]*: (\d+) pass and (\d+) skip"),
                "ci": re.compile(r"without flwr and opacus: (\d+) pass and (\d+) skip")}
# The closing line of a pytest run, for example "195 passed, 4 skipped, 4 deselected, 2 warnings in 20.30s".
SUITE_EVIDENCE = re.compile(r"^(?:=+ )?((?:\d+ (?:passed|failed|skipped|deselected|xfailed|xpassed|errors?|warnings?)(?:, )?)+) in [\d.]+s")
SUITE_PART = re.compile(r"(\d+) (passed|failed|skipped|errors?)")


def table_cell(readme: str, row: str, col: int) -> str:
    for line in readme.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and cells and cells[0].startswith(row):
            if col >= len(cells):
                raise KeyError(f"row '{row}' has no column {col}")
            return cells[col]
    raise KeyError(f"no README table row starts with '{row}'")


NUMBER = re.compile(r"-?\d+(?:\.(\d+))?(?:[eE]([-+]?\d+))?")


def parse_number(cell: str) -> tuple[float, int]:
    """First number in a cell and how many decimals it prints. For an exponent such as 1e-3 the
    decimals are counted at the exponent, so 1e-3 prints three decimals and 1.5e-3 prints four."""
    m = NUMBER.search(cell)
    if not m:
        raise ValueError(f"no number in '{cell}'")
    return float(m.group(0)), len(m.group(1) or "") - int(m.group(2) or 0)


def json_value(data, path: str) -> float:
    for part in path.split("/"):
        data = data[int(part)] if isinstance(data, list) else data[part]
    if isinstance(data, bool) or not isinstance(data, (int, float)):
        raise TypeError(f"value at '{path}' is not a number")
    return float(data)


def evidence_value(root: Path, file: str, path: str, scale: float = 1.0, minus_file: str = "", minus_path: str = "") -> float:
    """Raises FileNotFoundError, KeyError, IndexError, ValueError or TypeError when the evidence is absent or
    malformed, and ValueError when the value is not finite (NaN and infinity never pass a comparison)."""
    def one(f: str, pth: str) -> float:
        fp = root / f
        if not fp.is_file():
            raise FileNotFoundError(f"evidence file {f}")
        v = json_value(json.loads(fp.read_text()), pth)
        if not math.isfinite(v):
            raise ValueError(f"value at '{pth}' in {f} is not finite ({v})")
        return v
    value = one(file, path)
    if minus_path:
        value -= one(minus_file or file, minus_path)
    return value * scale


def compare(printed: float, decimals: int, value: float) -> tuple[str, str]:
    if not math.isfinite(value) or not math.isfinite(printed):
        return "MISSING", f"not a finite number (README {printed}, evidence {value})"
    tol = 0.5 * 10 ** -decimals + 1e-9
    if not abs(printed - value) <= tol:
        return "MISMATCH", f"README {printed} but evidence {value:.6g} (tolerance {tol:g})"
    return "OK", f"README {printed} evidence {value:.6g}"


def check_claim(readme: str, root: Path, c: Claim) -> tuple[str, str]:
    try:
        printed, decimals = parse_number(table_cell(readme, c.row, c.col))
    except (KeyError, ValueError) as e:
        return "MISSING", f"README: {e}"
    try:
        value = evidence_value(root, c.file, c.path, c.scale, c.minus_file, c.minus_path)
    except FileNotFoundError as e:
        return "MISSING", str(e)
    except (KeyError, IndexError, ValueError, TypeError) as e:
        return "MISSING", f"evidence path {c.path} in {c.file}: {e!r}"
    return compare(printed, decimals, value)


def check_prose(readme: str, root: Path, c: ProseClaim) -> tuple[str, str]:
    m = re.search(c.pattern, readme)
    if not m:
        return "MISSING", f"README prose does not match /{c.pattern}/"
    details = []
    for ref in c.refs:
        printed, decimals = parse_number(m.group(ref.group).replace(",", ""))     # "20,000" prints 20000
        try:
            value = evidence_value(root, ref.file, ref.path, ref.scale, ref.minus_file, ref.minus_path)
        except FileNotFoundError as e:
            return "MISSING", str(e)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            return "MISSING", f"evidence path {ref.path} in {ref.file}: {e!r}"
        status, detail = compare(printed, decimals, value)
        if status != "OK":
            return status, detail
        details.append(detail)
    return "OK", "; ".join(details)


def check_verdict(readme: str, root: Path, v: Verdict) -> tuple[str, str]:
    try:
        cell = table_cell(readme, v.row, v.col)
    except KeyError as e:
        return "MISSING", f"README: {e}"
    try:
        low = evidence_value(root, v.file, v.low_path)
    except FileNotFoundError as e:
        return "MISSING", str(e)
    except (KeyError, IndexError, ValueError, TypeError) as e:
        return "MISSING", f"evidence path {v.low_path} in {v.file}: {e!r}"
    says_above = re.search(r"\babove\b", cell) is not None
    says_not_beat = re.search(r"does not beat", cell) is not None
    if low > 0:
        ok, want = says_above and not says_not_beat, "above the floor"
    else:
        ok, want = says_not_beat and not says_above, "does not beat the floor"
    return ("OK", f"'{cell}' agrees with evidence ({want})") if ok else \
        ("MISMATCH", f"cell says '{cell}' but the paired interval (lower end {low:.4g}) means '{want}'")


def check_word(readme: str, root: Path, w: VerdictWord) -> tuple[str, str]:
    fp = root / w.file
    if not fp.is_file():
        return "MISSING", f"evidence file {w.file}"
    try:
        data = json.loads(fp.read_text())
        for part in w.path.split("/"):
            data = data[int(part)] if isinstance(data, list) else data[part]
    except (KeyError, IndexError, ValueError, TypeError) as e:
        return "MISSING", f"evidence path {w.path} in {w.file}: {e!r}"
    if isinstance(data, bool):
        passed = data
    elif data in ("PASS", "FAIL"):
        passed = data == "PASS"
    else:
        return "MISSING", f"value at {w.path} in {w.file} is neither a boolean nor PASS or FAIL: {data!r}"
    want, other = ("PASS", "FAIL") if passed else ("FAIL", "PASS")
    says_want = re.search(rf"\b{w.token} {want}\b", readme) is not None
    says_other = re.search(rf"\b{w.token} {other}\b", readme) is not None
    if says_want and not says_other:
        return "OK", f"README says '{w.token} {want}' as the evidence does"
    return "MISMATCH", f"evidence says {want} but README {'says ' + w.token + ' ' + other if says_other else 'does not say ' + w.token + ' ' + want}"


def check_phrase(readme: str, p: Phrase) -> tuple[str, str]:
    try:
        cell = table_cell(readme, p.row, p.col)
    except KeyError as e:
        return "MISSING", f"README: {e}"
    return ("OK", f"'{p.phrase}' present") if p.phrase in cell else ("MISMATCH", f"cell '{cell}' lacks '{p.phrase}'")


def parse_suite_line(text: str) -> dict | None:
    """Counts from the last real pytest closing line in the text, or None. Other lines are ignored."""
    last = None
    for ln in text.splitlines():
        m = SUITE_EVIDENCE.match(ln.strip())
        if m:
            last = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
            for n, word in SUITE_PART.findall(m.group(1)):
                last["error" if word.startswith("error") else word] = int(n)
    return last


def _version_key(path: Path) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", path.parent.name))


def newest_record(root: Path, name: str) -> Path | None:
    """The record `name` in the newest reports/results/privacy/v* directory that has one."""
    found = sorted((root / "reports/results/privacy").glob(f"v*/{name}"), key=_version_key)
    return found[-1] if found else None


def record_fields(text: str) -> dict[str, str]:
    fields = {}
    for ln in text.splitlines():
        m = re.match(r"^(date|commit|code_dirty|tree|python|packages): (.*)$", ln.strip())
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def staleness(root: Path, fields: dict[str, str]) -> tuple[str, str]:
    """OK when no code path changed since the recorded commit. MISMATCH when one did, or when the record was
    made with uncommitted code changes. MISSING when the commit is absent or git cannot answer (fail closed)."""
    commit = fields.get("commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return "MISSING", "record has no git commit line, so it cannot be compared with the code (rerun scripts/record_fast_suite.py)"
    if fields.get("code_dirty") != "no":
        return "MISMATCH", "record was made with uncommitted code changes"
    try:
        changed = _git(root, "diff", "--name-only", commit, "--", *CODE_PATHS).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        return "MISSING", f"git cannot compare {commit[:10]} with the working tree: {e!r}"
    if changed:
        more = "" if len(changed) <= 3 else f" and {len(changed) - 3} more"
        return "MISMATCH", f"code changed after the record (commit {commit[:10]}): {', '.join(changed[:3])}{more}. Rerun scripts/record_fast_suite.py"
    return "OK", f"no code change since {commit[:10]}"


def check_suite_line(readme: str, root: Path, kind: str = "full") -> tuple[str, str]:
    m = SUITE_README[kind].search(readme)
    if not m:
        return "MISSING", f"README quick start has no '{SUITE_README[kind].pattern}' line"
    f = newest_record(root, SUITE_FILES[kind])
    if f is None:
        return "MISSING", f"no {SUITE_FILES[kind]} under reports/results/privacy/v* (write it with scripts/record_fast_suite.py --kind {kind})"
    text = f.read_text()
    got = parse_suite_line(text)
    if got is None:
        return "MISSING", f"no pytest closing line in {f.relative_to(root)}"
    if got["failed"] or got["error"]:
        return "MISMATCH", f"saved pytest line has failures or errors: {got}"
    want = (int(m.group(1)), int(m.group(2)))
    have = (got["passed"], got["skipped"])
    if have != want:
        return "MISMATCH", f"README {want} but saved pytest line {have} in {f.relative_to(root)}"
    fields = record_fields(text)
    pk = fields.get("packages", "")
    absent = {n: re.search(rf"\b{n} absent\b", pk) is not None for n in ("flwr", "opacus")}
    present = {n: re.search(rf"\b{n} \d", pk) is not None for n in ("flwr", "opacus")}
    if kind == "ci" and not (all(absent.values()) and fields.get("tree", "").startswith("clean checkout")):
        return "MISMATCH", f"the record is not a clean checkout without flwr and opacus (tree '{fields.get('tree')}', packages '{pk}')"
    if kind == "full" and not all(present.values()):
        return "MISMATCH", f"the record was not made with flwr and opacus installed (packages '{pk}')"
    status, detail = staleness(root, fields)
    return (status, detail) if status != "OK" else ("OK", f"{want} in {f.relative_to(root)}, {detail}")


def check_guarantee_rows_held(readme: str, root: Path) -> tuple[str, str]:
    """Every row of the adaptive guarantee evidence held its prediction in both versions, and the README says how many rows."""
    m = re.search(r"All (\d+) rows per version held their predictions", readme)
    if not m:
        return "MISSING", "README does not say how many guarantee rows held"
    want = int(m.group(1))
    for version, rel in GUARANTEE.items():
        fp = root / rel
        if not fp.is_file():
            return "MISSING", f"evidence file {rel}"
        rows = json.loads(fp.read_text())["rows"]
        failed = [r["row"] for r in rows if not r["prediction_held"]]
        if len(rows) != want or failed:
            return "MISMATCH", f"{version}: README says {want} rows held, evidence has {len(rows)} rows and {len(failed)} that did not hold"
    cp = root / GUARANTEE_DIR / "cross_version" / "cross_version.json"
    if not cp.is_file() or not json.loads(cp.read_text()).get("prediction_held"):
        return "MISMATCH", "the cross version comparison is absent or did not hold"
    return "OK", f"{want} rows held in both versions and the cross version comparison held"


def check_tff_failures(readme: str, root: Path) -> tuple[str, str]:
    """The TFF evidence has exactly one failed check, in row F5, and the README calls it an arithmetic slip."""
    fp = root / TFF
    if not fp.is_file():
        return "MISSING", f"evidence file {TFF}"
    s = json.loads(fp.read_text())["summary"]
    failed = s["checks_failed"]
    if s["rows_failed"] != ["F5"] or len(failed) != 1 or failed[0]["row"] != "F5":
        return "MISMATCH", f"expected one failed check in F5, evidence has rows {s['rows_failed']} and {len(failed)} failed checks"
    if "arithmetic slip" not in readme or s["mismatch_found_rows"]:
        return "MISMATCH", "README does not call the failed check an arithmetic slip, or a mismatch was found"
    return "OK", "one failed check, in F5, and no delivered value mismatch"


def verify(readme_path: Path, root: Path, claims: list[Claim] | None = None, suite: bool = True,
           prose: list[ProseClaim] | None = None, verdicts: list[Verdict] | None = None,
           phrases: list[Phrase] | None = None, words: list[VerdictWord] | None = None) -> list[tuple[str, str, str]]:
    if not readme_path.is_file():
        return [("README", "MISSING", str(readme_path))]
    readme = readme_path.read_text()
    results = [(c.id, *check_claim(readme, root, c)) for c in (CLAIMS if claims is None else claims)]
    results += [(c.id, *check_prose(readme, root, c)) for c in (PROSE if prose is None else prose)]
    results += [(c.id, *check_verdict(readme, root, c)) for c in (VERDICTS if verdicts is None else verdicts)]
    results += [(c.id, *check_phrase(readme, c)) for c in (PHRASES if phrases is None else phrases)]
    results += [(f"word.{w.token}", *check_word(readme, root, w)) for w in (WORDS if words is None else words)]
    if prose is None:
        results.append(("v166.rows_held", *check_guarantee_rows_held(readme, root)))
        results.append(("v169.failures", *check_tff_failures(readme, root)))
    if suite:
        results += [(f"tests.fast_suite_count.{k}", *check_suite_line(readme, root, k)) for k in SUITE_FILES]
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--readme", type=Path, default=ROOT / "README.md")
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args(argv)
    results = verify(a.readme, a.root)
    for cid, status, detail in results:
        print(f"{status:9s} {cid}: {detail}")
    bad = [r for r in results if r[1] != "OK"]
    print(f"{len(results) - len(bad)} of {len(results)} README claims verified")
    return 1 if bad or not results else 0


if __name__ == "__main__":
    sys.exit(main())
