"""Selected exploratory paired comparisons, no claims of causality or adequate power."""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def analyze(path: Path) -> dict:
    source = json.loads(path.read_text())
    if sorted(source['aggregate']['completed_seeds']) != [241,242,243,244,245]:
        raise ValueError('Five matched seeds required')
    conditions = ('iid', 'dirichlet_balanced_0.1')
    results = {}
    for row in source['results']:
        key = (row['seed'], row['partition'])
        if key in results:
            raise ValueError('Duplicate training seed/condition')
        results[key] = row
    comparisons = {}
    for metric in ('test_accuracy', 'worst_client_accuracy'):
        values = []
        for seed in range(241,246):
            baseline = results[(seed,conditions[0])]
            shifted = results[(seed,conditions[1])]
            if baseline['test_cohort_sha256'] != shifted['test_cohort_sha256']:
                raise ValueError('Test cohort mismatch in paired seed')
            equal_budget_keys = ('train_samples_per_client', 'val_samples_per_client',
                                 'local_optimizer_batches_per_round_per_client')
            if any(row['metrics'][key] != baseline['metrics'][key]
                   for row in (baseline, shifted) for key in equal_budget_keys):
                raise ValueError('Client-size or optimizer-budget mismatch')
            values.append(100.0 * (shifted['metrics'][metric] - baseline['metrics'][metric]))
        x = np.asarray(values, dtype=float)
        mean, sd = float(x.mean()), float(x.std(ddof=1))
        margin = float(t.ppf(.975, df=4) * sd / np.sqrt(len(x)))
        # Exact sign-flip permutation test on matched seed differences.
        null = [abs(float(np.mean(x*np.array(signs))))
                for signs in product((-1,1),repeat=len(x))]
        p = float(sum(v >= abs(mean)-1e-10 for v in null) / len(null))
        comparisons[metric] = {'paired_difference_pp_by_seed': x.tolist(),
                               'mean_difference_pp': mean,
                               'sample_sd_pp': sd,
                               't_interval_95_pp': [mean-margin,mean+margin],
                               'exact_sign_flip_two_sided_p': p,
                               'n_paired_seeds': 5}
    return {'schema_version': 1, 'protocol': 'IID vs Dirichlet alpha=0.1 (selected exploratory contrast, equal-work)',
            'contrast_order': 'alpha_0.1_minus_iid',
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'comparisons': comparisons,
            'limitations': ['Exploratory paired comparisons; only five seeds and weak four-round model',
                            '95% t intervals require exchangeable independent seeds and approximate normality of paired differences',
                            'Sign-flip test assumes differences are symmetric under the null; limited resolution at n=5',
                            'Two outcomes examined; no multiplicity correction or causal claim',
                            'No formal DP or SecAgg+ guarantees apply']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=ROOT / 'reports/results/sprint5/balanced_heterogeneity/summary.json')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/results/sprint5/balanced_heterogeneity/paired_statistics.json')
    a=p.parse_args()
    out=analyze(a.input)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out['comparisons'], indent=2))
