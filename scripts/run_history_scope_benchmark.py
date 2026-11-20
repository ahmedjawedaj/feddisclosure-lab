"""Real three-institution HTTP benchmark of experiment-ID reset and release-policy lock.

Two isolated stacks with fresh databases. The attacker receives ONLY the
published federated answers, never institution rows or evaluator ground truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from feddisclosure.attacks.aggregate_difference import ObservedCount, infer_restricted_count
from scripts.run_local_stack import free_port, wait_ready


def run_policy(policy: str) -> dict:
    ports = [free_port() for _ in range(4)]
    processes, log_handles = [], []
    with tempfile.TemporaryDirectory(prefix=f'fdl-history-{policy}-') as folder:
        scratch = Path(folder)
        token, key = 'research-http-internal-token', 'research-http-key'
        try:
            for i, (name, n) in enumerate((('institution-a', 40), ('institution-b', 35), ('institution-c', 35))):
                env = dict(os.environ, FDL_INSTITUTION=name, FDL_ROWS=str(n), FDL_CASES='1',
                           FDL_DB_PATH=str(scratch / f'{name}.db'), FDL_MIN_CELL='5',
                           FDL_INTERNAL_TOKEN=token, FDL_RELEASE_POLICY=policy,
                           FDL_DP_ENABLED='0', PYTHONPATH=f'{ROOT / "src"}:{ROOT}')
                log = (scratch / f'{name}.log').open('w')
                log_handles.append(log)
                processes.append(subprocess.Popen([
                    sys.executable, '-m', 'uvicorn', 'services.client.app:app',
                    '--host', '127.0.0.1', '--port', str(ports[i]), '--no-access-log'],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
                wait_ready(f'http://127.0.0.1:{ports[i]}', processes)
            env = dict(os.environ, FDL_CLIENT_URLS=','.join(f'http://127.0.0.1:{p}' for p in ports[:3]),
                       FDL_INTERNAL_TOKEN=token, FDL_RESEARCH_KEY=key, FDL_RELEASE_POLICY=policy,
                       FDL_DP_ENABLED='0', PYTHONPATH=f'{ROOT / "src"}:{ROOT}')
            log = (scratch / 'gateway.log').open('w')
            log_handles.append(log)
            processes.append(subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 'services.gateway.app:app',
                '--host', '127.0.0.1', '--port', str(ports[3]), '--no-access-log'],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
            gateway = f'http://127.0.0.1:{ports[3]}'
            wait_ready(gateway, processes)
            observations, events = [], []
            sequence = [('all_records', 'experiment_a', policy),
                        ('negative_records', 'experiment_b', policy),
                        ('all_records', 'experiment_c', policy),
                        ('negative_records', 'experiment_d', 'naive' if policy == 'principal_history' else policy)]
            with httpx.Client(timeout=12) as client:
                for query, eid, selected_policy in sequence:
                    t0 = time.perf_counter()
                    response = client.post(f'{gateway}/v1/federated-count',
                        json={'query_id': query, 'experiment_id': eid, 'policy': selected_policy},
                        headers={'X-Research-Key': key})
                    if response.status_code not in (200, 403, 409):
                        raise RuntimeError(f'HTTP {response.status_code}: {response.text}')
                    count = response.json().get('count') if response.status_code == 200 else None
                    if count is not None:
                        observations.append(ObservedCount(query, count))
                    events.append({'query_id': query, 'experiment_id': eid, 'requested_policy': selected_policy,
                                   'http_status': response.status_code, 'released_count': count,
                                   'elapsed_ms': round((time.perf_counter()-t0)*1000, 3)})
            inferred = infer_restricted_count(observations)
            return {'policy_enforced_at_site_and_gateway': policy,
                    'events': events, 'attacker_inferred_restricted_count': inferred,
                    'evaluator_only': {'true_restricted_count': 3, 'correct_inference': inferred == 3},
                    'counts_released': sum(e['http_status'] == 200 for e in events),
                    'rejections': sum(e['http_status'] != 200 for e in events)}
        except Exception:
            for h in log_handles:
                h.flush()
            for logfile in scratch.glob('*.log'):
                print(f'--- {logfile.name} ---\n{logfile.read_text()}', file=sys.stderr)
            raise
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            for proc in processes:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
            for h in log_handles:
                h.close()


def benchmark(output: Path) -> dict:
    weak = run_policy('history')
    strong = run_policy('principal_history')
    if [e['http_status'] for e in weak['events'][:2]] != [200, 200] or weak['attacker_inferred_restricted_count'] != 3:
        raise RuntimeError('History reset bypass was not reproduced')
    if [e['http_status'] for e in strong['events']] != [200, 409, 200, 403]:
        raise RuntimeError('Protected history or policy enforcement failed')
    if strong['attacker_inferred_restricted_count'] is not None:
        raise RuntimeError('Protected site revealed inferable restricted count')
    result = {'schema_version': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'experiment': 'experiment_id_reset_vs_principal_history_three_site_http',
              'observer': 'authenticated_analytics_user_gateway_aggregate_answers_only',
              'scenarios': {'experiment_scoped_history': weak, 'principal_wide_history': strong},
              'repeated_query_allowed': strong['events'][2]['http_status'] == 200,
              'limitations': [
                  'One shared authenticated research principal; no production user-account separation',
                  'Two-query synthetic disclosure only, NOT a proof of statistical disclosure control',
                  'Other queries, side information, cross-policy releases, revoked tokens and repeated snapshots remain outside the tested guarantee',
                  'Per-site query history uses private membership IDs locally; never returned to the gateway',
                  'No differential privacy mechanism and no secure aggregation in this benchmark',
              ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/results/sprint1/history_scope_http.json'))
    args = parser.parse_args()
    result = benchmark(args.output if args.output.is_absolute() else ROOT / args.output)
    for name, scenario in result['scenarios'].items():
        print(f'{name}: statuses={[e["http_status"] for e in scenario["events"]]}, '
              f'inference={scenario["attacker_inferred_restricted_count"]}')
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
