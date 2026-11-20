#!/usr/bin/env bash
# Native Docker acceptance gate: persistent principal-history controls.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v docker >/dev/null 2>&1; then
  echo 'Docker is unavailable; principal-history Compose gate NOT verified.' >&2
  exit 2
fi
created_env=0
if [[ ! -e .env ]]; then touch .env; created_env=1; fi
export FDL_INTERNAL_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export FDL_RESEARCH_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export FDL_MIN_CELL=5
export COMPOSE_PROJECT_NAME="fdlhistory$(python -c 'import secrets; print(secrets.token_hex(3))')"
mkdir -p reports/results/sprint1
cleanup() {
  docker compose -f compose.yaml -f compose.history.yaml logs --no-color > reports/results/sprint1/docker_history_services.log 2>&1 || true
  docker compose -f compose.yaml -f compose.history.yaml down --volumes --remove-orphans || true
  [[ "$created_env" == 0 ]] || rm -f .env
}
trap cleanup EXIT
docker compose -f compose.yaml -f compose.history.yaml config -q
docker compose -f compose.yaml -f compose.history.yaml up --build -d --wait
python - <<'PY'
import json, os, httpx
url='http://127.0.0.1:8080/v1/federated-count'
seq=[('all_records','a','principal_history'),
     ('negative_records','b','principal_history'),
     ('all_records','c','principal_history'),
     ('negative_records','d','naive')]
observed=[]
with httpx.Client(timeout=12) as c:
    for query, eid, mode in seq:
        r=c.post(url, headers={'X-Research-Key': os.environ['FDL_RESEARCH_KEY']},
                 json={'query_id': query, 'experiment_id': eid, 'policy': mode})
        if r.status_code not in (200, 403, 409):
            raise RuntimeError(f'Unexpected HTTP {r.status_code} {r.text}')
        observed.append({'query_id':query,'experiment_id':eid,'policy':mode,
                         'status':r.status_code,'released_count':r.json().get('count') if r.status_code==200 else None})
assert [r['status'] for r in observed]==[200,409,200,403], observed
assert observed[0]['released_count']==110
assert observed[2]['released_count']==110
result={'backend':'native_docker_compose_with_principal_history','cases':observed,
        'limitations':['One synthetic principal and two-query attack','NOT differential privacy']}
with open('reports/results/sprint1/docker_principal_history.json','w') as f:
    json.dump(result,f,indent=2)
print('Native Docker history control accepted: statuses [200,409,200,403]')
PY
