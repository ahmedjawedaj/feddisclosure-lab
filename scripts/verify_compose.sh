#!/usr/bin/env bash
# Run on a Docker Engine + Compose v2 host only; saves honest infrastructure proof.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v docker >/dev/null 2>&1; then
  echo 'Docker Engine/Compose required. This gate is not verified.' >&2
  exit 2
fi
mkdir -p reports/results/sprint1
CREATED_ENV=0
if [[ ! -f .env ]]; then
  touch .env
  CREATED_ENV=1
fi
export FDL_INTERNAL_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
export FDL_RESEARCH_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
export FDL_MIN_CELL=5
export COMPOSE_PROJECT_NAME="fdlverify$(python -c 'import secrets;print(secrets.token_hex(3))')"
cleanup() {
  docker compose logs --no-color > reports/results/sprint1/docker_services.txt 2>&1 || true
  docker compose down -v --remove-orphans || true
  if [[ "$CREATED_ENV" == "1" ]]; then rm -f .env; fi
}
trap cleanup EXIT
docker compose config -q
docker compose up -d --build --wait
python scripts/verify_compose_isolation.py --output reports/results/sprint1/docker_isolation.json
python -m feddisclosure.cli analytics-benchmark --gateway http://127.0.0.1:8080 \
  --output reports/results/sprint1/docker_analytics.json
