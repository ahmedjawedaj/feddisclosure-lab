#!/usr/bin/env bash
# Replicates of the 2-round, 4-client SecAgg+ full-participation run (stochastic rounding differs per run).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
V=${1:-reports/results/secagg/v014_n4_replicates}
ENVD=${ENV_DIR:-/home/claude/runs}
mkdir -p "$V" "$ENVD"
for i in 1 2 3; do
  name=rep$i; [ -f "$V/$name.exit" ] && continue
  mkdir -p "$V/$name"
  python scripts/run_flower_docker.py --config configs/training/fedavg_digits_iid_4clients.yaml \
    --data-dir data/training/digits-docker-iid-4c-v014 --run-dir "$V/$name" --env-file "$ENVD/secagg-rep$i.env" \
    --app-dir experiments/flower_secagg --no-require-evaluation \
    --extra-run-config "use-secagg=true num-shares=4 reconstruction-threshold=2 timeout=60.0" --timeout 400 \
    > "$V/$name.stdout.log" 2> "$V/$name.stderr.log"
  echo "exit=$?" > "$V/$name.exit"; docker system prune -f >/dev/null 2>&1
done
echo done > "$V/replicates.done"
