#!/usr/bin/env bash
# SecAgg+ with FOUR institutions in the real Docker Deployment Runtime.
# Question: does the dropout outcome depend on the reconstruction threshold as the
# share-counting logic suggests? Each outcome is recorded whatever it is.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
V=${1:-reports/results/secagg/v014_n4}
ENVD=${ENV_DIR:-/home/claude/runs}
mkdir -p "$V" "$ENVD"
CFG=configs/training/fedavg_digits_iid_4clients.yaml
DATA=data/training/digits-docker-iid-4c-v014
APP=experiments/flower_secagg
run() { # name extra-run-config [runner args...]
  local name=$1 extra=$2; shift 2
  mkdir -p "$V/$name"
  python scripts/run_flower_docker.py --config $CFG --data-dir $DATA --run-dir "$V/$name" \
    --env-file "$ENVD/secagg4-$name.env" --app-dir $APP --no-require-evaluation \
    --extra-run-config "$extra" "$@" > "$V/$name.stdout.log" 2> "$V/$name.stderr.log"
  echo "exit=$?" > "$V/$name.exit"
  docker system prune -f >/dev/null 2>&1
}
run n4-t2-baseline "use-secagg=true num-shares=4 reconstruction-threshold=2 timeout=60.0" --timeout 500
run n4-plaintext-control "use-secagg=false" --timeout 500
run n4-t2-dropout-mid "use-secagg=true num-shares=4 reconstruction-threshold=2 timeout=30.0" \
    --fault-stop-client 3 --fault-when after-first-train --timeout 400
run n4-t3-dropout-mid "use-secagg=true num-shares=4 reconstruction-threshold=3 timeout=30.0" \
    --fault-stop-client 3 --fault-when after-first-train --timeout 400
echo done > "$V/scenarios.done"
