#!/usr/bin/env bash
# Upstream Flower SecAgg+ (SecAggPlusWorkflow + secaggplus_mod) in the real Docker
# Deployment Runtime. Fresh run dir per scenario. Fault scenarios are EXPECTED to be
# judged by what the protocol actually did; every outcome is recorded.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
V=${1:-reports/results/secagg/v014}
ENVD=${ENV_DIR:-$HOME/runs}
mkdir -p "$V" "$ENVD"
IID=configs/training/fedavg_digits_iid.yaml
IDATA=data/training/digits-docker-iid-v012
APP=experiments/flower_secagg
run() { # name extra-run-config [runner args...]
  local name=$1 extra=$2; shift 2
  mkdir -p "$V/$name"
  python scripts/run_flower_docker.py --config ${CFG:-$IID} --data-dir $IDATA --run-dir "$V/$name" \
    --env-file "$ENVD/secagg-$name.env" --app-dir $APP --no-require-evaluation \
    --extra-run-config "$extra" "$@" > "$V/$name.stdout.log" 2> "$V/$name.stderr.log"
  echo "exit=$?" > "$V/$name.exit"
}
run secagg-on-t2-n3   "use-secagg=true num-shares=3 reconstruction-threshold=2 timeout=60.0" --timeout 400
run plaintext-control "use-secagg=false" --timeout 400
CFG=configs/training/fedavg_digits_iid_1round.yaml run secagg-1round-collusion "use-secagg=true num-shares=3 reconstruction-threshold=2 timeout=60.0" --timeout 300
run secagg-dropout-before-run "use-secagg=true num-shares=3 reconstruction-threshold=2 timeout=30.0" \
    --fault-stop-client 2 --fault-when before-run --timeout 300
run secagg-dropout-mid-protocol "use-secagg=true num-shares=3 reconstruction-threshold=2 timeout=30.0" \
    --fault-stop-client 1 --fault-when after-first-train --timeout 300
echo done > "$V/scenarios.done"
