#!/usr/bin/env bash
# SecAgg+ mid-round dropout matrix in the real Docker Deployment Runtime.
# Victims are SIGKILLed during round 2, after every client finished key sharing.
# Prediction from upstream share counting (shares for a dropped client come from surviving
# holders): tolerated iff surviving holders >= reconstruction threshold. Upstream also requires
# threshold < num_shares. These runs are NOT certified as full participation (a killed client
# misses receipts). The protocol outcome (model written, rounds aggregated) is what is recorded.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
V=${1:-reports/results/secagg/v014_dropout_matrix}
ENVD=${ENV_DIR:-/home/claude/runs}
mkdir -p "$V" "$ENVD"
APP=experiments/flower_secagg
run() { # name cfg data n t victim-args...
  local name=$1 cfg=$2 data=$3 n=$4 t=$5; shift 5
  [ -f "$V/$name.exit" ] && return
  mkdir -p "$V/$name"
  python scripts/run_flower_docker.py --config $cfg --data-dir $data --run-dir "$V/$name" \
    --env-file "$ENVD/secagg-dm-$name.env" --app-dir $APP --no-require-evaluation \
    --extra-run-config "use-secagg=true num-shares=$n reconstruction-threshold=$t timeout=30.0" \
    "$@" --fault-when during-round-2 --no-fail-fast --timeout 400 \
    > "$V/$name.stdout.log" 2> "$V/$name.stderr.log"
  echo "exit=$?" > "$V/$name.exit"
  docker system prune -f >/dev/null 2>&1
}
C4=configs/training/fedavg_digits_iid_4clients.yaml; D4=data/training/digits-docker-iid-4c-v014
C3=configs/training/fedavg_digits_iid.yaml; D3=data/training/digits-docker-iid-v012
run n4-t2-kill1-r2 $C4 $D4 4 2 --fault-stop-client 3
run n4-t3-kill1-r2 $C4 $D4 4 3 --fault-stop-client 3
run n4-t2-kill2-r2 $C4 $D4 4 2 --fault-stop-client 2 --fault-stop-client 3
run n4-t3-kill2-r2 $C4 $D4 4 3 --fault-stop-client 2 --fault-stop-client 3
run n3-t2-kill1-r2 $C3 $D3 3 2 --fault-stop-client 2
# Plaintext control for n=4 (the earlier one omitted num-shares and was refused by the server guard)
if [ ! -f "$V/n4-plaintext-control.exit" ]; then
  mkdir -p "$V/n4-plaintext-control"
  python scripts/run_flower_docker.py --config $C4 --data-dir $D4 --run-dir "$V/n4-plaintext-control" \
    --env-file "$ENVD/secagg-dm-n4-plaintext.env" --app-dir $APP --no-require-evaluation \
    --extra-run-config "use-secagg=false num-shares=4 reconstruction-threshold=2" --timeout 500 \
    > "$V/n4-plaintext-control.stdout.log" 2> "$V/n4-plaintext-control.stderr.log"
  echo "exit=$?" > "$V/n4-plaintext-control.exit"
fi
# Single round, 4 clients: the quantisation bound applies directly (no round-to-round amplification)
if [ ! -f "$V/n4-1round-secagg.exit" ]; then
  mkdir -p "$V/n4-1round-secagg"
  python scripts/run_flower_docker.py --config configs/training/fedavg_digits_iid_4clients_1round.yaml --data-dir $D4 \
    --run-dir "$V/n4-1round-secagg" --env-file "$ENVD/secagg-dm-n4-1round.env" --app-dir $APP --no-require-evaluation \
    --extra-run-config "use-secagg=true num-shares=4 reconstruction-threshold=2 timeout=60.0" --timeout 400 \
    > "$V/n4-1round-secagg.stdout.log" 2> "$V/n4-1round-secagg.stderr.log"
  echo "exit=$?" > "$V/n4-1round-secagg.exit"
fi
echo done > "$V/scenarios.done"
