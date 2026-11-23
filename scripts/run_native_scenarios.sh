#!/usr/bin/env bash
# Sequential native Flower scenarios. Every scenario uses a FRESH run directory and
# env file; failures are expected (and recorded) for the fault-injection scenarios.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
V=${1:-reports/results/native_flower/v014}
ENVD=${ENV_DIR:-$HOME/runs}
mkdir -p "$ENVD"
run() { # name config datadir [extra args...]
  local name=$1 cfg=$2 data=$3; shift 3
  mkdir -p "$V/$name"
  python scripts/run_flower_docker.py --config "$cfg" --data-dir "$data" \
    --run-dir "$V/$name" --env-file "$ENVD/$name.env" "$@" \
    > "$V/$name.stdout.log" 2> "$V/$name.stderr.log"
  echo "exit=$?" > "$V/$name.exit"
}
ref() { # name config datadir
  python scripts/run_torch_reference.py --config "$2" --data-dir "$3" --output "$V/$1/torch_reference.json" > "$V/$1/torch_reference.stdout.log" 2>&1
  python scripts/verify_flower_parity.py --flower-dir "$V/$1/server" \
    --reference-json "$V/$1/torch_reference.json" \
    --reference-checkpoint "$V/$1/checkpoints/$(python - "$2" <<'PY'
import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["id"])
PY
)_final.pt" --manifest "$3/manifest.json" --output "$V/$1/flower_parity.json" --require-deployment \
    > "$V/$1/flower_parity.stdout.log" 2>&1
  echo "parity_exit=$?" >> "$V/$1.exit"
}
IID=configs/training/fedavg_digits_iid.yaml
NON=configs/training/fedavg_digits_non_iid.yaml
IDATA=data/training/digits-docker-iid-v012
NDATA=data/training/digits-docker-noniid-v014
run run-iid-repeat      $IID $IDATA;  ref run-iid-repeat $IID $IDATA
run run-noniid-labelskew $NON $NDATA; ref run-noniid-labelskew $NON $NDATA
run fault-before-run    $IID $IDATA --fault-stop-client 2 --fault-when before-run --timeout 150
run fault-after-round1  $IID $IDATA --fault-stop-client 1 --fault-when after-first-train --timeout 180
echo done > "$V/scenarios.done"
