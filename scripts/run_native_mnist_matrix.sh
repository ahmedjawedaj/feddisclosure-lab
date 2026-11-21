#!/usr/bin/env bash
# Native Docker + Flower full-MNIST runs (non-IID and extra IID seeds), each with an independent
# PyTorch reference and strict parity. Fail-fast is on, so a lost participant stops the run at once.
set -u
cd "$(dirname "$0")/.."
source ${VENV:-/home/claude/venv}/bin/activate
export PYTHONPATH=src:.
V=reports/results/native_flower/v014
ENVD=${ENV_DIR:-/home/claude/runs}
run() { # name config-file data-dir
  local N=$1 CFG=$2 DATA=$3 ID
  ID=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['id'])")
  [ -f "$V/$N.exit" ] && return
  mkdir -p "$V/$N"
  python scripts/run_flower_docker.py --config $CFG --data-dir $DATA --run-dir $V/$N --env-file $ENVD/$N.env \
    --timeout 1500 > $V/$N.stdout.log 2> $V/$N.stderr.log
  echo "exit=$?" > $V/$N.exit
  python scripts/run_torch_reference.py --config $CFG --data-dir $DATA --output $V/$N/torch_reference.json \
    > $V/$N/torch_reference.stdout.log 2>&1
  python scripts/verify_flower_parity.py --flower-dir $V/$N/server --reference-json $V/$N/torch_reference.json \
    --reference-checkpoint $V/$N/checkpoints/${ID}_final.pt --manifest $DATA/manifest.json \
    --output $V/$N/flower_parity.json --require-deployment > $V/$N/flower_parity.stdout.log 2>&1
  echo "parity_exit=$?" >> $V/$N.exit
  docker system prune -f >/dev/null 2>&1
}
run run-mnist-full-labelskew configs/training/fedavg_mnist_full_labelskew_native.yaml data/training/mnist-full-docker-labelskew-v014
run run-mnist-full-iid-seed2 configs/training/fedavg_mnist_full_native_seed2.yaml data/training/mnist-full-docker-iid-seed2-v014
run run-mnist-full-iid-seed3 configs/training/fedavg_mnist_full_native_seed3.yaml data/training/mnist-full-docker-iid-seed3-v014
echo done > $V/mnist_matrix.done
